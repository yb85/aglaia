"""Generate landing-page + doc illustration assets.

LOCAL BUILD TOOL — never run in CI. The committed assets under
``site/public/landing`` and ``docs/figures`` are the source of truth for the
site build; this script only regenerates them on demand.

Every illustration is derived from the three bundled example projects
(``test_data/test_{athanase,augustin,balthasar}/*.agl``). Those are committed
SLIM (raw + final pages only), so to rebuild the algorithm overlays — layout
boxes, column quad, sheet grid, binarized page — this script COPIES each
project to a temp dir, reprocesses it through the current ``book_curved_x2``
pipeline to restore every intermediate step image (full coverage), renders
from that copy, then deletes it. The reprocess is heavy (full dewarp on every
page) — fine for a local tool, never appropriate for CI.

    PYTHONPATH=. uv run python site/scripts/gen-landing-assets.py
"""
import base64
import io
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "site" / "public" / "landing"
OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
ATH = ROOT / "test_data" / "test_athanase" / "test-athanase.agl"
AUG = ROOT / "test_data" / "test_augustin" / "test-augustin-confessions-vii.agl"
BAL = ROOT / "test_data" / "test_balthasar" / "test-balthasar.agl"


# ── image helpers ────────────────────────────────────────────────────
def _img(blob: bytes) -> Image.Image:
    return Image.open(io.BytesIO(blob)).convert("RGB")


def fitw(im: Image.Image, w: int) -> Image.Image:
    return im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)


def fith(im: Image.Image, h: int) -> Image.Image:
    return im.resize((round(im.width * h / im.height), h), Image.LANCZOS)


def hstack(left: Image.Image, right: Image.Image, gut: int = 18) -> Image.Image:
    h = max(left.height, right.height)
    out = Image.new("RGB", (left.width + gut + right.width, h), "white")
    out.paste(left, (0, 0))
    out.paste(right, (left.width + gut, 0))
    return out


def crop_frac(im: Image.Image, left: float, top: float,
              right: float, bottom: float) -> Image.Image:
    w, h = im.size
    return im.crop((round(left * w), round(top * h),
                    round(right * w), round(bottom * h)))


def debanner(im: Image.Image) -> Image.Image:
    """Crop the dark debug header strip the inspector overlays carry: the
    consecutive near-black rows at the very top (capped at 8 % of height so a
    dark book edge can't eat the page)."""
    g = np.asarray(im.convert("L"))
    cut = 0
    for y in range(int(g.shape[0] * 0.08)):
        if g[y].mean() < 80:
            cut = y + 1
        else:
            break
    return im.crop((0, cut, im.width, im.height)) if cut else im


def save(im: Image.Image, name: str, q: int = 80) -> None:
    """Write a landing asset as WebP. PNG names → lossless; else lossy q≈80."""
    stem = name.rsplit(".", 1)[0]
    lossless = name.endswith(".png")
    p = OUT / f"{stem}.webp"
    im.save(p, format="WEBP", method=6,
            lossless=lossless, quality=(100 if lossless else q))
    print(p.name, im.size, f"{p.stat().st_size // 1024} KB")


def docfig(im: Image.Image, name: str) -> None:
    p = FIG / name
    im.save(p, quality=86, optimize=True)
    print(p, im.size)


def book_center(raw: Image.Image) -> int:
    """x of the book's spine ≈ midpoint of the bright paper span (hands /
    table are darker)."""
    g = np.asarray(raw.convert("L"), dtype=np.float32)
    col = np.convolve(g.mean(axis=0), np.ones(31) / 31, mode="same")
    bright = np.where(col > 0.62 * col.max())[0]
    return int((bright[0] + bright[-1]) // 2) if len(bright) >= 10 else g.shape[1] // 2


# ── reprocess a committed slim project to a throwaway full copy ───────
@contextmanager
def full_project(src_agl: Path):
    """Copy ``src_agl`` to a temp dir, reprocess it through book_curved_x2
    (restoring every intermediate step image the slim copy dropped), yield an
    open read-only-ish connection, then delete the temp copy. Local-only:
    the reprocess runs the full pipeline including dewarp."""
    from lib.storage.db import open_db
    tmpdir = Path(tempfile.mkdtemp(prefix="aglaia-landing-"))
    tmp = tmpdir / src_agl.name
    shutil.copy2(src_agl, tmp)
    try:
        subprocess.run(
            [sys.executable, str(ROOT / "aglaia.py"), str(tmp),
             "--headless", "--force-proc", "-p", "book_curved_x2"],
            cwd=str(ROOT), check=True, capture_output=True, text=True)
        conn = open_db(tmp)
        try:
            yield conn
        finally:
            conn.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def scan_ids(conn) -> list[int]:
    return [r[0] for r in conn.execute(
        "SELECT id FROM scans WHERE deleted_at IS NULL ORDER BY idx")]


def raw_img(conn, sid: int) -> Image.Image:
    r = conn.execute(
        "SELECT i.blob FROM nodes n JOIN images i ON i.id = n.image_id "
        "WHERE n.id = (SELECT root_node_id FROM scans WHERE id = ?)",
        (sid,)).fetchone()
    return _img(r[0])


def _leaf(conn, sid: int, branch: str) -> int:
    return conn.execute(
        "SELECT chosen_node_id FROM branches WHERE scan_id = ? AND branch_path = ?",
        (sid, branch)).fetchone()[0]


def final_img(conn, sid: int, branch: str) -> Image.Image:
    r = conn.execute(
        "SELECT i.blob FROM nodes n JOIN images i ON i.id = n.image_id WHERE n.id = ?",
        (_leaf(conn, sid, branch),)).fetchone()
    return _img(r[0])


def overlays(conn, sid: int, branch: str):
    from lib.storage.debug_renderers import render_chain_overlays
    return render_chain_overlays(conn, _leaf(conn, sid, branch))  # {label,url}/step


def ov_im(ov, i: int) -> Image.Image:
    return _img(base64.b64decode(ov[i]["url"].split(",", 1)[1]))


def step_img(conn, sid: int, branch: str, step: str) -> Image.Image:
    stem = conn.execute(
        "SELECT filestem FROM nodes WHERE scan_id = ? AND branch_label = ? LIMIT 1",
        (sid, branch)).fetchone()[0]
    r = conn.execute(
        "SELECT i.blob FROM nodes n JOIN images i ON i.id = n.image_id "
        "WHERE n.filestem = ? AND n.step_name = ?", (stem, step)).fetchone()
    return _img(r[0])


def before_after(conn, sid: int, w: int = 760, nudge: int = 0):
    """Before (raw spread, spine-centred) ↔ after (rectified A|B), same dims."""
    A, B = final_img(conn, sid, "A"), final_img(conn, sid, "B")
    after = hstack(A, B)
    asp = after.width / after.height
    frac = A.width / after.width
    raw = raw_img(conn, sid)
    rw, rh = raw.size
    bw = round(rh * asp)
    x0 = max(0, min(round(book_center(raw) - frac * bw) + nudge, rw - bw))
    before = raw.crop((x0, 0, x0 + bw, rh))
    a2 = fitw(after, w)
    b2 = fitw(before, w).resize(a2.size, Image.LANCZOS)
    return b2, a2


# ── athanase: before/after + stage slideshow + doc figs ──────────────
with full_project(ATH) as conn:
    sids = scan_ids(conn)
    s1 = sids[0]                                    # scan 1 = the landing page

    # Heavy-curl before/after (scan 1 has the spine bulge + hand).
    b, a = before_after(conn, s1)
    save(b, "curl-before.jpg")
    save(a, "curl-after.jpg")

    # Band slide: a couple of clean rectified pages.
    save(fith(final_img(conn, s1, "A"), 820), "band-slide-3.jpg", q=68)
    save(fith(final_img(conn, sids[1], "A"), 820), "band-slide-1.jpg", q=68)

    # 5-stage "start → finish" strip — the GUI inspector's own overlays.
    ov = overlays(conn, s1, "A")
    H = 760
    save(fith(crop_frac(debanner(ov_im(ov, 2)), 0.15, 0.03, 0.80, 1.0), H),
         "stage-1-detect.jpg")
    save(fith(step_img(conn, s1, "A", "05_pages_bw"), H), "stage-2-binarize.jpg")
    save(fith(debanner(ov_im(ov, 6)), H), "stage-3-trapezoid.jpg")
    save(fith(debanner(ov_im(ov, 7)), H), "stage-4-dewarp.jpg")
    save(fith(step_img(conn, s1, "A", "10_replay"), H), "stage-5-final.jpg")

    # Doc figures — the same overlays, wider.
    docfig(fitw(debanner(ov_im(ov, 6)), 900), "trap_example.jpg")
    docfig(fitw(debanner(ov_im(ov, 7)), 900), "dewarp_example.jpg")
    docfig(fitw(debanner(ov_im(ov, 2)), 1100), "layout_example.jpg")
    # Binarize doc fig: normalized input ↔ clean B&W output.
    bin_in = fith(step_img(conn, s1, "A", "04_dpi_normalize_output"), 820)
    bin_out = fith(step_img(conn, s1, "A", "05_pages_bw"), 820)
    docfig(hstack(bin_in, bin_out, 14), "binarize_example.jpg")


# ── augustin + balthasar: before/after + band slides ─────────────────
with full_project(AUG) as conn:
    s = scan_ids(conn)[1]
    b, a = before_after(conn, s)
    save(b, "before.jpg")
    save(a, "after.jpg")
    save(fith(final_img(conn, s, "A"), 820), "band-slide-2.jpg", q=68)
    save(fith(final_img(conn, scan_ids(conn)[3], "A"), 820), "band-slide-4.jpg", q=68)

with full_project(BAL) as conn:
    s = scan_ids(conn)[0]
    b, a = before_after(conn, s)
    save(b, "lit-before.jpg")
    save(a, "lit-after.jpg")
    save(fith(final_img(conn, s, "A"), 820), "band-slide-5.jpg", q=68)


# ── continuity-camera usage diagram (line-art, app_data) ─────────────
save(fitw(Image.open(ROOT / "lib" / "app_data" / "aglaia_usage.png").convert("RGB"), 760),
     "usage.jpg", q=86)

# NOTE: replay-distorted.webp / replay-restored.webp (the smart-replay
# showcase) are committed STATIC assets — their source was a one-off debug
# dump that has been retired, so this script no longer regenerates them.

print("ALL ASSETS OK")
