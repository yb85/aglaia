# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Per-user config DB (`aglaia-config.db`).

Two tables:

* `config(key TEXT PRIMARY KEY, value TEXT)` — JSON-encoded values so any
  Python literal round-trips. Use `set(key, py_value)` / `get(key, default)`.
* `recent_projects(path TEXT PRIMARY KEY, name TEXT, opened_at TEXT)` —
  most-recent project files for the startup picker.

Schema is created on first connect; default values are seeded from the
bundled `config/config_default.yaml` if the relevant key is missing.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml

from . import config_db_path, default_documents_dir


def _default_workers() -> int:
    """0 = AUTO — the chain derives a CPU-budget-aware count at start time
    (aglaia.worker_count.auto_workers). A manual override is stored only when
    the user moves the Settings slider off the 'auto' position."""
    return 0

# Bundled defaults — shipped INSIDE the package (aglaia/config/), both in
# source and in the PyInstaller bundle. `parents[1]` is the `aglaia/` package
# dir; `parents[2]` was the pre-refactor repo-root `config/`, which no longer
# exists on a fresh install / in the frozen app.
_DEFAULTS_YAML = Path(__file__).resolve().parents[1] / "config" / "config_default.yaml"


# ── canonical keys ────────────────────────────────────────────────────
# Listed here so the Settings dialog can iterate, and so `bootstrap`
# knows what to seed when the DB is fresh.

KEY_THEME = "theme"                   # "system" | "light" | "dark"
KEY_LANGUAGE = "language"             # "" = auto (QLocale.system()) | "en_US" | "fr_FR" | …
KEY_CWD_PROJECT = "cwd_project"       # str path
KEY_CWD_EXPORT = "cwd_export"         # str path
KEY_OCR_DEFAULTS = "ocr_defaults"     # {"engine": str, "languages": [str]}
KEY_EXPORT_DEFAULTS = "export_defaults"  # {"compression": str, "stage": str|None}
KEY_THUMB_SIZE = "thumb_size_default"  # int (px max card width)
KEY_WORKERS = "workers"                # int
KEY_WORKERS_RAM_WARN_DISMISSED = "workers_ram_warn_dismissed"  # bool
KEY_DEWARP_BATCH = "dewarp_batch"      # "auto" | "on" | "off" — GPU-batched dewarp
KEY_VOICE_CONTROL = "voice_control"    # bool
KEY_INPUT_DPI = "input_dpi"            # float
KEY_CAMERA_ID = "camera_id"            # int
KEY_VIEW_MODE = "scans_view_mode"      # "list" | "grid" | "gallery"
KEY_DISABLE_TIP_BUTTONS = "disable_tip_buttons"  # bool
KEY_MODELS_DIR = "models_dir"          # str — relative → APP_DATA, absolute = override; "" = built-in cache
KEY_LAYOUT_HEURISTIC_NO_WARN = "layout_heuristic_no_warn"  # bool
KEY_OCR_WORKERS = "ocr_workers"   # int — 0 = auto-size to available
                                   # GPU/VRAM/unified mem; ≥1 = explicit
                                   # llama-server `--parallel` slot count.
KEY_LIVE_OCR = "live_ocr"         # bool — when on, MainWindow auto-fires
                                   # an OCR pass on freshly-ready
                                   # branches after a 10 s grace window
                                   # (leaves the user time to delete
                                   # before OCR cost is spent).
KEY_OCR_DPI = "ocr_dpi"          # int — downsample input pages to this
                                  # DPI before sending to the active OCR
                                  # engine. 150 ≈ documented sweet spot
                                  # across Surya / PaddleOCR-VL / Apple
                                  # Vision; 100 = faster, 200 = sharper.
                                  # Single unified knob — same physics
                                  # apply regardless of which engine the
                                  # user picks.
# Legacy aliases — older configs may persist these keys. Read paths in
# the engines fall back to ``KEY_OCR_DPI`` when these are absent.
KEY_SURYA_OCR_DPI = "surya_ocr_dpi"
KEY_PADDLE_OCR_DPI = "paddle_ocr_dpi"
KEY_OCR_CONFIDENCE_GATE = "ocr_confidence_gate"  # float — per-line Vision
                                  # confidence below which a line is treated
                                  # as mis-read and offloaded to the
                                  # apple_docs complement engine. 0.7 default;
                                  # raise to offload more, lower to offload
                                  # less. Range (0,1].
KEY_SIDEBAR_TAB = "sidebar_active_tab"        # str | None — last active sidebar tab key
KEY_SIDEBAR_COLLAPSED = "sidebar_collapsed"   # bool — sidebar content-pane hidden?
KEY_DEBUG_OVERLAYS = "debug_overlays_shown"   # bool — Debug viewer's "Show
                                  # debug overlays" toggle, remembered across
                                  # sessions.
KEY_WELCOME_SEEN = "welcome_seen"  # bool — first-run welcome/permissions
                                   # screen has been shown + dismissed.
KEY_FILETYPE_ASSOC_DONE = "filetype_assoc_done"  # bool — auto-registered the
                                   # .agl ↔ app binding once (first .app launch).
KEY_MISTRAL_BATCH = "mistral_batch"  # bool — Cloud OCR (Mistral) submits a
                                   # batch job (cheaper, async) instead of a
                                   # synchronous OCR run. Remembered across
                                   # sessions; the OCR card's batch toggle.
KEY_MISTRAL_FOOTNOTES = "mistral_footnotes"  # str — "numeric" | "alphabetic" |
                                   # "off". Markdown-export post-process of the
                                   # stored Mistral output: superscript / (N)
                                   # footnotes → GFM. (Markdown export card.)
KEY_MISTRAL_HEADERS = "mistral_headers"  # bool — at markdown export, wrap the
                                   # page's running head / number in <header>/
                                   # <footer> tags (else keep them inline).
KEY_MODELS_PROMPT_DISMISSED = "models_prompt_dismissed"  # bool — user ticked
                                   # "don't show again" on the first-run model
                                   # install invite (EAST+Vosk / Vosk on mac).

BUILTIN_DEFAULTS: dict[str, Any] = {
    KEY_THEME: "system",
    KEY_LANGUAGE: "",
    KEY_CWD_PROJECT: str(default_documents_dir()),
    KEY_CWD_EXPORT: str(default_documents_dir()),
    KEY_OCR_DEFAULTS: {"engine": "apple_vision", "languages": ["fr-FR"]},
    KEY_EXPORT_DEFAULTS: {"compression": "auto", "stage": None,
                          "image_format": "jpg"},
    KEY_THUMB_SIZE: 150,
    KEY_WORKERS: _default_workers(),
    # "auto" = batched GPU dewarp on iff a CUDA JAX plugin is installed (the GPU
    # build); on CPU-only installs it stays off. "on"/"off" force it.
    KEY_DEWARP_BATCH: "auto",
    KEY_WORKERS_RAM_WARN_DISMISSED: False,
    KEY_VOICE_CONTROL: False,
    KEY_INPUT_DPI: 100.0,
    KEY_CAMERA_ID: 0,
    KEY_VIEW_MODE: "grid",
    KEY_DISABLE_TIP_BUTTONS: False,
    KEY_MODELS_DIR: "",
    KEY_LAYOUT_HEURISTIC_NO_WARN: False,
    KEY_SIDEBAR_TAB: None,
    KEY_SIDEBAR_COLLAPSED: False,
    KEY_OCR_WORKERS: 0,
    KEY_OCR_DPI: 200,
    KEY_OCR_CONFIDENCE_GATE: 0.7,
    KEY_LIVE_OCR: False,
    KEY_DEBUG_OVERLAYS: False,
    KEY_WELCOME_SEEN: False,
    KEY_FILETYPE_ASSOC_DONE: False,
    KEY_MISTRAL_BATCH: False,
    KEY_MISTRAL_FOOTNOTES: "numeric",
    KEY_MISTRAL_HEADERS: True,
    KEY_MODELS_PROMPT_DISMISSED: False,
}


# ── connection / schema ───────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS config (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recent_projects (
    path       TEXT PRIMARY KEY,
    name       TEXT,
    opened_at  TEXT NOT NULL,
    scan_count INTEGER
);

CREATE INDEX IF NOT EXISTS idx_recent_opened ON recent_projects(opened_at DESC);

CREATE TABLE IF NOT EXISTS plugins (
    path       TEXT PRIMARY KEY,   -- absolute path of the accepted .py file
    kind       TEXT NOT NULL,      -- "processors" | "ocr"
    sha256     TEXT NOT NULL,      -- content hash at acknowledge time
    status     TEXT NOT NULL,      -- "accepted" (only accepted rows persist)
    added_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS downloads (
    key        TEXT PRIMARY KEY,   -- DownloadTarget.key (core or plugin asset)
    status     TEXT NOT NULL,      -- "downloading" | "downloaded" | "failed"
    sha        TEXT,               -- optional content hash recorded on completion
    size_bytes INTEGER,            -- optional fetched size
    updated_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def open_db(path: Path | str | None = None) -> sqlite3.Connection:
    """Open (and lazily migrate) the per-user config DB."""
    path = Path(path) if path else config_db_path()
    conn = sqlite3.connect(str(path), isolation_level=None,
                           check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    # Lazy migration: add scan_count to pre-existing recent_projects tables.
    try:
        conn.execute("ALTER TABLE recent_projects ADD COLUMN scan_count INTEGER")
    except sqlite3.OperationalError:
        pass  # column already present
    return conn


@contextlib.contextmanager
def session(path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    conn = open_db(path)
    try:
        yield conn
    finally:
        conn.close()


# ── config KV API ────────────────────────────────────────────────────

def get(conn: sqlite3.Connection, key: str, default: Any = None) -> Any:
    row = conn.execute("SELECT value FROM config WHERE key = ?", (key,)).fetchone()
    if row is None:
        return BUILTIN_DEFAULTS.get(key, default)
    try:
        return json.loads(row["value"])
    except Exception:
        return default


def set(conn: sqlite3.Connection, key: str, value: Any) -> None:  # noqa: A001
    conn.execute(
        "INSERT INTO config (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, json.dumps(value)),
    )


def tip_buttons_disabled() -> bool:
    """Cheap one-shot read of the tip-buttons toggle. Opens its own
    session so call sites in widget constructors don't have to thread
    a connection through."""
    try:
        with session() as conn:
            return bool(get(conn, KEY_DISABLE_TIP_BUTTONS, False))
    except Exception:
        return False


def items(conn: sqlite3.Connection) -> dict[str, Any]:
    out: dict[str, Any] = dict(BUILTIN_DEFAULTS)
    for r in conn.execute("SELECT key, value FROM config").fetchall():
        try:
            out[r["key"]] = json.loads(r["value"])
        except Exception:
            continue
    return out


# ── recent projects ──────────────────────────────────────────────────

def remember_project(conn: sqlite3.Connection, project_path: Path | str,
                     name: str | None = None,
                     scan_count: int | None = None) -> None:
    p = str(Path(project_path).resolve())
    nm = name or Path(p).stem
    conn.execute(
        "INSERT INTO recent_projects (path, name, opened_at, scan_count) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(path) DO UPDATE SET name = excluded.name, "
        "opened_at = excluded.opened_at, "
        # Keep the prior count when this call doesn't supply one (so a
        # bare re-touch of the timestamp doesn't blank the badge).
        "scan_count = COALESCE(excluded.scan_count, recent_projects.scan_count)",
        (p, nm, _now(), scan_count),
    )


def list_recent_projects(conn: sqlite3.Connection,
                         limit: int = 20) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM recent_projects ORDER BY opened_at DESC LIMIT ?",
        (limit,),
    ).fetchall()


def forget_project(conn: sqlite3.Connection, project_path: Path | str) -> None:
    conn.execute(
        "DELETE FROM recent_projects WHERE path = ?",
        (str(Path(project_path).resolve()),),
    )


# ── plugins (drop-in .py trust registry) ─────────────────────────────
#
# Only *accepted* plugins live here. The startup trust gate writes a row
# when the user acknowledges a dropped file; discovery imports only rows
# whose stored sha256 still matches the file on disk. See
# `aglaia/app_data/plugins.py`.

def accepted_plugins(conn: sqlite3.Connection) -> dict[str, dict[str, str]]:
    """`{abs_path: {"kind", "sha256"}}` for every accepted plugin."""
    out: dict[str, dict[str, str]] = {}
    for r in conn.execute(
        "SELECT path, kind, sha256 FROM plugins WHERE status = 'accepted'"
    ).fetchall():
        out[r["path"]] = {"kind": r["kind"], "sha256": r["sha256"]}
    return out


def acknowledge_plugin(conn: sqlite3.Connection, kind: str,
                       path: Path | str, sha256: str) -> None:
    """Record (or refresh) an accepted plugin. Re-acknowledging a changed
    file updates the stored hash."""
    p = str(Path(path).resolve())
    conn.execute(
        "INSERT INTO plugins (path, kind, sha256, status, added_at) "
        "VALUES (?, ?, ?, 'accepted', ?) "
        "ON CONFLICT(path) DO UPDATE SET kind = excluded.kind, "
        "sha256 = excluded.sha256, status = excluded.status, "
        "added_at = excluded.added_at",
        (p, kind, sha256, _now()),
    )


def forget_plugin(conn: sqlite3.Connection, path: Path | str) -> None:
    conn.execute("DELETE FROM plugins WHERE path = ?",
                 (str(Path(path).resolve()),))


# ── download lifecycle state ─────────────────────────────────────────
#
# Persistent status for registered download targets (the catalogue itself
# lives in-memory in `aglaia/app_data/downloads.py`). Absence of a row means
# "never fetched", same convention as the plugins table. Disk remains ground
# truth; `downloads.download_status()` reconciles this table against it.

def set_download_status(conn: sqlite3.Connection, key: str, status: str,
                        *, sha: str | None = None,
                        size_bytes: int | None = None) -> None:
    conn.execute(
        "INSERT INTO downloads (key, status, sha, size_bytes, updated_at) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET status = excluded.status, "
        "sha = excluded.sha, size_bytes = excluded.size_bytes, "
        "updated_at = excluded.updated_at",
        (key, status, sha, size_bytes, _now()),
    )


def get_download_status(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute(
        "SELECT status FROM downloads WHERE key = ?", (key,)).fetchone()
    return row["status"] if row is not None else None


def clear_download_status(conn: sqlite3.Connection, key: str) -> None:
    conn.execute("DELETE FROM downloads WHERE key = ?", (key,))


def download_statuses(conn: sqlite3.Connection) -> dict[str, str]:
    """`{key: status}` for every recorded download."""
    return {r["key"]: r["status"]
            for r in conn.execute("SELECT key, status FROM downloads").fetchall()}


# ── bootstrap ───────────────────────────────────────────────────────

def bootstrap(conn: sqlite3.Connection) -> None:
    """Seed any missing keys from the bundled `config_default.yaml` and
    `BUILTIN_DEFAULTS`. Idempotent — re-running never overwrites the
    user's saved overrides."""
    existing = {r["key"] for r in conn.execute("SELECT key FROM config").fetchall()}

    # Bundled YAML → flat dict of (key → value) for the entries the user
    # is allowed to override at runtime.
    yaml_seed = _load_yaml_defaults()
    for key, value in yaml_seed.items():
        if key in existing:
            continue
        set(conn, key, value)
        existing.add(key)

    for key, value in BUILTIN_DEFAULTS.items():
        if key in existing:
            continue
        set(conn, key, value)


def _load_yaml_defaults() -> dict[str, Any]:
    """Pull the small subset of `config_default.yaml` that maps onto
    config keys exposed in the Settings dialog. Other YAML entries
    (keybindings, voice commands) are read at runtime directly."""
    out: dict[str, Any] = {}
    if not _DEFAULTS_YAML.is_file():
        return out
    try:
        data = yaml.safe_load(_DEFAULTS_YAML.read_text(encoding="utf-8"))
    except Exception:
        return out
    args = (data or {}).get("args") or {}
    # Note: `workers` deliberately NOT seeded from YAML — the bundled
    # default (6) ignored the machine's core count. BUILTIN_DEFAULTS
    # computes `min(ceil(ncores/2), 4)` instead.
    if "voice_control" in args:
        out[KEY_VOICE_CONTROL] = bool(args["voice_control"])
    if "camera_id" in args:
        out[KEY_CAMERA_ID] = int(args["camera_id"])
    return out
