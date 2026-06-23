# Lessons learned — Aglaïa processor design

This document captures pitfalls hit while building the trap / dewarp /
binarisation pipeline, with concrete rules of thumb to avoid repeating
them. Read it before touching any of the geometric processors
(`PageDewarper`, `TrapezoidalCorrection`, `PageDetector`,
`SkewFinder`) or their span / baseline plumbing.

The doc is split into three parts:

1. **Pitfalls** — mistakes encountered, what made them subtle, and the
   rule that prevents them.
2. **Implementation guidelines** — how to add a processor, wire it to
   YAML and the UI, give it a debug visualisation, and make it survive
   pipeline restarts.
3. **Useful patterns** — debugging recipes, probing tools, span-mask
   trick, multi-processing constraints, MLX/JAX gotchas.

---

## 1. Pitfalls

### 1.1 Absolute pixel constants in DPI-aware code

**Symptom**: code works on a 300 dpi corpus, silently degrades on a
150 dpi or 600 dpi input. Or: works on body text but rejects headings.

**Root cause**: hard-coding pixel thresholds (`if width > 30:`, etc.).
At different DPIs the same physical text has different pixel sizes.

**Examples we hit**:

- `_text_mask_dpi`'s char-CC filter used hardcoded `3 ≤ h ≤ 60` px,
  `2 ≤ w ≤ 100` px ranges. At 600 dpi body text is ≈ 80 px tall, above
  the upper bound. The filter then returned zero char-like CCs and the
  fallback DPI-only kernel kicked in silently — no obvious failure, just
  worse results.
- `EDGE_MAX_LENGTH = w // 3` (where `w` was image width). At a wide
  page (sw = 2576 px) this gave 858 px — large enough for the span
  builder to link the end-of-line-N cinfo to the start-of-line-N+1
  cinfo with a small angle, welding two text lines into one span.

**Rule**: every numeric threshold that has units of distance (pixels)
or area must be one of:

| Scaling | When to use |
|---------|-------------|
| `× DPI` | when text scale matches DPI (300 dpi → x-height ≈ 30 px) |
| `× h_med` (median char height) | preferred — DPI-independent and self-adapting to actual text size on the page |
| `× image dim` | only for visualisation grid steps; never for content filters |
| Pure ratio / angle | always fine (scale-free) |
| Pure count | flag as suspicious; usually wrong |

Use `h_med` (median height of char-like CCs in the binary mask) when
available. Compute it once per processor invocation and tie kernel
widths, thickness caps, edge bounds, and span-min-width to it. Fall
back to DPI-based fractions only when `h_med` can't be estimated
(too few char-like CCs).

**See**:
- `aglaia/processors/PageDewarper.py:_text_mask_dpi` — kernel = `2 × h_med`,
  TEXT_MAX_THICKNESS = `3 × h_med`, EDGE_MAX_LENGTH = `3 × h_med`,
  SPAN_MIN_WIDTH = `10 × h_med`.
- `aglaia/processors/TrapezoidalCorrection.py` — same scheme, mirrored.

### 1.2 Hidden assumption: input is binary (or grayscale)

**Symptom**: a processor expecting binary input gets grayscale (or
vice versa) when the replay pass defers binarisation. The downstream
Otsu / fixed threshold then runs on already-binary input and produces
weird artefacts.

**Examples we hit**:

- `_text_mask_dpi` did Otsu unconditionally. When the input was already
  BW (from Binarizer earlier in the chain), Otsu still ran but the
  bimodal threshold landed at an arbitrary value, sometimes inverting
  polarity. Fix: branch on `is_bw = img_buf.type == ImageType.BW`.
  For BW input use `bitwise_not(sgray)`; for grayscale use Otsu.
- `resize_to_analysis` used `INTER_AREA` which anti-aliases binary
  input back into grayscale. Fix: use `INTER_NEAREST` when `is_bw`.

**Rule**: every processor's `process()` should branch on
`img_buf.type`. Otsu / fixed thresholding on grayscale, polarity
detection on BW. Use `INTER_NEAREST` for resizing BW inputs to
preserve histogram bimodality.

### 1.3 Morphology kernel size = the silent killer

**Symptom**: span count drops in half on certain pages. Or: spans
form correctly but bbox heights are 3-4× expected line height
(multi-line bridging).

**Root cause**: horizontal MORPH_CLOSE with a kernel WIDER than the
text scale dilates each glyph into a wide stripe. If two glyphs on
neighbouring lines are vertically close (small inter-line gap, page
curl, or descender ↔ ascender contact), the stripes merge and the
connected component spans both lines.

**Examples we hit**:

- `kw = max(9, line_join_mm × dpi / 25.4)` with `line_join_mm = 4` at
  150 dpi gave `kw = 24`. On book pages with dense paragraphs, the
  24-px-wide stripes welded adjacent lines through any vertical
  ink filament (a single 1-2 px tall pixel chain between a descender
  and the next-line ascender was enough). Result: each "cinfo" covered
  2-3 lines vertically, and `TEXT_MAX_THICKNESS` then DROPPED these
  multi-line blobs wholesale, leaving 14 spans on a 30-line page.

**Rule**:

1. Kernel width = `2 × h_med` (one word-gap + safety margin). Tied to
   text scale.
2. Always run a small vertical OPEN before horizontal CLOSE to break
   1-2 px tall vertical bridges: `cv2.morphologyEx(ink, MORPH_OPEN,
   getStructuringElement(RECT, (1, h_med/6)))`. Costs a few isolated
   `i`-dots, gains separation of adjacent text lines.
3. Verify with a debug pane: render the morphed mask and check that
   each connected component is a single horizontal stripe per text
   line. Multi-line CCs = morphology too aggressive.

### 1.4 Multi-line bbox poisons the baseline fit

**Symptom**: per-span baseline_from_ink returns a line that drifts
through nowhere — neither at the line's typographic baseline, nor
horizontal, but at some weird tilt.

**Root cause**: when the span's bbox contains ink from MULTIPLE text
lines (because the cinfo is multi-line, see 1.3), per-column
bottom-most ink jumps between lines as x varies. At column x_A
the bottom is at line N+2's baseline; at column x_B at line N+1's
baseline. The fit averages or RANSAC-clusters these jumps and
produces a tilted line that fits NOTHING.

**Rule**: each baseline fit must see ONLY ink from its own span's
cinfos. Implementation: build a `span_mask` image (zeros except where
this span's cinfo tight masks are set), pass it to `baseline_from_ink`,
mask `crop = np.where(span_mask_crop > 0, crop, 0)` before computing
bottoms.

We proved this matters by comparing fits with/without mask on
`md1_008_A`: bottom-curl spans (23-29) had slope ≈ 0 without mask
(neighbour-line ink pulled the inlier cluster horizontal), and slope
≈ +0.07 with mask (matching the actual page tilt). 22/31 spans
changed when the mask was applied.

**See**:
- `aglaia/processors/geometry.py:baseline_from_ink(..., span_mask=...)`.
- `aglaia/processors/TrapezoidalCorrection.py` builds the per-span mask from
  cinfo tight masks (one image per span; OR'd union of cinfo
  rect-placed tight masks).

### 1.5 "Just average the residuals" — L2 / median trap

**Symptom**: baseline doesn't follow the obvious typographic baseline.
With L2 LSQ on per-column bottoms, the fit gets dragged by descenders
and page-curl outliers. With median (or theilsen), the fit lands on the
DOMINANT cluster — which is the curl region when the curl portion is
longer than the flat portion.

**Rule**:

- L2 fits are dragged by outliers. Don't use for baselines.
- Pure median / theilsen on a curved line picks the dominant
  per-column-bottom cluster. If the curl region has more columns than
  the flat region, the fit lands on the curl direction.
- RANSAC with a generous `eps` (~`0.15 × bbox_h`) and bbox-edge
  endpoints handled this best in practice. We tried multiple
  variations (mode-based reference, topmost-peak picking, longest
  contiguous run, multi-line auto-clip) and they all introduced
  their own failure modes (locks onto descender row, fragments at
  word gaps, wrong-direction tilt, etc.).

**Counter-intuitive lesson**: simple RANSAC with bbox-edge endpoints
beats every clever robust estimator we tried. Trust the user when
they say "RANSAC worked best" — try the elaborate fit first only when
you can prove the simple one fails on a specific case.

### 1.6 Pipeline yaml + UI + processor option — three places to wire

**Symptom**: option added to dataclass, default works, but user can't
see/change it in the GUI. Or: it's in the UI but not in YAML, so a
saved pipeline doesn't preserve it.

**Rule**: every new processor option must be wired in THREE places:

1. **Dataclass** (`aglaia/processors/<Processor>.py`): `field_name: type = default`
   in the `<Processor>Option` dataclass.
2. **Spec** (`aglaia/processors/option_specs.py`): a `_f` / `_i` / `_e` / `_b`
   entry in the `OPTION_SPECS["<Processor>"]` dict. Determines UI
   widget kind, range, step, help text.
3. **YAML** (`config/pipelines/*.yaml`): an entry in each pipeline
   that includes the processor. If the option only affects new
   defaults, the dataclass default + spec entry are enough; users get
   the new field in their editor automatically.

UI is auto-generated from `OPTION_SPECS` (both Qt
`PipelineEditorWidget` and Web `pipeline.py` endpoint pull from it).
Adding to specs surfaces the field in both editors with no extra
work.

### 1.7 Don't assume input polarity from the previous step

**Symptom**: a step expects BW input but gets grayscale (or vice
versa); Otsu / polarity detection produce odd outputs.

**Root cause**: a step's input polarity isn't pinned by its YAML
neighbour — the BW-preservation re-binarize (`binarize_fixed(127)`
across non-binarizing steps) and the end-of-chain replay pass (which
defers `PIXEL_VALUE` steps like the Binarizer) both move polarity around.

**Rule**: never infer input type from "the previous step is X". Read
`img_buf.type` explicitly.

### 1.8 multiprocessing.spawn caches modules at worker start

**Symptom**: edited a processor module, restarted the GUI, results
unchanged.

**Root cause**: workers are `spawn`-ed at process pool init time; they
each import their own copy of the module. Editing the source after
worker start doesn't update the running workers.

**Rule**: any code change to `aglaia/processors/*.py` or `aglaia/workers/*.py`
requires worker restart. The GUI pool persists across scan processing
runs; full app restart is the safest path. Verify your fix is live
by adding a one-shot `print()` and watching the log_queue.

### 1.9 Worker OOM kills + reenqueue drops

**Symptom**: random scans silently fail to produce dewarp output.
Logs show "Killed" / SIGKILL.

**Root cause**: worker hits the 3 GB phys_footprint watchdog, gets
SIGKILL'd. The in-flight scan is reenqueued, but any FURTHER scans
already pulled from the queue by that dead worker are LOST.

**Rule**:

- Clear MLX/JAX caches regularly inside processor `process()` to
  cap memory growth: `mx.clear_cache()`.
- Don't pre-fetch from the queue in workers; `Queue.get(timeout=…)`
  per-scan is fine.
- See `aglaia/workers/IntegratedProcessingChain.py` — single-scan-per-worker
  loop.

### 1.10 Hard-coded image bound checks (256 MB cap)

**Symptom**: debug image generated, but Qt viewer rejects it with
"QImageIOHandler: Rejecting image as it exceeds the current allocation
limit of 256 megabytes".

**Root cause**: a 3-pane composite at native 300 dpi can decode to
100+ MB. After base64 encode + browser/Qt decode buffer overhead, the
256 MB Qt cap is hit.

**Rule**: viewer-side downscaling is a render-only concern. Don't
let it propagate to storage or pipeline. `_png_data_url()` in
`aglaia/web/routes/debug.py` clamps max(width, height) ≤ 2400 px for
PNG encode. Storage (`images` table, debug_artifacts) keeps native
resolution.

---

## 2. Implementation guidelines

### 2.1 Adding a new processor

A processor lives in `aglaia/processors/<Name>.py` and exposes:

```python
@dataclass
class <Name>Option(AbstractProcessorOption):
    field_a: float = ...
    field_b: int = ...
    debug: bool = False

class <Name>(AbstractImageProcessor):
    def __init__(self, options: <Name>Option):
        super().__init__(options)
        self.field_a = options.field_a
        # …

    def process(self, img_buf: ImageBuffer) -> ImageBuffer:
        # MUST mutate img_buf in-place (buffer, dpi, meta).
        # MUST set img_buf.meta["status"] in finally.
        ...
```

Checklist:

1. **Dataclass** with sane defaults. All processor options live here.
2. **`option_specs.py`** entry for every field that should be user-editable.
3. **Pipeline YAML** entries in each pipeline that uses it.
4. **Default pipeline**: `config/pipelines/book_curved_x2.yaml` should include
   the processor if it's intended to be on by default.
5. **Status enum** — set `img_buf.meta["status"]` to `Status.OK`,
   `Status.FALLBACK`, or `Status.ERROR`. Used for UI badges.
6. **`debug_save(img, label, img_buf)`** — emit intermediate images
   via the `AbstractImageProcessor.debug_save` helper. These are
   surfaced in the GUI debug viewer.
7. **Persistence** — every step's node is recorded with its own stored
   image; the end-of-chain replay pass fuses the per-step geometric
   transforms into one warp from the raw image for the final output.
8. **Unit test** in `tests/dsp/test_<name>_processor.py` — at minimum,
   a "process passes through a clean page without throwing" smoke test.

### 2.2 Debug visualisation

Each processor that produces non-trivial output should have a
visualiser in `aglaia/web/routes/debug.py:_RENDERERS`. The renderer
takes `(img, parent, meta)` and returns
`[{"url": data_url, "label": str}]`.

Typical pattern: 3-pane composite (`source | mask | output`):

- LEFT: parent (source) + per-span colored overlays + baselines /
  spans / quad as appropriate.
- MIDDLE: morphological intermediate (unfiltered mask) — invaluable
  for debugging "why is the span set wrong?".
- RIGHT: this step's output (warped / binarised / etc.) + a
  reference grid.

Use `_png_data_url(arr)` for the data URL (auto-downscales). Label
text via `_label_bar(canvas, text)`. Span overlays via
`_overlay_spans(canvas, spans, used_idxs)`.

### 2.3 Pipeline-editor wiring

The Qt and Web pipeline editors are auto-generated from
`aglaia/processors/option_specs.py`. To add a field to the editor:

```python
OPTION_SPECS["MyProcessor"] = {
    "kernel_mult": _f(2.0, 0.5, 6.0, 0.1,
                      "Help text shown as tooltip."),
    "method": _e("default", ["default", "alt"],
                 "Choice between named alternatives."),
    "debug": _b(False, "Toggle intermediate dumps."),
    "count": _i(8, 1, 50, "Integer count."),
}
```

The Qt form uses a 2-column grid page for tall option blocks (see
`PipelineEditorWidget.set_step`).

### 2.4 Span/baseline-fit recipe (TrapezoidalCorrection-style)

When you need per-line baselines on a binary image:

```python
# 1. Compute h_med (median char-like CC height).
n_cc, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=4)
char_h = [int(s[3]) for s in stats[1:]
          if cc_h_min(dpi) <= s[3] <= cc_h_max(dpi)
          and cc_w_min(dpi) <= s[2] <= cc_w_max(dpi)]
h_med = float(np.median(char_h)) if len(char_h) >= 30 else 0.0

# 2. Break vertical bridges + horizontal close.
vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(h_med/6))))
ink_clean = cv2.morphologyEx(ink, cv2.MORPH_OPEN, vk)
kw = max(9, int(round(2.0 * h_med)))
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
morphed = cv2.morphologyEx(ink_clean, cv2.MORPH_CLOSE, kernel)

# 3. Configure page-dewarp's contour filter:
pd_cfg.TEXT_MAX_THICKNESS = max(10, int(round(3.0 * h_med)))
pd_cfg.TEXT_MIN_WIDTH = max(8, int(round(0.5 * h_med)))
pd_cfg.TEXT_MIN_HEIGHT = max(2, int(round(0.5 * h_med)))
pd_cfg.EDGE_MAX_LENGTH = max(20, int(round(3.0 * h_med)))
pd_cfg.EDGE_MAX_OVERLAP = max(2.0, 0.1 * h_med)
pd_cfg.SPAN_MIN_WIDTH = max(30, int(round(10.0 * h_med)))
pd_cfg.EDGE_MAX_ANGLE = 7.5  # library default

# 4. Get cinfos + assemble spans.
cinfos = get_contours("name", rgb, morphed)
spans = assemble_spans("name", rgb, pagemask, cinfos)

# 5. Per-span baseline with single-span mask.
for span in spans:
    span_mask = np.zeros(ink.shape, dtype=np.uint8)
    for ci in span:
        x, y, w, h = ci.rect
        sub = span_mask[y:y+h, x:x+w]
        sub |= ci.mask if ci.mask.dtype == np.uint8 else (ci.mask.astype(np.uint8) * 255)
    bb = bounding_box_of(span)
    bl = baseline_from_ink(ink, bb, span_mask=span_mask)
```

`baseline_from_ink` does:
- per-column bottom-most ink (with optional `span_mask` filter)
- windowed-median descender filter
- RANSAC with `eps = 0.15 × bbox_h`, 60 trials
- LSQ refit on inliers
- endpoints at `xs.min() / xs.max()` of descender-filtered bottoms

### 2.5 Pipeline configuration testing

Always test pipelines end-to-end:

```bash
# Trap-only on a single image:
uv run python -c "
import sys, cv2
sys.path.insert(0, '.')
from aglaia.ImageBuffer import ImageBuffer, ImageType
from aglaia.processors.TrapezoidalCorrection import TrapezoidalCorrection, TrapezoidalOption
raw = cv2.imread('/path/to/test.png', cv2.IMREAD_GRAYSCALE)
buf = ImageBuffer(buffer=raw, type=ImageType.BW)
buf.dpi = 300.0; buf.meta = {}; buf.branch_label = 'A'
out = TrapezoidalCorrection(TrapezoidalOption()).process(buf)
print(out.meta)
"
```

For end-to-end pipeline tests, use the web UI or capture GUI on a
fresh project with a known-good corpus.

---

## 3. Useful patterns

### 3.1 First-principles debugging

When a span / baseline / quad is wrong, ALWAYS:

1. **Save the actual data** to /tmp/ and inspect with `cv2.imread` +
   `cv2.imwrite` of crops. Don't trust just metadata.
2. **Reproduce the failure in a small probe script** (`/tmp/probe_*.py`).
   The probe is your tight loop. Anything reproducible in a probe is
   fixable.
3. **Print intermediate quantities**: per-column bottoms histogram,
   in-band counts, longest run lengths, etc. Numbers tell you
   WHERE the model breaks, not just THAT it does.
4. **Don't trust visual intuition alone**. The user may be reading
   a tilted line as horizontal or vice versa. Confirm with raw
   pixel values.

### 3.2 The `span_mask` trick

`page_dewarp`'s `assemble_spans` returns a list of cinfos per span.
Each cinfo has:

- `ci.rect = (x, y, w, h)` — bounding rect in source coords.
- `ci.mask` — tight binary mask of the contour shape, sized to
  `(h, w)`. Note: this is the SHAPE of the morphed CC, not the raw
  ink shape inside.

To build a per-span mask in source coords:

```python
sm = np.zeros(ink.shape, dtype=np.uint8)
for ci in span:
    x, y, w, h = ci.rect
    tm = ci.mask
    if tm.dtype != np.uint8:
        tm = tm.astype(np.uint8) * 255
    sub = sm[y:y+h, x:x+w]
    sub |= tm
```

Pass this `sm` to `baseline_from_ink(ink, bb, span_mask=sm)`. Inside,
`crop = np.where(sm_crop > 0, crop, 0)` zeros all ink that doesn't
belong to this span.

### 3.3 MLX / JAX cache hygiene

Long-running workers leak GPU/CPU memory through cached JIT'd
objectives. Inside `PageDewarper.process()` finally clause:

```python
try:
    import aglaia.processors.page_dewarp_mlx
    aglaia.processors.page_dewarp_mlx.clear_caches()
except Exception:
    pass
```

`clear_caches()` resets:
- The JIT'd value-and-grad function.
- Cached constants (page dims, etc.).
- MLX framework cache (`mx.clear_cache()` or `mx.metal.clear_cache()`).

Without this, the 600k-iter Powell run on an uncalibrated K matrix
leaks 14-58 GB. See `memory/project_camera_matrix_leak.md`.

### 3.4 PDF + sqlite project debugging

Each project is a `<slug>.scanproj.sqlite` file. Pipeline runs leave
nodes per step:

```sql
SELECT step_idx, step_name, processor_name, image_id, status_int
FROM nodes
WHERE scan_id = (SELECT id FROM scans WHERE idx = N)
  AND branch_label = 'A'
ORDER BY step_idx;
```

Extract any intermediate image:

```python
import sqlite3
con = sqlite3.connect("/path/to/project.scanproj.sqlite")
blob = con.execute("SELECT blob FROM images WHERE id = ?", (image_id,)).fetchone()[0]
open("/tmp/extracted.png", "wb").write(blob)
```

### 3.5 Renderer "missing variable" trap

If a debug renderer assigns `dpi = 300` early but later code paths
reference `dpi` from a different branch, NameError silently catches
in the outer `try / except`. The renderer returns the mask pane but
no spans (because the spans-list line wasn't reached). Symptom:
`spans=0` displayed despite a perfectly fine mask in the middle pane.

**Rule**: never rely on a variable being set "earlier" inside a
big try. Initialise defaults at the top of the function before any
try block.

### 3.6 "Restart workers" checklist

After any pipeline-code change:

1. Kill the GUI (Qt: cmd-Q).
2. Restart the entry script (`aglaia.py`).
3. Re-open the project.
4. Reprocess the affected scan.

If you forget step 1, you'll see stale results and assume your fix
didn't take effect. The most common form of "did not change anything"
is in fact a worker not restarted.

### 3.7 Observe, don't assume

Adopted from Karpathy: when debugging memory leaks, CPU usage,
process crashes, NEVER fix based on a guess. Always:

- `ps aux | grep python` to see process state.
- `tracemalloc` to confirm a Python object is leaking.
- `Activity Monitor` for phys_footprint per worker.
- `cv2.imwrite` of intermediate mask/morph/contour stages.

The number of times a guess turned out wrong but a measurement
revealed the actual cause is much higher than intuition suggests.

---

## 4. Quick reference: the constants table

For the `dewarp` / `trap` / `_text_mask_dpi` stack, here's the
canonical sizing. Replicate in any new processor that does
text-line geometry.

| Quantity | Formula | Notes |
|----------|---------|-------|
| Char-CC h-range | `0.04 × dpi ≤ h ≤ 0.45 × dpi` | DPI-derived window for char-like CCs |
| Char-CC w-range | `0.02 × dpi ≤ w ≤ 0.60 × dpi` | same |
| Horizontal close kernel | `kw = 2 × h_med` (floor 9 px) | line-bridging |
| Vertical break kernel | `(1, max(3, h_med / 6))` | breaks 1-2 px line bridges |
| TEXT_MAX_THICKNESS | `3 × h_med` (floor 10 px) | ≈ ascender + x-height + descender |
| TEXT_MIN_WIDTH | `0.5 × h_med` (floor 8 px) | narrowest legit glyph |
| TEXT_MIN_HEIGHT | `0.5 × h_med` (floor 2 px) | x-height/2 |
| EDGE_MAX_LENGTH | `3 × h_med` (floor 20 px) | wide-justified word gap |
| EDGE_MAX_OVERLAP | `0.1 × h_med` (floor 2 px) | overlap tolerance |
| EDGE_MAX_ANGLE | `7.5°` (library default) | scale-free |
| SPAN_MIN_WIDTH | `10 × h_med` (floor 30 px) | ≈ 5 words minimum |
| RANSAC eps (baseline) | `0.15 × bbox_h` (floor 1.5 px) | inlier band |
| RANSAC trials | `60` | original page-dewarp value |
| Descender threshold | `local_med + 0.25 × bbox_h` | drop descender outliers |

If `h_med` could not be estimated (fewer than 30 char-like CCs in
the mask), fall back to DPI-based fractions:

| Quantity | Fallback formula |
|----------|------------------|
| Kernel | `line_join_mm × dpi / 25.4` |
| TEXT_MAX_THICKNESS | `0.25 × dpi` |
| TEXT_MIN_WIDTH | `0.10 × dpi` |
| TEXT_MIN_HEIGHT | `0.01 × dpi` |
| EDGE_MAX_LENGTH | `0.5 × dpi` |
| EDGE_MAX_OVERLAP | `0.02 × dpi` |
| SPAN_MIN_WIDTH | `image_width / 20` |

---

## Appendix: things we tried and rejected

The history of attempts on the baseline-fit problem (`md1_008_A` and
similar curl-heavy pages):

1. **Original RANSAC** (`eps = 0.05 × h`) — fast, deterministic enough.
   Underestimated tilt on curl-heavy spans (≈ 0 slope when actual was
   0.08). Final choice: bumped `eps` to `0.15 × h` to recover tilt.
2. **Theil-Sen with median reference** — picks dominant cluster, which
   is the curl direction when curl spans more cols than flat. Wrong
   direction.
3. **Mode-based reference** — locks onto densest histogram bin. In
   multi-line bboxes lands on descender bottoms (deepest cluster).
4. **Topmost prominent peak** — picks the flat-baseline peak, then
   theilsen on in-band points. Gave the OPPOSITE slope direction
   because the in-band cluster spans both the flat baseline and
   the descender area, and the trend through them tilts opposite to
   the true tilt.
5. **Longest contiguous in-band run + theilsen refit** — runs fragment
   at every word gap (60-100 px), so the longest run is a single word
   (50-70 cols) and local theilsen on that gives noisy slope.
6. **Multi-line bbox auto-clip** — finds the bottom-most run of rows
   with substantial ink, clips bbox to that band. Works on synthetic
   data but on real pages with curl, no clear row valley exists.

Net of 6 attempts and a lot of frustration: simple RANSAC at
`eps = 0.15 × h` with the `span_mask` trick (1.4 above) gave the best
visual match to user-drawn ground-truth lines. Sometimes the elaborate
approach is wrong and the simple one is correct.
