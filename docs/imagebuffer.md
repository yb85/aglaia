# ImageBuffer

`lib/ImageBuffer.py:ImageBuffer` — the canonical envelope passed between every part of the system.

## Fields

| Field | Type | Purpose |
|---|---|---|
| `buffer` | `np.ndarray` | Pixel data. RGB (H,W,3), Gray (H,W), or BW (H,W with values in {0,255}). |
| `type` | `ImageType` | `COLOR`, `GRAY`, or `BW`. Determines write format (jpg/png) and helper conversion. |
| `dpi` | `float` | Image DPI. Used by DPIfixer, Binarizer template eval, page margin calc, dewarp. |
| `path` | `str?` | Source file path (raw image) — used for routing GUI events. |
| `parent` | `ImageBuffer?` | Parent buffer (set on child crops by PageDetector). |
| `parent_stem` | `str?` | Parent filestem cached as string so child copies don't drag the whole parent through pickling. |
| `filestem` | `str?` | Output filename stem (no extension). E.g. `mybook_001`, `mybook_001_A`. |
| `out_dir` | `str?` | Disk-write hint for `ImageBuffer.write` (see below). The live chain persists to the project DB and ignores it. |
| `scan_id`, `parent_node_id`, `pipeline_version_id`, `branch_path`, `branch_label`, `depth` | `int?`/`str` | DB tree context — carried by the chain so `Persister` can link each step's node to its scan, parent node and branch. |
| `children` | `list[ImageBuffer]` | Set by branching processors (PageDetector). |
| `meta` | `dict` | Per-buffer metadata. See below. |

## meta keys in use

| Key | Set by | Used by |
|---|---|---|
| `roi` | PageDetector (child polygon), SkewFinder (full rect after rotation), DPIfixer (scaled) | Binarizer `_apply_roi_mask` |
| `skew_angle` | SkewFinder | GUI display (`'skew'`) |
| `oob_stats` | PageDewarper | GUI display (`'oob'`) |
| `dewarp_success` | PageDewarper | GUI display (`'success'`) |
| `status` | PageDewarper | (informational — `int(Status)`) |
| `page_nums` | (not populated) | debug overlay |

When emitting `image_event`, `_emit_event` normalizes some keys for the GUI:

- `dewarp_success` → `success`
- `oob_stats` → `oob`
- `skew_angle` → `skew`

## Helpers

| Method | Behavior |
|---|---|
| `to_rgb()` | Returns 3-channel RGB ndarray (converts gray/RGBA). Non-mutating. |
| `to_gray()` | Returns single-channel grayscale. Non-mutating. |
| `to_bw()` | Returns Otsu-thresholded BW. Non-mutating. |
| `check_binary()` | True if 2D with ≤2 unique values. |
| `rescale(target_dpi, threshold)` | Resizes buffer to match `target_dpi` (skip if diff < threshold). Mutates. |
| `copy()` | Deep clone of `buffer` + deepcopy of `meta`. Does NOT deep-copy `parent` (keeps reference / `parent_stem` string). |
| `deskew(...)` | Estimate + apply skew in place. Prefer the `SkewFinder` processor. |

`binarize()`, `detectLayouts()`, `dewarp()` are duck-typing convenience helpers.

## write(collection, options, suffix=None, ...)

`ImageBuffer` keeps a general-purpose disk writer (`lib/ImageBuffer.py`). It is **not** called by the live `IntegratedProcessingChain` — that path persists each step to the project `.agl` SQLite DB via `lib/storage/persister.py`. The logic below documents the envelope's own write behaviour, used by off-chain helpers.

Decides where to save based on this priority:

1. `options["paths"][collection]` if set.
2. `options["paths"]["root"]` if set → `<root>/<collection>/<stem>.{ext}`.
3. Existing `self.out_dir` (PDF mode), with sibling-folder heuristics:
   - If `out_dir` equals the output root, redirect to its parent + collection.
   - If `collection == "output"`, respect `out_dir` exactly.
   - Otherwise rewrite to `<out_dir.parent>/<collection>`.
4. Fallback: `paths["output"].parent / collection`.

Filename:
- `self.filestem` (or `paths["filestem"]`, or `"capture"`) + optional `_<suffix>`.
- Extension: `.png` if `type == BW`, else `.jpg`.

Before writing, **same-stem conflicts in the target folder are unlinked** (jpg ↔ png hygiene).

Output format:
- `.jpg` — PIL save, RGB or L mode, quality=95, optimize=true, with `dpi=(dpi, dpi)` EXIF.
- `.png` — BW via PIL `'1'` mode (1bpp), or OpenCV `cv2.imwrite` for color.

`executor`: optional `ThreadPoolExecutor` for async writes — only relevant to off-chain `write` callers; the live chain does not use this method.

After a successful write, an `image_event` tuple is pushed on `log_queue` if `event_type` is set:

```python
('image_event',
 raw_path,          # absolute normalized path of the original raw file (walked via parent chain)
 filestem,
 event_type,        # usually the chain step's instance_name
 full_path,         # absolute normalized path of just-written file
 gui_meta,          # meta dict with key normalization
 parent_stem)
```

## Lifecycle examples

### Capture (single page)

```
raw_frame (cv2 BGR)
  → cv2.cvtColor BGR→RGB
  → ImageBuffer(buf=rgb, type=COLOR, dpi=capture_dpi, filestem='mybook_001', scan_id=...)
  → input queue.put(...)
  → worker runs full pipeline in place; persist_step → DB node+image per step
```

### PDF (2-page spread)

```
PDF page → PIL Image (extracted or rendered at median DPI)
  → raw stored as a scan in the project DB
  → ImageBuffer(COLOR, dpi, filestem=stem, scan_id=...)
  → input queue.put(ib)
  → worker pipeline: DPIfixer → SkewFinder → PageDetector
       PageDetector emits buf.children = [child_A, child_B]
       chain persists the parent node (marks it a branch point),
       persists each child node, re-injects each as a DB ref at step 4
  → each child: DPIfixer (300) → Binarizer → SkewFinder → PageDewarper
  → each step persisted as a DB node+image (branch labels A, B)
```

## Pitfalls

- **Always `copy()` before crossing process boundaries**. Copy a buffer before persisting or re-injecting it to avoid mutating the in-flight buffer.
- BW palette preservation: after a non-binarizing step runs on a BW input, the chain re-binarizes the result with `binarize_fixed(127)` so jpg compression artifacts don't accumulate.
- `roi` lives in **parent coordinates** until PageDetector remaps it via `cv2.intersectConvexConvex` (then it's in child coordinates).
- `dpi` is metadata only — the buffer is *not* automatically rescaled when you set it. Use `DPIfixer` (or `rescale()`).
