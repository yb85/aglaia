# Pipeline YAML

Pipelines live in `config/pipelines/*.yaml`. Default: `config/pipelines/book_curved_x2.yaml`. Override with `-p path/to/pipeline.yaml`.

## Schema

```yaml
name: "<human-readable name>"
pipeline:
  - name: "<step instance name>"        # required; becomes the node label (prefixed NN_)
    processor: "<ProcessorClassName>"   # required; an auto-discovered processor in aglaia/processors/
    options:
      <key>: <value>
      ...
```

Up to **99 steps** per pipeline (hard limit to keep the 2-digit prefix consistent — see `Initializer.create_processing_chain`).

## Instance names

For each step, `Initializer` computes `instance_name = f"{idx:02d}_{slugify(name, separator='_')}"`. Example: a step named `"pages_2ppf"` in position 3 becomes `03_pages_2ppf`. This string is the persisted node's `step_name` (and the `event_type` in `image_event`s) — **not** a filesystem directory.

The live `IntegratedProcessingChain` persists each step's output to the project's `.agl` SQLite file as a `nodes` row (+ an `images` row) via `aglaia/storage/persister.py` (`persist_step`). Every step stores its image, so every node has a real `image_id` (lineage and the stamped replay parameters live alongside it). The end-of-chain replay pass still fuses the per-step geometric transforms into one warp from the raw image for the final output.

## Unit-aware options

Processors that depend on the image's physical scale use plain mm/px
fields instead of templated strings. The Binarizer exposes:

```yaml
window_mm: 3.2     # preferred — converted via the buffer's DPI
window_px: 30      # used only when window_mm == 0
```

There is no expression evaluator — nothing to escape, nothing to load
from untrusted YAML.

## CLI overrides automatically injected

Some CLI flags override pipeline options at chain-build time (see `create_processing_chain`):

| Flag | Effect |
|---|---|
| `--debug` | Sets `step_opts["debug"] = True` on any step that already has a `debug` key |
| `--max-pages N` | Overrides `PageDetector.max_pages` |
| Camera calibration | If a `config/camera_params.json` exists, its matrix is injected into every `PageDewarper` step as `camera_matrix` + `camera_matrix_resolution` |

## Processors and their option dataclasses

See `docs/processors.md` for full per-processor details.

| `processor:` | Option dataclass | Source |
|---|---|---|
| `DPIfixer` | `DPIfixerOption` | `aglaia/processors/DPIfixer.py` |
| `SkewFinder` | `SkewFinderOption` | `aglaia/processors/SkewFinder.py` |
| `PageDetector` | `PageOption` | `aglaia/processors/PageDetector.py` |
| `Binarizer` | `BinarizerOption` | `aglaia/processors/Binarizer.py` |
| `TrapezoidalCorrection` | `TrapezoidalOption` | `aglaia/processors/TrapezoidalCorrection.py` |
| `PageDewarper` | `DewarpOption` | `aglaia/processors/PageDewarper.py` |
| `MarginSetter` | `MarginSetterOption` | `aglaia/processors/MarginSetter.py` |

Unknown processor names print a warning and are skipped.

## The default pipeline (annotated)

`config/pipelines/book_curved_x2.yaml`:

```yaml
name: "Standard Pipeline"
pipeline:
  - name: "dpi_clamp_input"
    processor: "DPIfixer"
    options:
      min_dpi: 100
      max_dpi: 300                   # Resample anything outside [100, 300] dpi

  - name: "capture_deskew"
    processor: "SkewFinder"
    options:
      max_angle: 30.0                # Search ±30° (camera scans can be tilted hard)
      min_angle: 0.5                 # Don't bother rotating below 0.5°
      apply_rotation: true
      k_cluster: 0                   # 0 = white border; >1 = k-means background detect

  - name: "pages_2ppf"
    processor: "PageDetector"
    options:
      margin_mm: 5                   # Margin around detected text bbox
      max_pages: 2                 # 2 pages per frame (book spread)
      processing_dpi: 150            # Downsample to 150dpi for detection only
      rescale_threshold: 0.01
      backend: auto                  # apple_vision → east → dbnet → heuristic
      min_contrast: 0.5              # drop bleed-through ghosts (<0.5), keep dim real pages

  - name: "dpi_normalize_output"
    processor: "DPIfixer"
    options:
      min_dpi: 300                   # Force 300dpi (both bounds same)
      max_dpi: 300

  - name: "pages_bw"
    processor: "Binarizer"
    options:
      method: "wolf++"
      window_mm_wolf: 5              # Wolf window in mm (family-specific option)
      k_wolf: 0.25                   # Threshold bias for the wolf family
      roi_shrink: 5                  # Erode ROI mask (kills border noise)
      morpho_close: 2                # Morphological close after threshold

  - name: "pages_deskew"
    processor: "SkewFinder"
    options:
       max_angle: 5.0                # Already roughly straight after dewarp; small range
       min_angle: 0.1
       apply_rotation: true
       k_cluster: 2                  # K-means BG detection for non-white pages

  - name: "pages_dewarp"
    processor: "PageDewarper"
    options:
      max_oob: 400                   # Reject dewarp if remap goes >400px out of bounds
      processing_dpi: 150            # Spans detected at 150dpi
      page_margin_mm: 15
      dewarp_margin: 15
      shear_cost: 0
      remap_decimate: 16
      focal_length: 1.2              # Overridden by camera calibration if present
      debug: false
```

Order matters. The default flow:

```
raw → clamp dpi → deskew page → detect pages (split) →
   each child: normalize to 300dpi → binarize → deskew → dewarp
```

After `PageDetector` produces children, each child is persisted as its own DB node (the parent node is marked a branch point) and re-injected — as a lightweight DB reference, not a pixel copy — at step 4 (`dpi_normalize_output`).

## Replay pass

After a branch finishes its forward run, the chain runs a **replay pass**
that recomputes the final image from the nearest persisted upstream image
with the minimum number of interpolations. This is what makes a mid-pipeline
binarize (`pages_bw` runs at step 5, before the warps) come out crisp:
replay defers it.

Replay is driven by each processor's `REPLAY_TRAIT`, not by manual ordering:

| Trait | Processors | Replay behaviour |
|---|---|---|
| `COORDINATE` | DPIfixer (resample), SkewFinder, TrapezoidalCorrection, PageDewarper | Coordinate remap `x'=f(x)`, value-preserving. A contiguous run **composes into one warp** (single interpolation) — e.g. the DPI upscale + keystone + dewarp fuse into one `cv2.remap` off the ROI-anchor crop. Set `AGLAIA_REPLAY_NOFUSE=1` to force the sequential per-step path. |
| `PIXEL_VALUE` | Binarizer | Value op `v'=g(v, neighbourhood)`. Pushed **as late as possible** — applied once, on the final geometry, so thresholding sees rectified text and a single quantisation. |
| `ROI` | PageDetector, MarginSetter | Changes the region (crop / branch / pad). A **fixed barrier**: replay never reorders across it, and a segment's ROI anchor image is the point replay starts from. |

A step **per-page disabled** (issue #68) is bypassed with a passthrough node
that stamps no `replay_kind`, so replay excludes it automatically — the
downstream transforms reconstruct from the un-transformed image. If *every*
replay-participating step on a branch is disabled, replay no-ops and the
forward terminal stands.

Replay ordering within a segment is derived: `COORDINATE` (pipeline order)
→ `PIXEL_VALUE` (last) → terminal `ROI`. Boundary/missing pixels introduced
by a warp (rotation corners, dewarp growth) are tracked with an ROI mask
carried through every coordinate step, so the late pixel-value op ignores
them. Result: equivalent to the forward pipeline but at higher fidelity
(fewer interpolations, one quantisation, value ops in final geometry).

## Authoring a new pipeline

1. Copy `config/pipelines/book_curved_x2.yaml`.
2. Reorder / add / remove steps. Use unique `name:` fields.
3. Pass it with `-p path/to/new.yaml`.
