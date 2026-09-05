# Processor architecture review

*2026-09-05. Requested by Yann: "they should not bleed too much one into the
other; schema enforcement; processor type should allow the pipeline to
properly chain them and do the replay without handling for a specific
processor — this enables pipeline modifications, plugin additions, and a better
overall architecture."*

The question behind the review: **could a plugin processor be dropped into any
position of a pipeline and have the chain, the replay, the debug view, the log
and the manual-override system all treat it correctly, with no code outside the
plugin knowing its name?** Today the answer is *yes for the chain and the
replay, no for everything the user looks at*. The evidence, then the plan.

## 1. What already works by type

The typing surface that exists is real and load-bearing:

| Declaration | Who reads it | Verdict |
|---|---|---|
| `REPLAY_TRAIT` — `COORDINATE` / `PIXEL_VALUE` / `ROI` | `Replay._ordered_replay_steps` fuses by trait; the chain decides "skippable" and "hand it the whole override map" by trait; the replay **anchor** is found by trait (`_node_trait == "roi"`), not by name | **good** — the comments say "PageDetector", the code says `ROI` |
| `replay_transform(params, in_wh)` → `AffineTransform \| SampleMapTransform` | the replay engine composes them; `Replay._anchor_erase` inverts them | **good**, and under-used (§3) |
| `BatchableTrait` | the chain's dewarp batcher | good |
| `inject_step_options(step_opts, args)` | `Initializer` at chain build | fine |
| `PROVIDES_META` | **nobody** | declared, unread, and wrong (§2.3) |
| `MetaKind` schema (`aglaia/meta_schema.py`, 710f003) | `Meta` on every write; `transform_geometry` | new; enforced by tests |

`IntegratedProcessingChain` and `Replay` contain **zero** processor-name
conditionals. The 70 name references outside `aglaia/processors/` are almost
all in the GUI (`DebugViewerTab` ≈ 30, `debug_renderers` 10, `oplog` 7) and in
comments.

## 2. Where processors bleed into each other

### 2.1 Undeclared reads of another processor's meta

| Key | Written by | Read by | What happens if the writer is removed or reordered |
|---|---|---|---|
| `page_side` | PageDetector | TrapezoidalCorrection, PageDewarper | reader gets `None`, silently changes behaviour |
| `char_h_frac` | TrapezoidalCorrection **and** PageDewarper, each from its own copy of the estimator | nobody, yet | not a dependency — a **duplication**: two implementations of one estimate that can disagree. The fix is the reverse of a dependency: one shared estimator, meta as its cache (§4.1b) |
| `roi` | PageDetector, SkewFinder, DPIfixer… | Binarizer, everyone | fine — it is the one key everybody understands |
| `parent_crop_xywh` | PageDetector | TrapezoidalCorrection | silently `None` |
| `erase` | any erase producer (plugins) | Binarizer, Replay | designed for this; the only key with a written contract (`erase.py`) |

Nothing declares "I need `char_h_frac`". A pipeline with the keystone step
removed is **valid to the loader** and produces a worse dewarp with no warning.
That is the bleed the user is pointing at: the coupling is real but invisible,
so a pipeline edit cannot be checked.

### 2.2 Coordinate carrying, by hand, five times

Each COORDINATE processor moves `roi`/`erase` through its own warp with its own
code: `SkewFinder` an inline affine, `TrapezoidalCorrection` a `_CARRIED_META`
allow-list plus `_erase.carry`, `PageDewarper` a pad-translate then a remap in
two separate steps, `DPIfixer` a scaling loop that forgot `erase` for months.
This is **#139**: the base class applies `replay_transform` to every geometric
key after `process()`. Forward and replay then share one geometry, and a
forgotten key becomes impossible.

### 2.3 `PROVIDES_META` is wrong

```
DPIfixer               roi  min_dpi
MarginSetter           status  margin_mm
SkewFinder             skew  roi  max_angle
```

`min_dpi`, `margin_mm`, `max_angle` are **options**, not meta keys.
`Binarizer` declares nothing and writes `replay_kind`/`replay_params`. Since
nothing reads `PROVIDES_META`, nobody noticed. A declaration nobody validates
documents a wish.

## 3. Where the pipeline needs a name

### 3.1 The debug view (`DebugViewerTab`, `debug_renderers`)

Four name-keyed tables and an `if processor == …` ladder decide, per built-in:
which renderer draws the pane (`_RENDERERS`), which fields an override owns
(`_STAGE_FIELDS`), which editor page shows (`_editor_pages`), the strip colour
(`_PROC_PALETTE`), and whether the polygon handle is a `roi` or a `quad`.
A plugin processor gets a **different** path — `register_debug_renderer` and
the `erase`-only editable pane added for StampRemover — so there are two
mechanisms for one job, and the built-ins use the one plugins cannot.

### 3.2 The log (`oplog`)

Per-processor summary formatting (`angle=+1.4°`, `α=0.012 β=-0.003`) is a
name-keyed table in the logger. A plugin step logs `(W×H@DPI)` and nothing
else.

### 3.3 Manual overrides

Which fields a stage owns is `_STAGE_FIELDS`, by name. `manual.py` knows
`quad`, `roi`, `layouts`, `curl`, `force`, `skew_deg` individually. A plugin
cannot own an override field except `erase`, which was wired by hand.

## 4. The plan — a processor protocol

Everything below is *declarative on the processor* and *generic in the host*.
The test of each item: **delete the word "PageDetector" from the host and
nothing breaks.**

| # | Declaration on the processor | Host mechanism it replaces | Issue |
|---|---|---|---|
| 1 | `REQUIRES_META = {"page_side", "char_h_frac"}` and a corrected `PROVIDES_META` (meta keys only, kinds from the schema) | `Initializer` validates the chain at build: every required key is provided by an earlier step, or `MissingProcessorError`-style refusal **before the run**. A test asserts each processor's actual `meta[...]` writes ⊆ PROVIDES and reads ⊆ REQUIRES ∪ own PROVIDES. | new |
| 2 | `replay_transform` (exists) | base class carries every geometric key; `_CARRIED_META` and the four hand-written carries are deleted | #139 |
| 3 | `EDITABLE = {"roi": MetaKind.POLYGON}` / `{"quad": …}` / `{"skew_deg": SCALAR}` and `render_debug(img, parent, meta)` on the class | `_RENDERERS`, `_STAGE_FIELDS`, `_editor_pages`, the `if processor ==` ladder, and the plugin-only `register_debug_renderer` path all collapse into one lookup on the class. Built-ins and plugins use the same door. | new |
| 4 | `summary(meta) -> str` on the class | `oplog`'s name table | new |
| 5 | colour by `REPLAY_TRAIT` | `_PROC_PALETTE` by name — the palette already *means* "what kind of step", so key it on the kind | new |
| 6 | `PROVIDES_META` cleanup (drop the options) | — | folded into 1 |
| 1b | **meta as cache, never as the only source.** A step that needs a page statistic (`char_h_frac`) reads it from meta if an earlier step left it, else computes it — through the **same shared function** the writer used. The pipeline works with or without the writer; the two steps agree by construction; the only thing meta buys is skipping a recomputation. | #143 |

Order: 1 first (it is the one that turns invisible coupling into a build-time
error), then 3 (largest deletion of host code), then #139, 4 and 5.

## 5. What this buys

- A pipeline edit that breaks a dependency is refused **before** 300 pages are
  processed with a silently degraded dewarp.
- A plugin processor that declares `EDITABLE` and `render_debug` gets the same
  debug pane, handles and "Clear override" as a built-in, with no host change.
- The replay's geometry and the forward pass's geometry are the same function.
- `grep PageDetector aglaia/workers aglaia/gui aglaia/storage` returns comments.
