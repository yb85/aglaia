# Changelog

All notable changes to Aglaïa are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0rc5] — 2026-09-05

The plugin store and a plugin API (#133), the CLI that drives all of it
(`aglaia plugins`, `--send-to`, `aglaia skill`), erase masks with a
StampRemover plugin to prove them, a declared schema for page metadata,
per-page manual tuning (milestone M9), editable capture shortcuts, and a
run of output defects that had been degrading pages silently — most with no
failing test, and all found by measuring real projects rather than by reading
the code.

### Added

- **A plugin store, and nothing ships inside the app** (#133). Plugins install
  from the registry (github.com/yb85/aglaia-plugins) or a local archive into
  `<APP_DATA>/plugins/<kind>/<slug>/`, in three kinds: processors (pipeline
  steps), OCR engines, and **destinations** — somewhere a finished export goes
  (Kindle, Calibre, a corpus). Each gets a `PluginContext`: settings in the
  config DB, secrets in the OS keychain under its own namespace, a scratch
  dir, a log line. The Plugins tab installs, updates in place when the
  registry has a newer version, disables, removes — and the install dialog
  says who wrote the plugin, because that is what the user is being asked to
  judge. A pipeline step whose plugin is missing or disabled **fails and says
  so**, instead of quietly doing less.

- **An exporter is an exporter.** Installed destinations appear in the Export
  tab as ordinary format cards beside PDF and Markdown, run by the same
  Export button; the send runs off the GUI thread with a spinner and a toast,
  because mail to a Kindle takes time. A destination that is not set up says
  so on the card, before the export runs.

- **The CLI reaches all of it.** `aglaia plugins list|search|install|update|
  toggle|remove|config` — `config` is a small interactive view over the
  plugin's declared fields, or `--set key=value` for scripts, `--test` to
  check the connection. `run`/`ocr` take `--send-to SLUG+SLUG` to hand the
  exports to destinations. `aglaia skill` prints the agent skill: what each
  command is for, what to ask the user, DPI arithmetic, recipes, gotchas — a
  test walks the Typer app and fails if any command or option is missing
  from it.

- **Erase masks** — `meta["erase"]`, a list of polygons any processor may add
  and every coordinate step carries through resample, deskew, keystone and
  dewarp; the binariser treats the inside as missing paper (`wolf++`), so a
  removed stamp leaves no halo, in the forward pass and at replay alike. The
  **StampRemover** plugin proves it: trace a stamp once in its own window,
  name it, export/import the library as JSON; it then finds the stamp on any
  page by SIFT features at native resolution (57 ms/page over 240 real pages,
  3/3 found, 0 false positives) and erases it with a `margin_mm` halo. Its
  debug pane shows the mask and the matched keypoints and lets the user add,
  move or delete mask polygons by hand — plugins can draw and own an editable
  debug pane.

- **A declared schema for `ImageBuffer.meta`** (#139). Every key has a kind —
  polygon, polygons, points, rect, scalar, label, opaque — and a coordinate
  transform moves every geometric key by its kind instead of each processor
  remembering its own list. Undeclared keys are dropped at a warp;
  `AGLAIA_META_STRICT=1` makes a write of one an error, which surfaced ten
  keys nobody had declared.

- **Cameras remember their transform.** Rotation and mirroring are stored per
  camera in the app database, so a webcam mounted upside-down stays upright
  across projects.

- **Rerun one page** from its card header, which now reads
  `Scan 012 · 300 dpi` with the delete control dimmed. **Jump to the first or
  last scan** in the gallery (Home/End, ⌘↑/⌘↓, shift-click the chevrons).

- **Tune a page by hand, in the debug view.** The view a user already opens to
  see what the pipeline decided is now where they correct it. Four stages take
  a manual value, stored per page-layout in a new `manual_overrides` table and
  honoured on the next run:

  | Stage | Control |
  |---|---|
  | SkewFinder | a rotation handle on the image and a slider |
  | PageDetector | the ROI polygon — drag a vertex, double-click an edge to add one |
  | TrapezoidalCorrection | the column quad — drag a corner (four corners, no more) |
  | PageDewarper | arch / tilt / spine-γ sliders, and **Force dewarp** |

  Ported from the iOS `ManualOverrides` model. An overridden estimator does not
  run at all: run-then-discard would be cheaper to write and wrong to live
  with, because the next rerun would re-derive its own answer and the
  correction would drift. The dewarp freezes the sheet at the user's curl and
  re-optimises only the pose — freeze the pose too and the sliders look inert,
  leave the shape free and every slider move snaps back.

  **Arch and tilt are not the fitted parameters.** The solver fits α and β,
  the sheet's slopes at the two page edges, and neither moves one visible
  thing on its own — which is what made them unusable by hand. Rotating the
  pair (`arch = (α−β)/2`, `tilt = (α+β)/2`) gives `z(0.5) = arch/4`, so arch
  alone sets the mid-page rise and tilt alone slides its crest. The grid
  previews the sheet live, in magenta over the fitted green, from the same
  builder the remap uses.

  Every spatial edit stores the frame it was drawn on: a polygon applied to a
  frame of another size is silently shifted, and a page that is quietly wrong
  is worse than one the pipeline decided alone.

- **Force dewarp.** Runs the fit past the `min_spans` guard and the `max_oob`
  gate. Both are right by default and both are sometimes wrong — a sparse page
  whose few spans are perfectly good, a wide fit read as runaway. Per page,
  never a default, and stamped so a bad page stays explainable.

- **A hand-edited page is marked** in the table, the card grid and the gallery
  by one quiet dot, whose tooltip names what was touched.

- **Capture shortcuts are editable, two per action** (#103). A pencil beside
  the shortcut legend opens a modal; clicking a slot **arms** it and the next
  key or combination pressed is what it becomes. Bindings persist per user and
  override the YAML `keycontrols`.

  The old matcher compared key NAMES against `event.text()` and a table of
  seven names, and never looked at `event.modifiers()` — so no combination was
  expressible at all. Matching moves to `QKeySequence`. That is what makes the
  case this was built for work: a **presentation remote** whose fullscreen
  button cycles between `Shift+F5` and `Esc`, both bound to capture.

- **Convex-hull page ROI** (`roi_hull`, default on), backported from iOS. The
  child ROI is the region the Binarizer keeps; the axis-aligned bbox of a
  slanted text block swallows its corners, which is exactly where the fingers
  holding the book sit.

### Removed

- **The three bundled destinations** (`aglaia/plugins/`), with the `cloud`
  extra's meaning. They loaded unconditionally, so "Export to Calibre server"
  sat in every install's Export tab whether or not anyone had a Calibre, and a
  Kindle plugin's SMTP settings existed in a build belonging to someone with no
  Kindle. Install them from the registry instead. `keyring` and `mistralai`
  become base dependencies — a card that raises ImportError when clicked is
  broken, not optional — and `--extra cloud` is kept as an empty alias so
  existing commands still work.

- **The CUDA build target.** It existed to batch the JAX page-dewarp on a GPU,
  back when the alternative was Powell. `backend: auto` now resolves to the LM
  solver, which fits the same sheet on CPU in ~0.25 s/page — about 100× faster
  than the path the GPU was racing — so `Aglaia-x86_64-cuda.AppImage` carried
  a slim CUDA payload for no gain. Its build job had also been failing since
  2026-07 (`xgrammar` publishes no wheel for the runner), so every release
  from rc3 on published without it regardless. The `cuda` extra and
  `Aglaia.spec`'s slim-CUDA block are kept for a local build.

### Changed

- **Every string the user reads was rewritten for the user** (#138), against a
  written standard (`docs/ui-writing.md`) distilled from thirty well-crafted
  open-source apps. No env var names, headers, module paths, spliced
  exceptions or design rationale in the UI: say what it is and where it goes,
  and leave the mechanism to the log. 97 strings retranslated; fr_FR complete.

- **`wolf++` is `wolf++` in both passes.** The forward pass mapped it to
  plain Wolf and only the replay used the mask-aware variant, so the two
  passes binarised differently and a stamp erased in one came out with a halo
  in the other. Cost measured marginal.

- **One char-height estimator** (#143). TrapezoidalCorrection and PageDewarper
  each measured text height their own way; now `text_metrics` does it once,
  and `meta["char_h_frac"]` is a cache the dewarper fills itself when the
  trapezoid step did not run.

- **The default page margin is 2 mm**, stated explicitly in all four shipped
  pipelines. They said 5 and 15 — inconsistent between workflows as well as
  within a page (#112).

- **`min_spans` 4 → 3** (#108). That floor was tuned for the retired Powell
  optimizer; the LM solver recovers a sheet from far fewer baselines, so the
  guard was refusing pages it fits well.

### Fixed

- **The stamp came back at replay.** `_anchor_erase` guarded on a frame
  that was never set, so the erase polygons were never mapped back to the
  anchor and the replay binarised the stamp in. Inverse composition through
  the stored transforms replaces the dead guard.

- **The forward pass erased far more than the mask.** A default 6 px halo in
  `fill_with_paper` ate 1625 text pixels around one stamp; halo is 0 and the
  margin is the user's `margin_mm`.

- **DPIfixer left erase polygons in pre-resample coordinates**, so a stamp
  traced at capture resolution was erased at the wrong place after DPI
  normalisation.

- **The erase editor hijacked every other stage** — it armed on any
  page that had ever stored an erase payload, so dragging a dewarp slider
  moved mask vertices. It arms only on a stage whose geometry carries `erase`.
  **Clicking a handle grabs the handle you clicked**, not the first one in
  range.

- **A processor installed from the store was invisible to the pipeline**,
  and got no `PluginContext`; the Plugins menu appeared twice.
  **Uninstalling a destination** deleted its files but left it registered, so
  it stayed on offer until restart.

- **Installs froze the window** — the download and the sha check ran on the
  GUI thread — and crawled on a dual-stack network; outbound HTTP now goes
  through one IPv4-preferring client, off the GUI thread.

- **The scan number vanished from the card header** on Retina: a predicted
  label width rounded the wrong way. The header stops predicting widths.

- **The project name was slugified** for the `.agl` and the derived export
  names. It is kept exactly as typed.

- **The margin you set was not the margin you got.** Measured over 40 pages of
  a real project asking for 5 mm: top and bottom exact, left and right never
  the requested value and varying 6.6 mm across pages (10.2–16.8 mm).
  `MarginSetter._enforce_width_floor` padded the page back out to the step's
  INPUT width whenever the crop came out narrower — which is every page,
  because that input is the dewarp canvas and stripping its whitespace is the
  crop's whole job. Horizontally the floor re-added exactly what the crop had
  removed. It becomes `width_floor`, **off by default**. Re-measured on the
  same 40 pages: every side 2.03 mm, zero spread; pages come out 16–25%
  narrower, which is the leftover dewarp whitespace going away (#112).

- **A reordered card vanished** (#105). `FlowLayout.removeWidget` hides the
  widget on the way out, and `hide()` sets the explicit hide flag that
  `addChildWidget` does not clear — so the card returned to the right slot and
  stayed invisible. Under it sat a second defect: the list and the gallery
  enumerated scans by `scan_id` while the grid ordered by `page_order`, so the
  two agreed only until the first reorder. One source of display order now
  serves all three views.

- **The dewarp composite dropped to a bare image mid-rerun** (#106).
  Committing a slider reruns the branch, and the rebuild cleared the rendered
  overlays for as long as the background render took — so the picture the live
  grid preview exists to be compared against disappeared exactly while the
  comparison was being made.

- **"No OS keychain was available" on a Mac whose Keychain works** (#107). The
  real cause is that `keyring` is not installed: it ships in the `cloud`
  extra, which the usual dev sync omits. One bare `except` around both the
  import and the write reported a missing *package* as a missing *backend*,
  and the key was silently downgraded to plain text. The dialog now warns
  before the key is typed.

- **One "Check result" click disabled the button for the session** (#111).
  `deleteLater()` freed the worker wrapper while the attribute still pointed
  at it, and `isRunning()` on a freed QThread raises rather than answering
  False — so the guard unwound out of the slot before the worker was ever
  constructed, and every later click was a no-op until the project was
  reopened.

- **The pipeline-mode artwork painted itself black** (#114). Those SVGs fill
  with `currentColor`, which `QSvgRenderer` does not resolve — it falls back
  to black, so the book artwork was near-invisible on the dark palette while
  the Lucide icon beside it, already routed through the tinting renderer,
  looked right. Measured on the bundled assets: the old path yields ink
  `rgb(0, 0, 0)`; tinted, the hero comes out `(239, 239, 239)` on dark and
  `(23, 23, 27)` on light.

- **Landscape page crops came out downscaled, still stamped 300 dpi.** The
  remap sized its output height against `ref_h`, but `page_dims` lives in
  norm2pix units whose scale is `0.5 · max(h, w)` — right only on a portrait
  crop. On a landscape one (a chapter opening, a part-title, anything wider
  than tall) the whole remap was scaled by `ref_h / ref_w`. Measured on one
  book: 26 of 272 dewarp nodes are landscape and **every one** was shrunk —
  a 1289×533 page landed at 520×224. Portrait crops are unaffected, so
  stamped projects replay to the size they have today. Inherited from upstream
  `page-dewarp`.

- **`min_contrast` deleted sparse real pages.** The filter exists to delete a
  bleed-through ghost — pale wherever its ink is — but measured `p95 − p5`
  over the layout's whole bbox, which makes it a density proxy: a sparse
  layout is mostly paper, so `p5` lands on paper too and the range collapses
  however black the ink. Over 141 spreads it deleted a **title page**, two
  chapter openings and two chronology date columns at 0.12–0.35, and did
  nothing else — on that book the filter never once did the job it exists for.
  Measured under the detection boxes instead, those five score 0.78–0.98 and
  the worst page of the whole book scores 0.68, with no threshold change.

- **The horizontal gap trim ate text.** It drops a box sitting a large gap
  past the dense cluster — a cable, a hand, a cup edge — and cannot tell that
  from a legitimate element that simply stands alone. It removed a lone `1996`
  from a chronology column, and cut the two longest lines of a short ragged
  block **mid-word**. The discriminator is the side: an intruder comes from
  outside the book, never from the gutter, so on a two-page spread the spine
  side is bounded by the crease instead. 139 of 141 spreads unchanged; the two
  that move are the two failures.

- **The dewarp debug grid drew the wrong surface.** It projected without the
  10-param camera's principal point or the fitted spine curl, and ignored the
  composite downscale — so the overlay sat up to 895 px from a correct output,
  reading as a broken dewarp. Debug-only; no output image was affected.

- **On Linux the Cloud OCR card never showed a key stored in the keychain.**
  The probe was deferred for the macOS Keychain, which prompts on read; the
  Linux Secret Service does not. And its only trigger needed a real engine
  switch, so a session that STARTS on Cloud OCR never probed at all. OCR
  itself was unaffected — only the status line lied.

## [0.1.0rc4] — 2026-08-31

Point release: cloud OCR was broken in every rc3 artifact.

### Fixed

- **Cloud OCR (Mistral) broken by a dependency upgrade.** `uv lock --upgrade`
  in the 0.1.0rc3 cycle moved `mistralai` 1.5.2 → 2.9.4, which is a
  restructure rather than an API bump: the top-level `mistralai` became a
  **namespace package** (no `__init__.py`) and `Mistral` moved to
  `mistralai.client`. Every run failed with
  `ImportError: cannot import name 'Mistral' from 'mistralai' (unknown
  location)`. Pinned to `<2` — the 1.x surface this code calls
  (`files.upload` / `get_signed_url` / `download`, `ocr.process`,
  `batch.jobs.create|get|cancel|list`) all needs re-verifying against a paid
  API before the ceiling can be lifted.

  Nothing caught it because cloud OCR needs a paid key and is never exercised
  in CI. Added `tests/workers/test_mistral_sdk_contract.py`: key-free,
  network-free checks that the SDK is shaped the way the code calls it, plus a
  guard on the pin itself — the latter runs without the `cloud` extra, so it
  fires in CI, which is where a future `uv lock --upgrade` would reintroduce
  this.

  **All rc3 artifacts were affected** — the DMG, the Windows installer and
  the AppImage all ship the `cloud` extra, so every one of them bundled the
  broken 2.9.4. rc4 exists to replace them.

## [0.1.0rc3] — 2026-08-30

A dewarp cycle: the sheet fit is ~100× faster and measurably straighter, the
four sheet models are down to one, and three defects that had been degrading
output silently — with no failing test — are fixed. Most of it is backported
from the iOS port (`aglaia-ios`), which had diverged ahead on the shared
algorithms.

Headline, measured over the fixture corpus (3 books, 12 pages) as mean
deviation of the output text baselines from a straight line:

| | curvature | per page |
|---|---|---|
| before | 2.04 px | 75.7 s |
| after | **0.92 px** | **0.66 s** |

### Added

- **LM dewarp solver** (`aglaia/processors/lm_solver.py`, #59). Levenberg-
  Marquardt with analytic Jacobians and the problem's bipartite sparsity —
  per-span *arrowhead* blocks eliminated in O(nᵢ), Schur complement down to one
  small solve per iteration. Now the default backend for the `cylindrical`
  sheet model (`backend: auto`), replacing ~10⁵ Powell objective evaluations
  with ~10-60 iterations: 28 s → 0.25 s per page on the test fixture, at a
  *better* final objective, on CPU with no GPU allocator to babysit. Ported
  from the iOS implementation; `AGLAIA_DEWARP_LM=0` keeps the old GPU chain.

- **Dewarp camera upgrades** (LM + `cylindrical` only, #60), from the iOS
  port's on-device research:
  - `principal_point` (default on) fits (cx, cy) alongside the pose. A
    homography composed with a perspective view of a curled page is a general
    projective camera the centred pinhole cannot represent — this is why the
    fit under-curled near the spine after `TrapezoidalCorrection`. Measured
    output line curvature 1.77 → 0.98 px on the fixture page.
  - `spine_weight_boost` / `spine_weight_zone` up-weight the binding-side
    residuals, which are otherwise a least-squares minority (2-4 px systematic
    under-curl at 300 dpi).
  - `spine_gammas` grid-searches a spine-localized directrix term
    `z += γ·exp(−|x−x_binding|/s)`, keeping the best objective (γ = 0 always
    competes). Only identifiable with the principal point; also supplies the
    surface slope the arc-length grid needs at the fold, fixing horizontal
    compression there.

  Replay params gained `camera_np` and `spine`; stamps written before this
  release default to the 8-param pinhole with no spine term, i.e. exactly the
  surface they were fitted on.

  Measured over `test_data/test_athanase` through `book_curved_x2`, mean
  output-baseline curvature: Powell 2.04 px @ 75.7 s/page → LM + the new
  camera 0.90 px @ 0.78 s/page. LM with the *old* 8-param camera is not
  uniformly better than Powell (2.96 px — it can park a shape DOF on the ±0.5
  curl clamp), which is why `principal_point` defaults on and a rejected LM
  fit now retries on Powell instead of falling straight through to grayscale.

- **Same-line footnotes** (`mistral_same_line`, default off). Critical editions
  pack several notes onto one physical line (`"(12) premier. (13) second."`).
  Every marker after the first never sits at a line start, so it never entered
  the entry set — and since a footnote is recognised by refs ∩ entries, it was
  never classified as a footnote **at all**: its ref stayed a bare `(13)` in
  the body and its text stayed glued to note 12. With the toggle on, markers
  inside a line that already starts with an entry marker also count, and such
  a line is split at each of them. Only markers already in that page's mapping
  cut (a citation inside a note never splits it), and the bare `N.` form never
  cuts mid-line (too ambiguous against dates and verse references).

- **Slope-based x decompression** (`slope_emphasis`, ships at k = 1).
  The arc-length grid corrects the *surface-length* term; it does not correct
  **projective foreshortening**, where the page recedes steeply near the spine
  and the camera compresses those glyphs however the surface is
  parameterised. `slope_emphasis` (k) weights the grid measure by
  `(1+z′²)^(k/2)` so steep regions claim more output width, and is carried in
  the replay stamp so a replayed page is sized like the live one.

  The effect scales with z′², so it is negligible on mild curl and large on
  strong: at k = 1, +0.3% output width at max|z′| = 0.1 but **+16.9%** at 1.67,
  with the steepest quarter's share of output columns going 0.318 → 0.403.

  **Ships at k = 1**, matching the iOS port where it is device-validated. That
  rests on iOS validation plus the maintainer's judgement, not on a local
  measurement: the A/B harness scores baseline straightness (vertical) and this
  changes horizontal scale, so it can only confirm no regression — which it
  does, 0.926 px vs 0.923 at the noise floor. `slope_emphasis: 0` recovers the
  exact previous geometry, and projects stamped before this option replay at 0
  regardless of the default.

- **Spine-aware keystone estimation** (`spine_aware`, default off). Near the
  binding the page curls out of plane, so the bottom-most ink sits
  progressively lower than the true baseline — evidence that is systematically
  *wrong*, not merely noisy. Using the binding side (PageDetector's
  `page_side`, which survives this step since the propagation fix above):
  baseline evidence in the fold zone is down-weighted (RANSAC band scales with
  the weight, candidates score by weight sum, refit is weighted least
  squares); the fold-side endpoint cluster gets a relaxed bandwidth because
  its members wobble; and a tilt disagreement backed by strong support on both
  sides is kept as real keystone instead of being reconciled away.

  Measured on a synthetic curled line, the weighting cuts the curl-induced
  baseline tilt by **57-69%**. **The fixture corpus does not corroborate it**:
  mean 0.926 → 0.929 px, median 0.879 → 0.897 (worse, ~9× the noise floor),
  worst page 1.206 → 1.128 (better), and the per-page benefit correlates
  **+0.275** with curl — the opposite of the mechanism's premise. Shipped
  **off** on that evidence, but kept and documented rather than dropped: the
  direct measurement is strong and the iOS port device-validated it on
  handheld captures, a curl regime the rig-capture fixtures do not cover.
  Turn it on for handheld or strongly-curled material.

- **A flat-fit rung in the dewarp failure ladder.** When a fit's remap
  overshoots the `max_oob` gate, the page now falls back to a zero-curl fit
  (pose and page dims around a flat sheet, ~6 ms) before being conceded to the
  grayscale passthrough. A zero-curl surface cannot produce the runaway remap
  that trips the gate, so this rung always has an answer — the page comes out
  perspective-corrected but not curl-corrected, rather than not processed.

  It also covers the one failure the rung above does not: LM parking a shape
  DOF on the ±0.5 curl clamp at a competitive objective but a wild remap (the
  case the iOS port answers with Powell, which costs 25-80 s here). Dormant on
  healthy pages — 0 firings across the 12-page corpus, output unchanged at
  0.923 px — and deliberately not recorded in the warm-start ring, since a
  curl of 0 by construction is not a measurement.

### Fixed

- **The keystone de-rotate no longer shears a page whose deskew failed.**
  `detect_column_quad_from_baselines` re-centres near-parallel tilted margins
  onto the median baseline angle, on the premise that the page was deskewed
  upstream so level baselines are ground truth. That premise fails exactly
  when `pages_deskew` returns 0 on a fanned page (no peak in its projection
  profile): the baselines keep their tilt, and margins disagreeing with them
  are real shear, not invented rotation. Un-gated it forced the sides onto the
  baseline angle — flat text lines, sheared verticals, the quad corner ~100 px
  off the ink. Now gated on `|median baseline angle| < 1°`. Backported from
  the iOS port, where it was found on a real capture and verified against the
  desktop reference byte-for-byte.

- **A headless run no longer abandons a page whose step outlives it** (#64).
  Completion decided the run had settled from a *silence timer*: once every
  scan had emitted one `branch_ready` and no further branch event arrived for
  8 s, it returned and the caller stopped the chain — with workers still
  mid-page. Those pages vanished: no node, no error, exit code 0. It now waits
  for the chain to report no work in flight (`is_idle`, debounced), and the
  worker holds an unconditional busy marker so that signal can't under-report
  (the resumable in-flight reference it used to rely on is only recorded for
  buffers carrying a `parent_node_id`).

  Measured on the same two spreads with the ~75 s/page `powell` backend:
  before, 2 of 4 pages lost in a 30 s "successful" run; after, 4 of 4 in 60 s.

### Removed

- **The twist/spline sheet models** (`sine_twist`, `bspline_twist`,
  `flat_spline`) and their options (`sheet_model`, `spline_modes`, `twist`,
  `knot_grading`, `flat_outer_penalty`). They were strictly more expressive on
  parameter count and strictly worse on results. A/B over the fixture corpus
  (3 books, 12 pages), mean deviation of the output baselines from straight:

  | model | mean | worst page |
  |---|---|---|
  | **cylindrical + spine curl** | **0.923 px** | **1.194** |
  | flat_spline | 1.679 px | 3.315 |
  | bspline_twist | 1.648 px | 3.053 |
  | sine_twist | 1.645 px | 3.051 |

  The cylindrical model wins on **every one of the 12 pages**. A fold
  concentrates curvature at the spine, which one localized exponential term
  captures robustly; the global spline families spread degrees of freedom
  where there is no curvature and overfit noisy spans.

  **Hard break**: a `.agl` node stamped with one of these models now raises on
  replay rather than being silently rebuilt against a different surface —
  re-process such a page from source.

  Removes ~900 lines: `sheet_models.py` 592 → 171, the spline objectives in
  the MLX and padded-JAX backends, the model plumbing in `PageDewarper` and
  the dewarp batcher, and two test modules. Output for the surviving model is
  bit-identical (verified: 0.923 px on all 12 pages, before and after).

## [0.1.0rc2] — 2026-07-02

Second release candidate. A large body of work landed since rc1: a subcommand
CLI, a phone-handoff bridge, a warm-pool job server, and a full OCR-engine
overhaul (local VLMs + cloud Mistral post-processing), plus many capture/GUI
stability fixes.

### Added

- **Subcommand CLI (Typer).** `aglaia [gui] [PROJECT]` (default), `run`, `ocr`,
  `server`, `setup`, `list`, `version`. `aglaia ocr` OCRs PDFs/images (or
  re-OCRs a `.agl`) with **no** processing chain — for already-clean docs.
- **Receive from phone.** A TLS receiver (QR-pinned, token-gated `/import`)
  plus an `.aglbundle` reader for the iOS handoff — capture on the phone, finish
  on the desktop.
- **Job server** (`server` extra). Warm-pool HTTP job API: run/list/check/get/
  delete/admin, processing, Mistral-batch backoff, downloads, email + admin.
- **Local VLM OCR.** An OCR-agnostic local VLM server layer with two engines —
  GLM-OCR and Baidu Unlimited-OCR (in-process MLX on macOS, vLLM on CUDA) — and
  a `DirectBlockOCR` trait so any block recogniser can complement `apple_docs`.
- **Per-engine OCR layers + export layer selection.** Keep multiple engines'
  results per page; pick which layer to export (PDF text layer + Markdown).
- **Mistral markdown post-processing.** Footnote conversion (LaTeX/Unicode
  superscripts and `(N)` → GFM `[^N]`, unique anchors that keep the original
  number) and header/footer extraction — with toggles on the Markdown export
  card, applied at export time (re-export reflects changes without re-OCR).
- **Central download registry** with resumable CLI downloads (retires
  `model-list.json`).
- **Dewarp warm-start** (curl seeded from recent same-side fits) and automatic
  discard of a degenerate trapezoid keystone.

### Changed

- **OCR post-processing is tied to markdown export, not OCR.** The raw engine
  output is stored; footnote/header-footer transforms run at export.
- **Surya** moved off its torch/GGUF stack onto the local VLM server layer.
- **Dropped the PaddleOCR-VL (`paddle_vl`) engine** — weak on Greek, heavy deps.
- Local VLM backend is **bundled by platform** (MLX / vLLM).
- Default `ocr_dpi` is **200** (matches the GUI); added `--ocr-dpi`.
- Cloud whole-document engines route through **one request** (fixes Mistral
  per-page billing); markdown export scan/branch markers are now
  `<!-- scan #N -->` + `<!-- branch N.A -->`.

### Fixed

- **QThread teardown crashes** ("Destroyed while thread is still running") on
  Mistral batch-check and on **deactivating voice control** — workers are now
  retained until they actually finish.
- **DPI-calibration "Trace manually" froze the window** (runaway width) and blew
  GUI RAM into the GBs — the trace canvas now paints directly instead of
  driving the layout via `setPixmap`.
- A fresh **capture session opens on the Capture tab** instead of the last-used
  sidebar tab.
- **Served-VLM degeneration loops** (repetition penalty), a **Cyrillic
  block-splice leak**, and script-anomaly garbage detection for complements.
- OCR **progress/ETA**: over-count (334/322), jumpy ETA, `s/page` labelling, and
  an idle watchdog that snapped the bar to 100%.
- **XY-cut page splitter** + absorb-smallest merge for DBnet 2-up scans; layout
  overlay renderer; per-page step toggle dead after reprocess; CLI now passes
  the page DPI so OCR honours the configured `ocr_dpi`.

## [0.1.0rc1] — 2026-06-27

First release candidate. Linux and Windows builds confirmed working; macOS
re-verified (full test suite + end-to-end headless chain on the MLX backend).

### Fixed

- **cv2 collision broke installs non-deterministically.** `page-dewarp` pulls a
  bare `opencv-python` (GUI build) while Aglaïa pins `opencv-python-headless`;
  both write the same `cv2/` directory. With `numpy<2.1` holding headless at
  4.11 and the GUI build floating to 4.13, a reinstall could leave a
  half-written `cv2/` (`cv2 has no attribute 'imdecode'`), and the bundle picked
  up whichever payload won. Pinned `opencv-python` to 4.11.0.86 so the shared
  `cv2/` is always one consistent payload.
- **Stuck per-card spinner.** A card could stay dimmed + spinning after the chain
  went idle and the progress bar read 100% (a dropped `branch_ready`). Idle
  reconciliation now sweeps any card still marked processing, even once the bar
  has finished.
- **Lost page-visibility toggle.** A hide/show whose `(scan, label)` matched no
  branch row was silently dropped and reappeared on reload; it now logs a loud
  `[visibility]` warning so the offending label can be diagnosed.

## [0.1.0a6] — 2026-06-26

Sixth alpha. Same as a5 plus a Windows build fix (a5's Windows installer
failed its release gate, so no a5 `.exe` shipped).

### Fixed

- **Windows "database is locked" under concurrent writers.** `journal_mode=DELETE`
  recreates the rollback journal per write; on Windows the file syscalls (plus
  AV scanning) make contended writes serialize past the old 5 s `busy_timeout`.
  Bumped to 20 s and added a bounded backoff-retry on the hot insert path (keeps
  the single-file `.agl` design — no WAL sidecars). Unblocks the Windows
  installer/ZIP.

## [0.1.0a5] — 2026-06-26

Fifth alpha. GPU Linux AppImage, faster dewarp, auto worker count.

### Added

- **Prebuilt slim-CUDA Linux AppImage** (`Aglaia-x86_64-cuda.AppImage`) for
  GPU-accelerated page dewarp on NVIDIA/Linux — no source / `--extra cuda`
  install needed. The dewarp is matmul-only L-BFGS-B, so the bundle ships only
  the CUDA libs it loads (cuBLAS, nvrtc, nvjitlink, ptxas, cupti, cudart) and
  drops ~2.6 GB of dead weight (cuDNN/NCCL/nvshmem/cuFFT/cuSPARSE/cuSOLVER) —
  1.3 GiB, under GitHub's 2 GiB release-asset cap. (#15)
- **Auto pipeline-worker count.** A worker count of `0` (config, `--workers 0`,
  or the Settings slider's leftmost notch) now means *auto* — sized to the CPU,
  platform-aware (Apple Silicon → performance-core count; x86 → ~half the
  physical cores). The pipeline sidebar shows `NN workers (auto|manual)`. Auto
  is the new default.

### Changed

- **Dewarp shape buckets right-sized** to real page geometry (measured ~45
  keypoints/line, not the ~70 the old caps assumed) plus finer steps — ~30%
  faster dewarp on GPU, ~40% on CPU, no memory or quality cost.

## [0.1.0a3] — 2026-06-25

Third alpha. Linux/GPU and tiling-WM fixes, plus dewarp robustness.

### Fixed

- **Dewarp produced no output on dense pages.** The padded JAX optimiser's
  over-cap fallback re-imported a function `install()` had already replaced
  with itself → infinite recursion (`maximum recursion depth exceeded`), which
  killed the dewarp branch. This was also the "QEMU" recursion crash — never
  QEMU-specific, just any page over the cap.
- **Sidebar "Tip the developer" link** opened a dead Ko-fi handle (redirected
  to the Ko-fi homepage); now points at the correct page.
- **Loading splash** was tiled/mangled by tiling window managers
  (Hyprland/sway/i3); it now floats.

### Changed

- **Over-cap dewarp** (dense or `baseline_source=both` pages — common now that
  line extraction is fixed) no longer falls back to the slow, cubic-only stock
  optimiser. It prunes text lines to fit (keeps the extremities, drops short
  lines in dense regions, protects sparse regions) and pads to size buckets
  (50 / 80 / 120 lines) so typical pages stay fast.

### Performance

- **Idle worker memory** is now released on Linux (`gc` + `malloc_trim`). An
  opt-in aggressive recycle (`AGLAIA_WORKER_IDLE_RECYCLE_S`) frees the
  JAX/CUDA resident stack (~1.6 GB → ~0.75 GB per worker) when idle.

## [0.1.0a2] — unreleased

Second alpha. Bug-fix pass over a1 from macOS release testing.

### Fixed

- **DPI estimation** (card + measure-a-distance) is applied as a per-session
  value again; it was silently lost in the frozen app (`camera_params.json`
  wrote to a read-only relative path). `camera_params.json` now stores only the
  camera matrix, under APP_DATA. Full chessboard calibration disabled for now (#16).
- **Quit crash** (SIGABRT) when closing mid-model-download — worker threads are
  now stopped on close.
- **Version** shown in About / Diagnostics / Bug report (was `0.0.0` / hardcoded
  `0.1.0`); added `aglaia --version`.
- **`roi_margin_mm`** now takes effect at any value (crop follows the extended
  ROI) — fixes DBnet clipping page margins.
- **Combo dropdowns** were see-through (transparent popup background).

### Performance

- A large project reprocess no longer balloons GUI memory (~3.9 GB → ~0.5 GB)
  or freezes the UI: stage thumbnails are deferred (spinner until the branch
  finishes, then render the final), and status-bar log updates are coalesced.
  Off-screen pixmap release for very large projects tracked in #17.

## [0.1.0a1] — 2026-06-24

First public **alpha**. Well tested on macOS; Linux and Windows are unverified.

### Added

- **End-to-end scanning pipeline.** Webcam capture or image/PDF import →
  deskew → ML page detection → per-page deskew → illumination-tolerant
  binarization → keystone + page-curvature (cubic-sheet) dewarp → a final
  *replay* pass that composes the geometric and morphological operators to
  avoid successive interpolation artifacts (especially on bilevel output).
- **Two entry points, one chain.** `aglaia <workspace>` (PySide6 capture GUI
  with voice control) and `aglaia <project.agl> --headless` (CLI batch) share
  the same multiprocess `IntegratedProcessingChain` and YAML pipeline.
- **Page detection backends.** `auto` resolves **DBnet → Apple Vision (macOS)
  → EAST**; DBnet (~5 MB ONNX) is the cross-platform default. Raises a clear
  error when no model is installed (no silent heuristic fallback).
- **OCR engines.** Apple Vision (macOS), Surya, PaddleOCR-VL (MLX), and Mistral
  Document AI (cloud, with a cheaper async batch mode). BCP-47 language
  selection; optional Markdown refinement.
- **Exports.** Searchable PDF (G4 / JBIG2 profiles) and structured Markdown,
  combinable in one run (`--export pdf:g4+md`).
- **First-run setup.** GUI onboarding wizard, and `aglaia --setup` — a Qt-free
  interactive TUI for CLI-only installs that picks/downloads models, seeds the
  default pipelines, and bootstraps the config. Headless runs refuse to start
  until configured.
- **Offline model downloader.** In-app (GUI) and via `--setup`; models are
  fetched on demand and live in the per-user app-data directory.
- **Voice control.** Vosk offline, constrained-grammar, cross-platform.
- **Extensibility.** Drop-in processors and OCR engines (auto-discovered), plus
  user plugins from the app-data folder gated by a startup trust prompt.
- **Cross-platform distribution.** Signed + notarized macOS DMG (Apple
  Silicon), Windows installer + portable ZIP, Linux AppImage, and
  `pip install aglaia` on any platform. Release artifacts use fixed "latest"
  names and ship `SHA256SUMS`.
- **Localization.** English and French UI (Qt translation catalogues).

### Known limitations

- The Windows build is **not code-signed**; SmartScreen warns on first run
  (bypassable via *More info → Run anyway*; verify with `SHA256SUMS-windows.txt`).
- Apple Vision can miss very faint, wide-spaced running heads — use DBnet or
  EAST for such pages.
- JAX Metal is disabled; the page dewarp runs on CPU (or CUDA/MLX where built).

[0.1.0rc5]: https://github.com/yb85/aglaia/releases/tag/v0.1.0rc5
[0.1.0rc4]: https://github.com/yb85/aglaia/releases/tag/v0.1.0rc4
[0.1.0rc3]: https://github.com/yb85/aglaia/releases/tag/v0.1.0rc3
[0.1.0rc2]: https://github.com/yb85/aglaia/releases/tag/v0.1.0rc2
[0.1.0rc1]: https://github.com/yb85/aglaia/releases/tag/v0.1.0rc1
[0.1.0a6]: https://github.com/yb85/aglaia/releases/tag/v0.1.0a6
[0.1.0a5]: https://github.com/yb85/aglaia/releases/tag/v0.1.0a5
[0.1.0a1]: https://github.com/yb85/aglaia/releases/tag/v0.1.0a1
