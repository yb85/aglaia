# GUI (aglaia)

PySide6 desktop app. Entry: `uv run aglaia gui [PROJECT]` — `gui` is the
default command, so `uv run aglaia` and `uv run aglaia ~/book.agl` open it too.
See the [CLI reference](./cli.md).

## Layout

A top tab bar switches between **Scans**, **Edit pipeline**, **Settings**,
**Log**, and any open image/debug viewer tabs. The **Scans** tab is the main
workspace: a collapsible left panel, the captured-scans area (grid / table /
gallery views), and a right-edge **sidebar** whose **ActivityBar** icons swap
the content pane between the **Capture**, **Import**, **Pipeline**, **OCR** and
**Export** tabs. The live webcam preview lives inside the **Capture** tab.

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Scans │ Edit pipeline │ Settings │ Log │ …          ← top tab bar         │
├─────────────────────────────────────────────────────────────────────────┤
│ collap- │  Captured scans                       │  Sidebar tab    │ Act-  │
│ sible   │  (grid · table · gallery)             │  content pane   │ ivity │
│ left    │                                       │  ┌───────────┐  │ Bar   │
│ panel   │  ┌─────────────────────────────────┐  │  │ Capture   │  │ [▣]   │
│         │  │ ScanItemWidget #N               │  │  │  webcam   │  │ [⤓]   │
│         │  │  raw → step01 → … → output      │  │  │  preview  │  │ [⫶]   │
│         │  └─────────────────────────────────┘  │  └───────────┘  │ [A]   │
│         │   …  (top = newest)                    │  Import·Pipe·   │ [⇪]   │
│         │                                       │  OCR · Export   │       │
├─────────────────────────────────────────────────────────────────────────┤
│  Status / Voice label                                  ← bottom status bar │
└─────────────────────────────────────────────────────────────────────────┘
```

Sidebar tab widgets live in `aglaia/gui/sidebar/tabs/` (`CaptureTab`,
`ImportTab`, `PipelineTab`, `OcrTab`, `ExportTab`); the `ActivityBar` +
content `QStackedWidget` are assembled in `aglaia/gui/sidebar/SidebarPanel.py`.

## Threads / processes

- **MainWindow** runs on the Qt main thread.
- **WebcamThread** (`aglaia/gui/WebcamThread.py`) — QThread, `cv2.VideoCapture`, applies rotation/mirror/flip per frame, emits `change_pixmap_signal`. `get_frame()` returns the latest BGR frame on demand. 30 FPS cap.
- **ProcessMonitor** (`aglaia/workers/ProcessMonitor.py`) — QThread that blocks on `log_queue.get(timeout=0.1)` and re-emits messages as Qt signals on the main thread. Handles `image_event`, `worker_started`, `log_info/warning`, `error`, and `timing` (printed via Rich).
- **VoiceWorker** (`aglaia/gui/VoiceWorker.py`) — QThread, Apple `SFSpeechRecognizer` + `AVAudioEngine`. Emits `command_detected(action)` and `transcription_update(text)`. Skipped if `pyobjc Speech` import fails.
- **Processing chain** — separate worker processes started by `IntegratedProcessingChain.start()`. Workers persist each step directly to the project `.agl` SQLite DB (no separate writer process).

## Workflow

Projects are a single SQLite `<slug>.agl` file — there are no per-step output directories on disk. Raw captures and imports become `scans` rows plus a raw root `nodes` row pointing at a `COLOR` image blob (`aglaia/storage/persister.py` `Persister`); every pipeline result is persisted as a further node. The only sibling files are slug-prefixed debug dirs and the export target.

1. `initialize(mode="capture")` (`aglaia/workers/Initializer.py`) parses args/config and builds `args.options`. For capture mode `args.options["paths"]` holds only `root`, `debug_prefix`, and `export` — no `raw`/`output` dirs.
2. `load_calibration()` reads `config/camera_params.json`. If present, `cv2.getOptimalNewCameraMatrix` is computed at capture time and each grabbed frame is undistorted before it is persisted (no on-disk save).
3. `create_processing_chain(args, log_queue, db_path=…)` builds the `IntegratedProcessingChain` (`aglaia/workers/Initializer.py`). `chain.start()` spawns the multiprocessing workers — they persist each step straight to the project DB (no separate writer process).
4. `load_existing_scans` rebuilds the right-hand panel **from the SQLite DB** (`ScanRepo.list_active` → `NodeRepo`), replaying every persisted node into its `ScanItemWidget` and seeding `current_idx` from the highest scan idx.
5. `WebcamThread`, `ProcessMonitor`, `VoiceWorker` start.
6. On user action:
   - **Scan** (key `Space`/`S`, voice `scan|check|next|photo`, SIFT auto-trigger, button — all funnel through `MainWindow.capture`): grab frame → undistort (if calibrated) → BGR→RGB → in one DB session create the scan + persist the COLOR blob + raw root node (`Persister.persist_image` / `persist_node`, `ScanRepo.set_root`) → spawn the raw `ScanItemWidget` → enqueue an `ImageBuffer` (carrying `scan_id`/`parent_node_id`/`pipeline_version_id`) on the chain input queue. No `.jpg` is written.
   - **Import** (Import tab → `_on_sidebar_import_requested` → `aglaia/workers/ImportHelpers.py`): `enqueue_image_files` / `enqueue_pdf_files` persist each image — and each PDF page, rendered per-page via pypdfium2 (`pdf_extract.render_page`) — as a scan + raw root node, emit a `scan_imported` `log_queue` event, and enqueue the `ImageBuffer`. `ProcessMonitor` re-emits the event; `MainWindow.on_scan_imported` spawns the raw widget immediately, before any worker stage completes.
   - **Trash/undo** (`Backspace`/`D`, voice `trash|delete|cancel`): pop last from history, then **soft-delete** the scan in the DB (`ScanRepo.soft_delete` sets `scans.deleted_at`) so it drops out of the active list. No blobs are removed.
   - **Quit** (⌘Q / Ctrl+Q via `QKeySequence.StandardKey`, voice `done|quit`): closes the window.
   - **Rotate** (`R`): cycles preview rotation by 90°.
7. Worker `image_event`s are routed by `scan_id` (`MainWindow.on_image_event` → `scan_widgets_by_scan[scan_id]`) to update the matching `ScanItemWidget` as each node lands. The user picks the kept page per branch in the widget; export happens via the Export tab (see below).
8. On close (`closeEvent`): stop the webcam/monitor/voice threads, close the thumbnail loader, and `shutil.rmtree(<workspace>/<output_dir_name>/._temp)` if that temp dir exists. No PDF is generated on close.

## ScanItemWidget

`aglaia/gui/ScanItemWidget.py`. One per captured scan. Shows the file's progression through pipeline steps:

- `raw` thumb → one thumb per pipeline step (`pipeline_steps` = the `instance_name`s computed in `MainWindow.__init__`).
- `output` thumb is the latest persisted result for the scan.
- Refresh timer polls every 2s for files that appeared on disk without an `image_event` (defensive against missed events).
- `restore_state(path, type)` is called on startup for every file that was already on disk.

## Per-page processor disable

Replaces the old exit-stage navigation (chevron step-back/forward, gallery
star, table select-as-chosen — all removed). Each page-layout can individually
**disable** a toggleable processor (linear COORDINATE/PIXEL_VALUE steps;
PageDetector and other ROI/branch-emitting steps are locked). Toggling writes a
`step_overrides` row and reruns that scan from raw (`set_step_disabled` →
`_reprocess_snaps_callback`); see [storage.md](storage.md#per-page-processor-disable-step_overrides).

The three views surface it differently, all via `MainWindow.cell_disable_states`
(`{node_id: (toggleable, disabled)}`) + `MainWindow.toggle_step_disabled`:

- **Table** (`ScansTableView`) — primary. Click a stage cell to toggle it;
  disabled cells get a red strike.
- **Grid** (`ScanItemWidget`) — keeps the chevrons (display nav only now). A
  round overlay on the displayed stage shows its pipeline index (or `R` for
  replay) — blue = active, red `✕` = disabled; click toggles. A 3px band at the
  thumbnail's top is a mini-map of the layout's disabled steps (one red slot per
  disabled stage), hidden when nothing is disabled.
- **Gallery** (`ScansGalleryView`) — a toggle (replacing the star) on the
  current stage; left/right still walks stages.

## Debug view / per-page editor (`DebugViewerTab`)

Click a stage thumb and a closable tab walks that page's chain, root → leaf.
Two panes: the stage strip on the left, the selected stage on the right.

**The strip** lists one row per pipeline step: a small thumbnail, the step
name, and a background that identifies the **processor**. It used to zebra on
the row INDEX, which carries no information — two adjacent look-alike stages
got two different shades and two unrelated stages got the same one, so
scrolling had no visible effect. Keyed by processor it reads as bands, and the
two DPIfixers and two SkewFinders of the default pipeline get a light/dark
variant so a repeat is still a seam. Thumbnails are small on purpose: at the
old 200-px portrait thumb a ten-step chain needed ~3700 px of strip, so the
whole thing was a scroll with no landmarks; a default `book_curved_x2` chain
now fits a normal window without scrolling.

**The stage pane** shows the per-processor overlay composite — spans,
baselines, the fitted quad and grid — rendered in the background by
`storage/debug_renderers.py`. There is no "show overlays" toggle: this view
exists to show the debug data.

**Manual tuning** (M9). Three stages can be corrected by hand; the edit is
stored per page-layout in `manual_overrides` and the page-branch is rerun.
See [storage.md](storage.md#per-page-manual-overrides-manual_overrides) and
[processors.md](processors.md).

| Stage | Control |
|---|---|
| SkewFinder | a rotation handle on the image and a slider, both on the same angle |
| PageDetector | the ROI polygon — drag a vertex, double-click an edge to add one |
| TrapezoidalCorrection | the column quad — drag a corner. **Four corners, no more**: a keystone is a projective map from exactly four points, so this polygon refuses insertion. When the step fell back and found no quad, the corners are seeded from the frame — that is precisely the page a user wants to draw one on |
| PageDewarper | sliders for **arch**, **tilt** and the spine γ, plus **Force dewarp**. The grid previews the sheet live |

**Arch and tilt are not the fitted parameters.** The solver fits α and β,
which are the sheet's slopes at the LEFT and RIGHT page edges (`z'(0) = α`,
`z'(1) = β`, and `z` is pinned to 0 at both). Neither moves one visible thing
on its own — every drag of either reshapes the whole surface — which is what
made them unusable by hand. The editor rotates the pair:

```
arch = (α − β)/2        tilt = (α + β)/2
α    = arch + tilt      β    = tilt − arch
```

`z(0.5) = (α − β)/8 = arch/4`, so **arch alone sets the mid-page rise** — the
arch of a bound page, the thing the eye reads — and **tilt alone slides its
crest** left or right. The arch slider shows that rise as a percentage of the
page width beside its raw value. Ranges cover the whole `delbrel-oc9` corpus
(276 fitted pages: |arch| ≤ 0.325, |tilt| ≤ 0.250, |γ| ≤ 0.100) at a 0.001
step, about one step per slider pixel.

**The grid previews live.** Dragging a curl slider redraws the sheet in
magenta over the fitted grid, from the same builder the remap uses
(`dewarp_grid_lattice`) with the row's stamp and the edited curl substituted —
so it is the surface, not an approximation of it. The pose is the last fit's:
a rerun re-optimises it around the frozen shape, so the final page shifts a
little. Magenta deliberately, not the renderer's green: green is what was
fitted, magenta is what the sliders are asking for.

**Force dewarp** runs the fit past the `min_spans` guard and the `max_oob`
gate. Both are right by default and both are sometimes wrong — a sparse page
whose few spans are perfectly good, a wide fit the gate reads as runaway. The
result may be worse; the node records `manual: force` (and `oob_forced` when
the gate was the thing overridden), so a bad page stays explainable.

Handles are vector, painted by `DebugEditCanvas.EditCanvas` over the raster.
The renderers hand them the geometry as numbers (`geom`), in the coordinates
of the **stage frame**, with two mappings beside it:

- `origin` — where that frame sits inside the composite: the label bar above
  it, the crop offset of a child drawn on its parent;
- `scale` — what `_png_data_url` shrinks the composite by on its way out (Qt's
  allocation cap). The picture on screen is not the composite.

A handle is drawn at `(point + origin) × scale`. Miss `origin` and every
handle sits a bar-height, or a crop, from its pixel; miss `scale` and the
error grows with the coordinate, which reads as everything shifted down and
right.

Every spatial edit stores the frame it was made on, so a polygon is validated
rather than silently rescaled.

**After a rerun** the branch's subtree is wiped and rewritten, so the tab's
leaf node is a dead row. `MainWindow._refresh_debug_tabs`, on `branch_ready`,
re-targets every open tab of that page-branch and re-keys `_debug_tabs` —
without it the tab keeps showing the pre-edit chain and the editor reads as
broken.

While that rerun is in flight the tab keeps the **previous composites** on
screen (`_adopt_stale_overlays`), row by row and only where the processor
still matches. Dropping them meant the dewarp's source | output picture was
replaced by the bare stage image plus the light Qt overlay for the whole
render — exactly while the user was comparing the live slider grid against
it (#106).

### The layout set (PageDetector rows)

On a PageDetector row the handles work on **every layout at once, in PARENT
coordinates** — the whole photo, not one child's crop. Before, the polygon
lived in the crop the detector had chosen and was clamped to it, so a page
could only ever be corrected *inwards*: a vertex could not be dragged out to
where the page really was, and the orange box was a wall.

- **Drag a vertex** anywhere on the photo; double-click an edge to insert one.
- **Trash badge** — a translucent disc at each layout's barycentre. The last
  layout keeps none: deleting it would leave the page with nothing to process.
- **Add badge** — top-right of the picture. Drops in a rectangle to drag into
  shape, offset from the ones already there so a second Add is not hidden
  under the first.

The set is stored once per scan on the **trunk** (`branch_path == ""`,
`manual_overrides.layouts` + `layouts_frame_wh`), because it decides how many
branches exist and so belongs to none of them. `PageDetector` reads it before
its own empty-page guard and lets it REPLACE detection: that is what makes a
deletion stick (a layout removed by hand is not found again next run), what
lets a layout be added to a page the detector saw as blank, and what lets the
crop follow the polygon instead of the other way round. `smart_merge` and
`max_pages` are skipped — they guard a guess, and this is not one.

Editing the set reruns the **whole scan**, not one branch: which branches
exist is exactly what is being decided, so resuming from the split point would
rerun children about to be renumbered or deleted. Each resulting child is
stamped `manual: layouts`, named for the instrument — only that one explains a
changed page count.

**Auto-process** (on by default, remembered for the session) reruns the page
once a value *settles* — never per drag step. `EditCanvas.edited` fires on
every mouse-move so the handle tracks the cursor; `edit_finished` is the
commit, emitted on release (and immediately for an atomic edit like a
double-click vertex insert). Persisting and rerunning on `edited` launched a
chain rerun per move event, hundreds deep into one drag, until memory ran out
and the app died (#116). The sliders debounce the same way, on
`sliderReleased`.

Turn it off to make several edits and run once —
the **Reprocess** button lights up, and is dimmed while auto-process is on
because there would be nothing for it to do. **Clear override** drops this
layout's stored values and restores the automatic result.

Manual tuning **survives a force rerun**: the pages come back as they were
corrected. The Force-rerun dialog therefore offers three ways out — *Cancel*,
*Reprocess all*, and *Reprocess all and clear manual overrides* (in the danger
colour: it throws away work done by hand, which no rerun can recover). The
third clears `manual_overrides` only; the per-page step **disables** live in
`step_overrides` and keep their own toggles in the scan views.

The slider ranges are chosen for manual tuning, not for the solver's freedom:
curl is clamped at ±0.5 internally but a page past ±0.35 is already extreme,
and a full-width slider over the solver's range would make every useful value
a two-pixel move.

## Hand-edited pages in the scan views

A page carrying manual overrides looked exactly like one the pipeline decided
alone, everywhere outside the debug editor. All three views now mark it with
one quiet dot — `widgets.ManualPip`, the primary accent, no glyph, no text —
whose tooltip names what was touched ("Hand-tuned: deskew angle, dewarp
curl").

| View | Where |
|---|---|
| Table | beside the branch label, where the eye reads the row's identity |
| Card grid | top-right of the layout thumbnail, clear of the disabled band (top edge) and the nav buttons |
| Gallery | top-right of the stage image, clear of the star (top-left) and the disabled glyph (centre) |

Quiet is the requirement: a hand-edited page is not a warning, so the mark
must not compete with the disable strike (red) or the trashed state, and it is
absent entirely on a page with no override — which is the common case, and the
reason the mark says anything at all.

All three read `MainWindow.manual_fields_for_layout(scan_id, branch_path)`,
which merges the layout's own payload with the pre-split trunk's. It is
memoised per scan like `cell_disable_states` — the views ask on every repaint
and it is one SQLite round-trip each — and the cache is dropped on
`branch_ready` and on every edit the editor writes.

## Calibration buttons

- **Full Calibration** — guides the user through capturing `calnum` (default 10) chessboard frames. Last sample is taken with board flat at "book distance" → its measured px-per-square sets the DPI. Calls `Calibrator.finalize_calibration` → `save_calibration(...)` → writes `config/camera_params.json`. Restart capture to pick up the new calibration.
- **Calibrate DPI** — single-sample, updates only the DPI field while keeping the existing camera matrix.

Print `assets/calibration/calibration-chessboard_A4_7x10sq_25mm.pdf` on real A4 (at 100%) as the calibration target — generate it with `scripts/gen_calibration_board.py`. Default board is 6×9 inner corners at 25mm squares (see `docs/calibration.md`).

## Voice commands

Defaults from `config/default.yml`:

```yaml
voicecontrols:
  scan:  [scan, check, next, photo]
  trash: [trash, delete, cancel]
  quit:  [done, quit]
  debounce_time: 2
```

Implementation: the recognizer runs continuously; only **new** words on each partial result are matched. A 2-second debounce prevents double-firing. Display label shows last ~10 words.

## Keybindings

Defaults:

```yaml
keycontrols:
  scan:   [Space, S]
  trash:  [Backspace, D]
  rotate: [R]
```

Quit (⌘Q / Ctrl+Q) and close-tab (⌘W / Ctrl+W) use platform-standard
shortcuts wired via `QKeySequence.StandardKey`; they are **not** configurable
in `keycontrols`.

**Editable in the app.** The capture panel's shortcut legend carries a small
pencil; it opens a modal with **two slots per action**. Clicking a slot *arms*
it — the next key or combination pressed is what it becomes, and focus leaves.
No record button, no timeout. Tab and Escape stay the dialog's, so an armed
field cannot trap the user in it; bind Escape from the other slot.

Two slots per action is not decoration. A **presentation remote**'s fullscreen
button typically cycles between `Shift+F5` and `Esc`, so driving capture from
one needs both bound to the same action — which is the case this was built
for.

Bindings are stored per user in the app-data config DB (`KEY_KEYBINDINGS`) as
`QKeySequence` portable strings, and override the YAML `keycontrols` per
action. An action the user **cleared** is stored as an empty list: that is a
decision, not an absence, so it does not fall back to the YAML default.

Matching goes through `QKeySequence` (`aglaia/gui/keybindings.py`). The
hand-rolled matcher it replaced compared key NAMES against `event.text()` and
a table of seven names, and never looked at `event.modifiers()` — so no
combination was expressible at all. Every legacy default (`Space`, `S`,
`Backspace`, `D`, `R`) parses as a `QKeySequence`, so a config written before
the change keeps working untouched.

### Priority over the focused widget

The bindings are taken by an **application-level event filter**
(`MainWindow.eventFilter`, installed on the `QApplication`), not by
`keyPressEvent`. That method is the last stop in Qt's propagation: whatever
has focus sees the press first, and anything it accepts never reaches the
window. So `PgUp` paged the list view's scroll area instead of capturing —
it worked in the gallery, which does not consume it — and the first press
after a click was eaten by whatever had just taken focus, which reads as
"I have to press twice" (#119). A filter on the window would not have helped:
that is still downstream of its own children.

The filter only pre-empts while capture is genuinely in front
(`_capture_keys_have_priority`): the window is active, no modal dialog is
open, the capture panel is visible, and focus is not in a text entry
(`keybindings.is_text_entry` — `QLineEdit`, `QTextEdit`, `QPlainTextEdit`,
`QAbstractSpinBox`, an editable `QComboBox`, and their subclasses). Those two
exclusions are what keep `s` a letter in a filename field and let the
keybinding recorder — itself a `QLineEdit`, inside a modal — **record** a
bound key rather than fire it. Auto-repeat is ignored, so a stuck key cannot
spray captures. `keyPressEvent` stays as the path for a press that reaches
the window on its own.

## Input transforms

`WebcamThread.set_transform(str)` parses a string like `"180+mirror"`: rotation in {0, 90, 180, 270}, plus optional `mirror` (horizontal) and `flip` (vertical). The GUI transform buttons mutate this state live.

## OCR tab

The sidebar **OCR** tab (`aglaia/gui/sidebar/tabs/OcrTab.py`) picks an engine via a
`RadioCardGroup` and fires `run_requested(engine, languages, mode, complement)`
→ `MainWindow._on_ocr_run_requested` → `OcrWorker`.

Engine cards:

- **Apple Document engine** (`apple_docs`) — **default on a capable Mac.**
  macOS 26 `VNRecognizeDocumentsRequest`: a structured, reading-ordered
  document tree (`meta.document`) plus a flat-line confidence pass. Lines
  Apple Vision can't read (non-Latin scripts like Greek — per-line
  confidence below the **confidence gate**) are cropped and re-OCR'd by a
  **complement** engine chosen in the card's *Complement engine* dropdown
  (**Surya** default, Paddle, or None). Fail-open: if the complement is
  unavailable the Vision text is kept. The gate is a system param (default
  **0.7**): env `AGLAIA_OCR_CONFIDENCE_GATE` → SQLite `KEY_OCR_CONFIDENCE_GATE`
  → default, resolved by `resolve_confidence_gate()`. Raise it to offload more
  lines, lower it to offload fewer. See `aglaia/workers/ocr/apple_docs.py`.
- **Apple Vision** (`apple_vision`) — the flat `VNRecognizeTextRequest`
  path with the geometric Markdown heuristics.
- **Surya** / **PaddleOCR-VL** — standalone VLM engines (needed off-mac and
  for full-page VLM runs).
- **Cloud OCR (Mistral)** (`mistral_cloud`) — **whole-document** engine. The
  selected pages are assembled into **one PDF** (bitonal scans → CCITT G4, the
  same codec as our exports; colour/grey → JPEG), uploaded once to Mistral's
  Document AI (`mistral-ocr-latest`), and the per-page Markdown spliced back
  into per-branch results (`meta.markdown`, rendered verbatim by md_export).
  Reads any script (Greek, etc.) off-device. `whole_doc = True` makes
  `OcrWorker` send every selected page (per run mode: missing / missing+stale
  / all) in one `recognize_batch` call. Mistral caps an upload at **1000
  pages / 50 MB** — over that, the engine **truncates** to the leading pages
  that fit, OCRs those, and leaves the rest *pending* (flagged
  `meta.truncated` → `OcrWorker` `fail()`s them); a Log-tab advisory tells
  the user to **run OCR again** to continue. Page mapping is positional
  (Mistral page *i* → the *i*-th selected scan). Needs the `cloud` extra
  (`uv sync --extra cloud`) and an API key. See
  `aglaia/workers/ocr/mistral_cloud.py`.

  *API key* — set via the card's **Set API key…** button (masked dialog).
  Resolution order (`aglaia/app_data/secrets.py`): env `MISTRAL_API_KEY` →
  `APP_DATA/.env` → **OS keychain** (`keyring`). `.env` is checked before the
  keychain so a dotenv-style dev never triggers a keychain unlock prompt.
  *Write* prefers the OS keychain, falling back to a cleartext `APP_DATA/.env`
  (0600) only when no keychain backend exists (headless Linux/Windows).
  Optional password-manager backends: `uv sync --extra keyring-bitwarden` /
  `--extra keyring-1password` (keyring auto-discovers them).

  *Key status line* — the card says where the key resolves from. Reading a
  keychain item pops a system password prompt on **macOS only**
  (`secrets.keychain_read_prompts`), so there the card probes the keychain
  only once the user engages Cloud OCR, and shows a neutral hint until then.
  On Linux (Secret Service) and Windows (Credential Locker) the read is
  silent, so the card probes at startup — deferring there only hid a key that
  was stored.

  *When there is no keychain* — the key falls back to a plaintext
  `APP_DATA/.env` (0600). `secrets.keychain_backend()` says which of the two
  reasons applies, and both the key dialog and the post-save message name it:
  `not_installed` (the `keyring` package is absent — it ships in the **cloud**
  extra, which `uv sync --extra dev --extra gui --extra macos` leaves out) or
  `no_backend` (keyring is there, nothing answered — headless Linux, bare
  Windows). Reporting the second for the first read as a broken macOS
  Keychain on a machine whose Keychain was fine (#107).

**Gating** (`aglaia/workers/ocr/apple_caps.py`): not macOS → both Apple cards
disabled ("macOS only"); macOS pre-26 → only the Document card disabled
("Requires macOS 26+"); macOS 26+ → both enabled. If the default card ends up
disabled, the tab falls back to the first enabled card. The Document engine
needs **no** Apple Intelligence.

## Export

The sidebar **Export** tab (`aglaia/gui/sidebar/tabs/ExportTab.py`) shows three
format cards picked via a radio group, then one **Export** button dispatched by
`MainWindow._on_export_clicked` on the selected key:

- **PDF** — `make_pdf("output")` → `create_pdf_from_db` assembles the chosen
  branch terminals into one PDF. Toggles: JBIG2/G4 compression, and an optional
  OCR text layer (tagged with the engine, e.g. `_appleOCR`).
- **Markdown** — `_export_markdown` → `write_markdown` (see
  [markdown_export.md](./markdown_export.md)). Card is disabled until OCR data
  exists (`set_markdown_available`).
- **Slim Aglaïa project** — `_export_slim_project` → `slim_export`, a pruned
  *copy* of the project DB (raw captures + chosen pages + their OCR only).

All three prompt for a destination with `QFileDialog.getSaveFileName`
(defaulting to the workspace dir + engine/DPI-tagged filename) and reveal the
written file in Finder on success (`_reveal_in_finder`).

## Menu bar

`MainWindow._build_menu_bar` populates `self.menuBar()`. Qt places it natively
per platform — the global top-of-screen bar on macOS, an in-window bar on
Windows/Linux — so the same code serves all three; the `QAction.MenuRole` hints
(Preferences/Quit/About) only matter on macOS and are harmless no-ops elsewhere.

- **File** — New / Open (round-trip through the launcher via
  `_confirm_then_restart`), **Slim-down current project…**, Close Project.
- **View** — Show Downloader, Close Tab, and the Table/Grid/Gallery selector.
- **Help** — Documentation, Report a Bug…, **About Aglaïa**.

**Slim-down current project** (`_on_slim_down_in_place`) is the *in-place*
sibling of the slim export: it confirms (intermediate states are dropped but
regenerable since originals are kept), then closes the project and arms an
`aglaia_restart="reopen"` round-trip. `main()` runs `slim_in_place` on the now-
free DB file (the chain has stopped) and reopens the same path — so the view
rebuilds against the slimmed project. Both paths share
`slim_export._prune_to_slim`.

**About Aglaïa** (`_open_about`, also reachable from the Settings tab's About
card) shows `AboutDialog` — a generated HTML page (`build_about_html`: version,
runtime stack, links, license) rendered in a `QTextBrowser` with links opened in
the system browser.
