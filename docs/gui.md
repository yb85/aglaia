# GUI (aglaia.py)

PySide6 desktop app. Entry: `uv run python aglaia.py <workspace_dir>`.

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

Sidebar tab widgets live in `lib/gui/sidebar/tabs/` (`CaptureTab`,
`ImportTab`, `PipelineTab`, `OcrTab`, `ExportTab`); the `ActivityBar` +
content `QStackedWidget` are assembled in `lib/gui/sidebar/SidebarPanel.py`.

## Threads / processes

- **MainWindow** runs on the Qt main thread.
- **WebcamThread** (`lib/gui/WebcamThread.py`) — QThread, `cv2.VideoCapture`, applies rotation/mirror/flip per frame, emits `change_pixmap_signal`. `get_frame()` returns the latest BGR frame on demand. 30 FPS cap.
- **ProcessMonitor** (`lib/workers/ProcessMonitor.py`) — QThread that blocks on `log_queue.get(timeout=0.1)` and re-emits messages as Qt signals on the main thread. Handles `image_event`, `worker_started`, `log_info/warning`, `error`, and `timing` (printed via Rich).
- **VoiceWorker** (`lib/gui/VoiceWorker.py`) — QThread, Apple `SFSpeechRecognizer` + `AVAudioEngine`. Emits `command_detected(action)` and `transcription_update(text)`. Skipped if `pyobjc Speech` import fails.
- **Processing chain** — separate worker processes started by `IntegratedProcessingChain.start()`. Workers persist each step directly to the project `.agl` SQLite DB (no separate writer process).

## Workflow

Projects are a single SQLite `<slug>.agl` file — there are no per-step output directories on disk. Raw captures and imports become `scans` rows plus a raw root `nodes` row pointing at a `COLOR` image blob (`lib/storage/persister.py` `Persister`); every pipeline result is persisted as a further node. The only sibling files are slug-prefixed debug dirs and the export target.

1. `initialize(mode="capture")` (`lib/workers/Initializer.py`) parses args/config and builds `args.options`. For capture mode `args.options["paths"]` holds only `root`, `debug_prefix`, and `export` — no `raw`/`output` dirs.
2. `load_calibration()` reads `config/camera_params.json`. If present, `cv2.getOptimalNewCameraMatrix` is computed at capture time and each grabbed frame is undistorted before it is persisted (no on-disk save).
3. `create_processing_chain(args, log_queue, db_path=…)` builds the `IntegratedProcessingChain` (`lib/workers/Initializer.py`). `chain.start()` spawns the multiprocessing workers — they persist each step straight to the project DB (no separate writer process).
4. `load_existing_scans` rebuilds the right-hand panel **from the SQLite DB** (`ScanRepo.list_active` → `NodeRepo`), replaying every persisted node into its `ScanItemWidget` and seeding `current_idx` from the highest scan idx.
5. `WebcamThread`, `ProcessMonitor`, `VoiceWorker` start.
6. On user action:
   - **Scan** (key `Space`/`S`, voice `scan|check|next|photo`, SIFT auto-trigger, button — all funnel through `MainWindow.capture`): grab frame → undistort (if calibrated) → BGR→RGB → in one DB session create the scan + persist the COLOR blob + raw root node (`Persister.persist_image` / `persist_node`, `ScanRepo.set_root`) → spawn the raw `ScanItemWidget` → enqueue an `ImageBuffer` (carrying `scan_id`/`parent_node_id`/`pipeline_version_id`) on the chain input queue. No `.jpg` is written.
   - **Import** (Import tab → `_on_sidebar_import_requested` → `lib/workers/ImportHelpers.py`): `enqueue_image_files` / `enqueue_pdf_files` persist each image — and each PDF page, rendered per-page via pypdfium2 (`pdf_extract.render_page`) — as a scan + raw root node, emit a `scan_imported` `log_queue` event, and enqueue the `ImageBuffer`. `ProcessMonitor` re-emits the event; `MainWindow.on_scan_imported` spawns the raw widget immediately, before any worker stage completes.
   - **Trash/undo** (`Backspace`/`D`, voice `trash|delete|cancel`): pop last from history, then **soft-delete** the scan in the DB (`ScanRepo.soft_delete` sets `scans.deleted_at`) so it drops out of the active list. No blobs are removed.
   - **Quit** (⌘Q / Ctrl+Q via `QKeySequence.StandardKey`, voice `done|quit`): closes the window.
   - **Rotate** (`R`): cycles preview rotation by 90°.
7. Worker `image_event`s are routed by `scan_id` (`MainWindow.on_image_event` → `scan_widgets_by_scan[scan_id]`) to update the matching `ScanItemWidget` as each node lands. The user picks the kept page per branch in the widget; export happens via the Export tab (see below).
8. On close (`closeEvent`): stop the webcam/monitor/voice threads, close the thumbnail loader, and `shutil.rmtree(<workspace>/<output_dir_name>/._temp)` if that temp dir exists. No PDF is generated on close.

## ScanItemWidget

`lib/gui/ScanItemWidget.py`. One per captured scan. Shows the file's progression through pipeline steps:

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

## Calibration buttons

- **Full Calibration** — guides the user through capturing `calnum` (default 10) chessboard frames. Last sample is taken with board flat at "book distance" → its measured px-per-square sets the DPI. Calls `Calibrator.finalize_calibration` → `save_calibration(...)` → writes `config/camera_params.json`. Restart capture to pick up the new calibration.
- **Calibrate DPI** — single-sample, updates only the DPI field while keeping the existing camera matrix.

Print `A4_chessboard.pdf` (in repo root) on real A4 to use as the calibration target. Default board is 5×8 inner corners at 30mm squares (see `config/default.yml:calibration`).

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

`_match_key` handles named keys (`Space`, `Backspace`, `Return`, `Enter`, `Escape`, `Tab`, `Delete`) and single-character keys.

## Input transforms

`WebcamThread.set_transform(str)` parses a string like `"180+mirror"`: rotation in {0, 90, 180, 270}, plus optional `mirror` (horizontal) and `flip` (vertical). The GUI transform buttons mutate this state live.

## OCR tab

The sidebar **OCR** tab (`lib/gui/sidebar/tabs/OcrTab.py`) picks an engine via a
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
  lines, lower it to offload fewer. See `lib/workers/ocr/apple_docs.py`.
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
  `lib/workers/ocr/mistral_cloud.py`.

  *API key* — set via the card's **Set API key…** button (masked dialog).
  Resolution order (`lib/app_data/secrets.py`): env `MISTRAL_API_KEY` →
  `APP_DATA/.env` → **OS keychain** (`keyring`). `.env` is checked before the
  keychain so a dotenv-style dev never triggers a keychain unlock prompt.
  *Write* prefers the OS keychain, falling back to a cleartext `APP_DATA/.env`
  (0600) only when no keychain backend exists (headless Linux/Windows).
  Optional password-manager backends: `uv sync --extra keyring-bitwarden` /
  `--extra keyring-1password` (keyring auto-discovers them).

**Gating** (`lib/workers/ocr/apple_caps.py`): not macOS → both Apple cards
disabled ("macOS only"); macOS pre-26 → only the Document card disabled
("Requires macOS 26+"); macOS 26+ → both enabled. If the default card ends up
disabled, the tab falls back to the first enabled card. The Document engine
needs **no** Apple Intelligence.

## Export

The sidebar **Export** tab (`lib/gui/sidebar/tabs/ExportTab.py`) shows three
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
