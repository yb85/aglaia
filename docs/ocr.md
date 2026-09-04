# OCR engines

OCR runs **off-chain** (after the pipeline), driven by `OcrWorker`
(`aglaia/workers/OcrWorker.py`). Engines live in `aglaia/workers/ocr/` behind a
common interface, selectable per document from the OCR tab.

## Engine interface (`aglaia/workers/ocr/engine.py`)

Subclass `OcrEngine`:

```python
class OcrEngine:
    name: str            # registry key
    display: str         # UI label
    description: str     # ≤80-char tagline for the picker
    available: bool      # True once deps loaded

    def recognize(self, image_rgb, languages, *, src_dpi=None) -> OcrResult: ...
    def recognize_batch(self, images_rgb, languages, *, src_dpis=None): ...  # optional
```

`OcrResult` is a JSON-serialisable dict (`engine`, `languages`, `page_w/h`,
`lines[]` with `text` / `bbox` / `confidence` / optional `quad`, `meta`)
and lands in `ocr_runs.result_json`.

Registration is by decorator — `@register` adds the class to
`ENGINE_REGISTRY`; `get_engine(name)` instantiates it. The OCR tab is
populated straight from the registry, so a [drop-in OCR
plugin](./processors.md) appears automatically.

## Bundled engines

| Engine (`name`) | Where | Notes |
|---|---|---|
| **apple_docs** | on-device | structured document OCR — recovers page (headings, blocks, reading order); the right choice for **Markdown** |
| **apple_vision** | on-device | line-based `VNRecognizeTextRequest`, Latin-first, **no page** — good for the searchable-**PDF** text layer, not for Markdown structure; **default** |
| **surya** | on-device | Qwen-VL served via `mlx-vlm`; `whole_doc` |
| **glm** | on-device | VLM served via `mlx-vlm` |
| **unlimited** | on-device | Baidu Unlimited-OCR (MLX port, in-process); whole-doc, per-page (window=1), DPI-independent. **Not offered to users yet** — the q4 weights it needs are unpublished, so the card reads *(missing)* on every machine but a developer's. Registered and benchmarked (see [ocr-benchmark](ocr-benchmark.md)); left out of the user-facing engine list until the model ships. |
| **mistral_cloud** | cloud | Mistral Document AI over HTTPS; reads any script; footnote + header/footer post-processing |

## Footnote lift (`aglaia/workers/ocr/md_postprocess.py`)

Applied at **markdown export** time, not at OCR import — re-exporting a project
reflects the current toggles and code without re-running (paid) OCR.

A footnote is recognised by an **intersection**: superscript refs in the body
(`$^{7}$`, `¹⁷⁰`, or numeric `(7)`) ∩ line-start entries in the footer
(`7. …`). The intersection is what keeps a stray citation from being lifted;
pairing is windowed ±1 page, because note numbers reset per page and a global
set would link page 2's "1" to page 500's "1".

**Same-line definitions** (`mistral_same_line`, default off). Critical editions
pack several notes onto one physical line:

    (12) premier. (13) second. (14) troisième.

Every marker after the first never sits at a line start, so it never enters the
entry set and is **never classified as a footnote at all** — its ref stays a
bare `(13)` in the body and its text stays glued to note 12. With the toggle
on, markers found inside a line that already *starts* with an entry marker also
count as entries, and such a line is split at each of them.

Two deliberate limits, both to avoid inventing footnotes:

- only markers already in that page's ref∩entry mapping cut a line, so a
  citation inside a note (`voir Migne (3) col. 44`) never splits it;
- the bare `N.` form never cuts mid-line — hopelessly ambiguous against dates,
  verse references and enumerations (`voir Jn 3. 16`). Only superscript and
  parenthesised markers cut.

Off by default because it changes how a page is segmented and only some books
are typeset this way. Backported from the iOS port (issue #70).

## Shared DPI + confidence knobs

Both live in `engine.py` (one place for picker, env, and DB key):

- **OCR DPI** — `resolve_ocr_dpi()`: env `AGLAIA_OCR_DPI` → config
  `ocr_dpi` → default (≈150). Every engine downsamples the page to this
  before inference (`downsample_to_dpi`).
- **Confidence gate** — `resolve_confidence_gate()` (env
  `AGLAIA_OCR_CONFIDENCE_GATE` → config `ocr_confidence_gate`, default
  0.7): per-line Vision confidence below which `apple_docs` offloads the
  line to its complement engine.

## Cloud key storage

`mistral_cloud` needs an API key. Stored in the OS keychain via `keyring`
(macOS Keychain / Windows Credential Locker / Linux Secret Service), with
a `0600` plaintext `<APP_DATA>/.env` fallback when no keychain is reachable
(`aglaia/app_data/secrets.py`). The key never touches the project DB or the
config DB. Install with `uv sync --extra cloud`.

**`keyring` ships in that same `cloud` extra**, and the usual dev syncs
(`--extra dev --extra gui --extra macos`) leave it out — so a source checkout
can have `mistralai` and no keychain at all, and the key lands in the
plaintext `.env`. `secrets.keychain_backend()` returns `(False,
"not_installed")` for that and `(False, "no_backend")` for a real absence of
any store, and the GUI names which one rather than blaming the OS keychain
(#107). The shipped macOS app always includes `cloud`, so this only bites
from source.

## Mistral batch OCR (async, cheaper)

The Cloud OCR card has a **batch toggle** (persisted; config key
`mistral_batch`). With it on, *Run OCR* submits a [Mistral Batch
API](https://docs.mistral.ai/studio-api/batch-processing) job
(`POST /v1/batch/jobs`, endpoint `/v1/ocr`) instead of OCR'ing
synchronously — ~50 % cheaper, processed asynchronously.

Flow (`aglaia/workers/ocr/mistral_batch.py`, `MistralBatchWorker`,
`OcrWorker(batch=True)`):

1. **Submit** — the selected branches' OCR runs are created (left
   *pending*), the pages assembled into capped PDF(s) reusing
   `MistralCloudEngine`'s 1000-page / 50 MB chunking (one batch job per
   chunk), uploaded as a JSONL batch input, and `batch.jobs.create(...)` is
   called with `metadata = {app: aglaia, aglaia_project: <full .agl path>,
   aglaia_chunk}`. Job ids + the page→run mapping (`run_ids` JSON) are
   stored in the project DB table `mistral_batch_jobs` (migration 0011).
2. **Pending** — while any job is pending the card disables Run and shows
   *“Batch job pending — submitted N ago”* with **Check result** and
   **Cancel** (confirm).
3. **Check result** — polls each pending job; for `SUCCESS`, downloads the
   output JSONL and writes each page's markdown back to its OCR run via
   `ocr_repo.finish` (dims from `ocr_runs → nodes → images`), then marks the
   job imported. `FAILED`/`TIMEOUT_EXCEEDED`/`CANCELLED` fail the runs. It is
   re-clickable: the poll runs in a `MistralBatchWorker` QThread handed to
   `MainWindow._track_worker(…, attr="_batch_worker")`, which clears the
   owning attribute when the thread ends. Leaving it set left a
   `deleteLater`-freed husk there, and `isRunning()` on a husk raises rather
   than returning False — so one click on a not-yet-ready job disabled the
   button for the whole session (#111). `_worker_alive` is the guard that
   treats "already deleted" as "not running".
4. **Jobs tab** — *View → Mistral OCR jobs…* (or the card's **Jobs** pill):
   a zebra table of every Aglaïa job on the account (`batch.jobs.list`,
   newest first); the job's `aglaia_project` metadata is a clickable link
   that opens that project (close-current confirm). A **finished** job
   (`SUCCESS`, `FAILED`, `TIMEOUT_EXCEEDED`, `CANCELLED`) carries a trash
   button that **dismisses** it: the Mistral Batch API is create / get /
   list / cancel with **no delete**, so the job itself is permanent account
   history and the button can only stop showing it. Being a view filter it
   is reversible — the count says how many are hidden, and *Show dismissed*
   brings them back with an undo button each. Dismissing also drops the
   project's own `mistral_batch_jobs` row, which is what makes a job
   "pending" for **Check result**; leaving it would keep asking about a job
   the user has just finished with. A live job (`RUNNING` / `QUEUED` /
   `CANCELLATION_REQUESTED`) gets no button: there is nothing to clear yet,
   and one beside a running row would read as *cancel*, which it is not.
   Hidden ids live in the app-data config DB
   (`KEY_MISTRAL_JOBS_DISMISSED`) — per user, not per project.

The key + SDK are the same `[cloud]` extra as the synchronous path; only
the submit/poll/fetch calls differ.

## Engine→GUI logging

Engines emit diagnostics via `engine_log(text, level)`. `OcrWorker`
installs a sink (`set_engine_log_sink`) routing them to the GUI Log tab;
outside the GUI they print to stdout.
