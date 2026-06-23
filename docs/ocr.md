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
| **surya** | on-device | VLM via bundled `llama-server` (llama.cpp); `whole_doc` |
| **paddle_vl** | on-device | PaddleOCR-VL via `mlx-vlm` |
| **mistral_cloud** | cloud | Mistral Document AI over HTTPS; reads any script |

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
a `0600` plaintext `<APP_DATA>/.env` fallback when no keychain backend is
available (`aglaia/app_data/secrets.py`). The key never touches the project
DB or the config DB. Install with `uv sync --extra cloud`.

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
   job imported. `FAILED`/`TIMEOUT_EXCEEDED`/`CANCELLED` fail the runs.
4. **Jobs tab** — *View → Mistral OCR jobs…* (or the card's **Jobs** pill):
   a zebra table of every Aglaïa job on the account (`batch.jobs.list`,
   newest first); the job's `aglaia_project` metadata is a clickable link
   that opens that project (close-current confirm).

The key + SDK are the same `[cloud]` extra as the synchronous path; only
the submit/poll/fetch calls differ.

## Engine→GUI logging

Engines emit diagnostics via `engine_log(text, level)`. `OcrWorker`
installs a sink (`set_engine_log_sink`) routing them to the GUI Log tab;
outside the GUI they print to stdout.
