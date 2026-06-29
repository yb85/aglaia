# Server

`aglaia server` is a long-running HTTP **job API**: a client submits an
`.aglbundle` (produced by [aglaia-bridge](https://aglaia.bibli.cc)) or a PDF, and
the server runs the same headless pipeline as `aglaia run` to produce a
**searchable PDF** (plus **Markdown** when OCR is on). The transient `.agl`
project is built in a per-job folder and **discarded** once the artifacts are
copied out — only the output files persist.

It is a warm-pool design (issue #52): one server process keeps the pipeline warm
and drains a queue of jobs in the background, so submitters don't pay startup
cost per request.

> Needs the `server` extra: `pip install "aglaia[server]"` (FastAPI, uvicorn,
> python-multipart). Code: `aglaia/server/{app,db,processor,mailer}.py`; the
> command is `aglaia/cli/commands/server.py`.

## Running it

```
aglaia server [--host HOST] [--port 4674] [--public-url URL]
```

| Option | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address. Use `0.0.0.0` to accept LAN/remote clients. |
| `--port` | `4674` | Port to listen on (`DEFAULT_PORT`). |
| `--public-url` | — | Public base URL written into download links in emails, e.g. `https://scan.example.com`. Stored in the `config` table. |

On first run the server **mints an admin secret** and prints it with the
admin-panel URL:

```
Aglaïa server → http://127.0.0.1:4674
  admin panel: http://127.0.0.1:4674/admin?secret=…
```

## Endpoints

| Method · Path | Auth | Purpose |
|---|---|---|
| `POST /run` | API key (`api_key` form field) | Submit a job. |
| `GET /list?api_key=` | API key | List the caller's jobs. |
| `GET /check/{job_id}?api_key=` | API key | Status; polls a pending Mistral batch *now*. |
| `GET /get/{job_id}?api_key=` | API key | Status + download URLs; also polls a pending batch. |
| `GET /download/{job_id}/{pdf\|md}` | **capability URL** (no key) | Download an artifact. |
| `POST /delete/{job_id}` | API key (`api_key` form field) | Delete a job (cancels a pending Mistral batch first). |
| `GET /admin?secret=` | admin secret | HTML stats + key list. |
| `POST /admin/keys?secret=&email=` | admin secret | Create an API key (returned once). |
| `POST /admin/keys/{key_id}/revoke?secret=` | admin secret | Revoke a key. |
| `GET /health` | none | `{"ok": true}`. |

### `POST /run`

Multipart form. `file` must be an `.aglbundle` / `.zip` (a bridge bundle) or a
`.pdf` — anything else is `400`.

| Field | Required | Meaning |
|---|---|---|
| `file` | yes | The upload (`.aglbundle`/`.zip` or `.pdf`). |
| `api_key` | yes | Caller's API key. |
| `email_notif` | no (`false`) | Email the download links when the job finishes. |
| `ocr` | no | OCR spec. **Omitted → no OCR** (plain image PDF). e.g. `mistral`, `auto`. |
| `dpi` | no | Force input DPI. |

Returns `{"job_id": …, "status": "pending"}`. The `job_id` is a CSPRNG
`token_urlsafe(16)` (≈128-bit) — see [capability URLs](#download-capability-urls).

## Job lifecycle

Jobs move through these statuses (`aglaia/server/db.py`):

`pending` → `processing` → (`ocr_pending` →) `done` · `failed` · `cancelled`

A background daemon thread (`JobProcessor`, started in the app lifespan) ticks
every ~2 s:

1. **`pending` → `processing`** — pick the oldest pending job, build a
   `CliConfig`, ingest the bundle/PDF into `proj/job.agl`, run the pipeline and
   exports (reuses `aglaia/workers/headless.py`).
2. **No OCR** → plain PDF; **OCR on** → PDF + OCR text layer + Markdown.
3. **`done`** — copy `proj/job.pdf` → `output.pdf` and (if OCR) `proj/job.md` →
   `output.md` into the job folder, then **delete `proj/`** (the `.agl` is never
   persisted). Send the completion email if requested.
4. **`failed`** — record the exception in the job's `error` column.

### Mistral batch OCR + backoff

On the server, Mistral OCR defaults to **batch** (`_normalize_ocr`: `mistral` →
`mistral:batch`; pass `mistral:streaming` to keep it synchronous). A batch
submission parks the job at **`ocr_pending`** and the server polls for the
result:

- on demand whenever a client hits `GET /check/{id}` or `GET /get/{id}`, and
- by the background ticker, with **exponential backoff** in minutes:
  `1, 2, 4, 8, 16, 32, 64, 128, 256, 512` (`BACKOFF_MINUTES`), after which it
  gives up and marks the job `failed` ("OCR batch timed out").

`POST /delete/{id}` (or a job already `ocr_pending`) cancels the in-flight
Mistral batch before removing the job.

## Download (capability URLs)

`GET /download/{job_id}/{pdf|md}` takes **no API key or token** — the unguessable
128-bit `job_id` *is* the secret (a capability URL). That's deliberate: it lets
the links in a completion email work directly when clicked. Every *other*
endpoint still requires the API key (or admin secret). `which` must be `pdf` or
`md`; a missing artifact is `404`.

`/get` and `/check` return the absolute download URLs (`pdf_url` / `md_url`),
built from the configured public base URL (`config.base_url`, set via
`--public-url`).

## Completion email

If a job was submitted with `email_notif=true`, on completion the server emails
the API key's address with the PDF (and Markdown, when present) download links.
SMTP settings live in the `config` table under `smtp`
(`{host, port, user, password, from, tls}`); if unset, sending is skipped
silently. The links use `config.base_url` as their prefix.

## Admin panel

`GET /admin?secret=` renders a small HTML page: job counts by status, and the
API-key table (id, email, active/revoked, created, job count). Manage keys with:

```bash
# create a key for an email (the raw key is returned once)
curl -X POST "http://HOST:4674/admin/keys?secret=SECRET" -F email=user@example.com

# revoke a key by id
curl -X POST "http://HOST:4674/admin/keys/3/revoke?secret=SECRET"
```

A created key is shown **once** (only its sha256 hash is stored).

## Storage layout

```
APP_DATA/
  aglaia-server.db          tables: api_keys, jobs, config
  server_data/
    <job_id>/
      input.zip | input.pdf   the upload
      proj/                    transient project (deleted after finalize)
      output.pdf               artifact (always, on success)
      output.md                artifact (only when OCR ran)
```

`APP_DATA` is the per-user app-data dir (see [app_data.md](app_data.md); macOS:
`~/Library/Application Support/Aglaia`). The DB path is
`APP_DATA/aglaia-server.db`; `create_app(db_path=…, data_dir=…, start_worker=…)`
overrides both (used by tests to drive the processor manually).
