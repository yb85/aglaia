---
title: Server
description: Run Aglaïa as a long-running HTTP job server — submit a phone bundle or a PDF, get a searchable PDF (and Markdown) back, with optional email notification.
---

`aglaia server` turns Aglaïa into a long-running **HTTP job API**. Instead of
opening the GUI on one machine, you point clients at a shared server: each one
uploads a document, the server ingests → processes → OCRs → exports it in the
background, and the finished **searchable PDF** (plus **Markdown** when OCR is
on) is available to download — optionally announced by email.

It pairs naturally with the [aglaia-bridge iOS companion](#from-the-phone):
capture pages on your phone, push an `.aglbundle` to the server, and collect the
PDF from any device.

## Install and run

The server is an opt-in extra (FastAPI + Uvicorn), not part of the base install:

```bash
pip install "aglaia[server]"
aglaia server                       # binds 127.0.0.1:4674 by default
```

Useful flags:

| Flag | Purpose |
|---|---|
| `--host 0.0.0.0` | accept LAN / remote clients (default is loopback only) |
| `--port N` | listen on a different port (default `4674`) |
| `--public-url https://scan.example.com` | base URL used in email download links |

On startup the server prints an **admin panel** link with a one-time secret:
`http://HOST:PORT/admin?secret=…`.

For the best OCR on hard pages, pair it with the `cloud` extra — Mistral
Document AI runs as a **batch** job (polled with exponential backoff), and the
completion email carries the download links.

## How a job flows

1. A client `POST`s a file to `/run` with its **API key**. The upload is either
   an `.aglbundle` (or `.zip`) from the phone app, or a plain `.pdf`.
2. A background worker ingests the upload into a transient project, runs the
   pipeline, optionally OCRs, and exports a PDF (and Markdown when OCR is on).
   The transient `.agl` is discarded; only the artifacts remain.
3. The client polls `/get` / `/check`, or — if `email_notif` was set — receives
   an email with direct download links when the job finishes.

## Endpoints

| Method · path | What it does |
|---|---|
| `POST /run` | submit a job — upload + `api_key`, plus `email_notif`, `ocr`, `dpi`. Returns a `job_id`. |
| `GET /list` | list your jobs (API key). |
| `GET /check/{job_id}` | refresh status; polls a pending Mistral batch now. |
| `GET /get/{job_id}` | job status + download URLs. |
| `GET /download/{job_id}/{pdf\|md}` | download an artifact — **no API key** (see below). |
| `POST /delete/{job_id}` | delete a job (cancels any in-flight Mistral batch). |
| `GET /admin?secret=…` | HTML dashboard: job counts + API keys. |
| `POST /admin/keys` · `POST /admin/keys/{id}/revoke` | create / revoke an API key (tied to an email). |
| `GET /health` | liveness check. |

### The capability download URL

Every endpoint requires the API key **except** `/download/{job_id}/{pdf|md}`.
There the unguessable `job_id` (a 128-bit random token) *is* the secret — a
classic **capability URL**. That's deliberate: it lets a download link in a
notification email work when clicked, without embedding the API key in the
message. Treat the link as you would the file itself.

## OCR and email

- **OCR** is off unless you ask for it. Pass `ocr` on `/run` (e.g. `mistral`,
  `apple`, `surya`). On the server, Mistral defaults to **batch** mode;
  `mistral:streaming` keeps it synchronous.
- When OCR is on, the job also produces a **Markdown** export alongside the PDF.
- Set `email_notif=true` to get a completion email with the capability download
  links (requires the server's mailer to be configured and a `--public-url`).

## From the phone

The **aglaia-bridge** iOS companion captures pages and produces an
`.aglbundle`. It can push that bundle straight to a desktop on the same network
(QR-bootstrapped, TLS-pinned — the desktop's Import tab shows *“Receive from
phone”*), or upload it to an `aglaia server` for unattended processing.

## Related resources

- [Install](/docs/install) — the `server` extra
- [CLI reference](/docs/reference/cli) — `aglaia server` flags
- [OCR engines](/docs/concepts/ocr-engines) — the engines a job can use
- [Export](/docs/concepts/export) — the PDF / Markdown a job produces
