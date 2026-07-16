# Bridge — phone as a scanner camera

Two modes let an [AglaiaBridge](https://github.com/yb85/aglaia-bridge) phone feed
the desktop:

- **Push** (`#47`, shipped) — scan pages offline into an `.aglbundle`, then
  one-shot upload over pinned TLS to `POST /import`. See `bridge_server.py` /
  `bridge_bundle.py` / `bridge_receive.py`.
- **Live** (`#49`, this doc) — the phone is a *tethered camera*: it streams a
  low-res preview to the desktop and snaps a **full-res still** on a
  desktop-triggered shutter (button / voice / DPI calibration). The desktop
  Bridge sidebar tab shows a pairing QR; the phone scans it and connects.

Live mode exists because Continuity Camera can't take full-res stills (the phone
camera is held exclusively by Continuity at ~1080p). Here the phone owns its own
camera: one `AVCaptureSession` runs a video-data output (preview) and a photo
output (full-res) at once.

## Trust model (shared with push)

The on-screen QR is the trusted side channel; the LAN is hostile.

- **TLS** — per-session ephemeral self-signed cert (`bridge_tls.py`) →
  confidentiality.
- **Fingerprint pinning** — the QR carries the cert's SHA-256; the phone pins
  it and refuses any mismatch or plain-HTTP downgrade → server authentication /
  no MITM. No CA, no hostname validation needed.
- **Single-use bearer token** — carried in the QR, required on every request →
  authorization (only the phone that scanned *this* QR can talk).

Each time the Bridge tab arms, a **fresh cert + token** are minted, so a QR is
useless once its session ends.

## Pairing QR

```
aglaia://v1?h=<host>&p=<port>&t=<token>&fp=<sha256-hex>&m=live
```

`m` selects the mode: absent → push (`/import`), `m=live` → this protocol.
`ReceiverInfo.qr_uri(mode="live")` builds it. An **old push-only app** ignores
`m`, POSTs `/import`, and gets a `404 {"error":"live-bridge session — update
AglaiaBridge"}` it surfaces as a normal upload failure. A **new app** that sees
an unknown `m` value refuses to connect.

## Wire protocol `bridge-live/1`

All requests are HTTPS with `Authorization: Bearer <token>`. Errors are JSON
`{"error": "..."}`. Server: stdlib `ThreadingHTTPServer` + `ssl`
(`bridge_live.py`, `BridgeLiveServer`) — no websocket/asyncio dependency.

| Endpoint | Dir | Body / headers | Response |
|---|---|---|---|
| `POST /v1/session` | phone→desktop | `{protocol:1, device, app, still_max:[w,h]}` | `200 {session, preview:{max_px:960, fps:12, jpeg_q:0.6}, poll_s:25}` · **409** if a session is already active (first phone wins) · **400** unsupported `protocol` |
| `POST /v1/frame` | phone→desktop | JPEG bytes; `X-Session`, `X-Seq` (monotonic), `Content-Type: image/jpeg` | `200 {}` · **410** session gone · **413** > 4 MiB |
| `GET /v1/command` | phone→desktop (long-poll) | — | blocks ≤ `poll_s` → `{"command":"none"}` (re-poll), `{"command":"capture","capture_id":"…"}`, or `{"command":"bye"}` |
| `POST /v1/still` | phone→desktop | JPEG (q≈0.9); `X-Session`, `X-Capture-Id` | `200 {}` (a still for an abandoned/late `capture_id` → 200 then discarded) · **410** session gone · **413** > 64 MiB |
| `POST /v1/bye` | phone→desktop | `{reason:"user"\|"background"}` | `200 {}` |

Status codes: `400` malformed, `401` bad token, `404` unknown path (incl.
`/import` on a live server), `409` session conflict, `410` session ended, `413`
too large.

### Semantics

- **Frames** are POSTed sequentially on a keep-alive connection — awaiting each
  response before sending the next *is* the backpressure (a slow desktop slows
  the phone). `X-Seq` lets the desktop skip already-shown frames and discard
  stragglers.
- **Commands**: the desktop holds **at most one** pending capture. A delivered
  but unanswered capture is failed desktop-side after its own timeout (default
  6 s) and is **never auto-reissued** — no duplicate stills.
- **Stills** correlate to their request by `X-Capture-Id`.
- **Liveness**: `POLL_SECONDS = 25` (under common 30 s idle timeouts);
  `LIVENESS_TIMEOUT = 10 s` with neither a frame nor a poll ⇒ the session is
  dead ⇒ the desktop tears down and shows a **fresh QR**.

## Session lifecycle

```
phone: scan QR → POST /v1/session ─────────────► desktop: on_session_started(device)
        ├─ frame loop:   POST /v1/frame (seq++) ─► latest_preview() advances
        └─ command loop: GET /v1/command (park) ◄─ request_still() queues "capture"
                          → POST /v1/still  ─────► request_still() returns the still
phone: user ends / backgrounded → POST /v1/bye ─► session cleared → fresh QR
desktop: tab closed / window closed → stop() ───► queues "bye", server shut down
```

## Failure modes

| Failure | Phone | Desktop |
|---|---|---|
| Lock / background | `scenePhase → end(.background)` → `POST /v1/bye` | bye (or 10 s silence) → teardown → fresh QR |
| Network drop mid-still | still POST fails → retry once → "desktop unreachable" | `request_still` times out (6 s) → `still_failed` toast; session survives if frames resume |
| Late still for abandoned id | — | `200`, payload discarded (logged) |
| Token replay after end | request → 401/410 | token+cert regenerate per pairing |
| Desktop closed while streaming | next poll → `bye`/refused → "desktop gone" | `stop()` queues bye then shuts down |
| Two phones scan one QR | 2nd hello → 409 "already connected" | first wins (single-session invariant) |

## Desktop architecture

- `aglaia/workers/bridge_live.py` — `BridgeLiveServer` (protocol + session state).
- `aglaia/gui/BridgeCameraThread.py` — a `WebcamThread` look-alike: emits the
  low-res preview via `change_pixmap_signal`, and `get_frame()` fetches a
  **remote full-res still** (`request_still`). Because it duck-types the webcam
  thread, `MainWindow.capture()`, voice, and zoom work unchanged.
- `aglaia/gui/sidebar/tabs/BridgeTab.py` + `aglaia/gui/bridge_live_controller.py`
  — the sidebar tab (QR ↔ live swap) and the Qt-signal marshaller.

**Zoom** is a desktop-side digital crop (center-crop `1/zoom`, resize back up) on
both preview and still — no wire command — so `effective_dpi = base × zoom`
keeps working. **Mutual exclusion**: bridge and the local webcam can't run at
once (`MainWindow.webcam_thread` is single-tenant).

## Device-free E2E

The `FakePhone` client in `tests/workers/test_bridge_live.py` drives the whole
protocol over pinned TLS. `tools/fake_bridge_phone.py <uri> --image page.jpg
[--still-scale N]` streams a downscaled image as preview and answers captures
with the full-size version — get `<uri>` from the Bridge tab's QR right-click →
"Copy pairing URI".

## Live-device checklist

- [ ] Pair on the same Wi-Fi **and** on a phone hotspot.
- [ ] Preview latency / fps acceptable; no thermal throttling over a chapter.
- [ ] Shutter fires from the button, `SPACE`, and voice ("photo").
- [ ] Still long-edge ≈ sensor max (≥ ~4000 px) — proves the `maxPhotoDimensions`
      fix (bare `AVCapturePhotoSettings()` is *not* max res).
- [ ] Card DPI calibration → capture → `capture_dpi` correct in the DB.
- [ ] Zoom 2× → recapture → DPI scales; framed card stays inside the still
      (preview video FoV vs photo FoV parity).
- [ ] Lock the phone mid-session → desktop shows a fresh QR.
- [ ] Kill the desktop mid-session → phone shows "desktop gone".
- [ ] Second phone scans the QR → "already connected" (409).
- [ ] An old push-only build scanning a live QR fails cleanly.
