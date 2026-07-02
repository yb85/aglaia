# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Voice control via Vosk (offline, constrained grammar).

The single voice backend on every platform — Apple's recognizers were dropped
(free-form transcription misheard the command words + multi-second latency).
The command vocabulary is tiny and fixed (config `voicecontrols`, e.g.
"photo" / "delete"), so the recognizer is constrained to exactly those words —
high accuracy, near-zero false fires, low latency, no cloud.

Optional deps (`voice` extra: vosk + sounddevice) and the model
(downloader: "Vosk — English (small)") are loaded lazily; missing either is
a graceful no-op.
"""

from __future__ import annotations

import json
import time

from PySide6.QtCore import QThread, Signal

from aglaia.gui import voice_transcript as _vt

try:
    import sounddevice  # noqa: F401
    import vosk  # noqa: F401
    HAS_VOSK = True
except Exception:  # pragma: no cover - optional
    HAS_VOSK = False

_SAMPLE_RATE = 16000


def _model_dir():
    """Path to the downloaded Vosk model dir, or None if absent."""
    try:
        from aglaia.app_data import models_dir
        d = models_dir() / "vosk-model-small-en-us"
        return d if d.is_dir() else None
    except Exception:
        return None


class VoskVoiceWorker(QThread):
    command_detected = Signal(str)
    transcription_update = Signal(str)

    def __init__(self, config=None):
        super().__init__()
        self.config = config
        # True from construction so a stop() during the (slow) model load is
        # honoured — run() must NOT re-set this to True after loading, else a
        # deactivate mid-load is silently overwritten and the loop runs on.
        self._running = True
        self._recent: list[tuple[str, str]] = []  # rolling (word, state)

    def _commands(self) -> dict[str, str]:
        """word → action map from config `voicecontrols` (matches the Apple
        worker)."""
        out: dict[str, str] = {}
        vc = (self.config or {}).get("voicecontrols", {}) if self.config else {}
        for action, words in vc.items():
            if action == "debounce_time":
                continue
            for w in words:
                out[str(w).lower()] = action
        return out or {"photo": "scan", "delete": "trash"}

    def run(self):
        if not HAS_VOSK:
            return
        model_path = _model_dir()
        if model_path is None:
            return  # model not downloaded yet — nothing to do
        commands = self._commands()
        grammar = json.dumps(sorted(set(commands)) + ["[unk]"])
        cooldown = float((self.config or {}).get("voicecontrols", {})
                         .get("debounce_time", 2.0)) if self.config else 2.0

        model = vosk.Model(str(model_path))
        rec = vosk.KaldiRecognizer(model, _SAMPLE_RATE, grammar)
        rec.SetWords(True)   # per-word confidences in the final result
        last_fire = 0.0

        # Only fire on FINAL results above this acoustic confidence. The
        # constrained grammar already routes true out-of-vocabulary speech to
        # "[unk]", but short words can still get shoehorned onto "photo" /
        # "delete" with low confidence — this is the second gate so we never
        # trigger on a word the user didn't actually say.
        MIN_CONF = 0.85

        def _commit_final(payload: dict) -> None:
            """Classify each word of a finalized utterance and fire only the
            confident command matches. Partials are ignored entirely — they
            flicker onto command words mid-utterance and caused false fires."""
            nonlocal last_fire
            results = payload.get("result") or []
            if not results:  # grammar/[unk] path may omit per-word data
                results = [{"word": w, "conf": 0.0}
                           for w in (payload.get("text") or "").split()]
            for r in results:
                word = str(r.get("word", "")).lower()
                if not word:
                    continue
                conf = float(r.get("conf", 0.0))
                is_cmd = (word in commands) and (conf >= MIN_CONF)
                if is_cmd and (time.time() - last_fire) > cooldown:
                    self.command_detected.emit(commands[word])
                    last_fire = time.time()
                    state = _vt.FIRED
                elif is_cmd:
                    state = _vt.DEBOUNCED
                else:
                    # Not a confident command (incl. "[unk]" and low-conf
                    # matches) → red, never fired. Don't shoehorn.
                    state = _vt.UNKNOWN
                self._recent.append((word, state))
                del self._recent[:-10]   # keep last 10
            if results:
                self.transcription_update.emit(_vt.words_html(self._recent))

        if not self._running:   # stopped during model load — don't open the mic
            return
        with sounddevice.RawInputStream(
                samplerate=_SAMPLE_RATE, blocksize=8000, dtype="int16",
                channels=1) as stream:
            while self._running:
                # A bad frame / decoder hiccup must never crash the thread
                # (and with it the app) — log and keep listening.
                try:
                    data, _overflow = stream.read(4000)
                    if rec.AcceptWaveform(bytes(data)):
                        _commit_final(json.loads(rec.Result()))
                except Exception:
                    continue

    def stop(self):
        self._running = False
