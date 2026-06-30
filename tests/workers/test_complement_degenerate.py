# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Complement degeneration guard: a recognition-only VLM (Surya/GLM) handed a
dense footnote block can loop, emitting one line with an incrementing counter
×100. That output must be rejected so it never splices into the Vision result
(observed: 390× "(N) Ibid., p. 188." polluting an apple_docs+surya export)."""

from __future__ import annotations

from aglaia.workers.ocr.apple_docs import _complement_degenerate


def test_repetition_loop_is_degenerate():
    loop = "\n".join(f"({n}) Ibid. , p. 188." for n in range(195, 296))
    assert _complement_degenerate(loop, n_source_lines=12) is True


def test_real_footnotes_survive():
    # Page numbers vary → lines stay distinct after the enumerator is stripped.
    real = "\n".join([
        "(194) Ibid., p. 148.", "(195) Op. cit., p. 22.",
        "(196) Cf. infra, ch. III.", "(197) Delbrêl, Œuvres, t. II.",
        "(198) Ibid., p. 203.", "(199) Lettre du 3 mai 1952.",
        "(200) Journal, p. 17.", "(201) Ibid., p. 19.",
        "(202) Méditations, p. 4.", "(203) Op. cit., p. 88.",
    ])
    assert _complement_degenerate(real, n_source_lines=10) is False


def test_normal_prose_survives():
    para = ("Proclamer au fond de soi-même, entre la foule et Dieu,\n"
            "la reconnaissance de ce que Dieu est.")
    assert _complement_degenerate(para, n_source_lines=2) is False


def test_short_output_never_tripped():
    # A handful of identical lines is too short to be a runaway loop.
    assert _complement_degenerate("a\na\na", n_source_lines=3) is False


def test_explosion_without_collapse_is_degenerate():
    # Many distinct lines but far more than the source block had → also a loop
    # signature (the VLM kept generating well past the block's content).
    txt = "\n".join(f"unique line number {n}" for n in range(40))
    assert _complement_degenerate(txt, n_source_lines=3) is True
