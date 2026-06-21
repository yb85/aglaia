# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""pipeline_step_descriptions — per-step essential/full blurbs."""

from lib.workers.Initializer import pipeline_step_descriptions


def test_descriptions_for_known_steps():
    pdef = {"pipeline": [
        {"name": "clamp", "processor": "DPIfixer",
         "options": {"min_dpi": 100, "max_dpi": 250}},
        {"name": "bw", "processor": "Binarizer",
         "options": {"method": "sauvola"}},
    ]}
    d = pipeline_step_descriptions(pdef)
    assert d["DPIfixer"][0] == "min_dpi 100 · max_dpi 250"
    assert "min_dpi: 100" in d["DPIfixer"][1]      # full
    # Binarizer is method-aware
    assert d["Binarizer"][0].startswith("method sauvola")


def test_unknown_processor_skipped():
    d = pipeline_step_descriptions({"pipeline": [
        {"name": "x", "processor": "Replay"},           # worker, not registered
        {"name": "d", "processor": "DPIfixer", "options": {}},
    ]})
    assert "Replay" not in d
    assert "DPIfixer" in d


def test_unknown_option_keys_ignored():
    # schema drift must not crash the blurb builder
    d = pipeline_step_descriptions({"pipeline": [
        {"processor": "DPIfixer", "options": {"min_dpi": 90, "bogus": 1}},
    ]})
    assert d["DPIfixer"][0] == "min_dpi 90 · max_dpi 300"
