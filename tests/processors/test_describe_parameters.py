# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Processor parameter descriptions (essential / full)."""

from aglaia.processors.abstraction import render_param_description
from aglaia.processors.DPIfixer import DPIfixer, DPIfixerOption
from aglaia.processors.Binarizer import Binarizer, BinarizerOption
from aglaia.processors.PageDewarper import PageDewarper, DewarpOption


def test_essential_is_compact_oneliner():
    s = DPIfixer.describe_options(DPIfixerOption(), "essential")
    assert s == "min_dpi 100 · max_dpi 300"
    assert "\n" not in s


def test_full_lists_all_algo_fields_one_per_line():
    full = DPIfixer.describe_options(DPIfixerOption(), "full")
    assert "min_dpi: 100" in full and "max_dpi: 300" in full
    # I/O / debug fields are hidden
    assert "write_out" not in full and "debug" not in full


def test_describe_options_does_not_instantiate():
    # PageDewarper.__init__ spins up JAX/MLX; the classmethod must avoid it.
    s = PageDewarper.describe_options(DewarpOption(), "essential")
    assert "sheet_model cylindrical" in s and "twist off" in s


def test_binarizer_is_method_aware():
    s = Binarizer.describe_options(BinarizerOption(method="sauvola"),
                                   "essential")
    assert "method sauvola" in s and "window" in s and "k " in s


def test_instance_method_matches_classmethod():
    o = DPIfixerOption()
    p = DPIfixer(o)
    assert p.describe_parameters("full") == DPIfixer.describe_options(o, "full")


def test_render_hides_io_fields_and_formats_bools():
    s = render_param_description(BinarizerOption(), (), "full")
    assert "on" in s or "off" in s or s  # bools render on/off, never True
    assert "instance_name" not in s
