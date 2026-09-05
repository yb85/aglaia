# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Every `ImageBuffer.meta` key is declared, and the declaration decides what a
warp does with it.

`DPIfixer` scaled `roi` and forgot `erase`, and nothing could have told it: a
key in a free-form dict has no kind. Now it does. The first test here is the one
that matters — it scans the source for every meta key literal and fails on an
undeclared one, so the schema cannot drift from the code. The regex-found-
something guard stops it passing vacuously.
"""
import os
import pathlib
import re

import numpy as np
import pytest

from aglaia import meta_schema as ms
from aglaia.meta_schema import META_SCHEMA, MetaKind

#: Writes and reads of ImageBuffer.meta, by the names the code uses for it.
KEY_RE = re.compile(
    r"""(?:\b(?:img_buf|buf|out|new_buf|input_buf|l_buf|r_buf|child|t|b|src|dst)\.meta
        |\bmeta)\s*(?:\[|\.get\(|\.setdefault\(|\.pop\()\s*["']([A-Za-z_][A-Za-z0-9_]*)["']""",
    re.X)

#: Dicts that share the variable name `meta` but are NOT ImageBuffer.meta.
NOT_META_FILES = {"md_export.py", "apple_docs.py", "mistral_cloud.py", "mistral_batch.py",
                  "debug_renderers.py", "ScanItemWidget.py", "DebugViewerTab.py"}


def _literals():
    found = {}
    for f in pathlib.Path("aglaia").rglob("*.py"):
        if f.name in NOT_META_FILES or f.name == "meta_schema.py":
            continue
        for m in KEY_RE.finditer(f.read_text("utf-8", errors="replace")):
            found.setdefault(m.group(1), set()).add(f.name)
    return found


def test_the_scan_finds_the_keys_we_know_about():
    lits = _literals()
    assert len(lits) > 15, f"only {len(lits)} keys found — did the regex break?"
    assert {"roi", "erase", "replay_kind"} <= set(lits)


def test_every_meta_key_in_the_source_is_declared():
    missing = {k: sorted(v) for k, v in _literals().items()
               if ms.kind_of(k) is None}
    assert not missing, (
        "meta keys written or read without a declared kind — add them to "
        f"aglaia/meta_schema.py: {missing}")


def test_private_keys_are_opaque_by_rule():
    assert ms.kind_of("_dewarp_ctx") is MetaKind.OPAQUE
    assert "_dewarp_ctx" not in META_SCHEMA


def test_the_geometric_keys_are_exactly_the_ones_a_warp_must_move():
    assert ms.geometric_keys() == {"roi", "erase", "parent_crop_xywh"}


class TestDeclare:
    def test_a_plugin_can_declare_a_key(self):
        ms.declare_meta("zz_probe", MetaKind.SCALAR)
        try:
            assert ms.kind_of("zz_probe") is MetaKind.SCALAR
        finally:
            META_SCHEMA.pop("zz_probe", None)

    def test_redeclaring_with_another_kind_is_refused(self):
        """Two producers that disagree about what a key is cannot both be
        carried correctly."""
        with pytest.raises(ValueError):
            ms.declare_meta("roi", MetaKind.SCALAR)


class TestStrictMode:
    def test_an_undeclared_write_raises_under_strict(self, monkeypatch):
        monkeypatch.setenv("AGLAIA_META_STRICT", "1")
        ms._warned.discard("zz_unknown")
        from aglaia.ImageBuffer import ImageBuffer, ImageType
        buf = ImageBuffer(np.zeros((4, 4), np.uint8), ImageType.GRAY, dpi=300.0)
        with pytest.raises(KeyError):
            buf.meta["zz_unknown"] = 1

    def test_a_declared_write_is_fine(self, monkeypatch):
        monkeypatch.setenv("AGLAIA_META_STRICT", "1")
        from aglaia.ImageBuffer import ImageBuffer, ImageType
        buf = ImageBuffer(np.zeros((4, 4), np.uint8), ImageType.GRAY, dpi=300.0)
        buf.meta["roi"] = [[0, 0], [1, 0], [1, 1]]
        buf.meta.setdefault("erase", []).append([[0, 0], [1, 0], [1, 1]])
        buf.meta.update({"page_side": "left"})

    def test_in_production_it_prints_once_and_carries_on(self, monkeypatch, capsys):
        monkeypatch.delenv("AGLAIA_META_STRICT", raising=False)
        ms._warned.discard("zz_loose")
        from aglaia.ImageBuffer import ImageBuffer, ImageType
        buf = ImageBuffer(np.zeros((4, 4), np.uint8), ImageType.GRAY, dpi=300.0)
        buf.meta["zz_loose"] = 1
        buf.meta["zz_loose"] = 2
        assert buf.meta["zz_loose"] == 2
        assert capsys.readouterr().out.count("zz_loose") == 1


class TestMetaSurvivesTransit:
    def test_assigning_a_plain_dict_wraps_it(self):
        from aglaia.ImageBuffer import ImageBuffer, ImageType, Meta
        buf = ImageBuffer(np.zeros((4, 4), np.uint8), ImageType.GRAY, dpi=300.0)
        buf.meta = {"roi": [[0, 0], [1, 0], [1, 1]]}
        assert isinstance(buf.meta, Meta)

    def test_pickle_round_trip(self):
        import pickle
        from aglaia.ImageBuffer import Meta
        m = Meta({"roi": [[0, 0], [1, 0], [1, 1]], "page_side": "left"})
        back = pickle.loads(pickle.dumps(m))
        assert isinstance(back, Meta) and back == m

    def test_json_and_deepcopy(self):
        import copy, json
        from aglaia.ImageBuffer import Meta
        m = Meta({"roi": [[0, 0], [1, 0], [1, 1]]})
        assert json.loads(json.dumps(m)) == {"roi": [[0, 0], [1, 0], [1, 1]]}
        assert isinstance(copy.deepcopy(m), Meta)


class TestTransformGeometry:
    """The one rule, applied to one meta."""

    @staticmethod
    def _double(pts):
        return pts * 2.0

    def test_polygons_are_moved_and_everything_else_copied(self):
        meta = {"roi": [[1, 1], [3, 1], [3, 3]],
                "erase": [[[0, 0], [1, 0], [1, 1]]],
                "page_side": "right", "skew_angle": 0.4,
                "replay_params": {"in_wh": [1, 1]}}
        out = ms.transform_geometry(meta, self._double)
        assert out["roi"] == [[2, 2], [6, 2], [6, 6]]
        assert out["erase"] == [[[0, 0], [2, 0], [2, 2]]]
        assert out["page_side"] == "right" and out["skew_angle"] == 0.4
        assert out["replay_params"] == {"in_wh": [1, 1]}

    def test_a_rect_goes_through_an_affine_as_its_corners_bbox(self):
        out = ms.transform_geometry({"parent_crop_xywh": [1, 2, 3, 4]}, self._double)
        assert out["parent_crop_xywh"] == [2, 4, 6, 8]

    def test_a_rect_is_dropped_through_a_nonlinear_map(self):
        """No axis-aligned rectangle survives a dewarp."""
        out = ms.transform_geometry({"parent_crop_xywh": [1, 2, 3, 4]},
                                    self._double, affine=False)
        assert "parent_crop_xywh" not in out

    def test_records_stay_with_their_producer(self):
        """`column_quad` describes the keystone step's INPUT frame; carrying
        it through the keystone warp would turn it into a lie."""
        out = ms.transform_geometry({"column_quad": [[0, 0], [1, 0], [1, 1], [0, 1]],
                                     "roi": [[0, 0], [1, 0], [1, 1]]}, self._double)
        assert "column_quad" not in out and "roi" in out

    def test_an_undeclared_key_is_dropped_not_passed_through(self, monkeypatch, capsys):
        """Passing it along untransformed IS the DPIfixer bug."""
        monkeypatch.delenv("AGLAIA_META_STRICT", raising=False)
        ms._warned.discard("zz_stray_pts")
        out = ms.transform_geometry({"zz_stray_pts": [[5, 5]], "roi": [[0, 0], [1, 0], [1, 1]]},
                                    self._double)
        assert "zz_stray_pts" not in out
        assert "zz_stray_pts" in capsys.readouterr().out

    def test_a_degenerate_polygon_in_a_set_is_dropped(self):
        out = ms.transform_geometry({"erase": [[[0, 0], [1, 1]], [[0, 0], [1, 0], [1, 1]]]},
                                    self._double)
        assert len(out["erase"]) == 1


def test_the_docs_table_lists_every_key():
    """docs/imagebuffer.md is the reference; it must not drift from the schema."""
    doc = pathlib.Path("docs/imagebuffer.md").read_text("utf-8")
    missing = [k for k in ms.APP_KEYS if f"`{k}`" not in doc]
    assert not missing, f"keys missing from docs/imagebuffer.md: {missing}"
