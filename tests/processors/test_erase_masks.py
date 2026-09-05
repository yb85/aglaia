# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Erase masks — the general API, and the edge that must not bleed.

`meta["erase"]` is the dual of `meta["roi"]`: a list of polygons naming regions
the pipeline must remove. The hard part is not removing them, it is removing
them without leaving a black rim where the removal met the page — which is
what these tests are mostly about.
"""
import cv2
import numpy as np
import pytest

from aglaia.ImageBuffer import ImageBuffer, ImageType
from aglaia.processors import erase
from aglaia.processors.Binarizer import Binarizer, BinarizerOption

SQUARE = [[40, 40], [120, 40], [120, 120], [40, 120]]


# ── the metadata contract ────────────────────────────────────────────

def test_polygons_round_trip():
    meta = {}
    erase.add(meta, SQUARE, source="stamp:lib-a")
    assert erase.get(meta) == [[[float(x), float(y)] for x, y in SQUARE]]
    assert meta["erase_sources"] == ["stamp:lib-a"]


def test_a_degenerate_polygon_is_dropped_at_the_door():
    """Two points is a segment. Carrying it to a consumer that would have to
    guess what it meant is worse than losing it."""
    meta = {}
    erase.add(meta, [[0, 0], [5, 5]])
    assert erase.get(meta) == []


def test_get_is_always_a_list():
    assert erase.get(None) == [] and erase.get({}) == []
    assert erase.get({"erase": None}) == []


def test_clear_removes_the_provenance_too():
    meta = {}
    erase.add(meta, SQUARE, source="s")
    erase.clear(meta)
    assert "erase" not in meta and "erase_sources" not in meta


# ── rasterisation: never anti-aliased ────────────────────────────────

def test_the_mask_has_only_two_values():
    """An anti-aliased edge is a row of mid-greys, and a mid-grey beside a
    threshold is a black pixel — the exact spurious rim to avoid."""
    m = erase.as_mask([SQUARE], (200, 200))
    assert set(np.unique(m).tolist()) == {0, 255}


def test_grow_expands_the_mask():
    small = erase.as_mask([SQUARE], (200, 200))
    big = erase.as_mask([SQUARE], (200, 200), grow=4)
    assert cv2.countNonZero(big) > cv2.countNonZero(small)


def test_no_polygons_means_no_mask():
    assert erase.as_mask([], (10, 10)) is None


# ── transport: it must arrive where the stamp is ─────────────────────

def test_a_rotation_carries_the_polygon():
    M = cv2.getRotationMatrix2D((100.0, 100.0), 90.0, 1.0)
    out = erase.transform_affine([SQUARE], M)
    assert len(out) == 1 and len(out[0]) == 4
    assert out[0] != [[float(x), float(y)] for x, y in SQUARE]


def test_an_identity_transform_changes_nothing():
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    out = erase.transform_affine([SQUARE], M)
    assert np.allclose(np.array(out[0]), np.array(SQUARE, dtype=float))


def test_a_homography_carries_the_polygon():
    H = np.array([[1.0, 0.1, 5.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    out = erase.transform_perspective([SQUARE], H)
    assert out[0][0][0] == pytest.approx(40 + 0.1 * 40 + 5)


def test_a_translation_carries_the_polygon():
    out = erase.transform_translate([SQUARE], -10, 5)
    assert out[0][0] == [30.0, 45.0]


def test_a_remap_carries_the_polygon_through_a_sampling_grid():
    """The dewarp has no closed form, so the polygon is rasterised, remapped
    and read back — the same trip the ROI makes."""
    h = w = 200
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    # A pure shift right by 20 px, expressed as a sampling grid.
    out = erase.transform_remap([SQUARE], (h, w), xs - 20, ys)
    assert len(out) == 1
    got = np.array(out[0])
    assert got[:, 0].min() == pytest.approx(60, abs=2)


def test_carry_moves_the_masks_between_buffers():
    src, dst = {}, {}
    erase.add(src, SQUARE, source="s")
    erase.carry(src, dst)
    assert erase.get(dst) == erase.get(src)
    assert dst["erase_sources"] == ["s"]


def test_carry_with_no_masks_writes_nothing():
    dst = {}
    erase.carry({}, dst)
    assert dst == {}


# ── the point of the exercise: no rim ────────────────────────────────

def _stamped_page():
    """A page of text with a dark 'stamp' blotted across part of it."""
    img = np.full((300, 300), 235, np.uint8)          # paper
    for y in range(30, 280, 20):                       # text lines
        img[y:y + 6, 20:280] = 40
    img[100:180, 90:210] = 25                          # the stamp
    return img


def test_the_paper_fill_leaves_no_step_at_the_boundary():
    """Fill with the paper level measured around the region, and the filled
    patch is not an edge at all — which is the whole reason it is not white."""
    img = _stamped_page()
    polys = [[[85, 95], [215, 95], [215, 185], [85, 185]]]
    erase.fill_with_paper(img, polys, halo_px=8)
    # The filled area must read as the PAPER of its neighbourhood — so the
    # comparison is against paper between the text lines, not against a block
    # that still contains ink.
    inside = img[130:150, 130:170].mean()
    paper_outside = img[138:148, 240:270].mean()   # between two text lines
    assert abs(inside - paper_outside) < 6


def test_the_fill_is_not_pure_white():
    """White would be a step against 235-grey paper, and a step is what Wolf
    marks a ring along."""
    img = _stamped_page()
    erase.fill_with_paper(img, [[[85, 95], [215, 95], [215, 185], [85, 185]]],
                          halo_px=8)
    assert img[140, 150] != 255


def test_each_region_takes_its_own_local_paper_level():
    """One global value cannot serve a page that is brighter at one edge —
    the reason the ROI's global bg-fill was abandoned."""
    img = np.zeros((200, 400), np.uint8)
    img[:, :200] = 120                       # dim half
    img[:, 200:] = 240                       # bright half
    img[60:140, 40:160] = 10                 # stamp on the dim half
    img[60:140, 240:360] = 10                # stamp on the bright half
    erase.fill_with_paper(
        img,
        [[[35, 55], [165, 55], [165, 145], [35, 145]],
         [[235, 55], [365, 55], [365, 145], [235, 145]]],
        halo_px=4)
    assert 100 < img[100, 100] < 140         # took the dim paper
    assert 220 < img[100, 300] < 255         # took the bright paper


def test_a_wolf_binarize_leaves_no_blob_where_the_stamp_was():
    """End to end, through the real binarizer.

    Three things, and the third is the one this test got wrong for a while.
    The erased region is white. The page just outside it is UNTOUCHED — this
    used to assert the opposite, that a band outside the polygon must also be
    white, which passed only because the fill extended half a Wolf window
    beyond it and wiped the text line above the stamp with it. And the
    binarizer invents no rim of its own along the boundary, which is the real
    "no blob" requirement and is measured against the same page binarized
    without any erase at all.
    """
    pytest.importorskip("doxapy")
    img = _stamped_page()
    poly = [[85, 95], [215, 95], [215, 185], [85, 185]]

    def _binarize(with_erase):
        buf = ImageBuffer(img.copy(), ImageType.GRAY, dpi=300.0)
        buf.filestem = "t"
        if with_erase:
            erase.add(buf.meta, poly)
        return Binarizer(BinarizerOption(method="wolf", window_px_wolf=30,
                                         k_wolf=0.25)).process(buf).buffer

    bw = _binarize(True)
    ref = _binarize(False)

    # 1. the erased region is pure white
    assert bw[100:180, 90:210].min() == 255

    # 2. the text line immediately above the polygon survives. `_stamped_page`
    #    rules a line every 20 px, so rows 90-96 are text, and the polygon
    #    starts at y=95 — a halo would take it.
    above = bw[88:94, 20:280]
    assert (above < 128).sum() > 100, "the text line above the stamp was eaten"

    # 3. nothing dark appears outside the polygon that was not there before
    outside = np.ones_like(bw, bool)
    outside[95:186, 85:216] = False
    invented = ((bw < 128) & (ref >= 128) & outside).sum()
    assert invented == 0, f"{invented} px of rim invented at the boundary"


def test_the_erased_region_is_white_even_without_doxapy():
    """The whiten pass is unconditional — it must not depend on which
    binarizer ran."""
    img = _stamped_page()
    n = erase.whiten(img, [[[85, 95], [215, 95], [215, 185], [85, 185]]],
                     grow=2)
    assert n > 0 and img[140, 150] == 255


def test_whitening_grows_the_polygon_to_take_the_rim_with_it():
    a = np.zeros((200, 200), np.uint8)
    b = np.zeros((200, 200), np.uint8)
    erase.whiten(a, [SQUARE], grow=0)
    erase.whiten(b, [SQUARE], grow=5)
    assert (b == 255).sum() > (a == 255).sum()


# ── replay: the mask is the better instrument ────────────────────────

def test_punch_subtracts_the_region_from_the_keep_mask():
    """In replay an erase region is not painted — it is removed from the
    keep-mask, and `wolf_masked` then excludes it from the local statistics
    AND whitens it. A truer exclusion than the forward pass's fill."""
    mask = np.full((200, 200), 255, np.uint8)
    out = erase.punch(mask, [SQUARE], (200, 200), grow=0)
    assert out[80, 80] == 0
    assert out[10, 10] == 255
    assert mask[80, 80] == 255, "must not mutate the caller's mask"


def test_punch_builds_a_mask_when_there_was_none():
    out = erase.punch(None, [SQUARE], (200, 200))
    assert out is not None and out[80, 80] == 0 and out[10, 10] == 255


def test_punch_with_nothing_to_erase_returns_the_mask_untouched():
    mask = np.full((10, 10), 255, np.uint8)
    assert erase.punch(mask, [], (10, 10)) is mask


def test_the_binarizer_stamps_what_it_erased():
    """A node has to record what was removed from it — a page that lost a
    region silently is a page nobody can audit later."""
    pytest.importorskip("doxapy")
    buf = ImageBuffer(_stamped_page(), ImageType.GRAY, dpi=300.0)
    buf.filestem = "t"
    erase.add(buf.meta, SQUARE)
    out = Binarizer(BinarizerOption(method="wolf")).process(buf)
    rp = out.meta["replay_params"]
    assert rp["erase"] and rp["erase_grow"] == 2
