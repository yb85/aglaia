"""Per-page manual parameter overrides in the processors (M9 #93, #94, #95).

The chain puts a layout's payload on the buffer for the duration of one
`process()` call; each of the three tunable steps takes its own field from it.
What has to hold, and is not obvious from the code:

- the ESTIMATOR must not run when the user has set the value — a rerun that
  re-derives its own angle would silently undo the correction;
- a spatial override is only valid on the frame it was drawn on;
- an honoured override is stamped, so a reader can tell a measurement from a
  decision;
- the dewarp freezes the sheet and re-optimises ONLY the pose, or the page
  swims as soon as one slider moves.
"""
import pickle
from pathlib import Path

import cv2
import numpy as np
import pytest

from aglaia.ImageBuffer import ImageBuffer, ImageType
from aglaia.processors.manual import manual_value, stamp_manual
from aglaia.processors.PageDetector import _manual_roi_for
from aglaia.processors.PageDewarper import DewarpOption, PageDewarper
from aglaia.processors.SkewFinder import SkewFinder, SkewFinderOption

FIX = Path(__file__).parent / "fixtures" / "dewarp_input_0.pkl"


# ── reading the payload ───────────────────────────────────────────────

def _buf(meta=None):
    b = ImageBuffer(np.full((40, 30, 3), 255, np.uint8), ImageType.COLOR,
                    dpi=300, filestem="t")
    b.meta.update(meta or {})
    return b


def test_absent_payload_reads_as_no_override():
    assert manual_value(_buf(), "skew_deg") is None
    assert manual_value(_buf({"manual_overrides": {}}), "skew_deg") is None


def test_spatial_override_is_dropped_on_a_different_frame():
    """A polygon drawn on an 800x600 frame means nothing on a 1600x1200 one.
    Applying it anyway would shift the page silently."""
    b = _buf({"manual_overrides": {"roi": [[0, 0]], "frame_wh": [800, 600]}})
    assert manual_value(b, "roi", frame_wh=(1600, 1200)) is None
    assert b.meta["manual_dropped"] == ["roi"]
    assert manual_value(b, "roi", frame_wh=(800, 600)) == [[0, 0]]


def test_a_payload_without_provenance_is_accepted():
    """Written before `frame_wh` existed: unvalidatable, not wrong."""
    b = _buf({"manual_overrides": {"roi": [[0, 0]]}})
    assert manual_value(b, "roi", frame_wh=(1600, 1200)) == [[0, 0]]


def test_stamp_is_a_set_not_a_log():
    b = _buf()
    stamp_manual(b, "curl")
    stamp_manual(b, "curl")
    assert b.meta["manual"] == ["curl"]


# ── SkewFinder (#93) ──────────────────────────────────────────────────

def _skew(**over):
    opts = dict(max_angle=30, min_angle=0.5, accuracy=0.1,
                apply_rotation=True, k_cluster=0)
    opts.update(over)
    return SkewFinder(SkewFinderOption(**opts))


def _slanted():
    """A block of ink rotated ~6 degrees — enough for the estimator to find
    an angle of its own, so a test can tell the two paths apart."""
    img = np.full((400, 400, 3), 255, np.uint8)
    img[150:250, 80:320] = 0
    m = cv2.getRotationMatrix2D((200, 200), 6.0, 1.0)
    img = cv2.warpAffine(img, m, (400, 400), borderValue=(255, 255, 255))
    return ImageBuffer(img, ImageType.COLOR, dpi=300, filestem="t")


def test_manual_angle_replaces_the_estimate():
    estimated = _skew().process(_slanted()).meta["skew"]
    b = _slanted()
    b.meta["manual_overrides"] = {"skew_deg": -1.25}
    out = _skew().process(b)
    assert out.meta["skew"] == -1.25
    assert out.meta["manual"] == ["skew_deg"]
    assert abs(estimated - (-1.25)) > 1.0        # the estimator disagreed


def test_manual_angle_lands_in_the_replay_stamp():
    """Replay re-applies the rotation from the stamp, not from the meta —
    a manual angle that missed it would come back estimated."""
    b = _slanted()
    b.meta["manual_overrides"] = {"skew_deg": -1.25}
    out = _skew().process(b)
    assert out.meta["replay_params"]["angle_deg"] == -1.25


def test_the_step_being_off_still_wins():
    """`apply_rotation: false` is the pipeline saying "do not rotate here".
    A manual angle tunes the step; it does not switch it back on."""
    b = _slanted()
    b.meta["manual_overrides"] = {"skew_deg": -1.25}
    out = _skew(apply_rotation=False).process(b)
    assert out.meta["skew"] == -1.25
    assert "replay_params" not in out.meta


# ── PageDetector (#94) ────────────────────────────────────────────────

def test_manual_roi_is_picked_by_the_branch_being_created():
    """The ROI is keyed by a label PageDetector is about to assign, so it
    reads the whole scan's map rather than its own buffer's payload."""
    b = _buf({"manual_overrides_all": {
        "A": {"roi": [[1, 2], [3, 4]], "frame_wh": [100, 200]},
        "B": {"roi": [[5, 6]], "frame_wh": [100, 200]}}})
    assert _manual_roi_for(b, "A", frame_wh=(100, 200)) == [[1.0, 2.0], [3.0, 4.0]]
    assert _manual_roi_for(b, "B", frame_wh=(100, 200)) == [[5.0, 6.0]]
    assert _manual_roi_for(b, "C", frame_wh=(100, 200)) is None


def test_manual_roi_drops_on_a_frame_mismatch():
    b = _buf({"manual_overrides_all": {
        "A": {"roi": [[1, 2]], "frame_wh": [100, 200]}}})
    assert _manual_roi_for(b, "A", frame_wh=(120, 200)) is None


def test_a_payload_with_no_roi_is_not_an_roi_override():
    b = _buf({"manual_overrides_all": {"A": {"skew_deg": 1.0}}})
    assert _manual_roi_for(b, "A", frame_wh=(100, 200)) is None


# ── PageDewarper (#95) ────────────────────────────────────────────────

def _page():
    m = pickle.load(open(FIX, "rb"))
    img = cv2.imdecode(np.frombuffer(m["buf_png"], np.uint8), cv2.IMREAD_UNCHANGED)
    b = ImageBuffer(img.copy(), m["type"], dpi=m["dpi"], filestem=m["filestem"])
    b.branch_label = m["branch_label"]
    b.meta = dict(m["meta"])
    b.meta["page_side"] = "left"
    return b


def _dewarp(**kw):
    return PageDewarper(DewarpOption(
        baseline_source="bottom", cubic_cost=0.0, focal_length=1.3,
        backend="lm", **kw))


@pytest.mark.skipif(not FIX.exists(), reason="dewarp fixture not present")
def test_manual_curl_is_the_fitted_sheet():
    """The user set the shape. It must come back unchanged — a solver that
    re-fitted alpha/beta around it would make every slider move snap back."""
    b = _page()
    b.meta["manual_overrides"] = {"curl": {"alpha": 0.12, "beta": -0.04}}
    out = _dewarp(spine_gammas="").process(b)
    params = out.meta["replay_params"]["params"]
    assert params[6] == pytest.approx(0.12, abs=1e-6)
    assert params[7] == pytest.approx(-0.04, abs=1e-6)
    assert "curl" in out.meta["manual"]


@pytest.mark.skipif(not FIX.exists(), reason="dewarp fixture not present")
def test_manual_curl_still_re_optimises_the_pose():
    """Only the SHAPE is frozen. Freezing the pose too would leave the page
    where the previous fit put it, and the sliders would look inert."""
    free = _dewarp(spine_gammas="").process(_page())
    b = _page()
    b.meta["manual_overrides"] = {"curl": {"alpha": 0.12, "beta": -0.04}}
    manual = _dewarp(spine_gammas="").process(b)
    pose_free = np.asarray(free.meta["replay_params"]["params"][:6])
    pose_manual = np.asarray(manual.meta["replay_params"]["params"][:6])
    assert not np.allclose(pose_free, pose_manual)


@pytest.mark.skipif(not FIX.exists(), reason="dewarp fixture not present")
def test_manual_gamma_is_stamped_as_the_spine():
    """Replay rebuilds the surface from `spine`; a gamma that missed it would
    remap the page against a different sheet."""
    b = _page()
    b.meta["manual_overrides"] = {"curl": {"alpha": 0.05, "beta": 0.0,
                                           "gamma": 0.04}}
    out = _dewarp().process(b)
    spine = out.meta["replay_params"]["spine"]
    assert spine is not None
    assert spine["gamma"] == pytest.approx(0.04, abs=1e-9)


@pytest.mark.skipif(not FIX.exists(), reason="dewarp fixture not present")
def test_no_override_leaves_the_fit_alone():
    out = _dewarp(spine_gammas="").process(_page())
    assert "curl" not in (out.meta.get("manual") or [])


# ── TrapezoidalCorrection (#100) ──────────────────────────────────────

def _blank(w=400, h=600):
    """A page the keystone estimator cannot read: no baselines at all. It
    falls back — which is exactly the page a user wants to draw a quad on."""
    return ImageBuffer(np.full((h, w, 3), 255, np.uint8), ImageType.COLOR,
                       dpi=300, filestem="t")


def _trap():
    from aglaia.processors.TrapezoidalCorrection import (
        TrapezoidalCorrection, TrapezoidalOption)
    return TrapezoidalCorrection(TrapezoidalOption())


def test_a_manual_quad_rectifies_a_page_the_estimator_gave_up_on():
    """The whole detection is what is being overridden, so the line-count
    guard — which exists to protect that detection — must not stand in the
    way of four points the user placed."""
    assert _trap().process(_blank()).meta["trapezoid_success"] is False
    b = _blank()
    b.meta["manual_overrides"] = {
        "quad": [[40, 60], [360, 55], [365, 545], [35, 540]],
        "frame_wh": [400, 600]}
    out = _trap().process(b)
    assert out.meta["trapezoid_success"] is True
    assert out.meta["column_edge_source"] == "manual"
    assert "quad" in out.meta["manual"]


def test_a_non_convex_manual_quad_is_refused():
    """A folded quad has no valid homography; warping through one folds the
    page over itself."""
    b = _blank()
    b.meta["manual_overrides"] = {
        "quad": [[40, 60], [360, 55], [35, 540], [365, 545]],   # crossed
        "frame_wh": [400, 600]}
    out = _trap().process(b)
    assert out.meta["trapezoid_success"] is False
    assert "not convex" in out.meta["fallback_reason"]


def test_a_manual_quad_from_another_frame_is_ignored():
    b = _blank()
    b.meta["manual_overrides"] = {
        "quad": [[40, 60], [360, 55], [365, 545], [35, 540]],
        "frame_wh": [800, 1200]}
    assert _trap().process(b).meta["trapezoid_success"] is False


# ── force dewarp (#101) ───────────────────────────────────────────────

@pytest.mark.skipif(not FIX.exists(), reason="dewarp fixture not present")
def test_force_runs_the_fit_past_the_span_count_guard():
    """`min_spans` protects the optimiser from an under-constrained fit. It
    is right by default and sometimes wrong — a sparse page whose few spans
    are perfectly good."""
    guarded = _dewarp(spine_gammas="", min_spans=999).process(_page())
    assert "too few spans" in guarded.meta.get("fallback_reason", "")

    b = _page()
    b.meta["manual_overrides"] = {"force": True}
    forced = _dewarp(spine_gammas="", min_spans=999).process(b)
    assert "too few spans" not in forced.meta.get("fallback_reason", "")
    assert "force" in forced.meta["manual"]
    assert forced.meta.get("replay_params") is not None


@pytest.mark.skipif(not FIX.exists(), reason="dewarp fixture not present")
def test_without_force_the_ladder_is_unchanged():
    out = _dewarp(spine_gammas="").process(_page())
    assert "force" not in (out.meta.get("manual") or [])
    assert out.meta.get("oob_forced") is None
