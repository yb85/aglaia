"""PageDewarper's LM backend end to end on a real page (issues #59, #60).

Covers the wiring the unit tests cannot: that `auto` picks LM for the
cylindrical model, that the #60 camera upgrades change the fit and are stamped
into the replay params, and — the one that would silently corrupt a project —
that replay rebuilds the SAME surface from those params. The grid geometry is
derived from `camera_np` + `spine`; drop either and replay would remap the page
against a different sheet.
"""
import pickle
from pathlib import Path

import cv2
import numpy as np
import pytest

from aglaia.ImageBuffer import ImageBuffer, ImageType
from aglaia.processors.PageDewarper import DewarpOption, PageDewarper

FIX = Path(__file__).parent / "fixtures" / "dewarp_input_0.pkl"
BASE = dict(baseline_source="bottom", cubic_cost=0.0, focal_length=1.3)


def _buf(page_side=None):
    m = pickle.load(open(FIX, "rb"))
    img = cv2.imdecode(np.frombuffer(m["buf_png"], np.uint8), cv2.IMREAD_UNCHANGED)
    b = ImageBuffer(img.copy(), m["type"], dpi=m["dpi"], filestem=m["filestem"])
    b.branch_label = m["branch_label"]
    b.meta = dict(m["meta"])
    if page_side is not None:
        b.meta["page_side"] = page_side
    return b


def _lm(**kw):
    return PageDewarper(DewarpOption(**{**BASE, "backend": "lm", **kw}))


def test_auto_resolves_to_lm():
    assert PageDewarper(DewarpOption(**BASE, backend="auto")).backend == "lm"


@pytest.mark.parametrize("retired", ["sine_twist", "bspline_twist",
                                     "flat_spline", "spline_twist"])
def test_retired_sheet_models_are_rejected_not_silently_approximated(retired):
    """A node stamped with one of these cannot be rebuilt against the
    cylindrical surface — replay must fail loudly, not draw a wrong page."""
    from aglaia.processors.sheet_models import canonical_model
    with pytest.raises(ValueError, match="retired"):
        canonical_model(retired)


def test_camera_upgrades_are_inert_on_the_other_backends():
    """#60 lives in the LM assembly only; a stale flag would stamp a camera
    the MLX/JAX/Powell objective never fitted."""
    d = PageDewarper(DewarpOption(**BASE, backend="powell",
                                  principal_point=True, spine_weight_boost=4.0,
                                  spine_gammas="0.05"))
    assert d.principal_point is False
    assert d.spine_weight_boost == 1.0
    assert d.spine_gammas == ()


def test_lm_dewarps_the_page_and_stamps_the_camera_it_fitted():
    out = _lm(principal_point=True, spine_weight_boost=1.0,
              spine_gammas="").process(_buf())
    assert out.meta["success"] is True
    rp = out.meta["replay_params"]
    assert rp["camera_np"] == 10
    assert rp["spine"] is None
    assert len(rp["params"]) > 10


def test_principal_point_off_keeps_the_eight_param_layout():
    out = _lm(principal_point=False, spine_weight_boost=1.0,
              spine_gammas="").process(_buf())
    assert out.meta["replay_params"]["camera_np"] == 8


def test_principal_point_straightens_the_output_text_lines():
    """The #60 headline: the keystone-composed camera is real and recoverable."""
    flat = _curvature(_lm(principal_point=False, spine_weight_boost=1.0,
                          spine_gammas="").process(_buf()).buffer)
    pp = _curvature(_lm(principal_point=True, spine_weight_boost=1.0,
                        spine_gammas="").process(_buf()).buffer)
    assert pp < flat, f"principal point should reduce curvature ({pp} vs {flat})"


def test_spine_features_need_a_resolvable_binding_side():
    """No page_side meta and binding_side=auto → the fit must stay unweighted
    rather than guess which edge is the gutter."""
    ctx, early = _lm(spine_weight_boost=4.0)._build_dewarp_problem(_buf())
    assert early is None
    assert ctx.binding_side is None and ctx.weights is None

    ctx2, _ = _lm(spine_weight_boost=4.0)._build_dewarp_problem(_buf("left"))
    assert ctx2.binding_side == "right"          # left page → binding right
    assert ctx2.weights is not None
    assert ctx2.weights.max() == pytest.approx(4.0)


def test_binding_side_option_overrides_missing_page_side_meta():
    ctx, _ = _lm(spine_weight_boost=4.0,
                 binding_side="left")._build_dewarp_problem(_buf())
    assert ctx.binding_side == "left"


def test_gamma_grid_can_only_improve_the_fit():
    """γ = 0 always competes, so a page the plain cubic wins is unchanged."""
    plain = _lm(spine_weight_boost=4.0, spine_gammas="").process(_buf("left"))
    grid = _lm(spine_weight_boost=4.0,
               spine_gammas="0.02, 0.05, 0.10").process(_buf("left"))
    spine = grid.meta["replay_params"]["spine"]
    if spine is None:                 # cubic won — the surface must be identical
        assert (grid.meta["replay_params"]["params"]
                == plain.meta["replay_params"]["params"])
    else:
        assert abs(spine["gamma"]) in (0.02, 0.05, 0.10)
        assert spine["s_x"] == pytest.approx(0.15 * grid.meta["replay_params"]
                                             ["model_dims"][0])


@pytest.mark.parametrize("gammas", ["", "0.02, 0.05, 0.10"])
def test_replay_rebuilds_the_lm_surface(gammas):
    """Replay reads camera_np + spine out of the stamp; if either were dropped
    it would remap against a DIFFERENT sheet — same page, silently wrong."""
    d = _lm(principal_point=True, spine_weight_boost=4.0, spine_gammas=gammas)
    src = _buf("left")
    raw, src_type = src.buffer.copy(), src.type
    out = d.process(src)
    rp = out.meta["replay_params"]

    im_x, im_y, pad = PageDewarper._replay_sample_map(raw.shape[:2], rp)
    padded = cv2.copyMakeBorder(raw, pad, pad, pad, pad, cv2.BORDER_CONSTANT,
                                value=[255, 255, 255])
    is_bw = src_type == ImageType.BW
    rep = cv2.remap(padded, im_x, im_y,
                    cv2.INTER_NEAREST if is_bw else cv2.INTER_CUBIC, None,
                    cv2.BORDER_CONSTANT,
                    (255, 255, 255) if padded.ndim == 3 else 255)
    if is_bw:
        _, rep = cv2.threshold(rep, 127, 255, cv2.THRESH_BINARY)

    assert rep.shape == out.buffer.shape
    # The two paths build the same grid but cast to float32 at different
    # points, so a handful of nearest-neighbour samples land a pixel apart.
    differing = float((rep != out.buffer).mean())
    assert differing < 1e-4, f"{differing:.2%} of pixels differ"


def _curvature(img):
    """Mean |deviation from a straight line| of the output text baselines, px."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, g.shape[1] // 40), 1))
    n, lab, st, _ = cv2.connectedComponentsWithStats(
        cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k), 8)
    devs = []
    for i in range(1, n):
        x, y, w, h, _a = st[i]
        if w < g.shape[1] * 0.3 or h > g.shape[0] * 0.05:
            continue
        ys, xs = np.nonzero(lab[y:y + h, x:x + w] == i)
        if xs.size < 50:
            continue
        cols = np.unique(xs)
        base = np.array([ys[xs == c].max() for c in cols], float)
        devs.append(np.mean(np.abs(base - np.polyval(
            np.polyfit(cols.astype(float), base, 1), cols.astype(float)))))
    assert devs, "no text lines found in the dewarped output"
    return float(np.mean(devs))


def test_a_rejected_fit_walks_the_whole_ladder_then_concedes():
    """Rungs, cheapest first: γ grid off, then a flat (zero-curl) fit, then
    grayscale. Each rung re-enters process(), whose own ladder starts one rung
    lower — that is what bounds the recursion.

    `max_oob = 0` rejects any remap that overshoots by even a pixel, which is
    what drives the ladder all the way down. Note the page still comes out
    dewarped: the flat rung's remap does not overshoot at all."""
    d = _lm(spine_gammas="0.02, 0.05, 0.10")
    d.max_oob = 0.0                    # force the gate to reject every remap
    rungs = []
    real_spine, real_flat = d._retry_without_spine, d._retry_flat

    def spy_spine(buf, orig, roi):
        rungs.append(("spine", tuple(d.spine_gammas)))
        return real_spine(buf, orig, roi)

    def spy_flat(buf, orig, roi):
        rungs.append(("flat", d._flat_fit))
        return real_flat(buf, orig, roi)

    d._retry_without_spine, d._retry_flat = spy_spine, spy_flat
    out = d.process(_buf())

    kinds = [r[0] for r in rungs]
    assert kinds.count("spine") >= 1 and kinds.count("flat") >= 1, rungs
    # The γ rung fires first, with the grid still populated.
    assert rungs[0] == ("spine", (0.02, 0.05, 0.10))
    # The flat rung is reached only after the γ rung has nothing left.
    assert ("flat", False) in rungs
    # Both flags restored for the next page — a fallback must not be sticky.
    assert d.spine_gammas == (0.02, 0.05, 0.10)
    assert d._flat_fit is False
    # Recursion terminated: each rung fired once per level, not repeatedly.
    assert rungs == [("spine", (0.02, 0.05, 0.10)), ("spine", ()),
                     ("flat", False)]
    # And the page survives: even under a gate this brutal the flat rung's
    # remap lands in bounds, because a zero-curl surface cannot overshoot.
    assert out.meta["success"] is True


def test_the_flat_rung_produces_a_usable_page_not_grayscale():
    """The point of the rung: a zero-curl surface cannot produce the runaway
    remap that trips the gate, so it always has an answer."""
    d = _lm(spine_gammas="")
    buf = _buf("left")
    ctx, early = d._build_dewarp_problem(buf)
    assert early is None
    d._flat_fit = True
    try:
        params = d._solve_lm(ctx)
    finally:
        d._flat_fit = False
    assert params[6] == pytest.approx(0.0) and params[7] == pytest.approx(0.0)
    assert ctx.spine is None
    ctx.params_initial = ctx.params
    out = d._finish_dewarp(buf, params, ctx)
    assert out.meta["success"] is True
    assert max(out.meta["oob"].values()) <= d.max_oob


def test_a_flat_fallback_does_not_poison_the_warm_start():
    """A flat fit's curl is 0 by construction, not by measurement. Recording
    it would drag the next same-side page's seed toward zero on the strength
    of a failure — the ring exists to carry a book's real curl forward."""
    d = _lm(spine_gammas="")
    d._flat_fit = True
    try:
        d.process(_buf("left"))
    finally:
        d._flat_fit = False
    assert not any(d._warm_curl.values()), "flat fallback was remembered"

    # …while an ordinary fit still is, or the warm start would do nothing.
    d.process(_buf("left"))
    assert any(d._warm_curl.values()), "a normal fit must seed the ring"


def test_no_spine_grid_means_nothing_to_back_off_to():
    d = _lm(spine_gammas="")
    assert d._retry_without_spine(_buf(), None, None) is None
