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
BASE = dict(sheet_model="cylindrical", twist=False, baseline_source="bottom",
            cubic_cost=0.0, focal_length=1.3)


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


def test_auto_resolves_to_lm_for_the_cylindrical_model():
    assert PageDewarper(DewarpOption(**BASE, backend="auto")).backend == "lm"


def test_auto_keeps_the_gpu_chain_for_spline_models():
    """LM's analytic Jacobian is the cubic sheet's — it cannot fit a spline."""
    d = PageDewarper(DewarpOption(**{**BASE, "sheet_model": "bspline_twist"},
                                  backend="auto"))
    assert d.backend != "lm"


def test_lm_requested_on_a_spline_model_falls_back_instead_of_misfitting():
    d = PageDewarper(DewarpOption(**{**BASE, "sheet_model": "flat_spline"},
                                  backend="lm"))
    assert d.backend != "lm"


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


def test_rejected_lm_fit_retries_on_powell_before_the_gray_fallback():
    """LM can park a shape DOF on the curl clamp at a competitive objective
    but a wild remap. Conceding straight to grayscale would lose a page that
    Powell still dewarps — so an OOB rejection must retry, not give up."""
    d = _lm()
    d.max_oob = 0.0                    # force the gate to reject any remap
    calls = []
    real = d._retry_with_powell

    def spy(buf, orig, roi):
        calls.append(d.backend)
        return real(buf, orig, roi)

    d._retry_with_powell = spy
    out = d.process(_buf())
    # First entry retries under Powell; the nested rejection sees backend
    # "powell", has nothing left to offer and returns None — that is what
    # stops the retry from recursing forever.
    assert calls == ["lm", "powell"]
    assert d.backend == "lm", "the backend must be restored for the next page"
    # The gate rejects everything here, so the page still ends in the gray
    # fallback — what matters is that Powell was given its turn.
    assert out.meta["success"] is False


def test_powell_backend_has_no_retry_to_offer():
    d = PageDewarper(DewarpOption(**BASE, backend="powell"))
    assert d._retry_with_powell(_buf(), None, None) is None
