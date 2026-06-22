# Aglaïa — DIDACTIC PLUGIN EXAMPLE (shipped, not auto-loaded)
#
# This file is a *complete, runnable* example of a drop-in processor plugin.
# It is shipped under examples/ purely as reference — it is NOT one of the
# built-in processors and is NOT registered unless you copy it into your
# Aglaïa plugin folder:
#
#     <APP_DATA>/plugins/processors/ExemplePluginDespeckler.py
#
# where <APP_DATA> on macOS is
#     ~/Library/Application Support/Aglaia/
# (or wherever $AGLAIA_APP_DATA_DIR points). On first launch after dropping
# the file in, the GUI's trust gate pops a warning — choose "Add (trust &
# run)" and its sha256 is recorded in the `plugins` table. Headless/CLI runs
# load only already-accepted plugins (no popup) and warn about pending ones.
#
# What this example demonstrates, end to end, with ZERO edits to the codebase:
#   1. The processor contract (SUMMARY / OPTIONS / OPTION_CLASS / process()).
#   2. Reading UPSTREAM METADATA — it consumes `meta["char_h_frac"]` (the
#      text scale stamped by TrapezoidalCorrection / PageDewarper) to size the
#      speckle threshold relative to the glyphs, instead of guessing in pixels.
#   3. Joining the END-OF-CHAIN REPLAY pass as a PIXEL_VALUE step — by setting
#      REPLAY_TRAIT and implementing apply_replay(), the replay engine applies
#      it last, on the final fused geometry, with no special-casing in
#      Replay.py.
#   4. DECLARING what metadata it stamps via PROVIDES_META (documentation +
#      discoverability for the next plugin author).
#
# Algorithm: remove "speckle" — tiny isolated ink blobs (scanner grain,
# binarisation noise) whose largest side is a small fraction of a character's
# height. A blob the size of a comma is kept; a 2 px fly-speck is wiped to the
# page background.

from dataclasses import dataclass

import cv2
import numpy as np

# Everything a plugin needs is imported from the PUBLIC contract modules —
# the same ones the built-in processors use. The plugin directory is placed
# on sys.path at load time, so `lib.*` resolves identically here and in the
# spawned multiprocessing workers.
from lib.ImageBuffer import ImageBuffer
from lib.processors.abstraction import (
    AbstractImageProcessor,
    AbstractProcessorOption,
    ReplayTrait,
)
from lib.processors.option_specs import _f, _i, _e
from lib.processors.utils import is_binary, to_gray


# ── options ────────────────────────────────────────────────────────────────
# A processor's tunables live on a dataclass that EXTENDS
# AbstractProcessorOption (which carries the shared debug/timeout plumbing).
# The OPTIONS dict below mirrors these fields with UI/validation metadata.
@dataclass
class ExemplePluginDespecklerOption(AbstractProcessorOption):
    # A blob is speckle when its largest side ≤ speckle_frac × glyph height.
    # 0.15 ≈ keep anything down to ~1/7 of a letter (commas, accents, dots on
    # i/j survive); wipe smaller dirt.
    speckle_frac: float = 0.15
    # Fallback maximum speckle side in pixels, used only when the upstream
    # text scale (`char_h_frac`) is unavailable on the buffer.
    min_size_px: int = 3
    # Connected-component connectivity: 8 (diagonal-touching counts) or 4.
    connectivity: int = 8


# ── shared core (used by BOTH the forward pass and replay) ───────────────────
# Factoring the pixel work into a free function keeps process() and
# apply_replay() in lock-step — the whole point of the replay contract is that
# the replayed step reproduces the forward step exactly.
def _despeckle(img: np.ndarray, *, speckle_frac: float, min_size_px: int,
               connectivity: int, char_h_frac: float) -> tuple[np.ndarray, int, int]:
    """Return (cleaned image, blobs removed, threshold px used)."""
    gray = to_gray(img) if img.ndim == 3 else img

    # Ink mask = the dark marks on a light page (or vice-versa). We binarise a
    # COPY only to *find* the blobs; the wipe happens on the original so a
    # grayscale/colour figure keeps its tones everywhere except the specks.
    light_bg = gray.mean() > 127
    if is_binary(img):
        ink = cv2.bitwise_not(gray) if light_bg else gray.copy()
    else:
        _, ink = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Resolve the speckle size. char_h_frac is glyph height ÷ page height
    # (dimensionless), so multiplying by THIS image's height gives glyph px at
    # the current resolution — robust to any upstream resample/warp.
    page_h = img.shape[0]
    if char_h_frac > 0:
        max_side = max(min_size_px, int(round(speckle_frac * char_h_frac * page_h)))
    else:
        max_side = min_size_px  # no text-scale hint → absolute fallback

    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity)
    speck = np.zeros(ink.shape, dtype=bool)
    removed = 0
    for i in range(1, n):  # 0 is the background label
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if max(w, h) <= max_side:
            speck |= (labels == i)
            removed += 1

    out = img.copy()
    bg = 255 if light_bg else 0
    if out.ndim == 3:
        out[speck] = (bg, bg, bg)
    else:
        out[speck] = bg
    return out, removed, max_side


# ── the processor ────────────────────────────────────────────────────────────
class ExemplePluginDespeckler(AbstractImageProcessor):
    name: str = "ExemplePluginDespeckler"
    # REQUIRED: shown in the GUI add-step menu and pipeline cards.
    SUMMARY = "Example plugin: remove ink specks smaller than a fraction of a glyph."
    # Links the OPTIONS below to the typed dataclass above.
    OPTION_CLASS = ExemplePluginDespecklerOption
    # PIXEL_VALUE → a value/neighbourhood op. The replay engine pushes it as
    # late as possible (after all geometry fuses) and calls apply_replay().
    REPLAY_TRAIT = ReplayTrait.PIXEL_VALUE
    # Field names surfaced in the pipeline card's one-line description.
    _ESSENTIAL_PARAMS = ("speckle_frac", "min_size_px", "connectivity")

    # DECLARE the metadata this step stamps. Documentary only — it does not
    # perform the stamping (process() does), but it tells the next author /
    # the generated docs what this processor contributes.
    PROVIDES_META = {
        "replay_kind": "'despeckle' — marks this step for the replay engine",
        "replay_params": "dict of the resolved despeckle parameters",
    }

    # REQUIRED: the option_name → ParamSpec map the GUI form and the YAML
    # loader read. Helpers: _f(float), _i(int), _e(enum), _b(bool), _s(str).
    OPTIONS = {
        "speckle_frac": _f(0.15, 0.0, 1.0, 0.01,
                           "Max speckle size as a fraction of glyph height "
                           "(needs char_h_frac from Trap/Dewarp upstream)."),
        "min_size_px": _i(3, 1, 50,
                          "Fallback max speckle side in px when the text "
                          "scale is unknown on the buffer."),
        "connectivity": _e("8", ["8", "4"],
                           "Connected-component connectivity for blob grouping."),
    }

    def __init__(self, options: ExemplePluginDespecklerOption):
        super().__init__(options)
        self.opt = options

    # The forward pass. Read upstream meta, do the work, stamp replay params.
    def process(self, buf: ImageBuffer) -> ImageBuffer:
        # CONSUME upstream metadata. char_h_frac is stamped by
        # TrapezoidalCorrection / PageDewarper; 0.0 means "not measured", and
        # _despeckle falls back to min_size_px.
        char_h_frac = float(buf.meta.get("char_h_frac", 0.0) or 0.0)
        connectivity = int(self.opt.connectivity)

        cleaned, removed, max_side = _despeckle(
            buf.buffer,
            speckle_frac=self.opt.speckle_frac,
            min_size_px=self.opt.min_size_px,
            connectivity=connectivity,
            char_h_frac=char_h_frac,
        )

        out = ImageBuffer(
            cleaned, buf.type, dpi=buf.dpi,
            filestem=buf.filestem,
            parent=buf,
            scan_id=buf.scan_id,
            parent_node_id=buf.parent_node_id,
            pipeline_version_id=buf.pipeline_version_id,
            depth=buf.depth,
            branch_label=buf.branch_label,
        )
        # Forward geometry/scale metadata a downstream step still needs — a
        # PIXEL_VALUE op leaves coordinates untouched, so the ROI mask and the
        # text scale remain valid.
        if (roi := buf.meta.get("roi")) is not None:
            out.meta["roi"] = roi
        if char_h_frac > 0:
            out.meta["char_h_frac"] = char_h_frac

        # STAMP the replay contract. The engine reads replay_kind to know this
        # node participates, and hands replay_params back to apply_replay().
        out.meta["replay_kind"] = "despeckle"
        out.meta["replay_params"] = {
            "speckle_frac": float(self.opt.speckle_frac),
            "min_size_px": int(self.opt.min_size_px),
            "connectivity": connectivity,
            "char_h_frac": char_h_frac,
        }

        # Per-call op-log line (drained by the chain worker).
        self.last_stats = {
            "removed": removed,
            "max_side_px": max_side,
            "char_h_frac": round(char_h_frac, 4),
        }
        return out

    # The replay pass. PIXEL_VALUE/ROI steps implement apply_replay (COORDINATE
    # steps implement replay_transform instead). It receives the raw buffer +
    # ROI mask + the params we stamped above, and returns (buf, mask).
    @classmethod
    def apply_replay(cls, buf, mask, params, ctx):
        cleaned, _removed, _max_side = _despeckle(
            buf,
            speckle_frac=float(params.get("speckle_frac", 0.15)),
            min_size_px=int(params.get("min_size_px", 3)),
            connectivity=int(params.get("connectivity", 8)),
            char_h_frac=float(params.get("char_h_frac", 0.0) or 0.0),
        )
        # Coordinates are unchanged, so the carried ROI mask passes through.
        return cleaned, mask
