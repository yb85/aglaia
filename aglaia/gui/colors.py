# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Theme color tokens.

Single source of truth for every color used by the Qt GUI. Importers
reference the ``COLOR_*`` constants below instead of hardcoding hex or
rgba literals. See ``docs/theme.md`` for rationale + token roles.

## Theme selection

Two palettes ship — ``_DARK`` and ``_LIGHT``. The active palette is
picked at module import time:

1. ``AGLAIA_THEME`` env var, if set to ``"light"`` or ``"dark"``.
2. Otherwise, ``KEY_THEME`` from the per-user config DB
   (``"system"`` → resolved against ``QPalette`` luminance, or
   defaulted to dark if Qt isn't initialised yet).
3. Fallback: ``_DARK``.

Switching themes at runtime requires an app restart for full fidelity:
``f"color: {COLOR_X};"`` strings baked at widget construction time
won't re-render. ``apply_modern_theme`` can repaint the qdarktheme
palette + the central ``_EXTRA_QSS``, which covers most chrome, but
inline-styled widgets stay on the imported palette until they're
reconstructed.

## Adding new tokens

If a hardcoded color appears that doesn't fit an existing token,
introduce a ``COLOR_TBD_<descriptor>`` entry in both ``_DARK`` and
``_LIGHT`` below. The TBD section at the bottom is gradually folded
into semantic tokens.
"""
from __future__ import annotations

import os


_DARK = {
    # ── Surfaces (background layers) ─────────────────────────────────
    "COLOR_BG": "#1f1f23",
    # Solid surface one shade brighter than ``COLOR_BG`` — used for
    # card bodies + header bands so they read as "front of page" even
    # when the user opts out of the alpha-tinted ``COLOR_BG_SURFACE``
    # treatment.
    "COLOR_BG_HINT": "#262626",
    "COLOR_BG_SURFACE": "rgba(255, 255, 255, 0.04)",
    "COLOR_BG_SURFACE_ALT": "rgba(255, 255, 255, 0.06)",
    "COLOR_BG_SURFACE_STRONG": "rgba(255, 255, 255, 0.08)",
    "COLOR_BG_INPUT": "rgba(255, 255, 255, 0.045)",
    "COLOR_BG_INPUT_HOVER": "rgba(255, 255, 255, 0.08)",
    "COLOR_BG_INPUT_FOCUS": "rgba(59, 130, 246, 0.06)",
    "COLOR_BG_INPUT_DISABLED": "rgba(255, 255, 255, 0.025)",
    "COLOR_BG_BUTTON": "#3f3f46",
    "COLOR_BG_BUTTON_HOVER": "#52525b",
    "COLOR_BG_BUTTON_PRESSED": "#27272a",
    "COLOR_BG_BUTTON_CHECKED": "#2563eb",
    "COLOR_BG_TOGGLE": "#27272a",
    "COLOR_BG_TOGGLE_ON": "#16a34a",
    "COLOR_BG_OVERLAY_SOFT": "rgba(255, 255, 255, 0.04)",
    "COLOR_BG_OVERLAY_HOVER": "rgba(255, 255, 255, 0.08)",
    "COLOR_BG_VIDEO": "#0a0a0a",
    "COLOR_BG_ZEBRA_EVEN": "#262626",
    "COLOR_BG_ZEBRA_ODD": "#1d1d1d",
    "COLOR_BG_TOAST": "rgba(30, 30, 30, 0.92)",
    "COLOR_BG_TOAST_QCOLOR": "rgba(30, 30, 30, 230)",
    # ── Scrims (black-alpha overlays) ─────────────────────────────────
    "COLOR_SCRIM_LIGHT": "rgba(0, 0, 0, 0.18)",
    "COLOR_SCRIM_MEDIUM": "rgba(0, 0, 0, 0.40)",
    "COLOR_SCRIM_STRONG": "rgba(0, 0, 0, 0.70)",
    # ── Outlines / borders ───────────────────────────────────────────
    "COLOR_OUTLINE": "rgba(255, 255, 255, 0.14)",
    "COLOR_OUTLINE_STRONG": "rgba(255, 255, 255, 0.22)",
    "COLOR_OUTLINE_SUBTLE": "rgba(255, 255, 255, 0.08)",
    "COLOR_OUTLINE_FAINT": "rgba(255, 255, 255, 0.06)",
    "COLOR_OUTLINE_GHOST": "rgba(255, 255, 255, 0.10)",
    "COLOR_OUTLINE_BUTTON": "#52525b",
    "COLOR_OUTLINE_BUTTON_STRONG": "#3f3f46",
    # ── Text ─────────────────────────────────────────────────────────
    "COLOR_FONT_PRIMARY": "#f0f0f0",
    "COLOR_FONT_SECONDARY": "rgba(255, 255, 255, 0.78)",
    "COLOR_FONT_MUTED": "rgba(255, 255, 255, 0.55)",
    "COLOR_FONT_DIM": "rgba(255, 255, 255, 0.45)",
    "COLOR_FONT_DISABLED": "rgba(255, 255, 255, 0.35)",
    "COLOR_FONT_INVERSE": "#ffffff",
    "COLOR_FONT_ON_BUTTON": "#e5e7eb",
    "COLOR_FONT_ON_TOGGLE": "#e5e7eb",
    "COLOR_FONT_PLACEHOLDER": "#9ca3af",
    "COLOR_FONT_LINK": "#c0c4cc",
    "COLOR_FONT_LINK_HOVER": "#ffffff",
    "COLOR_FONT_TIMING_NAME": "#e5e7eb",
    "COLOR_FONT_SECTION_LABEL": "#94a3b8",
    # ── Accents ──────────────────────────────────────────────────────
    "COLOR_PRIMARY": "#3b82f6",
    "COLOR_PRIMARY_HOVER": "#1d4ed8",
    "COLOR_PRIMARY_BG_SOFT": "rgba(59, 130, 246, 0.20)",
    "COLOR_PRIMARY_BG_STRONG": "rgba(59, 130, 246, 0.35)",
    "COLOR_PRIMARY_BORDER": "rgba(59, 130, 246, 0.55)",
    "COLOR_PRIMARY_BORDER_STRONG": "rgba(59, 130, 246, 0.70)",
    "COLOR_SECONDARY": "#f59e0b",
    "COLOR_SECONDARY_BG_SOFT": "rgba(245, 158, 11, 0.18)",
    "COLOR_SECONDARY_BG_HOVER": "rgba(245, 158, 11, 0.28)",
    "COLOR_SECONDARY_BORDER": "rgba(245, 158, 11, 0.55)",
    "COLOR_TERTIARY": "#a78bfa",
    "COLOR_TERTIARY_BG_SOFT": "rgba(167, 139, 250, 0.18)",
    "COLOR_ACCENT_BADGE_BG": "rgba(120, 170, 255, 0.18)",
    "COLOR_ACCENT_BADGE_FG": "rgba(180, 210, 255, 0.95)",
    "COLOR_ACCENT_BADGE_BORDER": "rgba(120, 170, 255, 0.35)",
    # ── Semantic ─────────────────────────────────────────────────────
    "COLOR_ERROR": "#ef4444",
    "COLOR_ERROR_STRONG": "#f43f5e",
    "COLOR_ERROR_BG_SOFT": "rgba(239, 68, 68, 0.15)",
    "COLOR_WARNING": "#f59e0b",
    "COLOR_SUCCESS": "#16a34a",
    "COLOR_SUCCESS_BORDER": "#15803d",
    "COLOR_SUCCESS_BG_SOFT": "rgba(34, 197, 94, 0.15)",
    # ── Medals ───────────────────────────────────────────────────────
    "COLOR_MEDAL_GOLD": "#fbbf24",
    "COLOR_MEDAL_SILVER": "#cbd5e1",
    "COLOR_MEDAL_BRONZE": "#d97706",
    # ── Timing-bar gradient ──────────────────────────────────────────
    "COLOR_TIMING_FAST": "#84cc16",
    "COLOR_TIMING_MID": "#eab308",
    "COLOR_TIMING_SLOW": "#f97316",
    "COLOR_TIMING_SLOWEST": "#ef4444",
    "COLOR_TIMING_STOP_GREEN": "#22c55e",
    "COLOR_TIMING_STOP_CHARTREUSE": "#a3e635",
    "COLOR_TIMING_STOP_DARK_ORANGE": "#ea580c",
    "COLOR_TIMING_STOP_RED": "#dc2626",
    "COLOR_TIMING_STOP_CRIMSON": "#b91c1c",
    "COLOR_TIMING_STOP_DARK_RED": "#7f1d1d",
    # ── Text on chrome surfaces ──────────────────────────────────────
    "COLOR_FONT_ON_TOAST": "#f4f4f5",
    # ── Shadows ──────────────────────────────────────────────────────
    "COLOR_SHADOW": "rgba(0, 0, 0, 90)",
    "COLOR_SHADOW_STRONG": "rgba(0, 0, 0, 140)",
    # ── TBDs ─────────────────────────────────────────────────────────
    "COLOR_TIPPING": "rgba(244,63,94,0.10)",
}


_LIGHT = {
    # ── Surfaces (background layers) ─────────────────────────────────
    "COLOR_BG": "#fafafa",
    # Solid surface one shade darker than ``COLOR_BG`` — same role as
    # the dark-mode entry: a card/header band sitting on top of the
    # page bg.
    "COLOR_BG_HINT": "#f3f4f6",
    "COLOR_BG_SURFACE": "rgba(0, 0, 0, 0.03)",
    "COLOR_BG_SURFACE_ALT": "rgba(0, 0, 0, 0.06)",
    "COLOR_BG_SURFACE_STRONG": "rgba(0, 0, 0, 0.10)",
    "COLOR_BG_INPUT": "#ffffff",
    "COLOR_BG_INPUT_HOVER": "rgba(0, 0, 0, 0.04)",
    "COLOR_BG_INPUT_FOCUS": "rgba(37, 99, 235, 0.10)",
    "COLOR_BG_INPUT_DISABLED": "rgba(0, 0, 0, 0.04)",
    "COLOR_BG_BUTTON": "#e4e4e7",
    "COLOR_BG_BUTTON_HOVER": "#d4d4d8",
    "COLOR_BG_BUTTON_PRESSED": "#a1a1aa",
    "COLOR_BG_BUTTON_CHECKED": "#2563eb",
    "COLOR_BG_TOGGLE": "#e4e4e7",
    "COLOR_BG_TOGGLE_ON": "#16a34a",
    "COLOR_BG_OVERLAY_SOFT": "rgba(0, 0, 0, 0.04)",
    "COLOR_BG_OVERLAY_HOVER": "rgba(0, 0, 0, 0.08)",
    "COLOR_BG_VIDEO": "#0a0a0a",
    "COLOR_BG_ZEBRA_EVEN": "#f4f4f5",
    "COLOR_BG_ZEBRA_ODD": "#e4e4e7",
    "COLOR_BG_TOAST": "rgba(245, 245, 247, 0.96)",
    "COLOR_BG_TOAST_QCOLOR": "rgba(245, 245, 247, 245)",
    # ── Scrims (black-alpha overlays — same in both palettes since
    # their job is to dim underlying content) ────────────────────────
    "COLOR_SCRIM_LIGHT": "rgba(0, 0, 0, 0.18)",
    "COLOR_SCRIM_MEDIUM": "rgba(0, 0, 0, 0.40)",
    "COLOR_SCRIM_STRONG": "rgba(0, 0, 0, 0.70)",
    # ── Outlines / borders ───────────────────────────────────────────
    "COLOR_OUTLINE": "rgba(0, 0, 0, 0.18)",
    "COLOR_OUTLINE_STRONG": "rgba(0, 0, 0, 0.30)",
    "COLOR_OUTLINE_SUBTLE": "rgba(0, 0, 0, 0.10)",
    "COLOR_OUTLINE_FAINT": "rgba(0, 0, 0, 0.06)",
    "COLOR_OUTLINE_GHOST": "rgba(0, 0, 0, 0.12)",
    "COLOR_OUTLINE_BUTTON": "#a1a1aa",
    "COLOR_OUTLINE_BUTTON_STRONG": "#71717a",
    # ── Text ─────────────────────────────────────────────────────────
    "COLOR_FONT_PRIMARY": "#18181b",
    "COLOR_FONT_SECONDARY": "rgba(0, 0, 0, 0.78)",
    "COLOR_FONT_MUTED": "rgba(0, 0, 0, 0.62)",
    "COLOR_FONT_DIM": "rgba(0, 0, 0, 0.50)",
    "COLOR_FONT_DISABLED": "rgba(0, 0, 0, 0.36)",
    "COLOR_FONT_INVERSE": "#ffffff",
    "COLOR_FONT_ON_BUTTON": "#18181b",
    "COLOR_FONT_ON_TOGGLE": "#18181b",
    "COLOR_FONT_PLACEHOLDER": "#71717a",
    "COLOR_FONT_LINK": "#3f3f46",
    "COLOR_FONT_LINK_HOVER": "#1d4ed8",
    "COLOR_FONT_TIMING_NAME": "#27272a",
    "COLOR_FONT_SECTION_LABEL": "#52525b",
    # ── Accents ──────────────────────────────────────────────────────
    "COLOR_PRIMARY": "#2563eb",
    "COLOR_PRIMARY_HOVER": "#1d4ed8",
    "COLOR_PRIMARY_BG_SOFT": "rgba(37, 99, 235, 0.12)",
    "COLOR_PRIMARY_BG_STRONG": "rgba(37, 99, 235, 0.22)",
    "COLOR_PRIMARY_BORDER": "rgba(37, 99, 235, 0.45)",
    "COLOR_PRIMARY_BORDER_STRONG": "rgba(37, 99, 235, 0.65)",
    "COLOR_SECONDARY": "#d97706",
    "COLOR_SECONDARY_BG_SOFT": "rgba(217, 119, 6, 0.16)",
    "COLOR_SECONDARY_BG_HOVER": "rgba(217, 119, 6, 0.26)",
    "COLOR_SECONDARY_BORDER": "rgba(217, 119, 6, 0.55)",
    "COLOR_TERTIARY": "#6d28d9",
    "COLOR_TERTIARY_BG_SOFT": "rgba(109, 40, 217, 0.14)",
    "COLOR_ACCENT_BADGE_BG": "rgba(37, 99, 235, 0.10)",
    "COLOR_ACCENT_BADGE_FG": "#1d4ed8",
    "COLOR_ACCENT_BADGE_BORDER": "rgba(37, 99, 235, 0.30)",
    # ── Semantic ─────────────────────────────────────────────────────
    "COLOR_ERROR": "#dc2626",
    "COLOR_ERROR_STRONG": "#e11d48",
    "COLOR_ERROR_BG_SOFT": "rgba(220, 38, 38, 0.12)",
    "COLOR_WARNING": "#d97706",
    "COLOR_SUCCESS": "#15803d",
    "COLOR_SUCCESS_BORDER": "#166534",
    "COLOR_SUCCESS_BG_SOFT": "rgba(22, 163, 74, 0.12)",
    # ── Medals ───────────────────────────────────────────────────────
    # Light-mode tints kept chromatically distinct: saturated yellow,
    # cool slate, copper-orange. The previous attempt at "darker for
    # contrast" collapsed all three into the same amber-brown band.
    "COLOR_MEDAL_GOLD": "#eab308",
    "COLOR_MEDAL_SILVER": "#94a3b8",
    "COLOR_MEDAL_BRONZE": "#c2410c",
    # ── Timing-bar gradient ──────────────────────────────────────────
    "COLOR_TIMING_FAST": "#65a30d",
    "COLOR_TIMING_MID": "#ca8a04",
    "COLOR_TIMING_SLOW": "#ea580c",
    "COLOR_TIMING_SLOWEST": "#dc2626",
    "COLOR_TIMING_STOP_GREEN": "#16a34a",
    "COLOR_TIMING_STOP_CHARTREUSE": "#65a30d",
    "COLOR_TIMING_STOP_DARK_ORANGE": "#c2410c",
    "COLOR_TIMING_STOP_RED": "#b91c1c",
    "COLOR_TIMING_STOP_CRIMSON": "#991b1b",
    "COLOR_TIMING_STOP_DARK_RED": "#7f1d1d",
    # ── Text on chrome surfaces ──────────────────────────────────────
    "COLOR_FONT_ON_TOAST": "#18181b",
    # ── Shadows ──────────────────────────────────────────────────────
    "COLOR_SHADOW": "rgba(0, 0, 0, 15)",
    "COLOR_SHADOW_STRONG": "rgba(0, 0, 0, 30)",
    # ── TBDs ─────────────────────────────────────────────────────────
    "COLOR_TIPPING": "rgba(225,29,72,0.10)",
    # QColor `#AARRGGBB` — flip the white-on-black overlays to black-on-white.
}


def _resolve_palette() -> tuple[str, dict[str, str]]:
    """Decide which palette to use at module import time."""
    # Explicit env-var override wins.
    env = os.environ.get("AGLAIA_THEME", "").strip().lower()
    if env == "light":
        return "light", _LIGHT
    if env == "dark":
        return "dark", _DARK
    # Otherwise read the saved KEY_THEME pref. Tolerate any failure
    # (config DB not bootstrapped yet at first launch, import cycle,
    # missing dep) by falling back to dark.
    try:
        from aglaia.app_data import db as cfg
        with cfg.session() as conn:
            pref = str(cfg.get(conn, cfg.KEY_THEME, "system") or "system")
    except Exception:
        pref = "system"
    if pref == "light":
        return "light", _LIGHT
    if pref == "dark":
        return "dark", _DARK
    # `pref == "system"` → resolve via darkdetect, the SAME oracle
    # qdarktheme's "auto" mode uses, so the chrome (qdarktheme stylesheet)
    # and the inline COLOR_* tokens can't disagree. The old QPalette
    # luminance probe was unreliable: qdarktheme themes via stylesheet and
    # leaves the palette Window role light, so the probe returned "light"
    # under a dark theme → dark fonts on dark chrome (broken on Linux).
    # Default to dark when darkdetect can't tell (None), matching prior
    # fallback behaviour.
    try:
        import darkdetect
        if str(darkdetect.theme() or "").lower() == "light":
            return "light", _LIGHT
    except Exception:
        pass
    return "dark", _DARK


_ACTIVE_NAME, _PALETTE = _resolve_palette()

# Inject all token names from the active palette into the module
# namespace so consumers keep doing `from aglaia.gui.colors import COLOR_X`
# without changes. Sanity-check that both palettes hold the same keys
# at import time so a future addition can't silently regress one side.
assert set(_DARK) == set(_LIGHT), (
    f"_DARK / _LIGHT palette key mismatch: "
    f"dark-only={sorted(set(_DARK)-set(_LIGHT))} "
    f"light-only={sorted(set(_LIGHT)-set(_DARK))}"
)
globals().update(_PALETTE)

# Cross-token alias kept for ergonomics. Resolved against the active
# palette so semantic uses (`COLOR_INFO` → "info accent") stay coherent.
COLOR_INFO = _PALETTE["COLOR_PRIMARY"]


def active_palette_name() -> str:
    """Return ``"dark"`` or ``"light"`` depending on what's installed."""
    return _ACTIVE_NAME


def qcolor(value):
    """Parse any palette-token string (``"#abc"``, ``"#abcdef"``,
    ``"#aarrggbb"``, ``"rgb(r, g, b)"``, ``"rgba(r, g, b, a)"``, or a
    bare CSS-style name like ``"white"``) into a PySide6 ``QColor``.

    Useful for places that need a ``QColor`` literal (``QPen``,
    ``QPainter.fillRect``, ``QPainter.setBrush``) instead of feeding
    the string through a QSS f-string — saves having to keep parallel
    ``_QCOLOR`` siblings of every token.
    """
    from PySide6.QtGui import QColor
    s = str(value).strip()
    if s.startswith("rgba(") and s.endswith(")"):
        parts = [p.strip() for p in s[len("rgba("):-1].split(",")]
        r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
        a_raw = parts[3]
        a = int(round(float(a_raw) * 255)) if "." in a_raw else int(a_raw)
        return QColor(r, g, b, a)
    if s.startswith("rgb(") and s.endswith(")"):
        parts = [int(p.strip()) for p in s[len("rgb("):-1].split(",")]
        return QColor(*parts)
    return QColor(s)


__all__ = sorted(
    list(_PALETTE.keys())
    + ["COLOR_INFO", "active_palette_name", "qcolor"]
)
