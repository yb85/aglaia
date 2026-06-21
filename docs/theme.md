# Theme color tokens

Single source of truth for colors used in the Qt GUI. Every QSS string,
inline `setStyleSheet(...)` call, and `QColor(...)` literal in
`lib/gui/` references one of the `COLOR_*` tokens defined in
`lib/gui/colors.py`. Two full palettes ship — **dark** and **light** —
and the active one is chosen at module import time, so the same token
name resolves to the right value for whichever theme is live.

## Rationale

Before tokenizing, ~27 GUI files hardcoded white-alpha overlays
(`rgba(255, 255, 255, 0.04/0.08/0.45/0.55)`) and light-gray hex codes
(`#c0c4cc`, `#94a3b8`, `#e5e7eb`, `#cbd5e1`) under the assumption of a
dark backdrop. The light theme was unusable as a result. Routing every
color use through a token made the light theme mechanical: `_LIGHT`
mirrors `_DARK` key-for-key (an import-time `assert` enforces it), so
flipping the active palette repaints the whole UI correctly.

## How the active palette is picked

`lib/gui/colors.py` holds two dicts, `_DARK` and `_LIGHT`. At import
time `_resolve_palette()` chooses one (resolution order):

1. `AGLAIA_THEME` env var (`"light"` / `"dark"`), if set.
2. Otherwise the `KEY_THEME` config key from the per-user config DB
   (`"light"` / `"dark"` / `"system"`).
3. `"system"` probes the live `QPalette` window luminance (≥128 → light);
   if Qt isn't initialised yet, falls back to **dark**.

The chosen palette's keys are injected into the module namespace via
`globals().update(...)`, so consumers keep doing
`from lib.gui.colors import COLOR_BG, COLOR_FONT_PRIMARY, ...` unchanged.
`active_palette_name()` reports `"dark"` / `"light"`; `qcolor(value)`
parses any token string (`#rgb`, `#rrggbb`, `rgb(...)`, `rgba(...)`,
named) into a `QColor` for painter / pen use without parallel `_QCOLOR`
siblings.

`apply_modern_theme(app, mode=...)` (in `lib/gui/theme.py`) wraps
`qdarktheme` + the central `_EXTRA_QSS`. Switching themes at runtime
needs an app restart for full fidelity: f-string QSS baked at widget
construction won't re-render until the widget is reconstructed.

## Conventions

- Tokens live in `lib/gui/colors.py` as keys of `_DARK` / `_LIGHT`.
- Always use them via `from lib.gui.colors import COLOR_BG, ...`.
- QSS strings interpolate via f-strings: `f"color: {COLOR_FONT_PRIMARY};"`.
- Adding a token means adding it to **both** `_DARK` and `_LIGHT`
  (the import-time `assert set(_DARK) == set(_LIGHT)` fails otherwise).
- For colors that don't yet fit a semantic token, introduce a
  `COLOR_TBD_<descriptor>` placeholder in both palettes (see "TBDs").

The tables below list both palette values per token.

## Surfaces (background layers)

| Token | Dark | Light | Use |
|---|---|---|---|
| `COLOR_BG` | `#1f1f23` | `#fafafa` | Window / page background. |
| `COLOR_BG_HINT` | `#262626` | `#f3f4f6` | Solid card body / header band (one shade off `COLOR_BG`). |
| `COLOR_BG_SURFACE` | `rgba(255,255,255,0.04)` | `rgba(0,0,0,0.03)` | Card / panel surface, 1 level above `COLOR_BG`. |
| `COLOR_BG_SURFACE_ALT` | `rgba(255,255,255,0.06)` | `rgba(0,0,0,0.06)` | Slightly elevated (selected card, hovered row). |
| `COLOR_BG_SURFACE_STRONG` | `rgba(255,255,255,0.08)` | `rgba(0,0,0,0.10)` | Most elevated (tooltip, popover). |
| `COLOR_BG_INPUT` | `rgba(255,255,255,0.045)` | `#ffffff` | QLineEdit / QSpinBox / QComboBox background. |
| `COLOR_BG_INPUT_HOVER` | `rgba(255,255,255,0.08)` | `rgba(0,0,0,0.04)` | Hover state for inputs. |
| `COLOR_BG_INPUT_FOCUS` | `rgba(59,130,246,0.06)` | `rgba(37,99,235,0.10)` | Focused input tint (matches primary). |
| `COLOR_BG_INPUT_DISABLED` | `rgba(255,255,255,0.025)` | `rgba(0,0,0,0.04)` | Read-only / disabled inputs. |
| `COLOR_BG_BUTTON` | `#3f3f46` | `#e4e4e7` | Default secondary button fill. |
| `COLOR_BG_BUTTON_HOVER` | `#52525b` | `#d4d4d8` | Button hover. |
| `COLOR_BG_BUTTON_PRESSED` | `#27272a` | `#a1a1aa` | Button pressed. |
| `COLOR_BG_BUTTON_CHECKED` | `#2563eb` | `#2563eb` | Toggled / checked button. |
| `COLOR_BG_TOGGLE` | `#27272a` | `#e4e4e7` | Pill toggle (Voice / Freehand) idle. |
| `COLOR_BG_TOGGLE_ON` | `#16a34a` | `#16a34a` | Pill toggle on. |
| `COLOR_BG_OVERLAY_SOFT` | `rgba(255,255,255,0.04)` | `rgba(0,0,0,0.04)` | Generic hover/idle tint. |
| `COLOR_BG_OVERLAY_HOVER` | `rgba(255,255,255,0.08)` | `rgba(0,0,0,0.08)` | Generic hover overlay. |
| `COLOR_BG_VIDEO` | `#0a0a0a` | `#0a0a0a` | Webcam preview placeholder pane (always dark). |
| `COLOR_BG_ZEBRA_EVEN` | `#262626` | `#f4f4f5` | Even zebra row. |
| `COLOR_BG_ZEBRA_ODD` | `#1d1d1d` | `#e4e4e7` | Odd zebra row. |
| `COLOR_BG_TOAST` | `rgba(30,30,30,0.92)` | `rgba(245,245,247,0.96)` | Toast surface (QSS). |
| `COLOR_BG_TOAST_QCOLOR` | `rgba(30,30,30,230)` | `rgba(245,245,247,245)` | Toast surface (QColor 0–255 alpha). |

## Scrims (black-alpha dimming overlays)

Same in both palettes — their job is to dim whatever sits underneath.

| Token | Value | Use |
|---|---|---|
| `COLOR_SCRIM_LIGHT` | `rgba(0,0,0,0.18)` | Light dim. |
| `COLOR_SCRIM_MEDIUM` | `rgba(0,0,0,0.40)` | Medium dim. |
| `COLOR_SCRIM_STRONG` | `rgba(0,0,0,0.70)` | Strong dim (modal backdrop). |

## Outlines / borders

| Token | Dark | Light | Use |
|---|---|---|---|
| `COLOR_OUTLINE` | `rgba(255,255,255,0.14)` | `rgba(0,0,0,0.18)` | Default border on inputs, cards. |
| `COLOR_OUTLINE_STRONG` | `rgba(255,255,255,0.22)` | `rgba(0,0,0,0.30)` | Hovered input border. |
| `COLOR_OUTLINE_SUBTLE` | `rgba(255,255,255,0.08)` | `rgba(0,0,0,0.10)` | Card divider, footer separators. |
| `COLOR_OUTLINE_FAINT` | `rgba(255,255,255,0.06)` | `rgba(0,0,0,0.06)` | Header underlines. |
| `COLOR_OUTLINE_GHOST` | `rgba(255,255,255,0.10)` | `rgba(0,0,0,0.12)` | Disabled / dashed borders. |
| `COLOR_OUTLINE_BUTTON` | `#52525b` | `#a1a1aa` | Button stroke. |
| `COLOR_OUTLINE_BUTTON_STRONG` | `#3f3f46` | `#71717a` | Pill toggle stroke. |

## Text

| Token | Dark | Light | Use |
|---|---|---|---|
| `COLOR_FONT_PRIMARY` | `#f0f0f0` | `#18181b` | Default text. |
| `COLOR_FONT_SECONDARY` | `rgba(255,255,255,0.78)` | `rgba(0,0,0,0.78)` | Buttons / secondary labels. |
| `COLOR_FONT_MUTED` | `rgba(255,255,255,0.55)` | `rgba(0,0,0,0.62)` | Help text, captions, "Subtle" labels. |
| `COLOR_FONT_DIM` | `rgba(255,255,255,0.45)` | `rgba(0,0,0,0.50)` | Empty-state text. |
| `COLOR_FONT_DISABLED` | `rgba(255,255,255,0.35)` | `rgba(0,0,0,0.36)` | Disabled labels. |
| `COLOR_FONT_INVERSE` | `#ffffff` | `#ffffff` | Text on accent backgrounds. |
| `COLOR_FONT_ON_BUTTON` | `#e5e7eb` | `#18181b` | Text on secondary buttons. |
| `COLOR_FONT_ON_TOGGLE` | `#e5e7eb` | `#18181b` | Text on idle toggle. |
| `COLOR_FONT_PLACEHOLDER` | `#9ca3af` | `#71717a` | Placeholder, icon-trailing actions. |
| `COLOR_FONT_LINK` | `#c0c4cc` | `#3f3f46` | Footer link text. |
| `COLOR_FONT_LINK_HOVER` | `#ffffff` | `#1d4ed8` | Hovered footer link. |
| `COLOR_FONT_TIMING_NAME` | `#e5e7eb` | `#27272a` | Timing-view row name. |
| `COLOR_FONT_SECTION_LABEL` | `#94a3b8` | `#52525b` | Section-label small-caps. |
| `COLOR_FONT_ON_TOAST` | `#f4f4f5` | `#18181b` | Text on toast surfaces. |

## Accents

| Token | Dark | Light | Use |
|---|---|---|---|
| `COLOR_PRIMARY` | `#3b82f6` | `#2563eb` | Brand blue. Focus rings, selected state, primary buttons. |
| `COLOR_PRIMARY_HOVER` | `#1d4ed8` | `#1d4ed8` | Primary hover. |
| `COLOR_PRIMARY_BG_SOFT` | `rgba(59,130,246,0.20)` | `rgba(37,99,235,0.12)` | Tinted accent surface (selected card). |
| `COLOR_PRIMARY_BG_STRONG` | `rgba(59,130,246,0.35)` | `rgba(37,99,235,0.22)` | List-item selected background. |
| `COLOR_PRIMARY_BORDER` | `rgba(59,130,246,0.55)` | `rgba(37,99,235,0.45)` | Card hover border. |
| `COLOR_PRIMARY_BORDER_STRONG` | `rgba(59,130,246,0.70)` | `rgba(37,99,235,0.65)` | Card pressed border. |
| `COLOR_SECONDARY` | `#f59e0b` | `#d97706` | Secondary accent (amber). |
| `COLOR_SECONDARY_BG_SOFT` | `rgba(245,158,11,0.18)` | `rgba(217,119,6,0.16)` | Tinted secondary surface. |
| `COLOR_SECONDARY_BG_HOVER` | `rgba(245,158,11,0.28)` | `rgba(217,119,6,0.26)` | Secondary hover surface. |
| `COLOR_SECONDARY_BORDER` | `rgba(245,158,11,0.55)` | `rgba(217,119,6,0.55)` | Secondary border. |
| `COLOR_TERTIARY` | `#a78bfa` | `#6d28d9` | Tertiary accent (violet). |
| `COLOR_TERTIARY_BG_SOFT` | `rgba(167,139,250,0.18)` | `rgba(109,40,217,0.14)` | Tinted tertiary surface. |
| `COLOR_ACCENT_BADGE_BG` | `rgba(120,170,255,0.18)` | `rgba(37,99,235,0.10)` | Badge background (info accent). |
| `COLOR_ACCENT_BADGE_FG` | `rgba(180,210,255,0.95)` | `#1d4ed8` | Badge text. |
| `COLOR_ACCENT_BADGE_BORDER` | `rgba(120,170,255,0.35)` | `rgba(37,99,235,0.30)` | Badge border. |

`COLOR_INFO` is a runtime alias of `COLOR_PRIMARY` (info accent), resolved
against the active palette.

## Semantic

| Token | Dark | Light | Use |
|---|---|---|---|
| `COLOR_ERROR` | `#ef4444` | `#dc2626` | Error / destructive icon, recording dot. |
| `COLOR_ERROR_STRONG` | `#f43f5e` | `#e11d48` | Stronger red (heart glow, tip-jar). |
| `COLOR_ERROR_BG_SOFT` | `rgba(239,68,68,0.15)` | `rgba(220,38,38,0.12)` | Tinted error surface. |
| `COLOR_WARNING` | `#f59e0b` | `#d97706` | Warning amber. |
| `COLOR_SUCCESS` | `#16a34a` | `#15803d` | Success green (toggle on). |
| `COLOR_SUCCESS_BORDER` | `#15803d` | `#166534` | Success darker border. |
| `COLOR_SUCCESS_BG_SOFT` | `rgba(34,197,94,0.15)` | `rgba(22,163,74,0.12)` | Tinted success surface. |

## Medals (engine accuracy badges)

| Token | Dark | Light | Use |
|---|---|---|---|
| `COLOR_MEDAL_GOLD` | `#fbbf24` | `#eab308` | Gold accuracy badge. |
| `COLOR_MEDAL_SILVER` | `#cbd5e1` | `#94a3b8` | Silver accuracy badge. |
| `COLOR_MEDAL_BRONZE` | `#d97706` | `#c2410c` | Bronze accuracy badge. |

## Timing-bar gradient

For `PipelineTimingView` wall-time bar segments. The `*_FAST/MID/SLOW/SLOWEST`
tokens drive the simple gradient; the `*_STOP_*` tokens are the multi-stop
gradient ramp.

| Token | Dark | Light | Use |
|---|---|---|---|
| `COLOR_TIMING_FAST` | `#84cc16` | `#65a30d` | Fast (lime). |
| `COLOR_TIMING_MID` | `#eab308` | `#ca8a04` | Medium (yellow). |
| `COLOR_TIMING_SLOW` | `#f97316` | `#ea580c` | Slow (orange). |
| `COLOR_TIMING_SLOWEST` | `#ef4444` | `#dc2626` | Slowest (red). |
| `COLOR_TIMING_STOP_GREEN` | `#22c55e` | `#16a34a` | Gradient stop. |
| `COLOR_TIMING_STOP_CHARTREUSE` | `#a3e635` | `#65a30d` | Gradient stop. |
| `COLOR_TIMING_STOP_DARK_ORANGE` | `#ea580c` | `#c2410c` | Gradient stop. |
| `COLOR_TIMING_STOP_RED` | `#dc2626` | `#b91c1c` | Gradient stop. |
| `COLOR_TIMING_STOP_CRIMSON` | `#b91c1c` | `#991b1b` | Gradient stop. |
| `COLOR_TIMING_STOP_DARK_RED` | `#7f1d1d` | `#7f1d1d` | Gradient stop. |

## Shadows

| Token | Dark | Light | Use |
|---|---|---|---|
| `COLOR_SHADOW` | `rgba(0,0,0,90)` | `rgba(0,0,0,15)` | Card drop shadow (QColor 0–255 alpha). |
| `COLOR_SHADOW_STRONG` | `rgba(0,0,0,140)` | `rgba(0,0,0,30)` | Modal / dialog shadow. |

## TBDs

Anything that can't be cleanly mapped is introduced as a
`COLOR_TBD_<descriptor>` placeholder in both palettes and gradually
folded into a semantic token. Current placeholder:

| Token | Dark | Light | Use |
|---|---|---|---|
| `COLOR_TIPPING` | `rgba(244,63,94,0.10)` | `rgba(225,29,72,0.10)` | Tip-jar / sponsoring tint. |
