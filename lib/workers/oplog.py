# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Unified operation logger.

Every long-running op (pipeline step, OCR run, export) emits one
structured line through this helper so the log strip + log tab read
the same way across subsystems:

  [pipeline.SkewFinder]  scan=42  layout=01  43ms   angle=+1.4°  rotated=yes
  [pipeline.PageDetector] scan=42  180ms  layouts=3  sizes=1024x780,512x780
  [pipeline.Binarizer]   scan=42  layout=02  64ms   blob_px[p0=4 p50=48 p100=2100]
  [pipeline.Trapezoidal] scan=42  layout=03  210ms  vp_dist=420mm  quad=210x297mm  spans=18
  [pipeline.PageDewarper] scan=42 layout=01  1.28s  α=0.012 β=-0.003 oob_max=42
  [ocr.Surya]            batch=4  8.2s   words=2340  elements{Text:14, Title:1, Table:1}
  [export.PDF]           /Users/y/scans/out.pdf  2.1MB  pages=12  1.1s

API:
  * ``format_op(name, elapsed_ms, **fields)`` returns the formatted
    line — pure function, side-effect-free, used by both queue-based
    chain workers and direct GUI-side callers.
  * ``emit(log_queue, level, name, elapsed_ms, **fields)`` pushes
    the formatted line through the chain's log_queue so it lands in
    the log strip + ProcessMonitor's rolling buffer.

Fields:
  * Reserved short keys: ``scan`` (scan id), ``layout`` (per-scan idx
    or list), ``batch`` (batch size for grouped ops).
  * Any other key is stringified — wrap structured values (lists,
    dicts) via the small helpers below.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional


_NAME_WIDTH = 24

# ── rich-markup palette ───────────────────────────────────────────
# Markup uses `rich` syntax: ``[cyan]...[/]``. ProcessMonitor renders
# it with rich.console for the terminal; Qt signals strip the markup
# (Qt status-bar labels are plain text) via ``strip_markup`` below.
_C_TAG_PIPE  = "bold cyan"
_C_TAG_OCR   = "bold magenta"
_C_TAG_EXP   = "bold green"
_C_TAG_DEF   = "bold white"
_C_SCOPE     = "yellow"     # scan=42 layout=01 batch=4
_C_TIME      = "bold yellow"
_C_KEY       = "dim"
_C_SIZE      = "blue"
_C_DELTA_HOT = "red"       # red when ms > 1500
_C_OK        = "green"

def _tag_color(name: str) -> str:
    head = name.split(".", 1)[0]
    if head == "pipeline": return _C_TAG_PIPE
    if head == "ocr":      return _C_TAG_OCR
    if head == "export":   return _C_TAG_EXP
    return _C_TAG_DEF


def _ms_color(ms: float | None) -> str:
    if ms is None:
        return _C_TIME
    if ms >= 5000:
        return _C_DELTA_HOT
    if ms >= 1500:
        return "yellow"
    return _C_OK


def strip_markup(text: str) -> str:
    """Drop rich-style ``[tag]...[/]`` markup. Cheap regex pass — used
    on the Qt path where status-bar labels render plain text. Also
    un-escapes the literal ``\\[`` / ``\\]`` sequences format_op emits
    so the Qt label shows real brackets, not the escape."""
    import re
    out = re.sub(r"\[/\]|\[[a-zA-Z #_]+\]", "", text)
    return out.replace("\\[", "[")


def fmt_size(wh: tuple[int, int] | None, dpi: float | None = None) -> str:
    """Render one ``WxH@DPI`` token. Returns ``"?"`` when wh is empty
    so the chain helper can still print a partial size chain rather
    than skipping the whole field."""
    if not wh:
        return "?"
    w, h = wh
    try:
        w = int(w)
        h = int(h)
    except (TypeError, ValueError):
        return "?"
    if dpi is None:
        return f"{w}×{h}"
    try:
        d = float(dpi)
    except (TypeError, ValueError):
        return f"{w}×{h}"
    if d <= 0:
        return f"{w}×{h}"
    return f"{w}×{h}@{d:g}"


def fmt_size_chain(input_wh_dpi: tuple[tuple[int, int], float] | None,
                    working_wh_dpi: tuple[tuple[int, int], float] | None,
                    output_wh_dpi: tuple[tuple[int, int], float] | None,
                    ) -> str:
    """Render the buffer-shape transition through one processor as
    ``(W×H@DPI — W×H@DPI -> W×H@DPI)``.

    Skips redundant middle / output tokens when the processor doesn't
    resize internally (i.e. input == working == output). That keeps
    the common case quiet — Binarizer / SkewFinder / MarginSetter only
    print ``(W×H@DPI)``. Resizers (PageDetector / PageDewarper /
    Replay) print the full triple."""

    def _norm(t):
        if t is None:
            return None
        wh, dpi = t
        if wh is None:
            return None
        return (tuple(wh), dpi)

    inp = _norm(input_wh_dpi)
    work = _norm(working_wh_dpi)
    out = _norm(output_wh_dpi)

    if inp is None:
        return ""
    inp_s = fmt_size(inp[0], inp[1])

    # No-resize fast path — middle and output identical to input.
    same_work = work is None or work == inp
    same_out = out is None or out == inp
    if same_work and same_out:
        return f"({inp_s})"

    work_s = fmt_size(work[0], work[1]) if work else inp_s
    out_s = fmt_size(out[0], out[1]) if out else inp_s
    return f"({inp_s}—{work_s}->{out_s})"


def fmt_ms(ms: float) -> str:
    """Compact elapsed-time format: 43ms / 1.28s / 2.1m."""
    try:
        v = float(ms)
    except (TypeError, ValueError):
        return "?"
    if v < 1000:
        return f"{v:.0f}ms"
    if v < 60_000:
        return f"{v / 1000:.2f}s"
    return f"{v / 60_000:.2f}m"


def fmt_bytes(n: int) -> str:
    n = int(n or 0)
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    if n < 1024 ** 3:
        return f"{n / 1024 ** 2:.1f} MB"
    return f"{n / 1024 ** 3:.2f} GB"


def fmt_list(values: Iterable[Any], *, sep: str = ",", max_items: int = 6) -> str:
    """Render an iterable as comma-joined text. Long lists truncated
    with ``+N more`` so log lines don't explode."""
    items = [str(v) for v in values]
    if len(items) <= max_items:
        return sep.join(items)
    return sep.join(items[:max_items]) + f"+{len(items) - max_items} more"


def fmt_counts(d: dict[str, int]) -> str:
    """Render ``{"Text":14,"Title":1}`` style maps as ``Text:14,Title:1``."""
    if not d:
        return "—"
    parts = [f"{k}:{v}" for k, v in sorted(d.items(),
                                            key=lambda kv: -int(kv[1]))]
    return "{" + ", ".join(parts) + "}"


def fmt_pct_dist(p0: float, p10: float, p50: float, p90: float, p100: float,
                 *, unit: str = "") -> str:
    """Compact 5-stop percentile distribution."""
    s = f"[p0={p0:.0f} p10={p10:.0f} p50={p50:.0f} p90={p90:.0f} p100={p100:.0f}"
    if unit:
        s += f" {unit}"
    return s + "]"


def _key_fmt(k: str, v: Any, *, color: bool) -> str:
    """Render one key=value pair. Dict / list shortcuts."""
    if isinstance(v, dict):
        body = fmt_counts(v)
    elif isinstance(v, (list, tuple, set)):
        body = fmt_list(v)
    elif isinstance(v, bool):
        body = "yes" if v else "no"
    elif isinstance(v, float):
        body = f"{v:g}"
    else:
        body = str(v)
    if not color:
        return f"{k}={body}"
    return f"[{_C_KEY}]{k}=[/]{body}"


def format_op(name: str, elapsed_ms: Optional[float] = None,
              *, color: bool = True,
              size_chain: Optional[str] = None,
              **fields: Any) -> str:
    """Build a single op-log line.

    ``name`` is the dotted subsystem tag (``pipeline.SkewFinder`` /
    ``ocr.Surya`` / ``export.PDF``). ``elapsed_ms`` is rendered second
    (after the reserved scope keys + ``size_chain``) so the eye lands
    on the timing column at a fixed offset.

    Returns text with rich-style markup when ``color=True``; otherwise
    plain text. Pass through ``strip_markup`` for Qt sinks.
    """
    raw_tag = f"[{name}]".ljust(_NAME_WIDTH)
    if color:
        # Literal '[' / ']' inside rich markup needs to be escaped with a
        # leading backslash so rich's parser doesn't read it as a style
        # name lookup (which then drops the inner text).
        # Only the opening `[` needs an escape so rich's parser doesn't
        # try to look up `pipeline.SkewFinder` as a style name.
        tag_text = f"\\[{name}]"
        visible_len = len(name) + 2
        padding = " " * max(0, _NAME_WIDTH - visible_len)
        tag = f"[{_tag_color(name)}]{tag_text}[/]{padding}"
    else:
        tag = raw_tag

    parts: list[str] = [tag]
    # Reserved scope keys, fixed order.
    for k in ("scan", "layout", "batch"):
        if k in fields:
            v = fields.pop(k)
            if color:
                parts.append(f"[{_C_KEY}]{k}=[/][{_C_SCOPE}]{v}[/]")
            else:
                parts.append(f"{k}={v}")
    if size_chain:
        if color:
            parts.append(f"[{_C_SIZE}]{size_chain}[/]")
        else:
            parts.append(size_chain)
    if elapsed_ms is not None:
        ms_text = fmt_ms(elapsed_ms)
        if color:
            parts.append(f"[{_ms_color(elapsed_ms)}]{ms_text}[/]")
        else:
            parts.append(ms_text)
    for k, v in fields.items():
        parts.append(_key_fmt(k, v, color=color))
    return "  ".join(parts)


def emit(log_queue, level: str, name: str,
         elapsed_ms: Optional[float] = None, **fields: Any) -> None:
    """Push the formatted op line through the chain's log_queue. Safe
    to call with ``log_queue=None`` (no-op) so plain CLI / tests can
    use the same helper without rigging a multiprocessing queue."""
    line = format_op(name, elapsed_ms=elapsed_ms, **fields)
    if log_queue is None:
        return
    try:
        log_queue.put((f"log_{level}", line))
    except Exception:
        pass
