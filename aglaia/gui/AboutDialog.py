# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""About box for the Qt GUI.

A single modeless dialog that renders a generated HTML "about" page —
app name, version, runtime stack, links, license. Reachable from both
the Help menu and the Settings tab, so the HTML lives in one builder
(`build_about_html`) and both entry points call `show_about`.
"""

from __future__ import annotations

import platform
from html import escape

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QTextBrowser, QVBoxLayout, QWidget,
)

# Canonical project links — kept in step with StartupWindow's constants.
HOMEPAGE_URL = "https://aglaia.bibli.cc"
DOCS_URL = "https://aglaia.bibli.cc/docs"
# Public-facing repo URL — the project is renamed `aglaia` when it leaves
# the dev phase and goes public (currently the private `aglaia` repo).
GIT_REPO = "https://github.com/yb85/aglaia"
SUPPORT_URL = "https://ko-fi.com/yb_85"
LICENSE_URL = "https://polyformproject.org/licenses/shield/1.0.0"


def app_version() -> str:
    """App version via the canonical resolver (frozen + from source)."""
    from aglaia.version import get_version
    return get_version()


def _logo_data_uri(target_h: int = 44) -> str | None:
    """Theme-appropriate Aglaïa wordmark as an `<img>` tag (base64 data
    URI — survives the frozen bundle without path resolution).

    Light theme → black wordmark (`aglaia-light.png`); dark theme → pale
    wordmark (`aglaia-dark.png`). Returns None if the asset is missing,
    so the caller falls back to a text `<h1>`."""
    try:
        import base64
        from PySide6.QtGui import QImage, QPalette
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        dark = False
        if app is not None:
            win = app.palette().color(QPalette.ColorRole.Window)
            dark = win.lightness() < 128
        fname = "aglaia-dark.png" if dark else "aglaia-light.png"
        from aglaia.assets import asset_path
        path = asset_path("brand", fname)
        if not path.is_file():
            return None
        raw = path.read_bytes()
        img = QImage()
        img.loadFromData(raw)
        w, h = img.width(), img.height()
        disp_w = int(round(w * target_h / h)) if h else target_h * 2
        b64 = base64.b64encode(raw).decode("ascii")
        return (f'<img src="data:image/png;base64,{b64}" '
                f'width="{disp_w}" height="{target_h}" alt="Aglaïa">')
    except Exception:
        return None


def build_about_html() -> str:
    """Generate the About page as a standalone HTML fragment. Styling is
    intentionally palette-neutral (no body background) so it reads on
    both the light and dark themes; only the accent link colour is set."""
    from aglaia.app_data import APP_NAME, APP_AUTHOR

    try:
        from PySide6 import __version__ as pyside_ver
        from PySide6.QtCore import qVersion
        qt_ver = qVersion()
    except Exception:
        pyside_ver = qt_ver = "?"

    name = escape(str(APP_NAME))
    author = escape(str(APP_AUTHOR))
    ver = escape(app_version())
    py = escape(platform.python_version())
    plat = escape(f"{platform.system()} {platform.release()} ({platform.machine()})")

    stack = [
        "PySide6 — GUI",
        "OpenCV · NumPy · SciPy · Pillow — image processing",
        "page-dewarp + JAX / MLX — cubic-sheet dewarp",
        "doxapy — binarization (Wolf / Sauvola)",
        "pikepdf · pypdfium2 — PDF I/O",
        "aglaia_jbig2 — JBIG2 compression",
        "Apple Vision · Surya · PaddleOCR · Mistral — OCR / layout",
        "Vosk — offline voice control",
    ]
    stack_items = "\n".join(f"    <li>{escape(s)}</li>" for s in stack)

    # Theme-aware wordmark in place of the text title; text fallback if
    # the logo asset is unavailable.
    logo_or_title = _logo_data_uri() or f"<h1>{name}</h1>"

    return f"""\
<html><head><style>
  a {{ color: #5b9bd5; text-decoration: none; }}
  h1 {{ margin: 0 0 2px 0; font-size: 22px; }}
  h2 {{ font-size: 13px; margin: 16px 0 4px 0; text-transform: uppercase;
        letter-spacing: 0.5px; opacity: 0.6; }}
  p, li {{ font-size: 13px; line-height: 1.5; }}
  ul {{ margin: 2px 0 0 0; padding-left: 18px; }}
  .sub {{ opacity: 0.7; font-size: 12px; }}
  .kv {{ font-size: 12px; opacity: 0.8; }}
</style></head><body>
  {logo_or_title}
  <p class="sub">Scanner &amp; page-extraction pipeline · v{ver}</p>

  <p>Webcam capture, image-processing chain, layout detection and OCR for
     turning book pages into clean, searchable scans.</p>

  <h2>Links</h2>
  <p>
    <a href="{HOMEPAGE_URL}">Homepage</a> ·
    <a href="{DOCS_URL}">Documentation</a> ·
    <a href="{GIT_REPO}">Source</a> ·
    <a href="{SUPPORT_URL}">Support the developer</a>
  </p>

  <h2>Built with</h2>
  <ul>
{stack_items}
  </ul>

  <h2>Runtime</h2>
  <p class="kv">
    Python {py}<br>
    Qt {escape(qt_ver)} · PySide6 {escape(str(pyside_ver))}<br>
    {plat}
  </p>

  <h2>License</h2>
  <p class="kv">© {author} · <a href="{LICENSE_URL}">PolyForm Shield 1.0.0</a></p>
</body></html>
"""


class AboutDialog(QDialog):
    """Modeless About window rendering `build_about_html` in a read-only
    browser with external links opened in the system browser."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from aglaia.app_data import APP_NAME
        self.setWindowTitle(self.tr("About {name}").format(name=APP_NAME))
        self.setMinimumSize(440, 520)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 8)

        browser = QTextBrowser(self)
        browser.setOpenExternalLinks(False)
        browser.setOpenLinks(False)
        browser.anchorClicked.connect(
            lambda url: QDesktopServices.openUrl(QUrl(url)))
        browser.setHtml(build_about_html())
        v.addWidget(browser, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        buttons.accepted.connect(self.close)
        v.addWidget(buttons)


def show_about(parent: QWidget | None = None) -> AboutDialog:
    """Open (or re-raise) the singleton About dialog on `parent`."""
    existing = getattr(parent, "_about_dialog", None) if parent else None
    if existing is not None and existing.isVisible():
        existing.raise_()
        existing.activateWindow()
        return existing
    dlg = AboutDialog(parent)
    if parent is not None:
        setattr(parent, "_about_dialog", dlg)
        dlg.finished.connect(lambda *_: setattr(parent, "_about_dialog", None))
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    return dlg
