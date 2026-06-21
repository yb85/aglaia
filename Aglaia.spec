# PyInstaller spec for Aglaïa.app (macOS).
#
# Build:
#   # 1. Populate vendor/llama-server/<plat>/ with the binary that
#   #    surya's llama.cpp backend will invoke. (Hosted by ggml-org/llama.cpp
#   #    on GitHub Releases.)
#   uv run python scripts/fetch_llama_server.py
#
#   # 2. Sync deps for the target platform.
#   uv sync --extra macos --extra dev --extra package --extra jbig2
#
#   # 3. Build the bundle.
#   uv run pyinstaller Aglaia.spec --clean --noconfirm
#
# Output: dist/Aglaïa.app
#
# Entry: `aglaia.py` — the Qt scan GUI. `pdf2scans.py` shares the same
# lib code and can be invoked via the bundled Python from Terminal.

from PyInstaller.utils.hooks import (
    collect_data_files, collect_dynamic_libs, collect_submodules,
)
from pathlib import Path
import os
import platform as _plat
import sys as _sys

REPO = Path(SPECPATH).resolve()
ICON_ICNS = str(REPO / "lib" / "app_data" / "Aglaia.icns")

# ── code signing ─────────────────────────────────────────────────────
#
# Developer ID Application identity used to sign the bundled exe + every
# .dylib/.so PyInstaller drops. The identity NEVER lives in version
# control — it's loaded from the gitignored ``.env`` at build time. See
# ``scripts/sign_and_notarize.sh`` for the post-build pass that re-signs
# the .app and runs notarytool, and ``.env.example`` for the shape.
#
# When ``AGLAIA_SIGN_IDENTITY`` is unset (no .env, CI without secrets,
# fork build) we fall back to an unsigned build — the .app still runs
# locally via the .command launcher but Gatekeeper will reject it.

def _load_dotenv():
    """Tiny env loader — avoids adding `python-dotenv` to the build deps."""
    env_path = REPO / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


_load_dotenv()
CODESIGN_IDENTITY = os.environ.get("AGLAIA_SIGN_IDENTITY") or None

# ── version ───────────────────────────────────────────────────────────
# Single source of truth at build time is the ``AGLAIA_VERSION`` env var,
# which the release CI sets from the git tag (``v1.2.3`` → ``1.2.3``).
# Local/dev builds with the var unset fall back to ``0.0.0-dev`` so a
# stray hardcoded number never ships as a "release".
_VERSION = (os.environ.get("AGLAIA_VERSION") or "").lstrip("v").strip() or "0.0.0-dev"
ENTITLEMENTS_FILE = str(REPO / "packaging" / "entitlements.plist")
if CODESIGN_IDENTITY is None:
    ENTITLEMENTS_FILE = None   # PyInstaller errors if entitlements set without identity


# ── platform-specific llama-server binary ────────────────────────────
#
# Surya 2.x runs through `llama-server` (llama.cpp HTTP backend). To
# avoid making users `brew install llama.cpp` after grabbing the .app
# we ship the binary right next to the Aglaïa executable. Drop the
# matching release into ``vendor/llama-server/<plat>/`` before running
# pyinstaller; ``scripts/fetch_llama_server.py`` automates this for
# any of the four supported (os, arch) combinations.
#
# Path resolved at runtime by ``lib/workers/ocr/surya.py``:
#   * frozen → ``sys._MEIPASS / "llama-server[.exe]"``
#   * dev    → ``vendor/llama-server/<plat>/llama-server[.exe]``
def _llama_server_subdir() -> str | None:
    sys_name = _plat.system().lower()         # "darwin" | "linux" | "windows"
    machine = _plat.machine().lower()         # "arm64" | "x86_64" | "amd64" | …
    if sys_name == "darwin":
        return "macos-arm64" if machine in ("arm64", "aarch64") else "macos-x64"
    if sys_name == "linux":
        return "linux-arm64" if machine in ("arm64", "aarch64") else "linux-x64"
    if sys_name == "windows":
        return "windows-arm64" if machine in ("arm64", "aarch64") else "windows-x64"
    return None


_llama_subdir = _llama_server_subdir()
_llama_binaries: list[tuple[str, str]] = []
if _llama_subdir is not None:
    _src_dir = REPO / "vendor" / "llama-server" / _llama_subdir
    if _src_dir.is_dir():
        # Include the binary + every shared library next to it (llama.cpp
        # ships several .dylib / .so / .dll files alongside the server).
        for entry in sorted(_src_dir.iterdir()):
            if not entry.is_file():
                continue
            # Place every file at the bundle root so the server picks up
            # its libs via the default loader search path. The relative
            # destination "." maps to MEIPASS on every platform.
            _llama_binaries.append((str(entry), "."))

# ── data files shipped inside the .app ───────────────────────────────
datas = [
    (str(REPO / "config"), "config"),
    (str(REPO / "lib" / "gui" / "icons"), "lib/gui/icons"),
    (str(REPO / "lib" / "app_data" / "model-list.json"),
     "lib/app_data"),
    (str(REPO / "lib" / "app_data" / "aglaia2-1024.png"),
     "lib/app_data"),
    # Theme-aware wordmarks for the About dialog title.
    (str(REPO / "lib" / "app_data" / "aglaia-light.png"),
     "lib/app_data"),
    (str(REPO / "lib" / "app_data" / "aglaia-dark.png"),
     "lib/app_data"),
    (str(REPO / "lib" / "app_data" / "Aglaia.icns"),
     "lib/app_data"),
    (str(REPO / "lib" / "app_data" / "AglaiaDoc.icns"),
     "lib/app_data"),
]

# Surya/transformers/huggingface_hub ship YAML configs + tokenizer
# resources that PyInstaller's static analysis misses. Pull them all in.
for pkg in ("surya", "transformers", "tokenizers", "huggingface_hub",
            "doxapy", "page_dewarp"):
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass

# ── hidden imports ───────────────────────────────────────────────────
# Plugin processors are imported via the registry lookup at runtime,
# so PyInstaller can't see them from the static graph.
hiddenimports = (
    collect_submodules("lib.processors")
    + collect_submodules("lib.workers")
    + ["pyobjc", "Vision", "Speech", "AVFoundation"]
    # JBIG2 encoder (maturin/PyO3). Editable installs leave the compiled
    # `_native` .so in the crate source dir, so name it explicitly or the
    # frozen app silently falls back to G4. Built when the build env was
    # synced with `--extra jbig2`.
    + ["aglaia_jbig2", "aglaia_jbig2._native"]
)

# Pull the compiled JBIG2 extension (.so) into the bundle.
_jbig2_binaries = collect_dynamic_libs("aglaia_jbig2")

block_cipher = None

a = Analysis(
    ["aglaia.py"],
    pathex=[str(REPO)],
    binaries=list(_llama_binaries) + list(_jbig2_binaries),
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # Optional Qt modules Aglaïa never touches — strip to shrink
        # the bundle by ~150 MB. Add back here if a feature breaks.
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtWebEngineQuick",
        "PySide6.QtWebChannel",
        "PySide6.QtWebSockets",
        "PySide6.Qt3DCore",
        "PySide6.Qt3DRender",
        "PySide6.Qt3DInput",
        "PySide6.Qt3DLogic",
        "PySide6.Qt3DAnimation",
        "PySide6.Qt3DExtras",
        "PySide6.QtCharts",
        "PySide6.QtDataVisualization",
        "PySide6.QtMultimedia",
        "PySide6.QtMultimediaWidgets",
        "PySide6.QtQuick",
        "PySide6.QtQuick3D",
        "PySide6.QtQuickWidgets",
        "PySide6.QtQml",
        "PySide6.QtPositioning",
        "PySide6.QtSensors",
        "PySide6.QtNfc",
        "PySide6.QtBluetooth",
        "PySide6.QtScxml",
        "PySide6.QtTextToSpeech",
        # Heavy dev-only deps we don't need at runtime.
        "matplotlib",
        "pytest",
        # Web stack stripped — Aglaïa is Qt-only now.
        "fastapi",
        "uvicorn",
        "jinja2",
        "sse_starlette",
        "starlette",
    ],
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Aglaia",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=CODESIGN_IDENTITY,
    entitlements_file=ENTITLEMENTS_FILE,
    icon=ICON_ICNS,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Aglaia",
)

app = BUNDLE(
    coll,
    name="Aglaia.app",
    icon=ICON_ICNS,
    bundle_identifier="cc.bibli.aglaia",
    info_plist={
        "CFBundleName": "Aglaïa",
        "CFBundleDisplayName": "Aglaïa",
        "CFBundleShortVersionString": _VERSION,
        "CFBundleVersion": _VERSION,
        "NSHighResolutionCapable": True,
        # Permissions Aglaïa needs at runtime.
        "NSCameraUsageDescription":
            "Aglaïa uses the camera to capture page scans.",
        "NSSpeechRecognitionUsageDescription":
            "Aglaïa uses speech recognition for voice control.",
        "NSMicrophoneUsageDescription":
            "Aglaïa uses the microphone for voice control.",
        # `.agl` document type binding — same UTI as
        # lib/app_data/filetype_register.plist_snippet().
        "CFBundleDocumentTypes": [{
            "CFBundleTypeName": "Aglaïa project",
            "CFBundleTypeRole": "Editor",
            "LSHandlerRank": "Owner",
            "LSItemContentTypes": ["cc.bibli.aglaia.project"],
            # macOS reads the file icon from this entry; the icon file
            # itself must live in Contents/Resources, which is where
            # PyInstaller drops anything declared in `datas`.
            "CFBundleTypeIconFile": "AglaiaDoc.icns",
        }],
        "UTExportedTypeDeclarations": [{
            "UTTypeIdentifier": "cc.bibli.aglaia.project",
            "UTTypeDescription": "Aglaïa project",
            "UTTypeConformsTo": ["public.database", "public.data"],
            "UTTypeTagSpecification": {
                "public.filename-extension": ["agl"],
                "public.mime-type": ["application/x-aglaia-project"],
            },
        }],
    },
)
