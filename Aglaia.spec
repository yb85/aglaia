# PyInstaller spec for Aglaïa — cross-platform (macOS .app / Windows onedir).
#
# Build:
#   # 1. Populate vendor/llama-server/<plat>/ with the binary that
#   #    surya's llama.cpp backend will invoke. (Hosted by ggml-org/llama.cpp
#   #    on GitHub Releases.)
#   uv run python scripts/fetch_llama_server.py
#
#   # 2. Sync deps for the target platform.
#   uv sync --extra macos --extra dev --extra package --extra jbig2   # macOS
#   uv sync --extra gui --extra voice --extra dev --extra package      # Windows
#
#   # 3. Build the bundle.
#   uv run pyinstaller Aglaia.spec --clean --noconfirm
#
# Output: macOS → dist/Aglaia.app   |   Windows → dist/Aglaia/Aglaia.exe
#
# Entry: `aglaia/__main__.py` — the Qt scan GUI (`aglaia <dir>` console script).
# The same module also drives the `--headless` CLI batch runner.

from PyInstaller.utils.hooks import (
    collect_all, collect_data_files, collect_dynamic_libs, collect_submodules,
)
from pathlib import Path
import os
import platform as _plat
import sys as _sys

REPO = Path(SPECPATH).resolve()
IS_MAC = _plat.system() == "Darwin"
IS_WIN = _plat.system() == "Windows"
ICON_ICNS = str(REPO / "aglaia" / "assets" / "app" / "Aglaia.icns")
ICON_ICO = str(REPO / "aglaia" / "assets" / "app" / "Aglaia.ico")
# PyInstaller wants the platform-native icon format (.icns on macOS, .ico on
# Windows); passing a mismatched format errors at build. None → default
# bootloader icon (Linux / missing file).
if IS_MAC:
    APP_ICON = ICON_ICNS
elif IS_WIN and Path(ICON_ICO).is_file():
    APP_ICON = ICON_ICO
else:
    APP_ICON = None

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
# Bake the version into the package so the FROZEN app can read it at runtime
# (the AGLAIA_VERSION env var is set at build time, not runtime). aglaia.version
# imports aglaia._version; writing it here, before Analysis, gets it bundled.
# Gitignored — a build artefact, never committed.
(REPO / "aglaia" / "_version.py").write_text(
    f'__version__ = "{_VERSION}"\n', encoding="utf-8")
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
# Path resolved at runtime by ``aglaia/workers/ocr/surya.py``:
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
# Static assets now live in the top-level `assets/` tree and are resolved at
# runtime via `aglaia.assets.asset_path` (-> <MEIPASS>/assets in the bundle), so
# they must ship under "assets/…". We ship only the subdirs the app loads at
# runtime — NOT the large site-only brand backgrounds (assets/brand/aglaia_bg*,
# aglaia_usage) — to keep the bundle lean.
# Assets + read-only config live INSIDE the aglaia package now and are resolved
# package-relative (aglaia.assets.asset_path / config_path -> aglaia/assets,
# aglaia/config). The bundle ships them at the SAME package-relative paths so
# `Path(__file__).parent / 'assets'` resolves under <MEIPASS>/aglaia. We ship
# only the runtime subdirs — NOT the large site-only brand backgrounds.
datas = [
    (str(REPO / "aglaia" / "config"), "aglaia/config"),
    # SQLite schema migrations (aglaia/storage/db.py globs these at every DB
    # open). Without them the frozen app finds zero .sql files, thinks the DB
    # is fully migrated, and creates NO tables → first query stalls.
    (str(REPO / "aglaia" / "storage" / "schema"), "aglaia/storage/schema"),
    (str(REPO / "aglaia" / "assets" / "icons"), "aglaia/assets/icons"),
    (str(REPO / "aglaia" / "assets" / "modes"), "aglaia/assets/modes"),
    # Theme-aware wordmarks (About dialog / startup) + the 1024 logo.
    (str(REPO / "aglaia" / "assets" / "brand" / "aglaia-light.png"), "aglaia/assets/brand"),
    (str(REPO / "aglaia" / "assets" / "brand" / "aglaia-dark.png"), "aglaia/assets/brand"),
    (str(REPO / "aglaia" / "assets" / "brand" / "aglaia2-1024.png"), "aglaia/assets/brand"),
    (str(REPO / "aglaia" / "app_data" / "model-list.json"), "aglaia/app_data"),
    # App + document icons. Kept at the "aglaia/app_data" bundle path the macOS
    # plist (CFBundleIconFile / CFBundleTypeIconFile) + filetype_register
    # reference by name — only the SOURCE path changed.
    (str(REPO / "aglaia" / "assets" / "app" / "Aglaia.icns"), "aglaia/app_data"),
    (str(REPO / "aglaia" / "assets" / "app" / "AglaiaDoc.icns"), "aglaia/app_data"),
    # CFBundleTypeIconFile resolves the document icon relative to
    # Contents/Resources/ ROOT — so the .agl icon must sit there, not only in
    # the aglaia/app_data subdir (where filetype_register reads it). Drop a
    # copy at the root for the Finder file icon.
    (str(REPO / "aglaia" / "assets" / "app" / "AglaiaDoc.icns"), "."),
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
    collect_submodules("aglaia.processors")
    + collect_submodules("aglaia.workers")
    # Apple frameworks only exist on macOS; listing them as hidden imports on
    # Windows/Linux makes PyInstaller emit noisy "module not found" warnings.
    + (["pyobjc", "Vision", "Speech", "AVFoundation"] if IS_MAC else [])
    # JBIG2 encoder (maturin/PyO3). Editable installs leave the compiled
    # `_native` .so in the crate source dir, so name it explicitly or the
    # frozen app silently falls back to G4. Built when the build env was
    # synced with `--extra jbig2`.
    + ["aglaia_jbig2", "aglaia_jbig2._native"]
    # Baked version module (written above). aglaia.version imports it inside a
    # try/except, which static analysis can miss — name it so it's bundled.
    + ["aglaia._version"]
)

# Pull the compiled JBIG2 extension (.so) into the bundle.
_jbig2_binaries = collect_dynamic_libs("aglaia_jbig2")

# mlx (Metal-accelerated dewarp) and vosk (offline voice) ship native libs
# plus data PyInstaller's static analysis misses — notably mlx's `.metallib`
# shader library, without which `import mlx.core` fails at Metal init and the
# app reports the backend unavailable. vosk's native `libvosk` is likewise not
# picked up. collect_all gathers datas + binaries + submodules for each.
_native_pkg_binaries = []
for _pkg in ("mlx", "vosk"):
    try:
        _d, _b, _h = collect_all(_pkg)
        datas += _d
        _native_pkg_binaries += _b
        hiddenimports += _h
    except Exception:
        pass

# ── slim CUDA bundle for GPU page-dewarp (Linux only) ────────────────
#
# Ship a GPU-capable AppImage by bundling jax[cuda12]'s plugin + the CUDA
# runtime libs the dewarp actually touches. The dewarp is L-BFGS-B over the
# reprojection cost — matmul / elementwise / reductions, NO convolution, FFT,
# sparse, or multi-GPU collectives — so most of the ~3.9 GB CUDA payload is
# dead weight. Bundling ONLY the loaded libs keeps the AppImage under GitHub's
# 2 GiB release-asset cap. Verified on an RTX 3090: GPU dewarp runs correctly
# with the excluded libs absent (PageDewarper sets JAX_SKIP_CUDA_CONSTRAINTS_CHECK
# so JAX's init-time version probe doesn't hard-fail on the missing libs and
# silently drop to CPU).
#
# Keep:    cublas (matmul), cuda_nvrtc + nvjitlink + cuda_nvcc/ptxas (XLA JIT),
#          cuda_cupti, cuda_runtime.
# Exclude: cudnn, nccl, nvshmem, cufft, cusparse, cusolver (~2.6 GB, unused).
#
# Only collected when the `cuda` extra is installed (jax_cuda12_plugin present);
# the macOS / Windows / CPU-Linux builds skip this block entirely.
_cuda_binaries = []
IS_LINUX = _plat.system() == "Linux"
if IS_LINUX:
    try:
        import jax_cuda12_plugin  # noqa: F401  — only with `uv sync --extra cuda`
        _HAS_CUDA = True
    except ImportError:
        _HAS_CUDA = False
    if _HAS_CUDA:
        from PyInstaller.utils.hooks import copy_metadata
        # PJRT plugin (xla_cuda_plugin.so, resolved relative to the package
        # __file__) + JAX's CUDA kernel extensions (cuda_plugin_extension,
        # _versions, …). collect_all preserves the package layout both rely on.
        for _pkg in ("jax_plugins.xla_cuda12", "jax_cuda12_plugin"):
            try:
                _d, _b, _h = collect_all(_pkg)
                datas += _d
                _cuda_binaries += _b
                hiddenimports += _h
            except Exception:
                pass
        # JAX discovers the GPU plugin through the `jax_plugins` entry point,
        # provided by the jax-cuda12-pjrt dist. PyInstaller drops dist metadata
        # by default → without this the frozen app never finds the plugin and
        # silently runs on CPU.
        try:
            datas += copy_metadata("jax-cuda12-pjrt")
        except Exception:
            pass
        # The CUDA runtime libs the dewarp loads. collect_all keeps each at
        # nvidia/<pkg>/lib/lib*.so — exactly where jax_plugins.xla_cuda12._load
        # looks (importlib.import_module("nvidia.<pkg>").__path__[0] / "lib").
        # The big unused libs (cudnn/nccl/nvshmem/cufft/cusparse/cusolver) are
        # simply not named here, so they never enter the bundle.
        for _pkg in ("nvidia.cublas", "nvidia.cuda_nvrtc", "nvidia.cuda_runtime",
                     "nvidia.cuda_cupti", "nvidia.cuda_nvcc", "nvidia.nvjitlink"):
            try:
                _d, _b, _h = collect_all(_pkg)
                datas += _d
                _cuda_binaries += _b
                hiddenimports += _h
            except Exception:
                pass

block_cipher = None

a = Analysis(
    ["aglaia/__main__.py"],
    pathex=[str(REPO)],
    binaries=list(_llama_binaries) + list(_jbig2_binaries) + list(_native_pkg_binaries) + list(_cuda_binaries),
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
    # codesign_identity / entitlements_file are macOS-only knobs. On Windows
    # signing happens post-build via signtool (see release.yml), so leave None.
    codesign_identity=CODESIGN_IDENTITY if IS_MAC else None,
    entitlements_file=ENTITLEMENTS_FILE if IS_MAC else None,
    icon=APP_ICON,
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

# BUNDLE wraps COLLECT into a macOS .app — a Darwin-only concept. On Windows
# the deliverable IS the COLLECT onedir (dist/Aglaia/Aglaia.exe), so skip BUNDLE
# there. `IS_MAC and BUNDLE(...)` short-circuits the call away off-macOS.
app = IS_MAC and BUNDLE(
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
        # aglaia/app_data/filetype_register.plist_snippet().
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
