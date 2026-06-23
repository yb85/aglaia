# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Register the `.agl` Aglaïa project file type with the host OS.

Platform notes
--------------

* **macOS**: file-type ↔ app binding lives in an .app bundle's
  `Info.plist` (`CFBundleDocumentTypes` + `UTExportedTypeDeclarations`).
  Registration probes for a real Aglaïa.app bundle in the running
  process (when frozen), then `/Applications/Aglaia.app`,
  `~/Applications/Aglaia.app`, and the dev `dist/Aglaia.app`.
  Falls back to `/Applications/Aglaia.app` as the canonical target
  when nothing is found — the user is expected to install the
  PyInstaller-built bundle there.

* **Linux**: write a freedesktop MIME XML under
  `~/.local/share/mime/packages/aglaia.xml` and a `.desktop` launcher
  under `~/.local/share/applications/aglaia.desktop`. Both are picked
  up after `update-mime-database` / `update-desktop-database`.

* **Windows**: write per-user registry entries under
  `HKCU\\Software\\Classes\\.agl` and `HKCU\\Software\\Classes\\bibli.cc.AglaiaProject`.

Public API:

    register_filetype()    — call the right platform branch.
    unregister_filetype()  — revert what `register_filetype` did.
    plist_snippet()        — Info.plist fragment for packagers.

MIME / UTI strings are centralised so callers don't drift:
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Single source of truth for the cross-platform identifiers. Bump the
# UTI revision (e.g. .v2) only if the schema changes incompatibly.
APP_NAME = "Aglaïa"
APP_BUNDLE_ID = "cc.bibli.aglaia"
PROJECT_UTI = "cc.bibli.aglaia.project"
PROJECT_MIME = "application/x-aglaia-project"
PROJECT_EXT_NODOT = "agl"  # do not include the dot — many APIs reject it
PROJECT_EXT = ".agl"


# ── public dispatch ────────────────────────────────────────────────

def filetype_registration_available() -> bool:
    """True when there's a real target to bind ``.agl`` to. On macOS that
    means an actual Aglaïa.app on disk (frozen process or an install) — a
    bare CLI / `python -m aglaia` / source run has none, so binding would
    point at the Python process. Linux registers a `.desktop` and works from
    any launch. The Settings button is disabled when this is False."""
    if sys.platform == "darwin":
        # Only when running AS the bundled .app (sys.frozen). A CLI /
        # `python -m aglaia` / source run shouldn't touch the system binding
        # (it would point .agl at the Python process), even if an .app
        # happens to exist on disk.
        if not getattr(sys, "frozen", False):
            return False
        app = _find_installed_app()
        return app is not None and app.is_dir()
    if sys.platform.startswith("linux"):
        return True
    return False


def register_filetype(*, app_path: Path | None = None,
                      icon_path: Path | None = None) -> tuple[bool, str]:
    """Register `.agl` with the host OS. Returns (ok, message)."""
    system = platform.system()
    try:
        if system == "Darwin":
            return _register_macos(app_path=app_path, icon_path=icon_path)
        if system == "Linux":
            return _register_linux(app_path=app_path, icon_path=icon_path)
        if system == "Windows":
            return _register_windows(app_path=app_path, icon_path=icon_path)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    return False, f"Unsupported platform: {system}"


def unregister_filetype() -> tuple[bool, str]:
    system = platform.system()
    try:
        if system == "Darwin":
            return _unregister_macos()
        if system == "Linux":
            return _unregister_linux()
        if system == "Windows":
            return _unregister_windows()
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    return False, f"Unsupported platform: {system}"


# ── macOS ─────────────────────────────────────────────────────────

def _find_installed_app() -> Path | None:
    """Best-effort lookup of the real Aglaïa.app on disk.

    Probe order:
      1. The current process, when running from inside a PyInstaller
         bundle (`sys.frozen` + `sys.executable` walks up to the `.app`).
      2. `/Applications/Aglaia.app` (standard install location).
      3. `~/Applications/Aglaia.app` (per-user install).
      4. `<repo-root>/dist/Aglaia.app` (developer build output).

    Returns the first path that has a `Contents/Info.plist`, or None.
    """
    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        try:
            for parent in Path(sys.executable).resolve().parents:
                if parent.suffix == ".app":
                    candidates.append(parent)
                    break
        except Exception:
            pass
    candidates += [
        Path("/Applications/Aglaia.app"),
        Path.home() / "Applications" / "Aglaia.app",
    ]
    try:
        repo_root = Path(__file__).resolve().parents[2]
        candidates.append(repo_root / "dist" / "Aglaia.app")
    except Exception:
        pass
    for p in candidates:
        if p.is_dir() and (p / "Contents" / "Info.plist").is_file():
            return p
    return None


def _macos_default_app_path() -> Path:
    """Canonical fallback when no installed bundle is found —
    `/Applications/Aglaia.app`. Used so the LS database always has a
    consistent target; the user installs the bundle there after build."""
    return Path("/Applications/Aglaia.app")


def _register_macos(*, app_path: Path | None,
                    icon_path: Path | None) -> tuple[bool, str]:
    """Bind `.agl` files to the Aglaïa app via Launch Services.

    Path resolution: explicit `app_path` argument > running-bundle / install
    locations probed by `_find_installed_app()` > `/Applications/Aglaia.app`.
    No shim is built — the real PyInstaller bundle already ships an
    Info.plist with `CFBundleDocumentTypes` + `UTExportedTypeDeclarations`
    + `CFBundleTypeIconFile=AglaiaDoc.icns`. `lsregister -f` just
    refreshes LS's database with that plist.

    Returns (True, msg) when LS was nudged, (False, msg) when no
    on-disk bundle could be located (LS would refuse to bind to a
    non-existent path).
    """
    if app_path is None:
        app_path = _find_installed_app() or _macos_default_app_path()
    if not app_path.is_dir():
        return False, (f"No Aglaïa.app found at {app_path}. Build with "
                       "`pyinstaller Aglaia.spec` and move the result to "
                       "/Applications/ (or pass --app-path) before "
                       "registering.")

    lsregister = (
        "/System/Library/Frameworks/CoreServices.framework/Versions/A/"
        "Frameworks/LaunchServices.framework/Versions/A/Support/lsregister"
    )
    if not Path(lsregister).exists():
        return False, "lsregister not found on this system."
    subprocess.run([lsregister, "-f", str(app_path)],
                   check=False, capture_output=True)
    return True, f"Registered {APP_NAME} → {app_path}"


def _unregister_macos() -> tuple[bool, str]:
    """Tell Launch Services to forget about the bundle. Walks the same
    candidate paths `_find_installed_app` does so any previously-built
    location gets unbound, not just `/Applications/`. Does not delete
    the on-disk bundle — that's the user's call."""
    lsregister = (
        "/System/Library/Frameworks/CoreServices.framework/Versions/A/"
        "Frameworks/LaunchServices.framework/Versions/A/Support/lsregister"
    )
    if not Path(lsregister).exists():
        return False, "lsregister not found on this system."
    removed: list[str] = []
    for cand in [_find_installed_app(), _macos_default_app_path(),
                 Path.home() / "Applications" / "Aglaia.app"]:
        if cand is not None and cand.is_dir():
            subprocess.run([lsregister, "-u", str(cand)],
                           check=False, capture_output=True)
            removed.append(str(cand))
    if not removed:
        return True, "Nothing to unregister."
    return True, "Unregistered: " + ", ".join(removed)


def _macos_plist(has_icon: bool) -> str:
    icon_line = "    <key>CFBundleIconFile</key>\n    <string>Aglaia.icns</string>\n" if has_icon else ""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0">\n'
        '<dict>\n'
        '    <key>CFBundleIdentifier</key>\n'
        f'    <string>{APP_BUNDLE_ID}</string>\n'
        '    <key>CFBundleName</key>\n'
        f'    <string>{APP_NAME}</string>\n'
        '    <key>CFBundleDisplayName</key>\n'
        f'    <string>{APP_NAME}</string>\n'
        '    <key>CFBundleExecutable</key>\n'
        '    <string>Aglaia</string>\n'
        '    <key>CFBundlePackageType</key>\n'
        '    <string>APPL</string>\n'
        '    <key>CFBundleVersion</key>\n'
        '    <string>1.0</string>\n'
        '    <key>CFBundleShortVersionString</key>\n'
        '    <string>1.0</string>\n'
        f'{icon_line}'
        '    <key>NSHighResolutionCapable</key>\n'
        '    <true/>\n'
        '    <key>CFBundleDocumentTypes</key>\n'
        '    <array>\n'
        '        <dict>\n'
        '            <key>CFBundleTypeName</key>\n'
        f'            <string>{APP_NAME} project</string>\n'
        '            <key>CFBundleTypeRole</key>\n'
        '            <string>Editor</string>\n'
        '            <key>LSHandlerRank</key>\n'
        '            <string>Owner</string>\n'
        '            <key>LSItemContentTypes</key>\n'
        '            <array>\n'
        f'                <string>{PROJECT_UTI}</string>\n'
        '            </array>\n'
        # Document icon — must live next to Aglaia.icns in Contents/Resources.
        '            <key>CFBundleTypeIconFile</key>\n'
        '            <string>AglaiaDoc.icns</string>\n'
        '        </dict>\n'
        '    </array>\n'
        '    <key>UTExportedTypeDeclarations</key>\n'
        '    <array>\n'
        '        <dict>\n'
        '            <key>UTTypeIdentifier</key>\n'
        f'            <string>{PROJECT_UTI}</string>\n'
        '            <key>UTTypeDescription</key>\n'
        f'            <string>{APP_NAME} project</string>\n'
        '            <key>UTTypeConformsTo</key>\n'
        '            <array>\n'
        '                <string>public.database</string>\n'
        '                <string>public.data</string>\n'
        '            </array>\n'
        '            <key>UTTypeTagSpecification</key>\n'
        '            <dict>\n'
        '                <key>public.filename-extension</key>\n'
        '                <array>\n'
        f'                    <string>{PROJECT_EXT_NODOT}</string>\n'
        '                </array>\n'
        '                <key>public.mime-type</key>\n'
        '                <array>\n'
        f'                    <string>{PROJECT_MIME}</string>\n'
        '                </array>\n'
        '            </dict>\n'
        '        </dict>\n'
        '    </array>\n'
        '</dict>\n'
        '</plist>\n'
    )


def plist_snippet() -> str:
    """Standalone Info.plist fragment for packagers (PyInstaller's
    `info_plist=` kwarg, briefcase, etc.). Same UTI / MIME / extension
    as the shim app produced by `register_filetype()`."""
    return _macos_plist(has_icon=False)


def _shquote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


# ── Linux ─────────────────────────────────────────────────────────

_LINUX_DESKTOP = '''[Desktop Entry]
Type=Application
Version=1.0
Name=Aglaïa
GenericName=Scan project editor
Comment=Aglaïa scan / OCR project editor
Exec={exec_cmd} %f
MimeType={mime};
Categories=Graphics;Office;
Terminal=false
StartupNotify=true
'''

_LINUX_MIME_XML = '''<?xml version="1.0" encoding="UTF-8"?>
<mime-info xmlns="http://www.freedesktop.org/standards/shared-mime-info">
    <mime-type type="{mime}">
        <comment>Aglaïa scan project</comment>
        <glob pattern="*.{ext}"/>
        <sub-class-of type="application/x-sqlite3"/>
    </mime-type>
</mime-info>
'''


def _linux_paths() -> tuple[Path, Path]:
    base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share")
    return (base / "applications" / "aglaia.desktop",
            base / "mime/packages" / "aglaia.xml")


def _find_installed_app_linux() -> Path | None:
    """Probe for the Aglaïa launcher binary on Linux. Frozen process →
    `sys.executable`. Otherwise standard install locations + dev dist."""
    if getattr(sys, "frozen", False):
        try:
            return Path(sys.executable).resolve()
        except Exception:
            pass
    candidates = [
        Path("/usr/local/bin/Aglaia"),
        Path("/usr/bin/Aglaia"),
        Path("/opt/aglaia/Aglaia"),
        Path.home() / "bin" / "Aglaia",
    ]
    try:
        repo_root = Path(__file__).resolve().parents[2]
        candidates.append(repo_root / "dist" / "Aglaia" / "Aglaia")
    except Exception:
        pass
    for p in candidates:
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return None


def _register_linux(*, app_path: Path | None,
                    icon_path: Path | None) -> tuple[bool, str]:
    if app_path is None:
        app_path = _find_installed_app_linux() or Path("/usr/local/bin/Aglaia")
    exec_cmd = _shquote(str(app_path))

    desktop_p, xml_p = _linux_paths()
    desktop_p.parent.mkdir(parents=True, exist_ok=True)
    xml_p.parent.mkdir(parents=True, exist_ok=True)
    desktop_p.write_text(_LINUX_DESKTOP.format(
        exec_cmd=exec_cmd, mime=PROJECT_MIME))
    xml_p.write_text(_LINUX_MIME_XML.format(
        mime=PROJECT_MIME, ext=PROJECT_EXT_NODOT))

    # Best-effort cache refresh.
    for cmd in (
        ["update-mime-database", str(xml_p.parent.parent)],
        ["update-desktop-database", str(desktop_p.parent)],
        ["xdg-mime", "default", "aglaia.desktop", PROJECT_MIME],
    ):
        if shutil.which(cmd[0]):
            subprocess.run(cmd, check=False, capture_output=True)
    return True, f"Wrote {desktop_p} and {xml_p}"


def _unregister_linux() -> tuple[bool, str]:
    desktop_p, xml_p = _linux_paths()
    removed = []
    for p in (desktop_p, xml_p):
        if p.exists():
            p.unlink()
            removed.append(str(p))
    for cmd in (
        ["update-mime-database", str(xml_p.parent.parent)],
        ["update-desktop-database", str(desktop_p.parent)],
    ):
        if shutil.which(cmd[0]):
            subprocess.run(cmd, check=False, capture_output=True)
    return True, "Removed: " + ", ".join(removed) if removed else "Nothing to remove."


# ── Windows ───────────────────────────────────────────────────────

def _find_installed_app_windows() -> Path | None:
    """Probe for `Aglaia.exe` on Windows. Frozen process → `sys.executable`.
    Otherwise the typical install + dev-build locations."""
    if getattr(sys, "frozen", False):
        try:
            return Path(sys.executable).resolve()
        except Exception:
            pass
    candidates: list[Path] = []
    program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
    local_appdata = os.environ.get("LOCALAPPDATA",
                                   str(Path.home() / "AppData" / "Local"))
    candidates += [
        Path(program_files) / "Aglaia" / "Aglaia.exe",
        Path(local_appdata) / "Programs" / "Aglaia" / "Aglaia.exe",
        Path(local_appdata) / "Aglaia" / "Aglaia.exe",
    ]
    try:
        repo_root = Path(__file__).resolve().parents[2]
        candidates.append(repo_root / "dist" / "Aglaia" / "Aglaia.exe")
    except Exception:
        pass
    for p in candidates:
        if p.is_file():
            return p
    return None


def _register_windows(*, app_path: Path | None,
                      icon_path: Path | None) -> tuple[bool, str]:
    try:
        import winreg  # type: ignore[import-not-found]
    except ImportError:
        return False, "winreg unavailable (Python without Windows support)."

    if app_path is None:
        program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
        app_path = (_find_installed_app_windows()
                    or Path(program_files) / "Aglaia" / "Aglaia.exe")
    exec_cmd = f'"{app_path}" "%1"'

    prog_id = "bibli.cc.AglaiaProject"
    classes = winreg.HKEY_CURRENT_USER, "Software\\Classes"

    with winreg.CreateKey(*classes, PROJECT_EXT) as k:  # type: ignore[arg-type]
        winreg.SetValueEx(k, "", 0, winreg.REG_SZ, prog_id)
        winreg.SetValueEx(k, "Content Type", 0, winreg.REG_SZ, PROJECT_MIME)

    with winreg.CreateKey(*classes, prog_id) as k:  # type: ignore[arg-type]
        winreg.SetValueEx(k, "", 0, winreg.REG_SZ, f"{APP_NAME} project")
    with winreg.CreateKey(*classes, prog_id + "\\shell\\open\\command") as k:  # type: ignore[arg-type]
        winreg.SetValueEx(k, "", 0, winreg.REG_SZ, exec_cmd)

    if icon_path is not None:
        with winreg.CreateKey(*classes, prog_id + "\\DefaultIcon") as k:  # type: ignore[arg-type]
            winreg.SetValueEx(k, "", 0, winreg.REG_SZ, str(icon_path))

    return True, f"Registered {prog_id} for {PROJECT_EXT}"


def _unregister_windows() -> tuple[bool, str]:
    try:
        import winreg  # type: ignore[import-not-found]
    except ImportError:
        return False, "winreg unavailable."
    base = winreg.HKEY_CURRENT_USER
    prog_id = "bibli.cc.AglaiaProject"
    for key in (
        f"Software\\Classes\\{prog_id}\\shell\\open\\command",
        f"Software\\Classes\\{prog_id}\\shell\\open",
        f"Software\\Classes\\{prog_id}\\shell",
        f"Software\\Classes\\{prog_id}\\DefaultIcon",
        f"Software\\Classes\\{prog_id}",
        f"Software\\Classes\\{PROJECT_EXT}",
    ):
        try:
            winreg.DeleteKey(base, key)
        except OSError:
            continue
    return True, f"Removed {prog_id} registration."


# ── CLI ────────────────────────────────────────────────────────────

def _main() -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Register / unregister the .agl Aglaïa project file type."
    )
    sub = ap.add_subparsers(dest="action", required=True)
    p_reg = sub.add_parser("register", help="Register .agl with the OS.")
    p_reg.add_argument("--icon", type=Path, default=None,
                       help="Optional icon (.icns on macOS, .ico on Windows).")
    p_reg.add_argument("--app-path", type=Path, default=None,
                       help="Override the launcher path (macOS: .app bundle dir; "
                            "Linux/Windows: executable).")
    sub.add_parser("unregister", help="Unregister .agl.")
    sub.add_parser("plist", help="Print the macOS Info.plist snippet for packagers.")
    args = ap.parse_args()
    if args.action == "register":
        ok, msg = register_filetype(app_path=args.app_path, icon_path=args.icon)
    elif args.action == "unregister":
        ok, msg = unregister_filetype()
    elif args.action == "plist":
        sys.stdout.write(plist_snippet())
        return 0
    else:
        return 1
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
