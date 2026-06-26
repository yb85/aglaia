#!/usr/bin/env bash
# Assemble a Linux AppImage from the PyInstaller onedir.
#
# Prereqs:
#   1. uv run pyinstaller Aglaia.spec --clean --noconfirm   # → dist/Aglaia/
#   2. `appimagetool` on PATH (https://github.com/AppImage/appimagetool releases)
#
# Env:
#   AGLAIA_VERSION  version string baked into the output name (default 0.0.0-dev)
#   ARCH            target arch for appimagetool (default x86_64)
#   SUFFIX          optional name suffix, e.g. "-cuda" → Aglaia-x86_64-cuda.AppImage
#
# Output: dist/Aglaia-<ARCH><SUFFIX>.AppImage
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$REPO/dist"
APPDIR="$DIST/Aglaia.AppDir"
VERSION="${AGLAIA_VERSION:-0.0.0-dev}"
ARCH="${ARCH:-x86_64}"
SUFFIX="${SUFFIX:-}"

if [ ! -d "$DIST/Aglaia" ]; then
  echo "error: $DIST/Aglaia not found — run pyinstaller Aglaia.spec first" >&2
  exit 1
fi

# ── assemble the AppDir ──────────────────────────────────────────────
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin"
cp -a "$DIST/Aglaia/." "$APPDIR/usr/bin/"

# Icon: AppImage spec wants it at the AppDir root AND under hicolor.
cp "$REPO/packaging/aglaia.png" "$APPDIR/aglaia.png"
install -Dm644 "$REPO/packaging/aglaia.png" \
  "$APPDIR/usr/share/icons/hicolor/256x256/apps/aglaia.png"
ln -sf usr/share/icons/hicolor/256x256/apps/aglaia.png "$APPDIR/.DirIcon"

# .desktop (root + standard location) and the .agl MIME type.
cp "$REPO/packaging/aglaia.desktop" "$APPDIR/aglaia.desktop"
install -Dm644 "$REPO/packaging/aglaia.desktop" \
  "$APPDIR/usr/share/applications/aglaia.desktop"
install -Dm644 "$REPO/packaging/aglaia-mime.xml" \
  "$APPDIR/usr/share/mime/packages/aglaia.xml"

# AppRun → exec the bundled PyInstaller binary, forwarding args (.agl path).
cat > "$APPDIR/AppRun" <<'EOF'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
export PATH="$HERE/usr/bin:$PATH"
exec "$HERE/usr/bin/Aglaia" "$@"
EOF
chmod +x "$APPDIR/AppRun"

# ── build ────────────────────────────────────────────────────────────
OUT="$DIST/Aglaia-${ARCH}${SUFFIX}.AppImage"
rm -f "$OUT"
# APPIMAGE_EXTRACT_AND_RUN avoids needing a FUSE mount for appimagetool
# itself (it's an AppImage) on CI runners without /dev/fuse.
ARCH="$ARCH" APPIMAGE_EXTRACT_AND_RUN=1 appimagetool "$APPDIR" "$OUT"
echo "wrote $OUT (version $VERSION)"
