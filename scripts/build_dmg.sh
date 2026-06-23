#!/usr/bin/env bash
# Build a compressed, themed DMG installer for Aglaïa.app.
#
# Pipeline:
#   1. Stage the .app + Applications symlink + bg image into a temp dir.
#   2. Create a read-write DMG large enough to hold them.
#   3. Mount it, run osascript to position icons and set the background,
#      stash window geometry inside .DS_Store.
#   4. Unmount, hdiutil convert to lzfse-compressed UDZO read-only.
#
# Output: dist/Aglaia-<version>.dmg
#
# Prereqs: `dist/Aglaia.app` exists and has been signed + notarized
# (otherwise the DMG passes Gatekeeper but the user's first launch
# of the .app inside still hits a quarantine prompt).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="${REPO_ROOT}/dist/Aglaia.app"
BG="${REPO_ROOT}/aglaia/assets/brand/aglaia_bg_arrow.png"
VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' \
    "${APP}/Contents/Info.plist" 2>/dev/null || echo '0.0.0')"
DMG_NAME="Aglaia-${VERSION}"
VOLNAME="Aglaïa"
FINAL_DMG="${REPO_ROOT}/dist/${DMG_NAME}.dmg"
RW_DMG="${REPO_ROOT}/dist/${DMG_NAME}-rw.dmg"

if [[ ! -d "$APP" ]]; then
    echo "✗ $APP not found. Run pyinstaller + sign_and_notarize first." >&2
    exit 1
fi
if [[ ! -f "$BG" ]]; then
    echo "✗ $BG not found." >&2
    exit 1
fi

rm -f "$FINAL_DMG" "$RW_DMG"

# Size the staging DMG: app size + 80 MB headroom.
APP_BYTES=$(du -sk "$APP" | cut -f1)
RW_SIZE_MB=$(( (APP_BYTES / 1024) + 80 ))

STAGE_DIR="$(mktemp -d -t aglaia-dmg)"
trap 'rm -rf "$STAGE_DIR"' EXIT

# Lay out the DMG window contents.
ditto "$APP" "$STAGE_DIR/Aglaia.app"
ln -s /Applications "$STAGE_DIR/Applications"
mkdir -p "$STAGE_DIR/.background"
# The brand bg is a large hi-res PNG (2424×1536). Resample to a 2× retina
# tile (1280×800) tagged 144 dpi so Finder renders it at the window's
# 640×400 *point* bounds (set in the osascript below) — crisp on Retina,
# correctly scaled on 1×. (The old bg was a 1× 640×400 image.)
sips -z 800 1280 -s dpiWidth 144 -s dpiHeight 144 \
    "$BG" --out "$STAGE_DIR/.background/background.png" >/dev/null
# Mark hidden so Finder doesn't render the dot-folder when the user
# mounts the DMG. Combination of chflags + SetFile catches both old
# (Carbon) and new (HFS) listings.
chflags hidden "$STAGE_DIR/.background"
SetFile -a V "$STAGE_DIR/.background" 2>/dev/null || true

echo "→ creating ${RW_SIZE_MB} MB read-write DMG"
hdiutil create -volname "$VOLNAME" \
    -srcfolder "$STAGE_DIR" \
    -ov \
    -format UDRW \
    -size "${RW_SIZE_MB}m" \
    "$RW_DMG" >/dev/null

echo "→ mounting"
MOUNT_OUT="$(hdiutil attach -readwrite -noverify -noautoopen "$RW_DMG")"
# Modern hdiutil output uses tab-separated columns: dev<TAB>type<TAB>mount.
# Some columns are blank for non-HFS slices. Match the last line with a
# "/Volumes/" mount-point (the actual file-system slice).
MOUNT_LINE="$(echo "$MOUNT_OUT" | grep '/Volumes/' | tail -1)"
MOUNT_DEV="$(echo "$MOUNT_LINE" | awk '{print $1}')"
MOUNT_POINT="$(echo "$MOUNT_LINE" | sed -E 's@.*(/Volumes/[^[:cntrl:]]*).*@\1@')"
if [[ -z "$MOUNT_POINT" || ! -d "$MOUNT_POINT" ]]; then
    echo "✗ couldn't parse mount point from:" >&2
    echo "$MOUNT_OUT" >&2
    exit 1
fi
echo "   mounted at $MOUNT_POINT"

# Let the volume settle before osascript pokes at .DS_Store.
sleep 2

echo "→ styling window via Finder"
osascript <<EOF
tell application "Finder"
    tell disk "$VOLNAME"
        open
        set current view of container window to icon view
        set toolbar visible of container window to false
        set statusbar visible of container window to false
        set the bounds of container window to {200, 120, 840, 520}
        set viewOptions to the icon view options of container window
        set arrangement of viewOptions to not arranged
        set icon size of viewOptions to 96
        set text size of viewOptions to 13
        set background picture of viewOptions to file ".background:background.png"
        -- Icon placement: app on the left, arrow → Applications on the right
        set position of item "Aglaia.app" of container window to {160, 240}
        set position of item "Applications" of container window to {480, 240}
        close
        open
        update without registering applications
        delay 1
    end tell
end tell
EOF

echo "→ syncing + unmounting"
# Finder leaves a .fseventsd cache + a .Trashes folder on the mounted
# volume. Both show up as siblings of Aglaia.app in plain `ls`. Strip
# them before unmounting so they don't ship inside the read-only DMG.
rm -rf "$MOUNT_POINT/.fseventsd" "$MOUNT_POINT/.Trashes" \
       "$MOUNT_POINT/.DS_Store.tmp"  2>/dev/null || true
sync
hdiutil detach "$MOUNT_DEV" >/dev/null

echo "→ converting to lzfse-compressed read-only"
hdiutil convert "$RW_DMG" \
    -format ULMO \
    -o "$FINAL_DMG" >/dev/null
rm -f "$RW_DMG"

echo "→ size:"
ls -lh "$FINAL_DMG" | awk '{print "  " $5 "  " $NF}'

# Re-sign the DMG so Gatekeeper accepts it.
if [[ -f "${REPO_ROOT}/.env" ]]; then
    # shellcheck disable=SC1091
    set -a; . "${REPO_ROOT}/.env"; set +a
fi
if [[ -n "${AGLAIA_SIGN_IDENTITY:-}" ]]; then
    echo "→ signing dmg"
    codesign --force --timestamp --sign "$AGLAIA_SIGN_IDENTITY" "$FINAL_DMG"
    codesign --verify --verbose=2 "$FINAL_DMG"
fi

echo "✓ $FINAL_DMG"
