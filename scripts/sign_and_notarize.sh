#!/usr/bin/env bash
# Re-sign Aglaia.app with the hardened runtime + Developer ID identity,
# then notarize via notarytool and staple the ticket.
#
# Prereqs (one-time per machine):
#   1. Developer ID Application cert + private key in login keychain.
#   2. App-specific password stored as a notarytool keychain profile:
#        xcrun notarytool store-credentials "aglaia-notary" \
#          --apple-id "you@example.com" \
#          --team-id  "ABR563Y282" \
#          --password "xxxx-xxxx-xxxx-xxxx"
#
# Usage:
#   uv run pyinstaller Aglaia.spec --clean --noconfirm
#   ./scripts/sign_and_notarize.sh
#
# Env-var overrides:
#   AGLAIA_SIGN_IDENTITY   — cert common-name (default: yann's cert)
#   AGLAIA_NOTARY_PROFILE  — keychain profile name (default: aglaia-notary)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="${REPO_ROOT}/dist/Aglaia.app"
ENT="${REPO_ROOT}/packaging/entitlements.plist"

# Load gitignored .env so the signing identity never appears on the
# command line / in scrollback / in CI logs.
if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    . "${REPO_ROOT}/.env"
    set +a
fi

IDENTITY="${AGLAIA_SIGN_IDENTITY:-}"
PROFILE="${AGLAIA_NOTARY_PROFILE:-aglaia-notary}"

if [[ -z "$IDENTITY" ]]; then
    echo "✗ AGLAIA_SIGN_IDENTITY unset. Copy .env.example → .env and fill it in." >&2
    exit 2
fi

if [[ ! -d "$APP" ]]; then
    echo "✗ $APP not found. Run pyinstaller first." >&2
    exit 1
fi
if [[ ! -f "$ENT" ]]; then
    echo "✗ entitlements file missing: $ENT" >&2
    exit 1
fi

# codesign with retry. Apple's secure-timestamp server (TSA) intermittently
# rejects requests when many fire back-to-back ("A timestamp was expected but
# was not found") — Pillow alone bundles ~15 small dylibs signed in a burst.
# Retry with linear backoff so a transient TSA hiccup doesn't fail the build.
# This is THE signer (PyInstaller is run WITHOUT an identity on purpose so it
# doesn't burst the TSA itself); every binary must end up signed + timestamped
# for notarization, which the --verify --strict pass below enforces.
_codesign_ts() {
    local n=1 max=6
    while true; do
        if codesign "$@"; then return 0; fi
        if (( n >= max )); then
            echo "✗ codesign failed after ${max} attempts: ${*: -1}" >&2
            return 1
        fi
        echo "  …codesign retry ${n}/${max} (TSA hiccup) — backing off ${n}s" >&2
        sleep "$n"
        ((n++))
    done
}

echo "→ signing $APP"
# Walk every dylib + so + binary first (PyInstaller leaves them unsigned).
# The --deep on the outer app sign only goes one level — we want full recursion.
find "$APP/Contents" \
    \( -name '*.dylib' -o -name '*.so' -o -perm +111 \) -type f \
    | while read -r f; do
        _codesign_ts --force --options runtime --timestamp \
            --sign "$IDENTITY" "$f" || true
    done

# Final wrap pass over the bundle itself (retry; an unrecoverable failure here
# aborts via set -e, which is what we want).
_codesign_ts --force --deep --options runtime --timestamp \
    --entitlements "$ENT" \
    --sign "$IDENTITY" \
    "$APP"

echo "→ verifying signature"
codesign --verify --deep --strict --verbose=2 "$APP"

echo "→ submitting to notarytool"
ZIP="${REPO_ROOT}/dist/Aglaia.zip"
ditto -c -k --keepParent "$APP" "$ZIP"
xcrun notarytool submit "$ZIP" \
    --keychain-profile "$PROFILE" \
    --wait

echo "→ stapling notarization ticket"
xcrun stapler staple "$APP"

echo "→ final gatekeeper assessment"
spctl --assess --type execute --verbose=2 "$APP"

rm -f "$ZIP"
echo "✓ $APP is signed, notarized, stapled."
