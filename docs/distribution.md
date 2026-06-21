# Distribution

How Aglaïa ships: a signed, notarized macOS `.app` in a DMG, built by
GitHub Actions on a version tag, downloaded from a slick landing page at
**aglaia.bibli.cc** with docs at **aglaia.bibli.cc/docs**.

> **Runtime is macOS-only.** Apple Vision (page/OCR) and Speech (voice
> control) are hard pyobjc dependencies, and `mlx` (dewarp) is Apple
> Silicon. Releases therefore target **Apple Silicon macOS** only.

## Release pipeline (`.github/workflows/release.yml`)

Trigger: push a tag `vX.Y.Z`.

```bash
git tag v0.2.0 -m "Aglaïa 0.2.0"
git push origin v0.2.0
```

The `macos-14` runner then:

1. derives the version from the tag (`v0.2.0` → `0.2.0`),
2. vendors `llama-server` (`scripts/fetch_llama_server.py`, Surya backend),
3. `uv sync --extra macos --extra gui --extra dev --extra package --extra jbig2`,
4. runs `pytest` (red tests → no release),
5. imports the Developer ID cert into a temporary keychain,
6. builds the `.app` with PyInstaller (signed inline — `AGLAIA_VERSION` +
   `AGLAIA_SIGN_IDENTITY` env),
7. `scripts/sign_and_notarize.sh` (hardened-runtime re-sign → notarytool
   → staple),
8. `scripts/build_dmg.sh` (themed, lzfse-compressed, signed DMG),
9. `shasum -a 256` → `SHA256SUMS.txt`,
10. publishes a GitHub Release with the DMG + checksum and auto-generated
    notes.

Version is single-sourced from the tag: `Aglaia.spec` reads
`AGLAIA_VERSION` (fallback `0.0.0-dev`), bakes it into the Info.plist, and
`build_dmg.sh` reads it back out for the DMG filename.

## Required GitHub secrets

Settings → Secrets and variables → Actions:

| Secret | What | How to get it |
|---|---|---|
| `APPLE_CERT_P12_BASE64` | Developer ID Application cert **+ private key** | Keychain Access → export the identity as `.p12`, then `base64 -i cert.p12 \| pbcopy` |
| `APPLE_CERT_PASSWORD` | password set during the `.p12` export | you choose it at export time |
| `APPLE_SIGN_IDENTITY` | cert common name | `Developer ID Application: Your Name (TEAMID)` — see `security find-identity -p codesigning -v` |
| `APPLE_ID` | Apple ID email | your developer account |
| `APPLE_TEAM_ID` | 10-char team id | appleid / developer portal |
| `APPLE_APP_SPECIFIC_PASSWORD` | app-specific password for notarytool | appleid.apple.com → Sign-In & Security → App-Specific Passwords |

These mirror the local `.env` (`AGLAIA_SIGN_IDENTITY`,
`AGLAIA_NOTARY_PROFILE`) described in `.env.example`. The cert/private key
never leave GitHub's encrypted secret store; the workflow drops them into
an ephemeral keychain that dies with the runner.

## Local build (no CI)

```bash
uv run python scripts/fetch_llama_server.py
uv sync --extra macos --extra gui --extra dev --extra package --extra jbig2
AGLAIA_VERSION=0.2.0 uv run pyinstaller Aglaia.spec --clean --noconfirm
./scripts/sign_and_notarize.sh     # needs .env (copy from .env.example)
./scripts/build_dmg.sh
```

Unsigned local builds work too (omit `.env`): the `.app` runs but
Gatekeeper flags it on first launch.

## Website + docs (`site/`)

Astro landing page + [Starlight](https://starlight.astro.build) docs in
one project, deployed to **Cloudflare Pages**
(`.github/workflows/site.yml`). The docs are generated from this very
`docs/` directory by `site/scripts/sync-docs.mjs`, so prose lives in one
place and never drifts. See `site/README.md`.

## Possible future additions

- **Sparkle auto-update** — ship an `appcast.xml` regenerated per release
  so the app self-updates.
- **Homebrew cask** — `brew install --cask aglaia`, bumped by CI on
  release.
- **Intel / universal2 builds** — second runner / target if non-Apple-
  Silicon demand appears (loses the MLX dewarp fast path).
