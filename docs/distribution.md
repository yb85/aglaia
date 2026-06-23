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

## Windows installer & code signing

The same tag triggers the `build-windows` job (`windows-latest`): it
PyInstaller-builds the onedir, compiles the Inno Setup installer
(`packaging/aglaia.iss` — Start-menu shortcut + `.agl` association), and
ships `Aglaia-windows-x64-setup.exe` + a portable ZIP.

Authenticode signing is **optional**. Set both secrets below to sign the
exe and the installer; leave them unset and an *unsigned* installer + ZIP
still ship (users get a SmartScreen "unknown publisher" prompt on first
run).

| Secret | What | How to get it |
|---|---|---|
| `WINDOWS_CERT_PFX_BASE64` | Authenticode cert **+ private key** as a `.pfx`, base64-encoded | see below |
| `WINDOWS_CERT_PASSWORD` | password protecting the `.pfx` | you choose it at export time |

> **CA reality (2023+):** OV/EV code-signing certs now require the private
> key on a hardware token or HSM, so a real CA won't hand you an
> exportable `.pfx`. Genuine SmartScreen trust means a cloud-signing
> service (Azure Trusted Signing, DigiCert KeyLocker, SSL.com eSigner),
> which would need the signing steps rewritten — `signtool /f cert.pfx`
> can't consume an HSM key.

### Self-signed cert (no SmartScreen trust)

Fits the existing workflow as-is. The signature is valid and timestamped
but the root isn't trusted, so the "unknown publisher" prompt remains.
Fine for internal distribution or validating the signing plumbing.
Generate one on Windows (PowerShell), keeping the `.pfx` **outside the
repo**:

```powershell
$cert = New-SelfSignedCertificate -Type CodeSigningCert `
  -Subject "CN=bibli.cc, O=bibli.cc" -KeyUsage DigitalSignature `
  -KeySpec Signature -KeyExportPolicy Exportable `
  -KeyAlgorithm RSA -KeyLength 3072 -HashAlgorithm SHA256 `
  -CertStoreLocation Cert:\CurrentUser\My -NotAfter (Get-Date).AddYears(5)

$pw = ConvertTo-SecureString "<choose-a-password>" -Force -AsPlainText
Export-PfxCertificate -Cert "Cert:\CurrentUser\My\$($cert.Thumbprint)" `
  -FilePath "$HOME\aglaia-codesign\aglaia-codesign.pfx" -Password $pw
Remove-Item "Cert:\CurrentUser\My\$($cert.Thumbprint)" -Force
```

Then push the secrets (web UI, or `gh` once authenticated):

```powershell
$b64 = [Convert]::ToBase64String(
  [IO.File]::ReadAllBytes("$HOME\aglaia-codesign\aglaia-codesign.pfx"))
$b64 | Out-File -Encoding ascii "$HOME\aglaia-codesign\WINDOWS_CERT_PFX_BASE64.txt"
gh secret set WINDOWS_CERT_PFX_BASE64 < "$HOME\aglaia-codesign\WINDOWS_CERT_PFX_BASE64.txt"
gh secret set WINDOWS_CERT_PASSWORD   # prompts, hidden input
```

Verify the cert signs before relying on CI (same `signtool` call the
workflow uses; `verify /pa` will report an untrusted chain for a
self-signed cert — that's expected):

```powershell
signtool sign /f aglaia-codesign.pfx /p <password> /fd SHA256 `
  /tr http://timestamp.digicert.com /td SHA256 path\to\Aglaia.exe
```

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
