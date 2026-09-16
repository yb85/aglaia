# Homebrew Cask — installs the signed, notarized Aglaïa.app from the
# GitHub release DMG (Apple Silicon). This is the recommended install for
# the GUI: a real .app (full Dock icon + name, Launchpad, Finder), no
# 2 GB source build. The CLI lives in the `aglaia-cli` formula / pip.
#
# CAVEATS
# - Needs a published GitHub release: the `vX.Y.Z` tag's DMG asset
#   (`Aglaia-X.Y.Z.dmg`) and its real sha256. Until a release is cut this
#   cask can't install — replace the placeholder sha256 below per tag:
#     shasum -a 256 dist/Aglaia-0.1.0rc6.dmg   (or read SHA256SUMS.txt in the release)
#   (The release CI already builds + uploads the DMG + SHA256SUMS.txt.)
cask "aglaia" do
  version "0.1.0rc6"
  sha256 "e20268a6b344c0b5002c9ebf6ba7555e1cf5efccb0f37b3012997892a8497f4b"

  url "https://github.com/yb85/aglaia/releases/download/v#{version}/Aglaia-#{version}.dmg",
      verified: "github.com/yb85/aglaia/"
  name "Aglaïa"
  desc "Webcam book scanner — capture, dewarp, binarize, OCR to PDF/Markdown"
  homepage "https://aglaia.bibli.cc/"

  # Apple Silicon only (MLX dewarp), built/notarized on macOS Sonoma.
  depends_on arch: :arm64
  depends_on macos: :sonoma

  app "Aglaia.app"

  zap trash: [
    "~/Library/Application Support/Aglaia",
    "~/Library/Caches/Aglaia",
    "~/Library/Logs/Aglaia",
    "~/Library/Preferences/cc.bibli.aglaia.plist",
    "~/Library/Saved Application State/cc.bibli.aglaia.savedState",
  ]
end
