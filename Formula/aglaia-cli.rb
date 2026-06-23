# Homebrew formula `aglaia-cli` — installs the `aglaia` command from source
# via uv. "cli" here means *launched from the command line* (`aglaia
# ~/scans/book`), NOT headless/GUI-less: this is the SAME full app as the
# Cask (GUI included), just started from a terminal instead of a .app
# double-click. The GUI app proper has its own Cask (`brew install --cask
# aglaia`, the notarized DMG). Token is `aglaia-cli` so the bare
# `brew install aglaia` resolves to the Cask, not this heavy source build —
# but the binary it installs is still `aglaia`.
#
# Lighter alternative: pass `--without-gui` for a CLI-ONLY install with no
# Qt/PySide6 (the headless batch pipeline, `aglaia --headless …`) — installs
# the lean base package, much smaller/faster to build.
#
# Pattern follows uv-based Homebrew installs (cf. github.com/thewisenerd/uvbrew):
# build an isolated venv with `uv`, install the project + its extras into it,
# and expose the `aglaia` console script. The package's entry point
# (`aglaia.__main__:run`) wires up multiprocessing before launching.
#
# CAVEATS
# - This builds a lot of native deps (jax, opencv, PySide6, pyobjc, …). It is a
#   heavy source install; for the macOS GUI app prefer the Cask (`brew install
#   --cask aglaia`) — the signed, notarized DMG.
# - The `jbig2` extra is omitted: it depends on the in-tree `aglaia_jbig2`
#   maturin crate (a local path dep, not on PyPI), so PDF export falls back to
#   CCITT G4 rather than JBIG2.
# - Replace `sha256` with the real release-tarball checksum when tagging:
#     curl -sL https://github.com/yb85/aglaia/archive/refs/tags/v0.1.0.tar.gz | shasum -a 256
class AglaiaCli < Formula
  desc "Webcam book scanner — capture, dewarp, binarize, OCR to PDF/Markdown"
  homepage "https://aglaia.bibli.cc"
  url "https://github.com/yb85/aglaia/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"
  license "LicenseRef-PolyForm-Shield-1.0.0"
  head "https://github.com/yb85/aglaia.git", branch: "main"

  option "without-gui", "CLI-only: install the lean base, no Qt/PySide6 (headless)"

  depends_on "uv" => :build
  depends_on "python@3.12"

  def install
    venv = libexec/"venv"
    python = formula_opt_bin("python@3.12")/"python3.12"

    # Isolated environment with the pinned interpreter.
    system "uv", "venv", venv, "--python", python

    # Default: project + GUI/macOS extras. `--without-gui`: lean base only
    # (no PySide6 → headless `aglaia --headless …`). jbig2 omitted — see
    # CAVEATS. `uv pip install` honours pyproject + uv.lock sources.
    extras = if build.without?("gui")
      ""
    else
      OS.mac? ? "[gui,macos]" : "[gui]"
    end
    system "uv", "pip", "install", "--python", venv/"bin/python", "#{buildpath}#{extras}"

    # Expose the console script with the venv's bin on PATH so its deps resolve.
    (bin/"aglaia").write_env_script venv/"bin/aglaia", PATH: "#{venv}/bin:$PATH"
  end

  test do
    # --help works without a workspace, display, or (with --without-gui) Qt.
    assert_path_exists bin/"aglaia"
    assert_path_exists libexec/"venv/bin/aglaia"
    assert_match "headless", shell_output("#{bin}/aglaia --help")
  end
end
