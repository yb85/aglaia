# Homebrew formula — installs the `aglaia` CLI/GUI from source via uv.
#
# Pattern follows uv-based Homebrew installs (cf. github.com/thewisenerd/uvbrew):
# build an isolated venv with `uv`, install the project + its extras into it,
# and expose the `aglaia` console script. The package's entry point
# (`aglaia.__main__:run`) wires up multiprocessing before launching.
#
# CAVEATS
# - This builds a lot of native deps (jax, opencv, PySide6, pyobjc, …). It is a
#   heavy source install; for the macOS GUI app most users want the signed,
#   notarized DMG instead — a `brew install --cask aglaia` Cask pointing at the
#   GitHub release would be the lighter path (add later).
# - The `jbig2` extra is omitted: it depends on the in-tree `aglaia_jbig2`
#   maturin crate (a local path dep, not on PyPI), so PDF export falls back to
#   CCITT G4 rather than JBIG2.
# - Replace `sha256` with the real release-tarball checksum when tagging:
#     curl -sL https://github.com/yb85/aglaia/archive/refs/tags/v0.1.0.tar.gz | shasum -a 256
class Aglaia < Formula
  desc "Webcam book scanner — capture, dewarp, binarize, OCR to PDF/Markdown"
  homepage "https://aglaia.bibli.cc"
  url "https://github.com/yb85/aglaia/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"
  license "LicenseRef-PolyForm-Shield-1.0.0"
  head "https://github.com/yb85/aglaia.git", branch: "main"

  depends_on "uv" => :build
  depends_on "python@3.12"

  def install
    venv = libexec/"venv"
    python = Formula["python@3.12"].opt_bin/"python3.12"

    # Isolated environment with the pinned interpreter.
    system "uv", "venv", venv, "--python", python

    # Install the project + the GUI/macOS extras into the venv. (jbig2 omitted
    # — see CAVEATS.) `uv pip install` honours pyproject + uv.lock sources.
    extras = OS.mac? ? "[gui,macos]" : "[gui]"
    system "uv", "pip", "install", "--python", venv/"bin/python", "#{buildpath}#{extras}"

    # Expose the console script with the venv's bin on PATH so its deps resolve.
    (bin/"aglaia").write_env_script venv/"bin/aglaia", PATH: "#{venv}/bin:$PATH"
  end

  test do
    # No workspace + no display → launching the GUI isn't meaningful in CI, so
    # just assert the console script and its venv entry are wired up.
    assert_predicate bin/"aglaia", :exist?
    assert_predicate libexec/"venv/bin/aglaia", :exist?
  end
end
