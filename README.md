<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Release][release-shield]][release-url]
[![CI][ci-shield]][ci-url]
[![Issues][issues-shield]][issues-url]
[![Python 3.12][python-shield]][python-url]
[![macOS][macos-shield]][macos-url]
[![License: PolyForm Shield][license-shield]][license-url]

<!-- PROJECT HEADER -->
<br />
<div align="center">
  <h1 align="center">Aglaïa</h1>

  <p align="center">
    Turn a webcam and a stack of pages into clean, deskewed, dewarped,
    searchable PDFs — locally, on your Mac.
    <br />
    <a href="https://aglaia.bibli.cc/docs"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://aglaia.bibli.cc">Website</a>
    ·
    <a href="https://github.com/yb85/aglaia/releases/latest">Download</a>
    ·
    <a href="https://github.com/yb85/aglaia/issues/new?labels=bug">Report Bug</a>
    ·
    <a href="https://github.com/yb85/aglaia/issues/new?labels=enhancement">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul><li><a href="#built-with">Built With</a></li></ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#how-it-works">How It Works</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

Scanning a book by phone is misery: curved pages, skew, glare, and a
hundred separate photos to wrangle. Flatbeds crack spines and take
forever. **Aglaïa** makes the webcam you already have behave like a
proper book scanner — and does the cleanup for you.

Point a camera at the page; Aglaïa captures, straightens, dewarps,
binarizes and OCRs whole books in one pass, then exports a searchable PDF
(invisible text layer over the image) or structured Markdown. It runs
entirely on your Mac — your pages never have to leave the machine.

A single `IntegratedProcessingChain` powers both the PySide6 capture GUI and
a headless CLI batch mode, driven by the same YAML-defined pipeline.

> [!NOTE]
> **macOS only.** Aglaïa hard-depends on Apple Vision (layout + OCR) and
> Speech (voice control), plus Apple Silicon for the MLX-accelerated page
> dewarper. There is no Windows/Linux build of the capture app.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![Python][python-shield]][python-url] managed with [uv](https://docs.astral.sh/uv/)
* **PySide6** — desktop GUI
* **OpenCV · NumPy · SciPy · Pillow** — image processing
* **page-dewarp + JAX/MLX** — cubic-sheet page dewarp
* **doxapy** — binarization (Wolf / Sauvola)
* **pikepdf · pypdfium2** — PDF I/O
* **Apple Vision · Speech** (pyobjc) — OCR, layout, voice
* **Surya · PaddleOCR-VL · Mistral Document AI** — OCR engines
* **SQLite (FTS5)** — project + full-text store

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* macOS on Apple Silicon
* [uv](https://docs.astral.sh/uv/getting-started/installation/)

  ```sh
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### Installation

**Download the app** — grab the latest signed, notarized DMG from the
[Releases page](https://github.com/yb85/aglaia/releases/latest), open it,
and drag **Aglaïa** to Applications.

**Or build from source:**

```sh
git clone https://github.com/yb85/aglaia.git
cd aglaia
uv sync --extra gui --extra macos
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE -->
## Usage

```sh
# Capture GUI (webcam + processing chain + voice control)
uv run python aglaia.py ~/scans/my-book

# Headless CLI batch — same chain, no Qt
uv run python aglaia.py ~/scans/my-book.agl --headless -p config/pipelines/book_curved_x2.yaml
```

Key flags: `-c/--config`, `-p/--pipeline`, `--workers`, `--make-pdf`,
`--debug`, `--input-dpi`, `--headless`. Capture-only: `--camera-id`,
`--voice-control`, `--transform "90|180|mirror|flip"`.

The import panel accepts multiple images and PDFs (per-page extract or
render). Drop EAST / PP-OCR models into `./model/` or `./models/`.

_For the full guide, see the [documentation](https://aglaia.bibli.cc/docs)._

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- HOW IT WORKS -->
## How It Works

```
capture → DPI fix → deskew → layout detect → dewarp → binarize → OCR → export
```

Every step is a pluggable processor defined in a YAML pipeline. Add your
own by dropping `lib/processors/<NewProc>.py` (the registry auto-discovers
it) — or, at runtime, drop a `.py` into `<APP_DATA>/plugins/` and approve
it in the trust prompt. See
[Architecture](https://aglaia.bibli.cc/docs/reference/architecture) and
[Processors](https://aglaia.bibli.cc/docs/reference/processors).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [ ] Drop-in plugin trust gate (processors + OCR engines)
- [ ] Sparkle in-app auto-update (appcast per release)
- [ ] Homebrew cask (`brew install --cask aglaia`)
- [ ] Intel / universal2 builds

See [open issues](https://github.com/yb85/aglaia/issues) for the full list.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Work is tracked via GitHub issues + milestones — one issue per discrete
unit of work. Before non-trivial work, open an issue. Branch names
reference it (`feat/123-slug`); PRs close via `Closes #N`.

1. Fork the project
2. Create your branch (`git checkout -b feat/123-amazing-feature`)
3. Make changes; keep `ruff`, `mypy --strict`, and `pytest` green
4. Commit (`git commit -m 'feat: add amazing feature'`)
5. Push and open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Source-available under the **[PolyForm Shield License 1.0.0](https://polyformproject.org/licenses/shield/1.0.0/)**
— see [`LICENSE`](LICENSE).

You may use, modify, and redistribute the software for **any purpose
except building a product that competes with it**. Otherwise free.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Yann Barbotin — yann.barbotin@gmail.com

Project: [github.com/yb85/aglaia](https://github.com/yb85/aglaia) ·
Website: [aglaia.bibli.cc](https://aglaia.bibli.cc)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [page-dewarp](https://github.com/lmmx/page-dewarp) — cubic-sheet dewarp
* [Surya](https://github.com/VikParuchuri/surya) · [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) · [Mistral Document AI](https://docs.mistral.ai/)
* [doxapy](https://github.com/brandonmpetty/Doxa) — binarization
* [Best-README-Template](https://github.com/othneildrew/Best-README-Template)
* Documentation patterns from [The Good Docs Project](https://www.thegooddocsproject.dev/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[release-shield]: https://img.shields.io/github/v/release/yb85/aglaia?style=for-the-badge
[release-url]: https://github.com/yb85/aglaia/releases/latest
[ci-shield]: https://img.shields.io/github/actions/workflow/status/yb85/aglaia/ci.yml?style=for-the-badge&label=CI
[ci-url]: https://github.com/yb85/aglaia/actions/workflows/ci.yml
[issues-shield]: https://img.shields.io/github/issues/yb85/aglaia?style=for-the-badge
[issues-url]: https://github.com/yb85/aglaia/issues
[python-shield]: https://img.shields.io/badge/python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org/
[macos-shield]: https://img.shields.io/badge/macOS-Apple%20Silicon-000000?style=for-the-badge&logo=apple&logoColor=white
[macos-url]: https://www.apple.com/macos/
[license-shield]: https://img.shields.io/badge/license-PolyForm%20Shield%201.0.0-orange?style=for-the-badge
[license-url]: https://polyformproject.org/licenses/shield/1.0.0/
