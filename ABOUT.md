# About Aglaïa

Aglaïa is a scanner / page-extraction pipeline: capture or import pages,
correct geometry, binarise, run OCR, export a clean PDF or Markdown.

This page lists the third-party libraries that make Aglaïa possible and
the scientific work the algorithms are based on.

---

## Libraries

Direct dependencies Aglaïa declares (`pyproject.toml`). Optional ones —
loaded only for a specific feature — are tagged; everything else is core.
Transitive sub-dependencies are not listed.

| Package | Version | Author / Maintainer | Repository | License |
|---|---|---|---|---|
| **aglaia_jbig2** (in-tree) | 0.1.0 | Aglaïa (this repo) | local, wraps [jbig2enc-rust](https://github.com/LegeApp/jbig2enc-rust) by *LegeApp* | MIT OR Apache-2.0 |
| **doxapy** | 0.9.4 | Brandon M. Petty (Doxa project) | https://github.com/brandonmpetty/Doxa | CC0 1.0 (public domain) |
| **jax** | 0.9.0 | Google (Matthew Johnson et al.) | https://github.com/jax-ml/jax | Apache-2.0 |
| **jaxlib** | 0.9.0 | Google | https://github.com/jax-ml/jax | Apache-2.0 |
| **keyring** *(optional — cloud OCR key storage)* | 25.7.0 | Jason R. Coombs (jaraco) | https://github.com/jaraco/keyring | MIT |
| **mistralai** *(optional — Mistral cloud OCR)* | 1.5.2 | Mistral AI | https://github.com/mistralai/client-python | Apache-2.0 |
| **ml-dtypes** | 0.5.4 | Google (JAX team) | https://github.com/jax-ml/ml_dtypes | Apache-2.0 |
| **mlx** *(macOS arm64)* | 0.31.2 | Apple ML Research | https://github.com/ml-explore/mlx | MIT |
| **mlx-metal** *(macOS arm64)* | 0.31.2 | Apple ML Research | https://github.com/ml-explore/mlx | MIT |
| **mlx-vlm** | 0.6.2 | Prince Canuma | https://github.com/Blaizzy/mlx-vlm | MIT |
| **numpy** | 2.0.2 | NumPy developers (Travis Oliphant et al.) | https://github.com/numpy/numpy | BSD-3-Clause |
| **opencv-python-headless** | 4.11.0.86 | OpenCV project (Olli-Pekka Heinisuo, builds) | https://github.com/opencv/opencv-python | Apache-2.0 |
| **paddleocr** (`[doc-parser]`) | 3.6.0 | PaddlePaddle (Baidu) | https://github.com/PaddlePaddle/PaddleOCR | Apache-2.0 |
| **paddlepaddle** | 3.3.1 | PaddlePaddle (Baidu) | https://www.paddlepaddle.org.cn/ | Apache-2.0 |
| **page-dewarp** | 0.2.7 | Matt Zucker (original) · Louis Maddox (PyPI fork) | https://github.com/lmmx/page-dewarp · original https://github.com/mzucker/page_dewarp | MIT |
| **pikepdf** | 10.7.3 | James R. Barlow | https://github.com/pikepdf/pikepdf | MPL-2.0 |
| **pillow** | 12.1.0 | Pillow maintainers (Alex Clark, Hugo van Kemenade et al.) | https://github.com/python-pillow/Pillow | MIT-CMU (HPND) |
| **platformdirs** | 4.10.0 | tox-dev (Bernát Gábor, Trent Mick et al.) | https://github.com/tox-dev/platformdirs | MIT |
| **psutil** | 7.2.2 | Giampaolo Rodola | https://github.com/giampaolo/psutil | BSD-3-Clause |
| **pypdfium2** | 5.9.0 | pypdfium2-team (Anaconda + community) | https://github.com/pypdfium2-team/pypdfium2 | Apache-2.0 OR BSD-3-Clause |
| **pyobjc-framework-avfoundation** *(macOS)* | 12.1 | Ronald Oussoren (PyObjC) | https://github.com/ronaldoussoren/pyobjc | MIT |
| **pyobjc-framework-cocoa** *(macOS)* | 12.1 | Ronald Oussoren (PyObjC) | https://github.com/ronaldoussoren/pyobjc | MIT |
| **pyobjc-framework-speech** *(macOS)* | 12.1 | Ronald Oussoren (PyObjC) | https://github.com/ronaldoussoren/pyobjc | MIT |
| **pyobjc-framework-vision** *(macOS)* | 12.1 | Ronald Oussoren (PyObjC) | https://github.com/ronaldoussoren/pyobjc | MIT |
| **pyqtdarktheme** *(macOS)* | 2.1.0 | Yunosuke Ohsugi | https://github.com/5yutan5/PyQtDarkTheme | MIT |
| **pyside6** *(macOS)* | 6.11.1 | The Qt Company | https://wiki.qt.io/Qt_for_Python | LGPL-3.0 / commercial |
| **python-slugify** | 8.0.4 | Val Neekman | https://github.com/un33k/python-slugify | MIT |
| **pyyaml** | 6.0.3 | Kirill Simonov · Ingy döt Net et al. | https://github.com/yaml/pyyaml | MIT |
| **rich** | 14.3.1 | Will McGugan (Textualize) | https://github.com/Textualize/rich | MIT |
| **scipy** | 1.17.0 | SciPy developers (Pauli Virtanen et al.) | https://github.com/scipy/scipy | BSD-3-Clause |
| **surya-ocr** | 0.20.0 | Vik Paruchuri (Datalab) | https://github.com/datalab-to/surya | Apache-2.0 |

### Native binaries reached through wrappers

| Library | Wrapper | Repository | License |
|---|---|---|---|
| jbig2enc-rust (lossless JBIG2 encoder) | `aglaia_jbig2` (PyO3) | https://github.com/LegeApp/jbig2enc-rust | MIT OR Apache-2.0 |
| llama.cpp `llama-server` (Surya VLM inference) | `surya-ocr` → bundled binary (`vendor/llama-server/`) | https://github.com/ggml-org/llama.cpp | MIT |
| Apple Vision (`VNRecognizeTextRequest`) | `pyobjc-framework-vision` | proprietary, ships with macOS | Apple SLA |
| Apple Speech (`SFSpeechRecognizer`) | `pyobjc-framework-speech` | proprietary, ships with macOS | Apple SLA |
| AVFoundation (camera capture) | `pyobjc-framework-avfoundation` | proprietary, ships with macOS | Apple SLA |
| qpdf (PDF object model — assembly + OCR layer) | `pikepdf` | https://github.com/qpdf/qpdf | Apache-2.0 |
| PDFium (PDF rendering, page → image) | `pypdfium2` | https://pdfium.googlesource.com/pdfium/ | BSD-3-Clause |
| MLX (Metal-backed array ops) | `mlx` | https://github.com/ml-explore/mlx | MIT |

### License

Aglaïa is released under the **[PolyForm Shield License 1.0.0](https://polyformproject.org/licenses/shield/1.0.0)**. The full text lives in [`LICENSE`](LICENSE) at the project root.

PolyForm Shield is a *source-available* license. Plain English:

* You may use, modify, deploy, and redistribute Aglaïa for **any
  purpose** — including commercial, internal, and academic use.
* You may **not** repackage Aglaïa to compete with it: hosting a
  service, library, or plug-in that markets itself as a substitute for
  Aglaïa is not permitted. Forking for personal / customer use is
  fine; selling a re-skin or "Aglaïa-as-a-service" is not.
* Inbound dependencies are compatible: every library bundled below is
  MIT, BSD, Apache-2.0, MPL-2.0, or LGPL-3.0, all of which can be
  combined under PolyForm Shield on the application code without
  conflict.
* PySide6 (LGPL-3.0) stays a dynamically-linked, swappable component
  per LGPL §4 — distribution bundles must let an end user replace the
  PySide6 binary with their own build.

---

## Scientific research

Algorithms and source ideas the pipeline directly builds on.

### Binarisation (1-bit page output)

* **Sauvola, J., & Pietikäinen, M. (2000).** *Adaptive document image binarisation.* Pattern Recognition, 33(2), 225–236. [https://doi.org/10.1016/S0031-3203(99)00055-2](https://doi.org/10.1016/S0031-3203(99)00055-2)
* **Wolf, C., & Jolion, J.-M. (2004).** *Extraction and recognition of artificial text in multimedia documents.* Pattern Analysis and Applications, 6(4), 309–326. (a.k.a. **Wolf** thresholding, the primary algorithm Aglaïa uses by default.)
* **Niblack, W. (1986).** *An Introduction to Digital Image Processing.* Prentice Hall.
* **Khurshid, K., Siddiqi, I., Faure, C., & Vincent, N. (2009).** *Comparison of Niblack inspired binarization methods for ancient documents.* SPIE 7247. (the **NICK** algorithm.)
* **Otsu, N. (1979).** *A threshold selection method from gray-level histograms.* IEEE Trans. Systems, Man, and Cybernetics, 9(1), 62–66.
* **Gatos, B., Pratikakis, I., & Perantonis, S. J. (2006).** *Adaptive degraded document image binarisation.* Pattern Recognition, 39(3), 317–327.
* **Implementation:** Doxa C++ library by Brandon M. Petty — [https://github.com/brandonmpetty/Doxa](https://github.com/brandonmpetty/Doxa)

### Page geometry — dewarp / cubic-sheet

* **Schneider, D. C., Schwarz, H. R., & Bertel, S. (2008).** *Detection of distortions in book pages from scanned images.* (Cubic-sheet model that inspired the page-dewarp library Aglaïa wraps.)
* **page_dewarp project — Matt Zucker (2016).** Original blog post + reference implementation: [https://mzucker.github.io/2016/08/15/page-dewarp.html](https://mzucker.github.io/2016/08/15/page-dewarp.html) · [https://github.com/mzucker/page_dewarp](https://github.com/mzucker/page_dewarp)
* **Stamatopoulos, N., Gatos, B., Pratikakis, I., & Perantonis, S. J. (2011).** *Goal-Oriented Rectification of Camera-Based Document Images.* IEEE Trans. Image Processing.

### Trapezoidal correction (vanishing-point keystone removal)

* **Mirmehdi, M., & Clark, A. F. (1993).** *Detection of vanishing points from text lines on document images.*
* **Hartley, R., & Zisserman, A. (2003).** *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press. — projective rectification background.
* The current TLS / SVD vanishing-point estimator targets a future upgrade to RANSAC + IRLS-L1 (see `lib/processors/TrapezoidalCorrection.py` and the project notes).

### Skew correction (projection-profile rotation)

* **Postl, W. (1986).** *Detection of linear oblique structures and skew scan in digitised documents.* Proc. ICPR.
* **Kavallieratou, E., Fakotakis, N., & Kokkinakis, G. (2002).** *Skew angle estimation for printed and handwritten documents using the Wigner–Ville distribution.* Image and Vision Computing, 20(11), 813–824.

### OCR

* **Apple Vision framework** — closed-source. Used through `VNRecognizeTextRequest` with the *accurate* recognition level. Aglaïa surfaces line bounding boxes + confidence + recognised string per detected line.
* **Surya OCR — Paruchuri, V. (2024).** Open-weights OCR + layout model used as an optional fallback engine. [https://github.com/datalab-to/surya](https://github.com/datalab-to/surya)

### Text-detection back-ends (layout)

* **Liao, M., Wan, Z., Yao, C., Chen, K., & Bai, X. (2020).** *Real-time Scene Text Detection with Differentiable Binarization.* AAAI. (**DB / DBNet**, used as one of the layout back-ends.) [https://arxiv.org/abs/1911.08947](https://arxiv.org/abs/1911.08947)
* **Zhou, X., Yao, C., Wen, H., Wang, Y., Zhou, S., He, W., & Liang, J. (2017).** *EAST: An Efficient and Accurate Scene Text Detector.* CVPR. [https://arxiv.org/abs/1704.03155](https://arxiv.org/abs/1704.03155)

### PDF compression / coding standards

* **ITU-T Recommendation T.6 (1988).** *Facsimile coding schemes and coding control functions for Group 4 facsimile equipment* (CCITT G4). — the fallback bitonal PDF encoder.
* **ISO/IEC 14492:2001.** *Information technology — Lossy/lossless coding of bi-level images* (**JBIG2**). — the default bitonal PDF encoder via [`jbig2enc-rust`](https://github.com/LegeApp/jbig2enc-rust) (LegeApp).

### Optimisation back-ends

* **Liu, D. C., & Nocedal, J. (1989).** *On the limited-memory BFGS method for large-scale optimization.* Mathematical Programming, 45(3), 503–528. — L-BFGS-B used by the dewarp parameter fitter (both the JAX-CPU and MLX backends).
* **Powell, M. J. D. (1964).** *An efficient method for finding the minimum of a function of several variables without calculating derivatives.* Computer Journal, 7(2), 155–162. — fallback optimiser when neither JAX nor MLX is available.

### Document image processing — general references

* **Bukhari, S. S., Shafait, F., & Breuel, T. M. (2009).** *Border Noise Removal of Camera-Captured Document Images Using Page-Frame Detection.* CBDAR.
* **Doermann, D., & Tombre, K. (eds., 2014).** *Handbook of Document Image Processing and Recognition.* Springer.

---

## Datasets used for benchmarking

* In-house corpus of French theological scans (Aglaïa development), 100–300 dpi flatbed + webcam captures.
* Public DIBCO competitions (Document Image Binarisation Contest) — used for binariser sanity checks. [https://vc.ee.duth.gr/dibco2019/](https://vc.ee.duth.gr/dibco2019/)

---

*Aglaïa is © 2025–2026 Yann Barbotin. Released under the PolyForm Shield License 1.0.0 — see [`LICENSE`](LICENSE).*
