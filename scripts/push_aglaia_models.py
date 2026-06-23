# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""One-shot: create + populate aglaia-models org's mirror repos on HF.

What it pushes (current state on disk):

  aglaia-models/surya-ocr-2-Q4_K_M-gguf  ←  custom Q4_K_M build + mmproj + configs
  aglaia-models/paddleocr-vl-1.5-4bit    ←  mirror of mlx-community/PaddleOCR-VL-1.5-4bit
  aglaia-models/east-text-detection      ←  mirror of OpenCV's EAST .pb
  aglaia-models/ppocrv4-det              ←  mirror of RapidOCR's ch_PP-OCRv4_det

Run:
    HF_HOME=$HOME/.cache/huggingface python scripts/push_aglaia_models.py

Idempotent. ``create_repo(exist_ok=True)`` skips already-created repos;
``upload_folder`` overwrites only what differs (xet dedupe handles the rest).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi


ORG = "aglaia-models"
# The app's per-user models dir (override with AGLAIA_MODELS_DIR). On macOS
# this resolves to ~/Library/Application Support/Aglaia/models.
_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root))
from aglaia.app_data import models_dir  # noqa: E402

MODELS_DIR = Path(os.environ.get("AGLAIA_MODELS_DIR") or models_dir())


@dataclass
class RepoSpec:
    repo: str                    # "<org>/<name>"
    local: Path                  # source folder OR single file
    is_folder: bool
    description: str             # short tagline for the model card
    upstream: str                # attribution string
    license_id: str = "apache-2.0"
    # Only these glob patterns are uploaded. Empty = upload everything
    # under ``local``. Lets us skip caches / partials when the source
    # folder gets reused later.
    allow_patterns: tuple[str, ...] = ()
    # Inverse — patterns excluded from upload (e.g. assets/ folder).
    ignore_patterns: tuple[str, ...] = field(
        default_factory=lambda: (
            ".gitattributes", ".cache/*", "*.partialdl", "__pycache__/*"
        )
    )


SPECS: list[RepoSpec] = [
    RepoSpec(
        repo=f"{ORG}/surya-ocr-2-Q4_K_M-gguf",
        local=MODELS_DIR / "surya-ocr-2-gguf",
        is_folder=True,
        description=(
            "Surya OCR 2 — Q4_K_M GGUF build for Aglaïa. Document OCR "
            "VLM (~0.6 B params) with bbox + reading-order structure."
        ),
        upstream="Quantized from datalab-to/surya-ocr-2-gguf via llama-quantize.",
        license_id="cc-by-nc-sa-4.0",
        # Skip the FP16 baseline + nested assets/ + docs — keep this
        # repo tight to the Q4_K_M weights + mmproj + the configs
        # llama-server needs at boot.
        allow_patterns=(
            "surya-2-Q4_K_M.gguf",
            "surya-2-mmproj.gguf",
            "*.json",
            "*.jinja",
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
            "chat_template.jinja",
            "generation_config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "video_preprocessor_config.json",
            "LICENSE",
        ),
    ),
    RepoSpec(
        repo=f"{ORG}/paddleocr-vl-1.5-4bit",
        local=MODELS_DIR / "PaddleOCR-VL-1.5-4bit",
        is_folder=True,
        description=(
            "PaddleOCR-VL 1.5 — MLX 4-bit. 0.9 B multilingual document "
            "VLM (109 languages). Native MD output via PP-DocLayoutV2 "
            "+ block-level VLM routing."
        ),
        upstream="Mirror of mlx-community/PaddleOCR-VL-1.5-4bit.",
        license_id="apache-2.0",
    ),
    RepoSpec(
        repo=f"{ORG}/east-text-detection",
        local=MODELS_DIR / "frozen_east_text_detection.pb",
        is_folder=False,
        description=(
            "EAST scene-text detector — frozen TensorFlow graph. Used "
            "as Aglaïa's text-region layout backend on platforms without "
            "Apple Vision."
        ),
        upstream=(
            "Mirror from gifflet/opencv-text-detection "
            "(commit c1e279bd4bf00889f25bf4fb6c169c8c0fdc619a)."
        ),
        license_id="apache-2.0",
    ),
    RepoSpec(
        repo=f"{ORG}/ppocrv4-det",
        local=MODELS_DIR / "ch_PP-OCRv4_det_infer.onnx",
        is_folder=False,
        description=(
            "PP-OCRv4 text detection — ONNX. Backup layout backend used "
            "by Aglaïa's DBNet path."
        ),
        upstream="Mirror of PaddleOCR's PP-OCRv4 det (via SWHL/RapidOCR).",
        license_id="apache-2.0",
    ),
]


_MODEL_CARD_TEMPLATE = """---
license: {license}
tags:
- aglaia
- ocr
---

# {name}

{description}

## Upstream

{upstream}

## How Aglaïa uses this

Pulled by Aglaïa's in-app model downloader (see
`aglaia/app_data/model-list.json`). Re-hosting here pins the file set we
ship and gives a single org to trust for the entire OCR stack.
"""


def write_model_card(local: Path, spec: RepoSpec) -> Path:
    """Write a minimal README.md inside the folder (or alongside the
    file) so the HF page isn't blank."""
    if spec.is_folder:
        target = local / "README.md"
    else:
        # Single-file specs upload only that file. We don't write a card
        # for them — the HF page will show a generated placeholder.
        return local
    content = _MODEL_CARD_TEMPLATE.format(
        license=spec.license_id,
        name=spec.repo.split("/", 1)[1],
        description=spec.description,
        upstream=spec.upstream,
    )
    target.write_text(content)
    return target


def push_one(api: HfApi, spec: RepoSpec) -> None:
    print(f"\n=== {spec.repo}")
    if not spec.local.exists():
        print(f"  ✗ source missing: {spec.local}")
        return
    api.create_repo(
        repo_id=spec.repo, repo_type="model",
        private=False, exist_ok=True,
    )
    print(f"  ✓ repo ready")

    if spec.is_folder:
        write_model_card(spec.local, spec)
        kwargs = dict(
            folder_path=str(spec.local),
            repo_id=spec.repo, repo_type="model",
        )
        if spec.allow_patterns:
            kwargs["allow_patterns"] = list(spec.allow_patterns)
        kwargs["ignore_patterns"] = list(spec.ignore_patterns)
        print(f"  ↑ uploading folder: {spec.local}")
        api.upload_folder(**kwargs)
    else:
        print(f"  ↑ uploading file: {spec.local}")
        api.upload_file(
            path_or_fileobj=str(spec.local),
            path_in_repo=spec.local.name,
            repo_id=spec.repo, repo_type="model",
        )
    print(f"  ✓ done → https://huggingface.co/{spec.repo}")


def main() -> int:
    api = HfApi()
    me = api.whoami()
    print(f"hf user: {me['name']}")
    org_names = [o["name"] for o in me.get("orgs", [])]
    if ORG not in org_names:
        print(f"✗ not a member of org {ORG}; current orgs: {org_names}",
              file=sys.stderr)
        return 1

    only = set(sys.argv[1:])
    for spec in SPECS:
        name = spec.repo.split("/", 1)[1]
        if only and name not in only and spec.repo not in only:
            continue
        push_one(api, spec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
