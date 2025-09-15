#!/usr/bin/env python3
"""
Pre-download translation/TTS models into the Hugging Face cache so the app runs offline.

Models:
- Helsinki-NLP/opus-mt-en-ko
- Helsinki-NLP/opus-mt-ko-en
- facebook/nllb-200-distilled-600M
- facebook/nllb-200-3.3B
- facebook/seamless-m4t-v2-large

Usage:
  python translation/download_models.py

Environment:
- Uses default HF cache (~/.cache/huggingface). Override with HF_HOME or HF_HUB_ENABLE_HF_TRANSFER=1 for faster downloads.
- Set HF_TOKEN if you have an authenticated mirror; not required for public models.
"""
from __future__ import annotations

import os
import sys
from typing import List

def ensure_hf_hub():
    try:
        import huggingface_hub  # noqa: F401
        return True
    except Exception:
        return False


def main() -> int:
    if not ensure_hf_hub():
        print("ERROR: huggingface_hub is not installed.\n"
              "Install it first: pip install -U huggingface_hub")
        return 2

    from huggingface_hub import snapshot_download

    repos: List[str] = [
        "Helsinki-NLP/opus-mt-en-ko",
        "Helsinki-NLP/opus-mt-ko-en",
        "facebook/nllb-200-distilled-600M",
        "facebook/nllb-200-3.3B",
        "facebook/seamless-m4t-v2-large",
    ]

    print("Hugging Face cache dir:", os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    print("Starting downloads (this can take a while)...")

    for repo in repos:
        try:
            print(f"\nDownloading {repo} ...")
            local_dir = snapshot_download(repo_id=repo, local_files_only=False)
            print(f"✓ Cached to: {local_dir}")
        except Exception as e:
            print(f"✗ Failed to download {repo}: {e}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

