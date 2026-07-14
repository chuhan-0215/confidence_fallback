#!/usr/bin/env python3
"""Download pre-trained Coconut checkpoint used in the paper demo."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "checkpoints"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("Install deps first: pip install -r requirements.txt")

    path = hf_hub_download(
        repo_id="Shibo-UCSD/coconut-theory",
        filename="checkpoint_300",
        local_dir=str(DEST),
    )
    print("Downloaded:", path)


if __name__ == "__main__":
    main()
