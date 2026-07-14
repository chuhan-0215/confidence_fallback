#!/usr/bin/env python3
"""Build GitHub-ready open-source zip for confidence_fallback."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_DIR = ROOT / "release" / "confidence_fallback"
ZIP_PATH = ROOT / "confidence_fallback_github.zip"

INCLUDE_DIRS = (
    "scripts",
    "model",
    "data",
    "configs",
    "figures",
    "submission_en",
)

INCLUDE_FILES = (
    "requirements.txt",
    "install.sh",
    "start.sh",
    "CODE_STRUCTURE.md",
    "README_GITHUB.md",
    "QUICKSTART_zh.md",
    "LICENSE",
    "download_confidence_fallback.ps1",
    "run_confidence_fallback.py",
    "verify_install.py",
)

RESULT_ARTIFACTS = (
    [
        "results/phase10/m2_enough_stop_head.pt",
        "inbox/000002/results/phase10/m2_enough_stop_head.pt",
    ],
    "results/phase10/m2_enough_stop_head.pt",
)

OPTIONAL_RESULT_FILES = (
    ("results/phase43/deploy_spec_v8_final.json", "results/phase43/deploy_spec_v8_final.json"),
    ("results/findings_summary.json", "results/findings_summary.json"),
)

EXCLUDE_DIR_NAMES = {
    "__pycache__",
    ".venv",
    ".cursor",
    ".git",
    "logs",
    "outbox",
    "inbox",
    "wheels",
    "checkpoints",
}

EXCLUDE_SUFFIXES = (".pyc", ".pyo", ".log")


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDE_DIR_NAMES:
        return True
    return path.suffix in EXCLUDE_SUFFIXES


def copy_tree(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise FileNotFoundError(f"Missing directory: {src}")
    for item in src.rglob("*"):
        if should_skip(item.relative_to(src)):
            continue
        rel = item.relative_to(src)
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def copy_first_existing(candidates: list[str], dst_rel: str) -> None:
    for src_rel in candidates:
        src = ROOT / src_rel
        if src.is_file():
            dst = RELEASE_DIR / dst_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return
    raise FileNotFoundError(f"Missing artifact (tried): {candidates}")


def stage_release() -> None:
    if RELEASE_DIR.exists():
        shutil.rmtree(RELEASE_DIR)
    RELEASE_DIR.mkdir(parents=True)

    for name in INCLUDE_DIRS:
        copy_tree(ROOT / name, RELEASE_DIR / name)

    for name in INCLUDE_FILES:
        src = ROOT / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing release file: {src}")
        dst_name = "README.md" if name == "README_GITHUB.md" else name
        shutil.copy2(src, RELEASE_DIR / dst_name)

    src_candidates, dst_rel = RESULT_ARTIFACTS
    copy_first_existing(list(src_candidates), dst_rel)

    for src_rel, out_rel in OPTIONAL_RESULT_FILES:
        src = ROOT / src_rel
        if src.is_file():
            dst = RELEASE_DIR / out_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    checkpoints_readme = RELEASE_DIR / "checkpoints" / "README.txt"
    checkpoints_readme.parent.mkdir(parents=True, exist_ok=True)
    checkpoints_readme.write_text(
        "Coconut checkpoint_300 is not bundled (about 60 MB).\n\n"
        "After creating the Python environment, run ONE of:\n"
        "  python scripts/download_checkpoint.py\n"
        "  bash install.sh\n\n"
        "Then verify:\n"
        "  python verify_install.py\n\n"
        "China mirror is used automatically when HF_ENDPOINT is unset:\n"
        "  export HF_ENDPOINT=https://hf-mirror.com\n",
        encoding="utf-8",
    )


def build_zip() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(RELEASE_DIR.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(RELEASE_DIR.parent))
    size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    print(f"Wrote {ZIP_PATH} ({size_mb:.2f} MB)")


def main() -> None:
    stage_release()
    build_zip()
    print("Download URL:")
    print("  http://39.105.119.125/others/000002/confidence_fallback_github.zip")
    print("  (use HTTP on Windows — HTTPS cert is not trusted)")


if __name__ == "__main__":
    main()
