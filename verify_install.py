#!/usr/bin/env python3
"""Check that the release tree has everything needed before running experiments."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    return None


def main() -> int:
    print("confidence_fallback install check\n")
    errors = 0

    required_files = [
        ROOT / "configs" / "symbol-2layer-8head-768dim.json",
        ROOT / "data" / "prosqa_test_graph_4_coconut.json",
        ROOT / "results" / "phase10" / "m2_enough_stop_head.pt",
        ROOT / "run_confidence_fallback.py",
        ROOT / "scripts" / "download_checkpoint.py",
        ROOT / "model" / "coconut.py",
    ]
    print("Required files:")
    for path in required_files:
        if path.is_file():
            ok(str(path.relative_to(ROOT)))
        else:
            fail(f"missing {path.relative_to(ROOT)}")
            errors += 1

    checkpoint = ROOT / "checkpoints" / "checkpoint_300"
    print("\nCoconut checkpoint:")
    if checkpoint.is_file():
        ok("checkpoints/checkpoint_300 present")
    else:
        warn("checkpoints/checkpoint_300 missing — run: python scripts/download_checkpoint.py")

    print("\nPython imports:")
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "model"))
    for mod in ("torch", "transformers", "numpy"):
        if importlib.util.find_spec(mod):
            ok(mod)
        else:
            fail(f"pip package missing: {mod}")
            errors += 1

    try:
        from evaluate_coconut import load_dataset  # noqa: WPS433
        samples = load_dataset(ROOT / "data" / "prosqa_test_graph_4_coconut.json", None)
        ok(f"ProsQA dataset loads ({len(samples)} samples)")
    except Exception as exc:
        fail(f"dataset load: {exc}")
        errors += 1

    print("\nNext steps:")
    if not checkpoint.is_file():
        print("  1. python scripts/download_checkpoint.py")
        print("  2. python scripts/smoke_test.py --device cpu")
        print("  3. python run_confidence_fallback.py --device cuda --seed 99")
    else:
        print("  python scripts/smoke_test.py --device cpu")
        print("  python run_confidence_fallback.py --device cuda --seed 99")

    if errors:
        print(f"\n{errors} required item(s) missing.")
        return 1
    print("\nEnvironment looks ready (except checkpoint if warned above).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
