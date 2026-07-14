#!/usr/bin/env python3
"""Quick smoke test on a few ProsQA samples (no full 419 run)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase25"))
sys.path.insert(0, str(ROOT / "scripts" / "phase32"))
sys.path.insert(0, str(ROOT / "model"))

from _fallback_eval import eval_confidence_fallback, setup_fallback_stack  # noqa: E402
from phase23._phase23_common import load_full_dataset  # noqa: E402
from phase32._phase32_common import TRANSFER_THR, m2_head_ready  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke test confidence_fallback on N samples.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--n", type=int, default=5, help="number of samples (default 5)")
    ap.add_argument("--tau", type=float, default=TRANSFER_THR)
    args = ap.parse_args()

    if not m2_head_ready():
        raise SystemExit("Missing results/phase10/m2_enough_stop_head.pt")

    model, tokenizer, device, profile = load_model_bundle(args.device)
    head, struct_floor, knn_floor, knn_thr, pfn = setup_fallback_stack(
        model, tokenizer, device, profile
    )
    samples = load_full_dataset()[: max(1, args.n)]
    row = eval_confidence_fallback(
        head,
        model,
        tokenizer,
        samples,
        device=device,
        seed=args.seed,
        profile=profile,
        struct_floor=struct_floor,
        knn_floor=knn_floor,
        knn_thr=knn_thr,
        pfn=pfn,
        fallback_thr=args.tau,
    )
    print(
        json.dumps(
            {
                "smoke_test": True,
                "n_samples": len(samples),
                "accuracy": row["accuracy"],
                "fallback_rate": row.get("fallback_rate"),
                "mean_stop_n": row.get("mean_stop_n"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
