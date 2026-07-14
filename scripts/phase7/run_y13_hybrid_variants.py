#!/usr/bin/env python3
"""Y13 · hybrid 变体：first probe 与 retry 策略组合。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _hybrid_eval import evaluate_hybrid
from _phase7_common import load_model_bundle, load_test_split, timed_run, write_phase7_result

VARIANTS = [
    ("n_eq_d", "soft_floor"),
    ("n_d_minus1", "soft_floor"),
    ("n_eq_d", "hop_split"),
    ("n_eq_3", "soft_floor"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--min-n", type=int, default=2)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        test_set = load_test_split()
        rows = []
        for first_mode, retry_mode in VARIANTS:
            row = evaluate_hybrid(
                model,
                tokenizer,
                test_set,
                first_mode=first_mode,
                retry_mode=retry_mode,
                cap=args.cap,
                min_n=args.min_n,
                device=device,
                seed=args.seed,
                profile=profile,
            )
            rows.append(row)
        rows.sort(key=lambda r: (-r["accuracy"], r["mean_forward_probes"]))
        return {
            "variants": rows,
            "eval_split": "test_40pct",
            "sample_count": len(test_set),
            "seed": args.seed,
            "device": str(device),
            "insight": "对比 first=n=d vs n=d-1；retry 用 hop_split 是否更稳。",
        }

    path = timed_run(run_body, "y13_hybrid_variants", "Y13 · hybrid 变体", device=args.device)
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    write_phase7_result("y13_hybrid_variants", data)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
