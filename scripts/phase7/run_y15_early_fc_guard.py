#!/usr/bin/env python3
"""Y15 · early_fc_guard hybrid：d≥4 时忽略 fc=1，再与 Y8 基线对比。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _hybrid_eval import evaluate_hybrid
from _phase7_common import load_model_bundle, load_test_split, timed_run, write_phase7_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--min-n", type=int, default=2)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--guard-depth", type=int, default=4)
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        test_set = load_test_split()
        baseline = evaluate_hybrid(
            model,
            tokenizer,
            test_set,
            first_mode="n_eq_d",
            retry_mode="soft_floor",
            cap=args.cap,
            min_n=args.min_n,
            device=device,
            seed=args.seed,
            profile=profile,
        )
        guarded = evaluate_hybrid(
            model,
            tokenizer,
            test_set,
            first_mode="n_eq_d",
            retry_mode="early_fc_guard",
            cap=args.cap,
            min_n=args.min_n,
            device=device,
            seed=args.seed,
            profile=profile,
            guard_depth=args.guard_depth,
        )
        return {
            "comparison": [baseline, guarded],
            "eval_split": "test_40pct",
            "sample_count": len(test_set),
            "seed": args.seed,
            "guard_depth": args.guard_depth,
            "device": str(device),
            "insight": "针对 Y6 early_fc_deep 失败；若 acc↑ 则合并进生产 hybrid。",
        }

    path = timed_run(run_body, "y15_early_fc_guard", "Y15 · early_fc_guard", device=args.device)
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    write_phase7_result("y15_early_fc_guard", data)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
