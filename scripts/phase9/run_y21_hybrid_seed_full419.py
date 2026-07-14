#!/usr/bin/env python3
"""Y21 · hybrid seed 鲁棒性（全量 419）。"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase9_common import DEPLOY_DEFAULTS, evaluate_hybrid, load_full_dataset, load_model_bundle, timed_run, write_phase9_result

SEEDS = [0, 1, 2, 3, 4, 42, 99]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cap", type=int, default=DEPLOY_DEFAULTS["cap"])
    ap.add_argument("--min-n", type=int, default=DEPLOY_DEFAULTS["min_n"])
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        dataset = load_full_dataset()
        accs, probes = [], []
        for seed in SEEDS:
            row = evaluate_hybrid(
                model, tokenizer, dataset, cap=args.cap, min_n=args.min_n,
                device=device, seed=seed, profile=profile,
            )
            accs.append(row["accuracy"])
            probes.append(row["mean_forward_probes"])
        return {
            "robustness": {
                "hybrid_n_eq_d_then_soft_floor": {
                    "seeds": SEEDS,
                    "accuracies": accs,
                    "mean_acc": round(statistics.mean(accs), 4),
                    "stdev_acc": round(statistics.pstdev(accs), 4) if len(accs) > 1 else 0.0,
                    "min_acc": round(min(accs), 4),
                    "max_acc": round(max(accs), 4),
                    "mean_probes": round(statistics.mean(probes), 3),
                }
            },
            "eval_split": "full",
            "sample_count": len(dataset),
            "min_n": args.min_n,
            "device": str(device),
            "insight": "全量 419 seed 方差；部署报告应写 min_acc 下限。",
        }

    path = timed_run(run_body, "y21_hybrid_seed_full419", "Y21 · hybrid 全量 seed", device=args.device)
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    write_phase9_result("y21_hybrid_seed_full419", data)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
