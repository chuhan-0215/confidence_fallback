#!/usr/bin/env python3
"""Y12 · two_probe hybrid seed 鲁棒性（同 Y5 口径）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase6"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase7_common import load_model_bundle, load_test_split, timed_run, write_phase7_result
from _hybrid_eval import evaluate_two_probe_hybrid

SEEDS = [0, 1, 2, 3, 4, 42, 99]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--min-n", type=int, default=2)
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        test_set = load_test_split()
        accs = []
        probes = []
        for seed in SEEDS:
            row = evaluate_two_probe_hybrid(
                model, tokenizer, test_set,
                cap=args.cap, min_n=args.min_n, device=device, seed=seed, profile=profile,
            )
            accs.append(row["accuracy"])
            probes.append(row["mean_forward_probes"])
        import statistics
        return {
            "robustness": {
                "two_probe_n_eq_d_then_fc": {
                    "seeds": SEEDS,
                    "accuracies": accs,
                    "mean_acc": round(statistics.mean(accs), 4),
                    "stdev_acc": round(statistics.pstdev(accs), 4) if len(accs) > 1 else 0.0,
                    "min_acc": round(min(accs), 4),
                    "max_acc": round(max(accs), 4),
                    "mean_probes": round(statistics.mean(probes), 3),
                }
            },
            "eval_split": "test_40pct",
            "sample_count": len(test_set),
            "min_n": args.min_n,
            "device": str(device),
            "insight": "Y8 hybrid 是否 seed-stable；min_acc≥95% 则升为默认部署。",
        }

    path = timed_run(run_body, "y12_hybrid_seed_robustness", "Y12 · hybrid seed 鲁棒", device=args.device)
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    write_phase7_result("y12_hybrid_seed_robustness", data)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
