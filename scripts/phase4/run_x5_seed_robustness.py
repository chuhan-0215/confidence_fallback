#!/usr/bin/env python3
"""X5 · 提示随机性鲁棒性：shuffle_edges 多种子下方差。"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase4_common import load_model_bundle, timed_run  # noqa: E402
from evaluate_coconut import load_dataset  # noqa: E402
from run_auto_submit_experiment import evaluate_policy, make_policies  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=120)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    args = ap.parse_args()
    seed_list = [int(x) for x in args.seeds.split(",")]

    model, tokenizer, device, profile = load_model_bundle(args.device)
    dataset = load_dataset(ROOT / "data" / "prosqa_test_graph_4_coconut.json", args.max_samples)
    policies = make_policies()

    report = {}
    for name in ("fixed_3", "auto_route"):
        accs = []
        for seed in seed_list:
            r = evaluate_policy(
                model,
                tokenizer,
                dataset,
                policies[name],
                device,
                seed=seed,
                eval_profile=profile,
            )
            accs.append(r["accuracy"])
        report[name] = {
            "seeds": seed_list,
            "accuracies": accs,
            "mean": round(statistics.mean(accs), 4),
            "stdev": round(statistics.pstdev(accs), 4) if len(accs) > 1 else 0.0,
            "min": round(min(accs), 4),
            "max": round(max(accs), 4),
        }

    path = timed_run(
        lambda: {"robustness": report, "sample_count": len(dataset)},
        "x5_seed_robustness",
        "X5 · 提示随机性鲁棒性",
        device=args.device,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
