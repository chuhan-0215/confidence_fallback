#!/usr/bin/env python3
"""Y14 · 部署 Pareto v2（seed=99 统一口径，含 two_probe）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase5"))
sys.path.insert(0, str(ROOT / "scripts" / "phase6"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase7_common import load_model_bundle, load_test_split, timed_run, write_phase7_result
from boundary_budget import blind_depth
from evaluate_coconut import expected_answer
from phase5._phase5_common import make_predict_fn
from phase5.run_y4_gap_upfront_rules import rule_n_d
from run_adaptive_stop_experiment import predict_at_n
from phase6.run_y8_two_probe_hybrid import evaluate_two_probe_hybrid
from run_auto_submit_experiment import evaluate_policy, make_policies
from stop_head_tracks import evaluate_hop_split_first_correct_stop, evaluate_soft_floor_first_correct_stop


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
        predict_fn = make_predict_fn(model, tokenizer, device, profile)
        policies = make_policies(cap=args.cap)
        rows = []

        for name in ("fixed_3", "auto_route"):
            r = evaluate_policy(model, tokenizer, test_set, policies[name], device, cap=args.cap, eval_profile=profile)
            rows.append({"strategy": name, "accuracy": r["accuracy"], "mean_forward_probes": 1.0})

        correct_nd = 0
        for idx, sample in enumerate(test_set):
            n = rule_n_d(sample, args.cap)
            pred = predict_at_n(
                model, tokenizer, sample, n, device,
                seed=args.seed + idx * 31, eval_profile=profile,
            )
            if pred == expected_answer(sample, profile):
                correct_nd += 1
        rows.append({
            "strategy": "n_eq_d",
            "accuracy": round(correct_nd / len(test_set), 4),
            "mean_forward_probes": 1.0,
        })

        for label, fn in (
            ("soft_floor_fc", evaluate_soft_floor_first_correct_stop),
            ("hop_split_fc", evaluate_hop_split_first_correct_stop),
        ):
            r = fn(
                model, tokenizer, test_set, cap=args.cap, min_n=args.min_n,
                device=device, seed=args.seed, predict_fn=predict_fn,
                expected_fn=expected_answer, eval_profile=profile,
            )
            rows.append({
                "strategy": label,
                "accuracy": r["accuracy"],
                "mean_forward_probes": r.get("mean_forward_probes", r.get("mean_probes", 3.0)),
                "timing_acc": r.get("timing_acc"),
            })

        hybrid = evaluate_two_probe_hybrid(
            model, tokenizer, test_set, cap=args.cap, min_n=args.min_n,
            device=device, seed=args.seed, profile=profile,
        )
        rows.append({
            "strategy": hybrid["strategy"],
            "accuracy": hybrid["accuracy"],
            "mean_forward_probes": hybrid["mean_forward_probes"],
            "one_probe_success_rate": hybrid.get("one_probe_success_rate"),
        })

        rows.sort(key=lambda x: (-x["accuracy"], x["mean_forward_probes"]))
        return {
            "pareto": rows,
            "eval_split": "test_40pct",
            "sample_count": len(test_set),
            "seed": args.seed,
            "min_n": args.min_n,
            "device": str(device),
            "insight": "seed=99 统一口径；确认 two_probe 是否在 acc-probe 前沿。",
        }

    path = timed_run(run_body, "y14_deploy_pareto_v2", "Y14 · 部署 Pareto v2", device=args.device)
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    write_phase7_result("y14_deploy_pareto_v2", data)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
