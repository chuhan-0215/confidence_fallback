#!/usr/bin/env python3
"""Y1 · 4跳预算专项：d=4 用 n∈{3,4,d-1}，测 acc 与 ε=1 tradeoff。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase24_common import (
    CAP, MIN_N, SEED, filter_by_hop, is_deployable_mvp, is_eps_deployable,
    load_full_dataset, timed_run, timing_metrics, write_phase24_result,
)
from boundary_budget import blind_depth, make_d_minus_one_budget_fn, make_structure_budget_fn
from evaluate_coconut import expected_answer
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import first_correct_step


def make_hop4_budget_fn(four_fn, *, min_n: int, cap: int):
    def _fn(sample: dict) -> int:
        d = blind_depth(sample)
        if d < 4:
            return max(min_n, min(cap, d))
        return four_fn(sample)
    return _fn


@torch.no_grad()
def eval_budget(model, tokenizer, samples, budget_fn, device, seed, profile, label: str):
    correct = 0
    stop_ns, fcs = [], []
    for idx, sample in enumerate(samples):
        n = budget_fn(sample)
        pred = predict_at_n(model, tokenizer, sample, n, device, seed=seed + idx * 31, eval_profile=profile)
        expected = expected_answer(sample, profile)
        fc, _ = first_correct_step(
            model, tokenizer, sample, cap=CAP, device=device, seed=seed + idx * 31,
            predict_fn=lambda s, nn, ss: predict_at_n(model, tokenizer, s, nn, device, seed=ss, eval_profile=profile),
            expected_fn=expected_answer, eval_profile=profile,
        )
        if pred == expected:
            correct += 1
        stop_ns.append(n)
        fcs.append(fc)
    row = {
        "accuracy": round(correct / len(samples), 4),
        "total": len(samples),
        "mean_stop_n": round(sum(stop_ns) / len(samples), 2),
        "params": {"uses_oracle": False, "strategy": label, "single_forward": True},
    }
    row.update(timing_metrics(stop_ns, fcs))
    row["deployable_mvp"] = is_deployable_mvp(row)
    row["eps_deployable"] = is_eps_deployable(row)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        full = load_full_dataset()
        hop4 = filter_by_hop(full, 4)
        struct = make_structure_budget_fn(min_n=MIN_N, cap=CAP)
        variants = [
            ("hop4_n4_baseline", make_hop4_budget_fn(lambda s: max(MIN_N, min(CAP, 4)), min_n=MIN_N, cap=CAP)),
            ("hop4_n3", make_hop4_budget_fn(lambda s: 3, min_n=MIN_N, cap=CAP)),
            ("hop4_d_minus_one", make_hop4_budget_fn(
                lambda s: max(MIN_N, min(CAP, blind_depth(s) - 1)), min_n=MIN_N, cap=CAP)),
            ("global_d_minus_one", make_d_minus_one_budget_fn(min_n=MIN_N, cap=CAP)),
            ("global_structure_d", struct),
        ]
        results = []
        for label, bfn in variants:
            for split_name, samples in (("full_419", full), ("hop4_only", hop4)):
                row = eval_budget(model, tokenizer, samples, bfn, device, SEED, profile, label)
                row["split"] = split_name
                results.append(row)

        full_rows = [r for r in results if r["split"] == "full_419"]
        hop4_rows = [r for r in results if r["split"] == "hop4_only"]
        best_full = max(full_rows, key=lambda r: (r["accuracy"], r.get("timing_eps1") or 0))
        best_hop4_eps = max(hop4_rows, key=lambda r: (r.get("timing_eps1") or 0, r["accuracy"]))
        bl = next(r for r in full_rows if r["params"]["strategy"] == "global_structure_d")
        return {
            "results": results,
            "best_full": best_full,
            "best_hop4_eps1": best_hop4_eps,
            "baseline_structure_d": bl,
            "full_419": best_full,
            "insight": "P23 Z2：4跳 ε=1 仅 46%；专项 n=3/d-1 能否抬 4跳 ε 且保全量 acc。",
            "mentor_brief": (
                f"Y1 4跳专项：全量最优 {best_full['params']['strategy']} acc {best_full['accuracy']:.1%} "
                f"ε=1 {best_full.get('timing_eps1', 0):.1%}；4跳 ε 最优 {best_hop4_eps['params']['strategy']} "
                f"{best_hop4_eps.get('timing_eps1', 0):.1%}。"
            ),
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "y1_hop4_budget", "Y1 · 4跳预算", device=args.device)
    write_phase24_result("y1_hop4_budget", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
