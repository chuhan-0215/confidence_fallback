#!/usr/bin/env python3
"""Z4 · 4跳不对称规则：d≥4 且候选深度不同 → budget=3，否则 structure_d。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase23_common import (
    CAP, MIN_N, SEED, is_deployable_mvp, is_eps_deployable,
    load_full_dataset, load_splits, timed_run, timing_metrics, write_phase23_result,
)
from boundary_budget import make_asymmetry_rule_budget_fn, make_structure_budget_fn
from evaluate_coconut import expected_answer
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import first_correct_step


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
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        full = load_full_dataset()
        variants = [
            ("structure_d", make_structure_budget_fn(min_n=MIN_N, cap=CAP)),
            ("asymmetry_rule", make_asymmetry_rule_budget_fn(min_n=MIN_N, cap=CAP)),
        ]
        results = []
        for label, bfn in variants:
            row = eval_budget(model, tokenizer, full, bfn, device, SEED, profile, label)
            row["deployable_mvp"] = is_deployable_mvp(row)
            row["eps_deployable"] = is_eps_deployable(row)
            results.append(row)

        best = max(results, key=lambda r: (r["accuracy"], r.get("timing_eps1") or 0))
        bl = next(r for r in results if r["params"]["strategy"] == "structure_d")
        return {
            "results": results,
            "best": best,
            "full_419": best,
            "baseline_structure_d": bl,
            "insight": "4跳不对称专项：asymmetry_rule 能否 beat structure_d 93.6%？",
            "mentor_brief": (
                f"Z4 不对称：最优 {best['params']['strategy']} acc {best['accuracy']:.1%}；"
                f"structure_d {bl['accuracy']:.1%}。"
            ),
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "z4_asymmetry_budget", "Z4 · 不对称预算", device=args.device)
    write_phase23_result("z4_asymmetry_budget", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
