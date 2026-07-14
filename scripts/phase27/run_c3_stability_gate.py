#!/usr/bin/env python3
"""C3 · 稳定性门控全量：答案连续稳定 k 步即停，无 Stop Head。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase27_common import CAP, SEED, load_full_dataset, load_json, timed_run, write_phase27_result
from boundary_budget import blind_depth, make_structure_budget_fn
from evaluate_coconut import expected_answer
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import _rich_step_features, first_correct_step


@torch.no_grad()
def eval_stability_gate(model, tokenizer, samples, *, cap, streak_min, min_n, device, seed, profile):
    correct = stop_sum = 0
    stop_ns, fcs = [], []
    struct = make_structure_budget_fn(min_n=min_n, cap=cap)
    pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
    for idx, sample in enumerate(samples):
        expected = expected_answer(sample, profile)
        floor_n = max(min_n, min(cap, struct(sample)))
        fc, preds = first_correct_step(
            model, tokenizer, sample, cap=cap, device=device, seed=seed + idx * 31,
            predict_fn=pfn, expected_fn=expected_answer, eval_profile=profile,
        )
        stop_n = cap
        final = preds.get(cap, "")
        prev, streak = "", 0
        for n in range(1, cap + 1):
            final = preds[n]
            _, streak, _ = _rich_step_features(final, prev, streak)
            prev = final
            stop_n = n
            if n >= floor_n and streak >= streak_min:
                break
        if final == expected:
            correct += 1
        stop_sum += stop_n
        stop_ns.append(stop_n)
        fcs.append(fc)
    from phase23._phase23_common import timing_metrics, is_deployable_mvp, is_eps_deployable
    row = {
        "accuracy": round(correct / len(samples), 4),
        "total": len(samples),
        "mean_stop_n": round(stop_sum / len(samples), 2),
        "params": {"streak_min": streak_min, "min_n": min_n, "mode": "stability_gate", "uses_oracle": False},
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
        sweep = []
        for streak in (2, 3):
            for min_n in (2, 3):
                sweep.append(eval_stability_gate(
                    model, tokenizer, full, cap=8, streak_min=streak, min_n=min_n,
                    device=device, seed=SEED, profile=profile,
                ))
        best = max(sweep, key=lambda r: (r["accuracy"], r.get("timing_eps1") or 0))
        return {
            "sweep": sweep,
            "best": best,
            "full_419": best,
            "insight": "W4 CPU 证伪 timing=0%；全量 419 再验证稳定性早停范式。",
            "mentor_brief": f"C3 稳定门控最优 streak={best['params']['streak_min']} acc {best['accuracy']:.1%}。",
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "c3_stability_gate", "C3 · 稳定门控", device=args.device)
    write_phase27_result("c3_stability_gate", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
