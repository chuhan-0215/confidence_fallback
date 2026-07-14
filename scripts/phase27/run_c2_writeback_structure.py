#!/usr/bin/env python3
"""C2 · 写回消融 + structure_d：第 4 步后停写回，按题深单次前向。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase27_common import CAP, MIN_N, SEED, load_full_dataset, timed_run, write_phase27_result
from phase23._phase23_common import is_deployable_mvp, timing_metrics
from boundary_budget import blind_depth, make_structure_budget_fn
from coconut_feedback import apply_feedback_config, default_feedback_strategies
from evaluate_coconut import expected_answer
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import first_correct_step


@torch.no_grad()
def eval_writeback_struct(model, tokenizer, samples, *, device, seed, profile, feedback_id: str):
    strat = next(s for s in default_feedback_strategies() if s["id"] == feedback_id)
    apply_feedback_config(model, strat)
    struct = make_structure_budget_fn(min_n=MIN_N, cap=CAP)
    correct = 0
    stop_ns, fcs = [], []
    pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
    for idx, sample in enumerate(samples):
        expected = expected_answer(sample, profile)
        n = max(MIN_N, min(CAP, struct(sample)))
        pred = predict_at_n(model, tokenizer, sample, n, device, seed=seed + idx * 31, eval_profile=profile)
        fc, _ = first_correct_step(
            model, tokenizer, sample, cap=CAP, device=device, seed=seed + idx * 31,
            predict_fn=pfn, expected_fn=expected_answer, eval_profile=profile,
        )
        if pred == expected:
            correct += 1
        stop_ns.append(n)
        fcs.append(fc)
    row = {
        "accuracy": round(correct / len(samples), 4),
        "total": len(samples),
        "mean_stop_n": round(sum(stop_ns) / len(samples), 2),
        "params": {"uses_oracle": False, "mode": "writeback_structure_d", "feedback_id": feedback_id},
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
        results = []
        for fid in ("baseline", "zero_after4", "residual_zero4"):
            results.append(eval_writeback_struct(
                model, tokenizer, full, device=device, seed=SEED, profile=profile, feedback_id=fid,
            ))
        best = max(results, key=lambda r: r["accuracy"])
        return {
            "results": results,
            "best": best,
            "full_419": best,
            "insight": "Exp9 写回消融 + 题深路由：改推理不改权重能否抬 acc？",
            "mentor_brief": f"C2 写回+ablation 最优 {best['params']['feedback_id']} acc {best['accuracy']:.1%}。",
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "c2_writeback_structure", "C2 · 写回消融", device=args.device)
    write_phase27_result("c2_writeback_structure", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
