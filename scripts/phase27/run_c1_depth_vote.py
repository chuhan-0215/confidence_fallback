#!/usr/bin/env python3
"""C1 · 题深邻域投票：n∈{d-1,d,d+1} 多数表决，无 Stop Head。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase27_common import CAP, MIN_N, SEED, load_full_dataset, load_json, timed_run, write_phase27_result
from phase23._phase23_common import is_deployable_mvp, is_eps_deployable, timing_metrics
from boundary_budget import blind_depth
from evaluate_coconut import expected_answer
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import first_correct_step


@torch.no_grad()
def eval_depth_vote(model, tokenizer, samples, *, device, seed, profile):
    correct = 0
    stop_ns, fcs = [], []
    pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
    for idx, sample in enumerate(samples):
        expected = expected_answer(sample, profile)
        d = blind_depth(sample)
        cands = sorted({max(MIN_N, min(CAP, x)) for x in (d - 1, d, d + 1)})
        votes = Counter()
        for n in cands:
            votes[predict_at_n(model, tokenizer, sample, n, device, seed=seed + idx * 31 + n, eval_profile=profile)] += 1
        pred = votes.most_common(1)[0][0]
        fc, _ = first_correct_step(
            model, tokenizer, sample, cap=CAP, device=device, seed=seed + idx * 31,
            predict_fn=pfn, expected_fn=expected_answer, eval_profile=profile,
        )
        stop_n = max(MIN_N, min(CAP, d))
        if pred == expected:
            correct += 1
        stop_ns.append(stop_n)
        fcs.append(fc)
    row = {
        "accuracy": round(correct / len(samples), 4),
        "total": len(samples),
        "mean_stop_n": round(sum(stop_ns) / len(samples), 2),
        "params": {"uses_oracle": False, "mode": "depth_vote", "candidates": "d-1,d,d+1"},
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
        row = eval_depth_vote(model, tokenizer, full, device=device, seed=SEED, profile=profile)
        p25 = load_json("phase25/a1_fallback_finetune_latest.json")
        return {
            "full_419": row,
            "baseline_champion": (p25.get("best_thr_row") or {}).get("accuracy"),
            "insight": "不训练、不 head：题深邻域投票能否接近 confidence_fallback？",
            "mentor_brief": f"C1 题深投票 acc {row['accuracy']:.1%} mean_n {row['mean_stop_n']}。",
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "c1_depth_vote", "C1 · 题深投票", device=args.device)
    write_phase27_result("c1_depth_vote", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
