#!/usr/bin/env python3
"""C5 · 多种子集成投票：structure_d 三种子多数表决。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase27_common import CAP, MIN_N, SEED, VOTE_SEEDS, load_full_dataset, load_json, timed_run, write_phase27_result
from phase23._phase23_common import timing_metrics
from boundary_budget import make_structure_budget_fn
from evaluate_coconut import expected_answer
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import first_correct_step


@torch.no_grad()
def eval_multi_seed_vote(model, tokenizer, samples, *, device, profile, seeds):
    struct = make_structure_budget_fn(min_n=MIN_N, cap=CAP)
    correct = 0
    stop_ns, fcs = [], []
    pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
    for idx, sample in enumerate(samples):
        expected = expected_answer(sample, profile)
        n = max(MIN_N, min(CAP, struct(sample)))
        votes = Counter()
        for s in seeds:
            votes[predict_at_n(model, tokenizer, sample, n, device, seed=s + idx * 31, eval_profile=profile)] += 1
        pred = votes.most_common(1)[0][0]
        fc, _ = first_correct_step(
            model, tokenizer, sample, cap=CAP, device=device, seed=SEED + idx * 31,
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
        "params": {"seeds": list(seeds), "mode": "multi_seed_vote", "uses_oracle": False},
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
        row = eval_multi_seed_vote(model, tokenizer, full, device=device, profile=profile, seeds=VOTE_SEEDS)
        p25 = load_json("phase25/a1_fallback_finetune_latest.json")
        return {
            "full_419": row,
            "baseline_champion": (p25.get("best_thr_row") or {}).get("accuracy"),
            "insight": "不增训练：多种子投票能否无 head 抬 acc？",
            "mentor_brief": f"C5 多种子投票 acc {row['accuracy']:.1%}（3×前向）。",
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "c5_multi_seed_vote", "C5 · 多种子投票", device=args.device)
    write_phase27_result("c5_multi_seed_vote", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
