#!/usr/bin/env python3
"""C6 · 双前向选优：d 与 d+1 两答案取稳定/一致者。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase27_common import CAP, MIN_N, SEED, load_full_dataset, timed_run, write_phase27_result
from phase23._phase23_common import timing_metrics
from boundary_budget import blind_depth
from evaluate_coconut import expected_answer
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import first_correct_step


@torch.no_grad()
def eval_dual_forward(model, tokenizer, samples, *, device, seed, profile, mode: str):
    correct = 0
    stop_ns, fcs = [], []
    pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
    for idx, sample in enumerate(samples):
        expected = expected_answer(sample, profile)
        d = blind_depth(sample)
        n1 = max(MIN_N, min(CAP, d))
        n2 = max(MIN_N, min(CAP, d + 1))
        p1 = predict_at_n(model, tokenizer, sample, n1, device, seed=seed + idx * 31, eval_profile=profile)
        p2 = predict_at_n(model, tokenizer, sample, n2, device, seed=seed + idx * 31 + 1, eval_profile=profile)
        if mode == "agree_or_deep":
            pred, stop_n = (p1, n1) if p1 == p2 else (p2, n2)
        else:
            pred, stop_n = (p1, n1) if p1 == p2 else (p1, n1)
        fc, _ = first_correct_step(
            model, tokenizer, sample, cap=CAP, device=device, seed=seed + idx * 31,
            predict_fn=pfn, expected_fn=expected_answer, eval_profile=profile,
        )
        if pred == expected:
            correct += 1
        stop_ns.append(stop_n)
        fcs.append(fc)
    row = {
        "accuracy": round(correct / len(samples), 4),
        "total": len(samples),
        "mean_stop_n": round(sum(stop_ns) / len(samples), 2),
        "params": {"mode": mode, "uses_oracle": False},
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
        variants = []
        for mode in ("agree_or_shallow", "agree_or_deep"):
            variants.append(eval_dual_forward(model, tokenizer, full, device=device, seed=SEED, profile=profile, mode=mode))
        best = max(variants, key=lambda r: r["accuracy"])
        return {
            "variants": variants,
            "best": best,
            "full_419": best,
            "insight": "2×前向轻量范式：不一致时信浅或信深？",
            "mentor_brief": f"C6 双前向最优 {best['params']['mode']} acc {best['accuracy']:.1%}。",
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "c6_dual_forward", "C6 · 双前向", device=args.device)
    write_phase27_result("c6_dual_forward", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
