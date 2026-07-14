#!/usr/bin/env python3
"""X3 · 预算错位压力测试：n = blind_depth + Δ，观察准确率如何随 Δ 变化。"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase4_common import load_model_bundle, timed_run  # noqa: E402
from boundary_budget import blind_depth  # noqa: E402
from evaluate_coconut import expected_answer, load_dataset  # noqa: E402
from run_adaptive_stop_experiment import predict_at_n  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--deltas", default="-2,-1,0,1,2")
    args = ap.parse_args()
    deltas = [int(x) for x in args.deltas.split(",")]

    model, tokenizer, device, profile = load_model_bundle(args.device)
    dataset = load_dataset(ROOT / "data" / "prosqa_test_graph_4_coconut.json", args.max_samples)

    by_delta = {}
    by_depth_delta = defaultdict(lambda: {"correct": 0, "total": 0})

    for delta in deltas:
        correct = total = 0
        for idx, sample in enumerate(dataset):
            d = blind_depth(sample)
            n = max(1, min(args.cap, d + delta))
            pred = predict_at_n(
                model, tokenizer, sample, n, device, seed=42 + idx, eval_profile=profile
            )
            exp = expected_answer(sample, profile)
            total += 1
            if pred == exp:
                correct += 1
            key = f"d{d}_delta{delta:+d}"
            by_depth_delta[key]["total"] += 1
            if pred == exp:
                by_depth_delta[key]["correct"] += 1
        by_delta[str(delta)] = {
            "accuracy": round(correct / total, 4) if total else 0,
            "correct": correct,
            "total": total,
            "mean_n_offset": delta,
        }

    heatmap = {
        k: round(v["correct"] / v["total"], 4) if v["total"] else 0
        for k, v in sorted(by_depth_delta.items())
    }

    path = timed_run(
        lambda: {
            "by_delta": by_delta,
            "heatmap_samples": heatmap,
            "sample_count": len(dataset),
            "insight": "Δ=0 应接近 auto_route；Δ<0 测欠思考；Δ>0 测过思考/噪声",
        },
        "x3_misbudget_stress",
        "X3 · 预算错位压力测试",
        device=args.device,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
