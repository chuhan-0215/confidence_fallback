#!/usr/bin/env python3
"""B3 · 冠军 vs 简单主推：五种子对照终表。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "phase25"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase26_common import ROBUST_SEEDS, load_full_dataset, load_json, stats, timed_run, write_phase26_result
from _fallback_eval import eval_confidence_fallback, setup_fallback_stack
from boundary_budget import make_structure_budget_fn
from evaluate_coconut import expected_answer
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n


@torch.no_grad()
def eval_structure_d(model, tokenizer, samples, struct_floor, device, seed, profile):
    correct = 0
    for idx, sample in enumerate(samples):
        n = max(3, min(8, struct_floor(sample)))
        pred = predict_at_n(model, tokenizer, sample, n, device, seed=seed + idx * 31, eval_profile=profile)
        if pred == expected_answer(sample, profile):
            correct += 1
    return round(correct / len(samples), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        full = load_full_dataset()
        head, struct_floor, knn_floor, knn_thr, pfn = setup_fallback_stack(model, tokenizer, device, profile)
        struct_fn = make_structure_budget_fn(min_n=3, cap=8)
        rows = []
        for seed in ROBUST_SEEDS:
            sd = eval_structure_d(model, tokenizer, full, struct_fn, device, seed, profile)
            fb = eval_confidence_fallback(
                head, model, tokenizer, full, device=device, seed=seed, profile=profile,
                struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
                fallback_thr=0.48,
            )
            rows.append({
                "seed": seed,
                "structure_d": sd,
                "fallback": fb["accuracy"],
                "delta": round(fb["accuracy"] - sd, 4),
                "fallback_rate": fb["fallback_rate"],
            })
        sd_stats = stats([r["structure_d"] for r in rows])
        fb_stats = stats([r["fallback"] for r in rows])
        win_count = sum(1 for r in rows if r["delta"] > 0)
        return {
            "rows": rows,
            "structure_d_stats": sd_stats,
            "fallback_stats": fb_stats,
            "fallback_wins_seeds": win_count,
            "full_419": {"accuracy": max(r["fallback"] for r in rows), "params": {"fallback_thr": 0.48}},
            "insight": "P25 A1：fallback 五种子 μ=93.9% 低于 seed99 峰值；与 structure_d 逐种子对比。",
            "mentor_brief": (
                f"B3 种子对照：fallback μ={fb_stats['mean']:.1%} structure_d μ={sd_stats['mean']:.1%}；"
                f"fallback 赢 {win_count}/5 种子。"
            ),
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "b3_champion_seed_table", "B3 · 种子对照", device=args.device)
    write_phase26_result("b3_champion_seed_table", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
