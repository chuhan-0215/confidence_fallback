#!/usr/bin/env python3
"""B1 · 跳数分阈值：3跳永不回退，4跳用更低 thr。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "phase25"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase26_common import load_full_dataset, load_json, timed_run, write_phase26_result
from _fallback_eval import run_knn_path, setup_fallback_stack
from boundary_budget import blind_depth
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import _rich_step_features, extract_latent_hidden, first_correct_step


@torch.no_grad()
def eval_hop_adaptive(head, model, tokenizer, samples, *, device, seed, profile,
                      struct_floor, knn_floor, knn_thr, pfn, hop4_thr: float):
    head.eval()
    correct = fallback_count = 0
    hop3_fb = hop4_fb = 0
    for idx, sample in enumerate(samples):
        expected = expected_answer(sample, profile)
        hop = blind_depth(sample)
        n0 = max(3, min(8, struct_floor(sample)))
        prompt0 = build_eval_prompt(sample, n0, seed=seed + idx * 31,
            choice_order=profile.choice_order, shuffle_edges=profile.prompt_mode != "fixed_edges")
        ids0 = torch.tensor([tokenizer.encode(prompt0, add_special_tokens=False)], device=device)
        hid0 = extract_latent_hidden(model, ids0, pass_idx=n0 - 1).to(device)
        pred0 = predict_at_n(model, tokenizer, sample, n0, device, seed=seed + idx * 31, eval_profile=profile)
        ab0, st0, ch0 = _rich_step_features(pred0, "", 1)
        prob0 = torch.sigmoid(head(
            hid0.unsqueeze(0), torch.tensor([n0], device=device),
            torch.tensor([ab0], device=device), torch.tensor([st0], device=device),
            torch.tensor([ch0], device=device),
        )).item()
        do_fallback = hop >= 4 and prob0 < hop4_thr
        if do_fallback:
            fallback_count += 1
            hop4_fb += 1
            final, _ = run_knn_path(
                head, model, tokenizer, sample, device=device, seed=seed + idx * 31,
                profile=profile, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            )[:2]
        else:
            final = pred0
        if final == expected:
            correct += 1
    return {
        "accuracy": round(correct / len(samples), 4),
        "total": len(samples),
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / len(samples), 4),
        "hop4_fallback": hop4_fb,
        "params": {"hop4_thr": hop4_thr, "hop3_fallback": False, "mode": "hop_adaptive"},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        full = load_full_dataset()
        head, struct_floor, knn_floor, knn_thr, pfn = setup_fallback_stack(model, tokenizer, device, profile)
        sweep = []
        for thr in (0.38, 0.40, 0.42, 0.45, 0.48):
            row = eval_hop_adaptive(
                head, model, tokenizer, full, device=device, seed=99, profile=profile,
                struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
                hop4_thr=thr,
            )
            sweep.append(row)
        best = max(sweep, key=lambda r: (r["accuracy"], -r["fallback_rate"]))
        p25 = load_json("phase25/a1_fallback_finetune_latest.json")
        return {
            "sweep": sweep,
            "best": best,
            "full_419": best,
            "baseline_p25_global": (p25.get("best_thr_row") or {}).get("accuracy"),
            "insight": "P25 A2：4跳专项回退 95.0% < 全局 95.2%；试 3跳不回退+4跳低 thr。",
            "mentor_brief": (
                f"B1 跳数分阈：4跳 thr={best['params']['hop4_thr']} "
                f"acc {best['accuracy']:.1%} fallback {best['fallback_rate']:.1%}。"
            ),
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "b1_hop_adaptive_thr", "B1 · 跳数分阈", device=args.device)
    write_phase26_result("b1_hop_adaptive_thr", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
