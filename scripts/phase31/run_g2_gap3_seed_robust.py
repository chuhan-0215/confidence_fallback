#!/usr/bin/env python3
"""G2 · upper=0.551 gap3 方案五种子稳健性。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "phase25"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fallback_eval import eval_confidence_fallback, run_knn_path, setup_fallback_stack
from _phase31_common import CAP, GAP_INDICES, MIN_N, ROBUST_SEEDS, load_json, stats, timed_run, write_phase31_result
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase23._phase23_common import load_full_dataset
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import _rich_step_features, extract_latent_hidden, first_correct_step


@torch.no_grad()
def eval_upper0551(head, model, tokenizer, samples, *, device, seed, profile,
                    struct_floor, knn_floor, knn_thr, pfn):
    correct = gap_hit = 0
    for idx, sample in enumerate(samples):
        expected = expected_answer(sample, profile)
        n0 = max(MIN_N, min(CAP, struct_floor(sample)))
        pred0 = predict_at_n(model, tokenizer, sample, n0, device, seed=seed + idx * 31, eval_profile=profile)
        prompt0 = build_eval_prompt(sample, n0, seed=seed + idx * 31,
            choice_order=profile.choice_order, shuffle_edges=profile.prompt_mode != "fixed_edges")
        ids0 = torch.tensor([tokenizer.encode(prompt0, add_special_tokens=False)], device=device)
        hid0 = extract_latent_hidden(model, ids0, pass_idx=n0 - 1).to(device)
        ab0, st0, ch0 = _rich_step_features(pred0, "", 1)
        prob0 = torch.sigmoid(head(
            hid0.unsqueeze(0), torch.tensor([n0], device=device),
            torch.tensor([ab0], device=device), torch.tensor([st0], device=device),
            torch.tensor([ch0], device=device),
        )).item()
        if prob0 < 0.551:
            final, _, _, _ = run_knn_path(
                head, model, tokenizer, sample, device=device, seed=seed + idx * 31,
                profile=profile, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            )
        else:
            final = pred0
        if final == expected:
            correct += 1
            if idx in GAP_INDICES:
                gap_hit += 1
    return round(correct / len(samples), 4), gap_hit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        full = load_full_dataset()
        head, sf, kf, kt, pfn = setup_fallback_stack(model, tokenizer, device, profile)
        gap_rows, champ_rows = [], []
        for seed in ROBUST_SEEDS:
            acc, gh = eval_upper0551(
                head, model, tokenizer, full, device=device, seed=seed, profile=profile,
                struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn,
            )
            gap_rows.append({"seed": seed, "accuracy": acc, "gap_hit": gh})
            fb = eval_confidence_fallback(
                head, model, tokenizer, full, device=device, seed=seed, profile=profile,
                struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, fallback_thr=0.48,
            )
            champ_rows.append(fb["accuracy"])
        gs, cs = stats([r["accuracy"] for r in gap_rows]), stats(champ_rows)
        return {
            "gap3_rows": gap_rows,
            "gap3_stats": gs,
            "champion_stats": cs,
            "gap3_full_rate": sum(1 for r in gap_rows if r["gap_hit"] >= 3),
            "gap_indices": list(GAP_INDICES),
            "mentor_brief": (
                f"G2 upper=0.551：μ={gs['mean']:.1%} max={gs['max']:.1%} "
                f"gap3率={sum(1 for r in gap_rows if r['gap_hit']>=3)}/5；冠军μ={cs['mean']:.1%}。"
            ),
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "g2_gap3_seed_robust", "G2 · gap3种子", device=args.device)
    write_phase31_result("g2_gap3_seed_robust", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
