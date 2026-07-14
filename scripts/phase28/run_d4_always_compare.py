#!/usr/bin/env python3
"""D4 · 高置信也比对：struct 与 knn 答案不同则比置信度（全量）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "phase25"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fallback_eval import run_knn_path, setup_fallback_stack
from _phase28_common import CAP, GAP_INDICES, MIN_N, timed_run, write_phase28_result
from boundary_budget import blind_depth
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase23._phase23_common import is_deployable_mvp, is_eps_deployable, load_full_dataset, timing_metrics
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import _rich_step_features, extract_latent_hidden, first_correct_step


@torch.no_grad()
def eval_always_compare(head, model, tokenizer, samples, *, device, seed, profile,
                        struct_floor, knn_floor, knn_thr, pfn):
    head.eval()
    correct = disagree = 0
    gap_hit = 0
    stop_ns, fcs = [], []
    for idx, sample in enumerate(samples):
        expected = expected_answer(sample, profile)
        n0 = max(MIN_N, min(CAP, struct_floor(sample)))
        ps = predict_at_n(model, tokenizer, sample, n0, device, seed=seed + idx * 31, eval_profile=profile)
        prompt0 = build_eval_prompt(sample, n0, seed=seed + idx * 31,
            choice_order=profile.choice_order, shuffle_edges=profile.prompt_mode != "fixed_edges")
        ids0 = torch.tensor([tokenizer.encode(prompt0, add_special_tokens=False)], device=device)
        hid0 = extract_latent_hidden(model, ids0, pass_idx=n0 - 1).to(device)
        ab0, st0, ch0 = _rich_step_features(ps, "", 1)
        prob_s = torch.sigmoid(head(
            hid0.unsqueeze(0), torch.tensor([n0], device=device),
            torch.tensor([ab0], device=device), torch.tensor([st0], device=device),
            torch.tensor([ch0], device=device),
        )).item()
        pk, nk, _, prob_k = run_knn_path(
            head, model, tokenizer, sample, device=device, seed=seed + idx * 31,
            profile=profile, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
        )
        fc, _ = first_correct_step(
            model, tokenizer, sample, cap=CAP, device=device, seed=seed + idx * 31,
            predict_fn=pfn, expected_fn=expected_answer, eval_profile=profile,
        )
        if ps != pk:
            disagree += 1
            final, stop_n = (pk, nk) if prob_k >= prob_s else (ps, n0)
        else:
            final, stop_n = ps, n0
        if final == expected:
            correct += 1
            if idx in GAP_INDICES:
                gap_hit += 1
        stop_ns.append(stop_n)
        fcs.append(fc)
    row = {
        "accuracy": round(correct / len(samples), 4),
        "total": len(samples),
        "disagree_count": disagree,
        "gap_hit": gap_hit,
        "mean_stop_n": round(sum(stop_ns) / len(samples), 2),
        "params": {"mode": "always_compare", "uses_oracle": False},
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
        head, sf, kf, kt, pfn = setup_fallback_stack(model, tokenizer, device, profile)
        row = eval_always_compare(
            head, model, tokenizer, full, device=device, seed=99, profile=profile,
            struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn,
        )
        return {
            "full_419": row,
            "gap_indices": list(GAP_INDICES),
            "baseline_p25": 0.9523,
            "mentor_brief": (
                f"D4 全量比对：acc {row['accuracy']:.1%} "
                f"分歧 {row['disagree_count']} gap_hit {row['gap_hit']}/3。"
            ),
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "d4_always_compare", "D4 · 全量比对", device=args.device)
    write_phase28_result("d4_always_compare", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
