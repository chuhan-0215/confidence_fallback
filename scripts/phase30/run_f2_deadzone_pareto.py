#!/usr/bin/env python3
"""F2 · 死区阈值 Pareto：细扫 upper_thr，绘 acc vs gap_hit 前沿。"""
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
from _phase30_common import CAP, GAP_INDICES, MIN_N, timed_run, write_phase30_result
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase23._phase23_common import is_deployable_mvp, is_eps_deployable, load_full_dataset, timing_metrics
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import _rich_step_features, extract_latent_hidden, first_correct_step


@torch.no_grad()
def eval_upper_thr(head, model, tokenizer, samples, *, device, seed, profile,
                   struct_floor, knn_floor, knn_thr, pfn, upper_thr):
    head.eval()
    correct = fallback_count = 0
    gap_hit = 0
    stop_ns, fcs = [], []
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
        fc, _ = first_correct_step(
            model, tokenizer, sample, cap=CAP, device=device, seed=seed + idx * 31,
            predict_fn=pfn, expected_fn=expected_answer, eval_profile=profile,
        )
        if prob0 < upper_thr:
            fallback_count += 1
            final, stop_n, _, _ = run_knn_path(
                head, model, tokenizer, sample, device=device, seed=seed + idx * 31,
                profile=profile, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            )
        else:
            final, stop_n = pred0, n0
        if final == expected:
            correct += 1
            if idx in GAP_INDICES:
                gap_hit += 1
        stop_ns.append(stop_n)
        fcs.append(fc)
    row = {
        "accuracy": round(correct / len(samples), 4),
        "gap_hit": gap_hit,
        "fallback_rate": round(fallback_count / len(samples), 4),
        "mean_stop_n": round(sum(stop_ns) / len(samples), 2),
        "params": {"upper_thr": upper_thr},
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
        grid = (0.48, 0.49, 0.50, 0.52, 0.55, 0.551, 0.555, 0.56, 0.58, 0.60)
        sweep = []
        for upper in grid:
            sweep.append(eval_upper_thr(
                head, model, tokenizer, full, device=device, seed=99, profile=profile,
                struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, upper_thr=upper,
            ))
        gap3 = [r for r in sweep if r["gap_hit"] >= 3]
        best_acc = max(sweep, key=lambda r: r["accuracy"])
        best_gap = max(gap3, key=lambda r: r["accuracy"], default=None)
        return {
            "sweep": sweep,
            "best_acc": best_acc,
            "best_gap3": best_gap,
            "gap_indices": list(GAP_INDICES),
            "champion_p25": 0.9523,
            "union_ceiling": 0.9594,
            "insight": "P28 D2：0.55捞2/3(缺189@0.5501)；0.60捞3/3但acc降。细扫找Pareto膝点。",
            "mentor_brief": (
                f"F2 Pareto：acc冠军 upper={best_acc['params']['upper_thr']} {best_acc['accuracy']:.1%}；"
                f"gap3最优 upper={(best_gap or {}).get('params', {}).get('upper_thr', '—')} "
                f"{(best_gap or {}).get('accuracy', 0):.1%}。"
            ),
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "f2_deadzone_pareto", "F2 · 死区Pareto", device=args.device)
    write_phase30_result("f2_deadzone_pareto", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
