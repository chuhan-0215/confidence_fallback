#!/usr/bin/env python3
"""D2 · 中置信扩展回退：prob0 < upper_thr 时强制走 knn 路径。"""
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
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase23._phase23_common import is_deployable_mvp, is_eps_deployable, load_full_dataset, timing_metrics
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import _rich_step_features, extract_latent_hidden, first_correct_step


@torch.no_grad()
def eval_upper_thr_fallback(head, model, tokenizer, samples, *, device, seed, profile,
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
        "total": len(samples),
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / len(samples), 4),
        "gap_hit": gap_hit,
        "mean_stop_n": round(sum(stop_ns) / len(samples), 2),
        "params": {"upper_thr": upper_thr, "mode": "mid_conf_zone"},
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
        sweep = []
        for upper in (0.48, 0.55, 0.60, 0.65):
            row = eval_upper_thr_fallback(
                head, model, tokenizer, full, device=device, seed=99, profile=profile,
                struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, upper_thr=upper,
            )
            sweep.append(row)
        best = max(sweep, key=lambda r: (r["accuracy"], r.get("gap_hit", 0), -r["fallback_rate"]))
        return {
            "sweep": sweep,
            "best": best,
            "full_419": best,
            "gap_indices": list(GAP_INDICES),
            "baseline_p25": 0.9523,
            "p27_best": 0.9475,
            "insight": "P27 证伪替代范式；扩大回退触发区能否捞回 idx 111/189/261。",
            "mentor_brief": (
                f"D2 中置信扩展：最优 upper={best['params']['upper_thr']} "
                f"acc {best['accuracy']:.1%} gap_hit {best.get('gap_hit', 0)}/3。"
            ),
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "d2_mid_conf_zone", "D2 · 中置信扩展", device=args.device)
    write_phase28_result("d2_mid_conf_zone", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
