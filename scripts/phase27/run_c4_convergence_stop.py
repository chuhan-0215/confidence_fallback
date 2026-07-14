#!/usr/bin/env python3
"""C4 · 隐层收敛早停：hidden cosine 收敛 OR 答案稳定，无训练。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase27_common import CAP, MIN_N, SEED, load_full_dataset, timed_run, write_phase27_result
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import _rich_step_features, extract_latent_hidden, first_correct_step


@torch.no_grad()
def eval_convergence_stop(model, tokenizer, samples, *, cap, cos_thr, patience, min_n, device, seed, profile):
    from boundary_budget import blind_depth
    correct = stop_sum = 0
    stop_ns, fcs = [], []
    pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
    for idx, sample in enumerate(samples):
        expected = expected_answer(sample, profile)
        d = blind_depth(sample)
        floor_n = max(min_n, min(cap, d))
        fc, preds = first_correct_step(
            model, tokenizer, sample, cap=cap, device=device, seed=seed + idx * 31,
            predict_fn=pfn, expected_fn=expected_answer, eval_profile=profile,
        )
        stop_n = cap
        final = preds.get(cap, "")
        prev_pred, streak = "", 0
        prev_h = None
        conv_count = 0
        for n in range(1, cap + 1):
            final = preds[n]
            ab, streak, ch = _rich_step_features(final, prev_pred, streak)
            prev_pred = final
            prompt = build_eval_prompt(sample, n, seed=seed + idx * 31 + n,
                choice_order=profile.choice_order, shuffle_edges=profile.prompt_mode != "fixed_edges")
            ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
            hid = extract_latent_hidden(model, ids, pass_idx=n - 1).to(device)
            if prev_h is not None:
                cos = torch.nn.functional.cosine_similarity(hid.unsqueeze(0), prev_h.unsqueeze(0)).item()
                conv_count = conv_count + 1 if cos >= cos_thr else 0
            prev_h = hid
            stop_n = n
            if n >= floor_n and (conv_count >= patience or streak >= 2):
                break
        if final == expected:
            correct += 1
        stop_sum += stop_n
        stop_ns.append(stop_n)
        fcs.append(fc)
    from phase23._phase23_common import timing_metrics, is_deployable_mvp
    row = {
        "accuracy": round(correct / len(samples), 4),
        "total": len(samples),
        "mean_stop_n": round(stop_sum / len(samples), 2),
        "params": {"cos_thr": cos_thr, "patience": patience, "min_n": min_n, "mode": "convergence_stop", "uses_oracle": False},
    }
    row.update(timing_metrics(stop_ns, fcs))
    row["deployable_mvp"] = is_deployable_mvp(row)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        full = load_full_dataset()
        sweep = []
        for cos_thr in (0.95, 0.98):
            for patience in (1, 2):
                sweep.append(eval_convergence_stop(
                    model, tokenizer, full, cap=CAP, cos_thr=cos_thr, patience=patience, min_n=MIN_N,
                    device=device, seed=SEED, profile=profile,
                ))
        best = max(sweep, key=lambda r: r["accuracy"])
        return {
            "sweep": sweep,
            "best": best,
            "full_419": best,
            "insight": "实验16/19 收敛信号全量复现：无 head 能否自停？",
            "mentor_brief": f"C4 收敛早停最优 acc {best['accuracy']:.1%} mean_n {best['mean_stop_n']}。",
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "c4_convergence_stop", "C4 · 收敛早停", device=args.device)
    write_phase27_result("c4_convergence_stop", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
