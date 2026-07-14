#!/usr/bin/env python3
"""E3 · 置信扫描选步：n∈[min_n,d+1] 选 head 停步概率最高的 n，单次提交。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "phase25"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase29_common import CAP, MIN_N, load_json, timed_run, write_phase29_result
from boundary_budget import blind_depth
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase23._phase23_common import is_deployable_mvp, load_full_dataset, timing_metrics
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import _rich_step_features, extract_latent_hidden, first_correct_step


@torch.no_grad()
def eval_conf_scan_pick(head, model, tokenizer, samples, *, device, seed, profile):
    from _fallback_eval import setup_fallback_stack
    head, _, _, _, pfn = setup_fallback_stack(model, tokenizer, device, profile)
    head.eval()
    correct = 0
    stop_ns, fcs = [], []
    for idx, sample in enumerate(samples):
        expected = expected_answer(sample, profile)
        d = blind_depth(sample)
        hi = min(CAP, d + 1)
        best_n, best_prob, best_pred = MIN_N, -1.0, ""
        prev, streak = "", 0
        for n in range(MIN_N, hi + 1):
            pred = predict_at_n(model, tokenizer, sample, n, device, seed=seed + idx * 31 + n, eval_profile=profile)
            ab, streak, ch = _rich_step_features(pred, prev, streak)
            prev = pred
            prompt = build_eval_prompt(sample, n, seed=seed + idx * 31 + n,
                choice_order=profile.choice_order, shuffle_edges=profile.prompt_mode != "fixed_edges")
            ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
            hid = extract_latent_hidden(model, ids, pass_idx=n - 1).to(device)
            prob = torch.sigmoid(head(
                hid.unsqueeze(0), torch.tensor([n], device=device),
                torch.tensor([ab], device=device), torch.tensor([streak], device=device),
                torch.tensor([ch], device=device),
            )).item()
            if prob > best_prob:
                best_prob, best_n, best_pred = prob, n, pred
        fc, _ = first_correct_step(
            model, tokenizer, sample, cap=CAP, device=device, seed=seed + idx * 31,
            predict_fn=pfn, expected_fn=expected_answer, eval_profile=profile,
        )
        if best_pred == expected:
            correct += 1
        stop_ns.append(best_n)
        fcs.append(fc)
    row = {
        "accuracy": round(correct / len(samples), 4),
        "total": len(samples),
        "mean_stop_n": round(sum(stop_ns) / len(samples), 2),
        "params": {"mode": "conf_scan_pick", "uses_oracle": False},
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
        from _fallback_eval import setup_fallback_stack
        head, _, _, _, _ = setup_fallback_stack(model, tokenizer, device, profile)
        row = eval_conf_scan_pick(head, model, tokenizer, full, device=device, seed=99, profile=profile)
        return {
            "full_419": row,
            "baseline_p25": 0.9523,
            "mentor_brief": f"E3 置信扫描选步：acc {row['accuracy']:.1%} mean_n {row['mean_stop_n']}。",
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "e3_conf_scan_pick", "E3 · 置信扫描", device=args.device)
    write_phase29_result("e3_conf_scan_pick", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
