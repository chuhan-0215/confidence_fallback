#!/usr/bin/env python3
"""X5 · structure_d vs knn 互补解剖：并集上界、分歧按跳数分布。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase22_common import CAP, MIN_N, SEED, load_full_dataset, load_splits, timed_run, write_phase22_result
from boundary_budget import (
    build_prompt_budget_labels, make_d4_knn_budget_fn, make_structure_budget_fn, train_d4_knn_bank,
)
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import (
    _rich_step_features, calibrate_rich_threshold, extract_latent_hidden,
    first_correct_step, split_train_val_samples,
)
from _phase22_common import load_m2_head_state, load_rich_head, FINE_GRID


@torch.no_grad()
def predict_structure_d(model, tokenizer, sample, device, seed, profile):
    d = __import__("boundary_budget", fromlist=["blind_depth"]).blind_depth(sample)
    n = max(MIN_N, min(CAP, d))
    return predict_at_n(model, tokenizer, sample, n, device, seed=seed, eval_profile=profile)


@torch.no_grad()
def predict_knn_m2(head, model, tokenizer, sample, device, seed, profile, floor_fn, thr):
    from boundary_budget import blind_depth
    floor_n = max(MIN_N, min(CAP, floor_fn(sample)))
    fc, preds = first_correct_step(
        model, tokenizer, sample, cap=CAP, device=device, seed=seed,
        predict_fn=lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile),
        expected_fn=expected_answer, eval_profile=profile,
    )
    stop_n = CAP
    final_pred = preds.get(CAP, "")
    prev, streak = "", 0
    for n in range(1, CAP + 1):
        final_pred = preds[n]
        ab, streak, ch = _rich_step_features(final_pred, prev, streak)
        prev = final_pred
        prompt = build_eval_prompt(sample, n, seed=seed + n,
            choice_order=profile.choice_order, shuffle_edges=profile.prompt_mode != "fixed_edges")
        ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
        hid = extract_latent_hidden(model, ids, pass_idx=n - 1).to(device)
        prob = torch.sigmoid(head(
            hid.unsqueeze(0), torch.tensor([n], device=device),
            torch.tensor([ab], device=device), torch.tensor([streak], device=device),
            torch.tensor([ch], device=device),
        )).item()
        stop_n = n
        if n >= floor_n and prob >= thr:
            break
    return final_pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        train_set, _ = load_splits()
        full = load_full_dataset()
        train_sub, val_sub = split_train_val_samples(train_set, val_ratio=0.2, seed=43)
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)

        train_rows = build_prompt_budget_labels(
            model, tokenizer, train_sub, cap=CAP, min_n=MIN_N, device=device, seed=42,
            predict_fn=pfn, expected_fn=expected_answer, eval_profile=profile,
        )
        knn_bank, _ = train_d4_knn_bank(train_rows, feature_key="joint_features")
        floor_fn = make_d4_knn_budget_fn(knn_bank, k=5, min_n=MIN_N, cap=CAP)
        head = load_rich_head(device, load_m2_head_state(device))
        thr, _ = calibrate_rich_threshold(
            head, model, tokenizer, val_sub, cap=CAP, min_n=MIN_N,
            thresholds=FINE_GRID, device=device, seed=SEED, predict_fn=pfn,
            expected_fn=expected_answer, build_prompt_fn=build_eval_prompt, eval_profile=profile,
            optimize="accuracy",
        )

        from boundary_budget import blind_depth
        struct_ok = knn_ok = union_ok = both_ok = disagree = 0
        struct_only = knn_only = 0
        by_hop = defaultdict(lambda: {"struct_ok": 0, "knn_ok": 0, "union_ok": 0, "n": 0})

        for idx, sample in enumerate(full):
            expected = expected_answer(sample, profile)
            ps = predict_structure_d(model, tokenizer, sample, device, SEED + idx * 31, profile)
            pk = predict_knn_m2(head, model, tokenizer, sample, device, SEED + idx * 31, profile, floor_fn, thr)
            s_ok = ps == expected
            k_ok = pk == expected
            if s_ok:
                struct_ok += 1
            if k_ok:
                knn_ok += 1
            if s_ok or k_ok:
                union_ok += 1
            if s_ok and k_ok:
                both_ok += 1
            if s_ok and not k_ok:
                struct_only += 1
            if k_ok and not s_ok:
                knn_only += 1
            if ps != pk:
                disagree += 1
            hop = blind_depth(sample)
            by_hop[str(hop)]["n"] += 1
            by_hop[str(hop)]["struct_ok"] += int(s_ok)
            by_hop[str(hop)]["knn_ok"] += int(k_ok)
            by_hop[str(hop)]["union_ok"] += int(s_ok or k_ok)

        total = len(full)
        return {
            "structure_d_acc": round(struct_ok / total, 4),
            "knn_m2_acc": round(knn_ok / total, 4),
            "union_acc": round(union_ok / total, 4),
            "both_correct": both_ok,
            "struct_only_correct": struct_only,
            "knn_only_correct": knn_only,
            "disagree_count": disagree,
            "by_hop": dict(by_hop),
            "union_gain_vs_knn": round((union_ok - knn_ok) / total, 4),
            "union_gain_vs_struct": round((union_ok - struct_ok) / total, 4),
            "insight": "并集 acc 上界：structure_d 与 knn 错误是否互补？能否指导融合策略。",
            "mentor_brief": (
                f"X5 互补：structure_d {struct_ok/total:.1%} knn {knn_ok/total:.1%} "
                f"并集 {union_ok/total:.1%}；仅 structure {struct_only} 仅 knn {knn_only}。"
            ),
            "sample_count": total,
            "device": str(device),
        }

    path = timed_run(run_body, "x5_complement_audit", "X5 · 互补解剖", device=args.device)
    write_phase22_result("x5_complement_audit", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
