#!/usr/bin/env python3
"""Y3 · 错题解剖：structure_d vs knn 错题按跳数/不对称分类。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase24_common import (
    CAP, FINE_GRID, MIN_N, SEED, load_full_dataset, load_json, load_m2_head_state,
    load_rich_head, load_splits, timed_run, write_phase24_result,
)
from boundary_budget import _asymmetry_bin, blind_depth, build_prompt_budget_labels, make_d4_knn_budget_fn, make_structure_budget_fn, train_d4_knn_bank
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import calibrate_rich_threshold, first_correct_step, split_train_val_samples


@torch.no_grad()
def predict_struct(model, tokenizer, sample, device, seed, profile):
    d = blind_depth(sample)
    n = max(MIN_N, min(CAP, d))
    return predict_at_n(model, tokenizer, sample, n, device, seed=seed, eval_profile=profile), n


@torch.no_grad()
def predict_knn(head, model, tokenizer, sample, device, seed, profile, floor_fn, thr):
    from stop_head import _rich_step_features, extract_latent_hidden
    floor_n = max(MIN_N, min(CAP, floor_fn(sample)))
    fc, preds = first_correct_step(
        model, tokenizer, sample, cap=CAP, device=device, seed=seed,
        predict_fn=lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile),
        expected_fn=expected_answer, eval_profile=profile,
    )
    stop_n = CAP
    final = preds.get(CAP, "")
    prev, streak = "", 0
    for n in range(1, CAP + 1):
        final = preds[n]
        ab, streak, ch = _rich_step_features(final, prev, streak)
        prev = final
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
    return final, stop_n, fc


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
        head = load_rich_head(device, load_m2_head_state(device))
        train_rows = build_prompt_budget_labels(
            model, tokenizer, train_sub, cap=CAP, min_n=MIN_N, device=device, seed=42,
            predict_fn=pfn, expected_fn=expected_answer, eval_profile=profile,
        )
        knn_floor = make_d4_knn_budget_fn(train_d4_knn_bank(train_rows)[0], k=5, min_n=MIN_N, cap=CAP)
        thr, _ = calibrate_rich_threshold(
            head, model, tokenizer, val_sub, cap=CAP, min_n=MIN_N, thresholds=FINE_GRID,
            device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt, eval_profile=profile, optimize="accuracy",
        )

        struct_wrong = knn_wrong = both_wrong = struct_only = knn_only = 0
        by_hop = {}
        asym_counter = Counter()
        fc_gap_counter = Counter()

        for idx, sample in enumerate(full):
            expected = expected_answer(sample, profile)
            ps, ns = predict_struct(model, tokenizer, sample, device, SEED + idx * 31, profile)
            pk, nk, fc = predict_knn(head, model, tokenizer, sample, device, SEED + idx * 31, profile, knn_floor, thr)
            s_ok = ps == expected
            k_ok = pk == expected
            hop = blind_depth(sample)
            asym = _asymmetry_bin(sample, cap=CAP)
            key = str(hop)
            by_hop.setdefault(key, {"struct_ok": 0, "knn_ok": 0, "n": 0})
            by_hop[key]["n"] += 1
            by_hop[key]["struct_ok"] += int(s_ok)
            by_hop[key]["knn_ok"] += int(k_ok)
            if not s_ok:
                struct_wrong += 1
                asym_counter[f"hop{hop}_asym{asym}"] += 1
                if fc is not None:
                    fc_gap_counter[str(ns - fc)] += 1
            if not k_ok:
                knn_wrong += 1
            if not s_ok and not k_ok:
                both_wrong += 1
            if s_ok and not k_ok:
                struct_only += 1
            if k_ok and not s_ok:
                knn_only += 1

        p23 = load_json("phase22/x5_complement_audit_latest.json")
        return {
            "struct_wrong": struct_wrong,
            "knn_wrong": knn_wrong,
            "both_wrong": both_wrong,
            "struct_only_correct": struct_only,
            "knn_only_correct": knn_only,
            "union_miss": both_wrong,
            "by_hop": by_hop,
            "struct_wrong_asym": dict(asym_counter),
            "struct_wrong_fc_gap": dict(fc_gap_counter),
            "baseline_union_acc": p23.get("union_acc"),
            "insight": "剩余错题结构：并集缺口主要来自 both_wrong 还是可互补？",
            "mentor_brief": (
                f"Y3 错题：struct 错 {struct_wrong} knn 错 {knn_wrong} 双错 {both_wrong}；"
                f"仅 struct 对 {struct_only} 仅 knn 对 {knn_only}。"
            ),
            "sample_count": len(full),
            "device": str(device),
        }

    path = timed_run(run_body, "y3_error_taxonomy", "Y3 · 错题解剖", device=args.device)
    write_phase24_result("y3_error_taxonomy", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
