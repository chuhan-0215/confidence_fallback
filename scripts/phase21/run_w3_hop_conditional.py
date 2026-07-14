#!/usr/bin/env python3
"""W3 · 跳数条件化 M2：预测 floor → 动态 min_n → M2 在线停步。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase21_common import (
    CAP, FINE_GRID, SEED, is_deployable_mvp, is_feasible,
    load_full_dataset, load_m2_head_state, load_rich_head, load_splits, timed_run, write_phase21_result,
)
from boundary_budget import (
    blind_depth, build_prompt_budget_labels, make_d4_knn_budget_fn,
    make_d4_rich_binary_budget_fn, train_d4_knn_bank, train_d4_weighted_binary_mlp,
)
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import (
    _rich_step_features, calibrate_rich_threshold, extract_latent_hidden,
    first_correct_step, split_train_val_samples,
)

MIN_N = 2


@torch.no_grad()
def evaluate_hop_conditional_m2(
    head, model, tokenizer, samples, *, cap, threshold, device, seed,
    predict_fn, expected_fn, build_prompt_fn, eval_profile, floor_fn, global_min_n,
):
    head.eval()
    correct = total = stop_sum = timing_hits = timing_total = 0
    stop_hist = {}
    for idx, sample in enumerate(samples):
        expected = expected_fn(sample, eval_profile)
        floor_n = max(global_min_n, min(cap, floor_fn(sample)))
        fc, preds = first_correct_step(
            model, tokenizer, sample, cap=cap, device=device, seed=seed + idx * 31,
            predict_fn=predict_fn, expected_fn=expected_fn, eval_profile=eval_profile,
        )
        stop_n = cap
        final_pred = preds.get(cap, "")
        prev, streak = "", 0
        for n in range(1, cap + 1):
            final_pred = preds[n]
            ab, streak, ch = _rich_step_features(final_pred, prev, streak)
            prev = final_pred
            prompt = build_prompt_fn(sample, n, seed=seed + idx * 31 + n,
                choice_order=eval_profile.choice_order,
                shuffle_edges=eval_profile.prompt_mode != "fixed_edges")
            ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
            hid = extract_latent_hidden(model, ids, pass_idx=n - 1).to(device)
            prob = torch.sigmoid(head(
                hid.unsqueeze(0),
                torch.tensor([n], device=device),
                torch.tensor([ab], device=device),
                torch.tensor([streak], device=device),
                torch.tensor([ch], device=device),
            )).item()
            stop_n = n
            if n >= floor_n and prob >= threshold:
                break
        total += 1
        if final_pred == expected:
            correct += 1
        stop_sum += stop_n
        stop_hist[str(stop_n)] = stop_hist.get(str(stop_n), 0) + 1
        if fc is not None:
            timing_total += 1
            if stop_n == fc:
                timing_hits += 1
    acc = correct / total if total else 0.0
    return {
        "accuracy": round(acc, 4), "correct": correct, "total": total,
        "mean_stop_n": round(stop_sum / total, 2) if total else 0.0,
        "stop_n_histogram": stop_hist,
        "stop_timing_acc": round(timing_hits / timing_total, 4) if timing_total else None,
        "stop_timing_hits": timing_hits, "stop_timing_total": timing_total,
        "params": {
            "threshold": threshold, "cap": cap, "global_min_n": global_min_n,
            "mode": "hop_conditional_m2", "uses_oracle": False,
        },
        "strategy": "hop_conditional_m2",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        train_set, test_set = load_splits()
        full = load_full_dataset()
        train_sub, val_sub = split_train_val_samples(train_set, val_ratio=0.2, seed=43)
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)

        train_rows = build_prompt_budget_labels(
            model, tokenizer, train_sub, cap=CAP, min_n=MIN_N, device=device, seed=42,
            predict_fn=pfn, expected_fn=expected_answer, eval_profile=profile,
        )
        val_rows = build_prompt_budget_labels(
            model, tokenizer, val_sub, cap=CAP, min_n=MIN_N, device=device, seed=43,
            predict_fn=pfn, expected_fn=expected_answer, eval_profile=profile,
        )
        rich_head, rich_meta = train_d4_weighted_binary_mlp(train_rows, val_rows, feature_key="rich_features")
        knn_bank, knn_meta = train_d4_knn_bank(train_rows, feature_key="joint_features")

        state = load_m2_head_state(device)
        if not state:
            raise FileNotFoundError("需要 m2_enough_stop_head.pt")
        head = load_rich_head(device, state)

        floor_variants = [
            ("blind_depth", lambda s: blind_depth(s)),
            ("rich_mlp_floor", make_d4_rich_binary_budget_fn(rich_head, min_n=MIN_N, cap=CAP, device=device)),
            ("knn_joint_floor", make_d4_knn_budget_fn(knn_bank, k=5, min_n=MIN_N, cap=CAP)),
        ]

        sweep = []
        for floor_name, floor_fn in floor_variants:
            thr, _ = calibrate_rich_threshold(
                head, model, tokenizer, val_sub, cap=CAP, min_n=MIN_N,
                thresholds=FINE_GRID, device=device, seed=SEED, predict_fn=pfn,
                expected_fn=expected_answer, build_prompt_fn=build_eval_prompt, eval_profile=profile,
                optimize="timing",
            )
            for split_name, samples in (("test", test_set), ("full_419", full)):
                row = evaluate_hop_conditional_m2(
                    head, model, tokenizer, samples, cap=CAP, threshold=thr,
                    device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
                    build_prompt_fn=build_eval_prompt, eval_profile=profile,
                    floor_fn=floor_fn, global_min_n=MIN_N,
                )
                sweep.append({
                    "floor": floor_name, "threshold": thr, "split": split_name,
                    "accuracy": row["accuracy"], "stop_timing_acc": row.get("stop_timing_acc"),
                    "mean_stop_n": row.get("mean_stop_n"),
                    "feasible": is_feasible(row),
                    "deployable_mvp": is_deployable_mvp(row),
                })

        full_rows = [r for r in sweep if r["split"] == "full_419"]
        best = max(full_rows, key=lambda r: ((r.get("stop_timing_acc") or 0), r["accuracy"]))
        best_mvp = max(full_rows, key=lambda r: (r["deployable_mvp"], r["accuracy"]))
        return {
            "sweep": sweep,
            "rich_meta": rich_meta,
            "knn_meta": knn_meta,
            "best": best,
            "best_deployable_mvp": best_mvp,
            "full_419": next(r for r in full_rows if r["floor"] == best["floor"]),
            "feasible": any(r["feasible"] for r in full_rows),
            "deployable_mvp": any(r["deployable_mvp"] for r in full_rows),
            "insight": "跳数条件化：用结构/模型预测 floor 放宽 min_n，M2 负责在线细停；测「新问题→步数→停」链路。",
            "mentor_brief": (
                f"W3 跳数条件化：最优 {best['floor']} timing {best.get('stop_timing_acc', 0):.1%} "
                f"acc {best['accuracy']:.1%}。"
            ),
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "w3_hop_conditional", "W3 · 跳数条件化 M2", device=args.device)
    write_phase21_result("w3_hop_conditional", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
