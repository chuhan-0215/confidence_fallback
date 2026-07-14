#!/usr/bin/env python3
"""P3 · 前缀 kNN floor + M2 在线停步（模型 latent 定 min_n，无 BFS）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase14_common import CAP, FINE_GRID, MIN_N, SEED, is_deployable_mvp, is_feasible, load_m2_head_state, load_rich_head, load_splits, timed_run, write_phase14_result
from boundary_budget import build_teacher_budget_rows, make_d4_knn_budget_fn, train_d4_knn_bank
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from run_auto_submit_experiment import evaluate_policy, make_policies
from stop_head import calibrate_rich_threshold, evaluate_rich_stop, first_correct_step, split_train_val_samples


@torch.no_grad()
def evaluate_rich_stop_with_floor(
    head, model, tokenizer, samples, *, cap, threshold, device, seed,
    predict_fn, expected_fn, build_prompt_fn, eval_profile, floor_fn,
):
    """Per-sample dynamic min_n from prefix kNN floor."""
    head.eval()
    correct = total = stop_sum = timing_hits = timing_total = 0
    stop_hist = {}
    for idx, sample in enumerate(samples):
        expected = expected_fn(sample, eval_profile)
        floor_n = max(MIN_N, min(cap, floor_fn(sample)))
        fc, preds = first_correct_step(
            model, tokenizer, sample, cap=cap, device=device, seed=seed + idx * 31,
            predict_fn=predict_fn, expected_fn=expected_fn, eval_profile=eval_profile,
        )
        stop_n = cap
        final_pred = preds.get(cap, "")
        prev_pred = ""
        streak = 0
        for n in range(1, cap + 1):
            final_pred = preds[n]
            from stop_head import _rich_step_features, extract_latent_hidden, encode_answer_bucket
            answer_bucket, streak, changed = _rich_step_features(final_pred, prev_pred, streak)
            prev_pred = final_pred
            prompt = build_prompt_fn(sample, n, seed=seed + idx * 31 + n,
                choice_order=eval_profile.choice_order,
                shuffle_edges=eval_profile.prompt_mode != "fixed_edges")
            input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
            hidden = extract_latent_hidden(model, input_ids, pass_idx=n - 1).to(device)
            step_t = torch.tensor([n], dtype=torch.long, device=device)
            ans_t = torch.tensor([answer_bucket], dtype=torch.long, device=device)
            st_t = torch.tensor([streak], dtype=torch.long, device=device)
            ch_t = torch.tensor([changed], dtype=torch.float32, device=device)
            prob = torch.sigmoid(head(hidden.unsqueeze(0), step_t, ans_t, st_t, ch_t)).item()
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
        "params": {"threshold": threshold, "cap": cap, "mode": "prefix_knn_floor", "uses_oracle": False},
        "strategy": "prefix_knn_floor_stop",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        train_set, test_set = load_splits()
        train_sub, val_sub = split_train_val_samples(train_set, val_ratio=0.2, seed=43)
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)

        train_rows = build_teacher_budget_rows(
            model, tokenizer, train_sub, cap=CAP, min_n=MIN_N, device=device, seed=42,
            predict_fn=pfn, expected_fn=expected_answer, eval_profile=profile,
        )
        knn_bank, knn_meta = train_d4_knn_bank(train_rows, feature_key="prompt_hidden")
        floor_fn = make_d4_knn_budget_fn(knn_bank, k=5, min_n=MIN_N, cap=CAP)

        state = load_m2_head_state(device)
        if not state:
            raise FileNotFoundError("需要 m2_enough_stop_head.pt")
        head = load_rich_head(device, state)

        best_thr = 0.5
        best_row = None
        best_score = -1.0
        for thr in FINE_GRID:
            row = evaluate_rich_stop_with_floor(
                head, model, tokenizer, val_sub, cap=CAP, threshold=thr, device=device, seed=SEED,
                predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt,
                eval_profile=profile, floor_fn=floor_fn,
            )
            score = (row.get("stop_timing_acc") or 0) * 5 + row["accuracy"]
            if score > best_score:
                best_score = score
                best_thr = thr
                best_row = row

        test_row = evaluate_rich_stop_with_floor(
            head, model, tokenizer, test_set, cap=CAP, threshold=best_thr, device=device, seed=SEED,
            predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt,
            eval_profile=profile, floor_fn=floor_fn,
        )
        baseline = evaluate_rich_stop(
            head, model, tokenizer, test_set, cap=CAP, min_n=MIN_N, threshold=0.5,
            device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt, eval_profile=profile,
        )
        baseline["params"]["uses_oracle"] = False
        policies = make_policies(cap=CAP)
        auto = evaluate_policy(model, tokenizer, test_set, policies["auto_route"], device, cap=CAP, eval_profile=profile)
        return {
            "val_calibration": best_row,
            "knn_meta": knn_meta,
            "threshold": best_thr,
            "baseline_m2": baseline,
            "test": test_row,
            "feasible": is_feasible(test_row),
            "deployable_mvp": is_deployable_mvp(test_row),
            "auto_route_test_acc": auto["accuracy"],
            "insight": "Coconut 前缀 kNN 定每题 floor，M2 head 在线停；无 BFS 扫步。",
            "eval_split": "test_40pct",
            "sample_count": len(test_set),
            "device": str(device),
        }

    path = timed_run(run_body, "p3_prefix_knn_floor", "P3 · 前缀 floor", device=args.device)
    import json
    write_phase14_result("p3_prefix_knn_floor", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
