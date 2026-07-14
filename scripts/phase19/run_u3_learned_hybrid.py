#!/usr/bin/env python3
"""U3 · 学习型 hybrid：首探 n=d，未中则 M2 head 在线扫（无 oracle 停步）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase7"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase19_common import (
    CAP, FINE_GRID, MIN_N, SEED, is_deployable_mvp, is_feasible,
    load_full_dataset, load_m2_head_state, load_rich_head, load_splits, timed_run, write_phase19_result,
)
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from phase7._hybrid_eval import first_probe_n
from run_adaptive_stop_experiment import predict_at_n
from stop_head import _rich_step_features, extract_latent_hidden, first_correct_step, split_train_val_samples


@torch.no_grad()
def evaluate_learned_hybrid(
    head, model, tokenizer, samples, *, cap, min_n, threshold, first_mode, device, seed,
    predict_fn, expected_fn, build_prompt_fn, eval_profile,
):
    head.eval()
    correct = total = stop_sum = timing_hits = timing_total = probe_sum = 0
    stop_hist = {}
    one_shot = 0
    for idx, sample in enumerate(samples):
        expected = expected_fn(sample, eval_profile)
        sseed = seed + idx * 31
        fc, preds = first_correct_step(
            model, tokenizer, sample, cap=cap, device=device, seed=sseed,
            predict_fn=predict_fn, expected_fn=expected_fn, eval_profile=eval_profile,
        )
        n0 = first_probe_n(first_mode, sample, cap)
        pred0 = predict_fn(sample, n0, sseed)
        probes = 1
        stop_n = n0
        final_pred = pred0

        if pred0 != expected:
            prev_pred = ""
            streak = 0
            start = max(min_n, n0 + 1)
            for n in range(start, cap + 1):
                final_pred = preds[n]
                answer_bucket, streak, changed = _rich_step_features(final_pred, prev_pred, streak)
                prev_pred = final_pred
                prompt = build_prompt_fn(sample, n, seed=sseed + n,
                    choice_order=eval_profile.choice_order,
                    shuffle_edges=eval_profile.prompt_mode != "fixed_edges")
                input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
                hidden = extract_latent_hidden(model, input_ids, pass_idx=n - 1).to(device)
                prob = torch.sigmoid(head(
                    hidden.unsqueeze(0),
                    torch.tensor([n], device=device),
                    torch.tensor([answer_bucket], device=device),
                    torch.tensor([streak], device=device),
                    torch.tensor([changed], device=device),
                )).item()
                probes += 1
                stop_n = n
                if prob >= threshold:
                    break
        else:
            one_shot += 1

        total += 1
        if final_pred == expected:
            correct += 1
        stop_sum += stop_n
        probe_sum += probes
        stop_hist[str(stop_n)] = stop_hist.get(str(stop_n), 0) + 1
        if fc is not None:
            timing_total += 1
            if stop_n == fc:
                timing_hits += 1

    acc = correct / total if total else 0.0
    return {
        "accuracy": round(acc, 4), "correct": correct, "total": total,
        "mean_stop_n": round(stop_sum / total, 2) if total else 0.0,
        "mean_probes": round(probe_sum / total, 2) if total else 0.0,
        "one_probe_rate": round(one_shot / total, 4) if total else 0.0,
        "stop_n_histogram": stop_hist,
        "stop_timing_acc": round(timing_hits / timing_total, 4) if timing_total else None,
        "stop_timing_hits": timing_hits, "stop_timing_total": timing_total,
        "params": {"min_n": min_n, "threshold": threshold, "first_mode": first_mode,
                   "mode": "learned_hybrid", "uses_oracle": False},
        "strategy": "learned_hybrid_stop",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        _, test_set = load_splits()
        full_set = load_full_dataset()
        train_set, _ = load_splits()
        _, val_sub = split_train_val_samples(train_set, val_ratio=0.2, seed=43)
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
        head = load_rich_head(device, load_m2_head_state(device))

        sweep = []
        for thr in FINE_GRID:
            row = evaluate_learned_hybrid(
                head, model, tokenizer, val_sub, cap=CAP, min_n=MIN_N, threshold=thr,
                first_mode="n_eq_d", device=device, seed=SEED, predict_fn=pfn,
                expected_fn=expected_answer, build_prompt_fn=build_eval_prompt, eval_profile=profile,
            )
            sweep.append({
                "threshold": thr, "accuracy": row["accuracy"],
                "stop_timing_acc": row.get("stop_timing_acc"),
                "mean_probes": row.get("mean_probes"), "feasible": is_feasible(row),
            })

        best = max(sweep, key=lambda p: ((p["stop_timing_acc"] or 0), p["accuracy"]))
        full_row = evaluate_learned_hybrid(
            head, model, tokenizer, full_set, cap=CAP, min_n=MIN_N, threshold=best["threshold"],
            first_mode="n_eq_d", device=device, seed=SEED, predict_fn=pfn,
            expected_fn=expected_answer, build_prompt_fn=build_eval_prompt, eval_profile=profile,
        )
        test_row = evaluate_learned_hybrid(
            head, model, tokenizer, test_set, cap=CAP, min_n=MIN_N, threshold=best["threshold"],
            first_mode="n_eq_d", device=device, seed=SEED, predict_fn=pfn,
            expected_fn=expected_answer, build_prompt_fn=build_eval_prompt, eval_profile=profile,
        )
        return {
            "sweep": sweep,
            "best": best,
            "full_419": full_row,
            "test": test_row,
            "feasible": is_feasible(full_row),
            "deployable_mvp": is_deployable_mvp(full_row),
            "insight": "hybrid 97% acc 但 timing 30%；U3 用 head 替代 soft_floor 扫步。",
            "sample_count": len(full_set),
            "device": str(device),
        }

    path = timed_run(run_body, "u3_learned_hybrid", "U3 · learned hybrid", device=args.device)
    import json
    write_phase19_result("u3_learned_hybrid", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
