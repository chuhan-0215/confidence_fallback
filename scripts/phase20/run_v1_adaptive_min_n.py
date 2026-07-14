#!/usr/bin/env python3
"""V1 · 自适应 min_n：n=2 高置信+稳定才允许停，否则 min_n=3。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase20_common import (
    CAP, FINE_GRID, SEED, TAU2_GRID, is_deployable_mvp, is_feasible, load_json,
    load_full_dataset, load_m2_head_state, load_rich_head, load_splits, timed_run, write_phase20_result,
)
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import _rich_step_features, extract_latent_hidden, first_correct_step, split_train_val_samples


@torch.no_grad()
def evaluate_adaptive_min_n(
    head, model, tokenizer, samples, *, cap, tau2, tau3, streak_min, device, seed,
    predict_fn, expected_fn, build_prompt_fn, eval_profile,
):
    head.eval()
    correct = total = stop_sum = timing_hits = timing_total = 0
    stop_hist = {}
    n2_stops = 0
    for idx, sample in enumerate(samples):
        expected = expected_fn(sample, eval_profile)
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
            answer_bucket, streak, changed = _rich_step_features(final_pred, prev_pred, streak)
            prev_pred = final_pred
            prompt = build_prompt_fn(sample, n, seed=seed + idx * 31 + n,
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
            stop_n = n
            if n == 2 and prob >= tau2 and streak >= streak_min:
                n2_stops += 1
                break
            if n >= 3 and prob >= tau3:
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
        "n2_stop_count": n2_stops,
        "stop_n_histogram": stop_hist,
        "stop_timing_acc": round(timing_hits / timing_total, 4) if timing_total else None,
        "stop_timing_hits": timing_hits, "stop_timing_total": timing_total,
        "params": {
            "tau2": tau2, "tau3": tau3, "streak_min": streak_min,
            "cap": cap, "mode": "adaptive_min_n", "uses_oracle": False,
        },
        "strategy": "adaptive_min_n_stop",
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
        for tau2 in TAU2_GRID:
            for tau3 in FINE_GRID:
                for streak_min in (2, 3):
                    row = evaluate_adaptive_min_n(
                        head, model, tokenizer, val_sub, cap=CAP, tau2=tau2, tau3=tau3,
                        streak_min=streak_min, device=device, seed=SEED, predict_fn=pfn,
                        expected_fn=expected_answer, build_prompt_fn=build_eval_prompt, eval_profile=profile,
                    )
                    pt = {
                        "tau2": tau2, "tau3": tau3, "streak_min": streak_min,
                        "accuracy": row["accuracy"], "stop_timing_acc": row.get("stop_timing_acc"),
                        "mean_stop_n": row.get("mean_stop_n"), "n2_stop_count": row.get("n2_stop_count"),
                        "feasible": is_feasible(row),
                    }
                    sweep.append(pt)

        best = max(sweep, key=lambda p: ((p["stop_timing_acc"] or 0), p["accuracy"]))
        full_row = evaluate_adaptive_min_n(
            head, model, tokenizer, full_set, cap=CAP,
            tau2=best["tau2"], tau3=best["tau3"], streak_min=best["streak_min"],
            device=device, seed=SEED, predict_fn=pfn,
            expected_fn=expected_answer, build_prompt_fn=build_eval_prompt, eval_profile=profile,
        )
        test_row = evaluate_adaptive_min_n(
            head, model, tokenizer, test_set, cap=CAP,
            tau2=best["tau2"], tau3=best["tau3"], streak_min=best["streak_min"],
            device=device, seed=SEED, predict_fn=pfn,
            expected_fn=expected_answer, build_prompt_fn=build_eval_prompt, eval_profile=profile,
        )
        p19 = load_json("phase19/u6_proof_rollup_latest.json")
        return {
            "sweep": sweep,
            "best": best,
            "full_419": full_row,
            "test": test_row,
            "feasible": is_feasible(full_row),
            "deployable_mvp": is_deployable_mvp(full_row),
            "baseline_phase19_timing": (p19.get("best") or {}).get("timing"),
            "insight": "P19 U5：late_stop 57% 主因 fc<min_n；V1 允许 n=2 高置信停步吃 fc=2。",
            "sample_count": len(full_set),
            "device": str(device),
        }

    path = timed_run(run_body, "v1_adaptive_min_n", "V1 · adaptive min_n", device=args.device)
    import json
    write_phase20_result("v1_adaptive_min_n", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
