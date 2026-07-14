#!/usr/bin/env python3
"""U1 · M2 + patience 停步（连续 k 步 head≥thr 才停，防 min_n 早停）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase19_common import (
    CAP, FINE_GRID, MIN_N, SEED, is_deployable_mvp, is_feasible,
    load_full_dataset, load_m2_head_state, load_rich_head, load_splits, timed_run, write_phase19_result,
)
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import _rich_step_features, calibrate_rich_threshold, extract_latent_hidden, first_correct_step, split_train_val_samples


@torch.no_grad()
def evaluate_patience_stop(
    head, model, tokenizer, samples, *, cap, min_n, threshold, patience, device, seed,
    predict_fn, expected_fn, build_prompt_fn, eval_profile,
):
    head.eval()
    correct = total = stop_sum = timing_hits = timing_total = 0
    stop_hist = {}
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
        high_run = 0
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
            high_run = high_run + 1 if prob >= threshold else 0
            if n >= min_n and high_run >= patience:
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
        "params": {"min_n": min_n, "threshold": threshold, "patience": patience,
                   "cap": cap, "mode": "patience_stop", "uses_oracle": False},
        "strategy": "patience_rich_stop",
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
        train_sub, val_sub = split_train_val_samples(train_set, val_ratio=0.2, seed=43)
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
        head = load_rich_head(device, load_m2_head_state(device))

        best = None
        sweep = []
        for patience in (1, 2, 3):
            for thr in FINE_GRID:
                row = evaluate_patience_stop(
                    head, model, tokenizer, val_sub, cap=CAP, min_n=MIN_N, threshold=thr,
                    patience=patience, device=device, seed=SEED, predict_fn=pfn,
                    expected_fn=expected_answer, build_prompt_fn=build_eval_prompt, eval_profile=profile,
                )
                pt = {"patience": patience, "threshold": thr, **{k: row[k] for k in ("accuracy", "stop_timing_acc", "mean_stop_n")}, "feasible": is_feasible(row)}
                sweep.append(pt)
                if best is None or (pt["stop_timing_acc"] or 0, pt["accuracy"]) > ((best.get("stop_timing_acc") or 0), best.get("accuracy", 0)):
                    best = {**pt, "row": row}

        assert best
        full_row = evaluate_patience_stop(
            head, model, tokenizer, full_set, cap=CAP, min_n=MIN_N,
            threshold=best["threshold"], patience=best["patience"], device=device, seed=SEED,
            predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt, eval_profile=profile,
        )
        test_row = evaluate_patience_stop(
            head, model, tokenizer, test_set, cap=CAP, min_n=MIN_N,
            threshold=best["threshold"], patience=best["patience"], device=device, seed=SEED,
            predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt, eval_profile=profile,
        )
        return {
            "sweep": sweep,
            "best": {k: best[k] for k in ("patience", "threshold", "accuracy", "stop_timing_acc")},
            "full_419": full_row,
            "test": test_row,
            "feasible": is_feasible(full_row),
            "deployable_mvp": is_deployable_mvp(full_row),
            "insight": "Phase18 全停在 n=3；patience 允许延后停步抬 timing。",
            "sample_count": len(full_set),
            "device": str(device),
        }

    path = timed_run(run_body, "u1_patience_stop", "U1 · patience", device=args.device)
    import json
    write_phase19_result("u1_patience_stop", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
