#!/usr/bin/env python3
"""V3 · first_correct 标签重训 + min_n=2 timing 校准（推理无 oracle）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase20_common import (
    CAP, FINE_GRID, FIXED_3_ACC, SEED, is_deployable_mvp, is_feasible,
    load_full_dataset, load_m2_head_state, load_rich_head, load_splits, timed_run, write_phase20_result,
)
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import (
    build_rich_stop_examples_for_samples,
    calibrate_rich_threshold,
    evaluate_rich_stop,
    split_train_val_samples,
    train_rich_stop_head,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        train_set, test_set = load_splits()
        full_set = load_full_dataset()
        train_sub, val_sub = split_train_val_samples(train_set, val_ratio=0.2, seed=43)
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
        init_state = load_m2_head_state(device)

        train_ex = build_rich_stop_examples_for_samples(
            model, tokenizer, train_sub, cap=CAP, device=device, seed=42,
            predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt,
            eval_profile=profile, label_mode="first_correct",
        )
        val_ex = build_rich_stop_examples_for_samples(
            model, tokenizer, val_sub, cap=CAP, device=device, seed=43,
            predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt,
            eval_profile=profile, label_mode="first_correct",
        )
        head, train_metrics = train_rich_stop_head(
            train_ex, val_ex, epochs=args.epochs, device=device, init_state=init_state,
        )
        ckpt = ROOT / "results" / "phase20" / "v3_fc_min2_head.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": head.state_dict(), "train_metrics": train_metrics, "label_mode": "first_correct"}, ckpt)

        sweep = []
        best_thr = 0.5
        best_min_n = 2
        best_timing = -1.0
        for min_n in (2, 3):
            thr, cal = calibrate_rich_threshold(
                head, model, tokenizer, val_sub, cap=CAP, min_n=min_n, device=device, seed=SEED,
                predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt,
                eval_profile=profile, optimize="timing", min_accuracy=FIXED_3_ACC, thresholds=FINE_GRID,
            )
            row = evaluate_rich_stop(
                head, model, tokenizer, val_sub, cap=CAP, min_n=min_n, threshold=thr,
                device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
                build_prompt_fn=build_eval_prompt, eval_profile=profile,
            )
            row["params"]["uses_oracle"] = False
            sweep.append({
                "min_n": min_n, "threshold": thr, "accuracy": row["accuracy"],
                "stop_timing_acc": row.get("stop_timing_acc"), "mean_stop_n": row.get("mean_stop_n"),
                "feasible": is_feasible(row), "calibration": cal,
            })
            t = row.get("stop_timing_acc") or 0
            if t > best_timing or (t == best_timing and row["accuracy"] > 0):
                best_timing = t
                best_thr = thr
                best_min_n = min_n

        full_row = evaluate_rich_stop(
            head, model, tokenizer, full_set, cap=CAP, min_n=best_min_n, threshold=best_thr,
            device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt, eval_profile=profile,
        )
        test_row = evaluate_rich_stop(
            head, model, tokenizer, test_set, cap=CAP, min_n=best_min_n, threshold=best_thr,
            device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt, eval_profile=profile,
        )
        full_row["params"]["uses_oracle"] = False
        test_row["params"]["uses_oracle"] = False

        return {
            "train_metrics": train_metrics,
            "label_mode": "first_correct",
            "sweep": sweep,
            "best": {"min_n": best_min_n, "threshold": best_thr},
            "full_419": full_row,
            "test": test_row,
            "feasible": is_feasible(full_row),
            "deployable_mvp": is_deployable_mvp(full_row),
            "insight": "P19：late_stop 主因 min_n=3 挡 fc=1/2；fc 标签 + min_n=2 直打 timing。",
            "sample_count": len(full_set),
            "device": str(device),
        }

    path = timed_run(run_body, "v3_fc_min2_train", "V3 · fc+min2 训练", device=args.device)
    import json
    write_phase20_result("v3_fc_min2_train", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
