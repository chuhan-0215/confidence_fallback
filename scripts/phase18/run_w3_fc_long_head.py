#!/usr/bin/env python3
"""W3 · 冻结 Coconut · first_correct 长训 + timing 校准 + min_n=3。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase18_common import (
    CAP, FINE_GRID, FIXED_3_ACC, MIN_N_BEST, SEED, is_deployable_mvp, is_feasible,
    load_full_dataset, load_rich_head, load_splits, timed_run, write_phase18_result,
)
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from run_auto_submit_experiment import evaluate_policy, make_policies
from stop_head import (
    RichStopExample,
    build_rich_stop_examples_for_samples,
    calibrate_rich_threshold,
    evaluate_rich_stop,
    split_train_val_samples,
    train_rich_stop_head,
)


def _oversample_fc(examples: list[RichStopExample], factor: int = 5) -> list[RichStopExample]:
    out = list(examples)
    for ex in examples:
        if ex.should_stop > 0.5:
            out.extend([ex] * (factor - 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        train_set, test_set = load_splits()
        full_set = load_full_dataset()
        train_sub, val_sub = split_train_val_samples(train_set, val_ratio=0.2, seed=43)
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)

        train_ex = build_rich_stop_examples_for_samples(
            model, tokenizer, train_sub, cap=CAP, device=device, seed=42,
            predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt,
            eval_profile=profile, label_mode="first_correct",
        )
        train_ex = _oversample_fc(train_ex, factor=5)
        val_ex = build_rich_stop_examples_for_samples(
            model, tokenizer, val_sub, cap=CAP, device=device, seed=43,
            predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt,
            eval_profile=profile, label_mode="first_correct",
        )
        head, train_metrics = train_rich_stop_head(train_ex, val_ex, epochs=args.epochs, device=device)
        ckpt = ROOT / "results" / "phase18" / "w3_fc_long_head.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": head.state_dict(), "train_metrics": train_metrics}, ckpt)

        thr, cal = calibrate_rich_threshold(
            head, model, tokenizer, val_sub, cap=CAP, min_n=MIN_N_BEST, device=device, seed=SEED,
            predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt,
            eval_profile=profile, optimize="timing", min_accuracy=FIXED_3_ACC, thresholds=FINE_GRID,
        )
        test_row = evaluate_rich_stop(
            head, model, tokenizer, test_set, cap=CAP, min_n=MIN_N_BEST, threshold=thr,
            device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt, eval_profile=profile,
        )
        full_row = evaluate_rich_stop(
            head, model, tokenizer, full_set, cap=CAP, min_n=MIN_N_BEST, threshold=thr,
            device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt, eval_profile=profile,
        )
        test_row["params"]["uses_oracle"] = False
        full_row["params"]["uses_oracle"] = False
        policies = make_policies(cap=CAP)
        auto = evaluate_policy(model, tokenizer, full_set, policies["auto_route"], device, cap=CAP, eval_profile=profile)

        return {
            "train_metrics": train_metrics,
            "oversample_factor": 5,
            "calibration": cal,
            "threshold": thr,
            "test": test_row,
            "full_419": full_row,
            "feasible": is_feasible(full_row),
            "deployable_mvp": is_deployable_mvp(full_row),
            "auto_route_full_acc": auto["accuracy"],
            "insight": "冻结 Coconut；fc 标签 5× 过采样 + timing 校准；低成本抬 timing。",
            "eval_split": "full_419",
            "sample_count": len(full_set),
            "device": str(device),
        }

    path = timed_run(run_body, "w3_fc_long_head", "W3 · fc 长训", device=args.device)
    import json
    write_phase18_result("w3_fc_long_head", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
