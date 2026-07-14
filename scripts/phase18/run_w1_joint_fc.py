#!/usr/bin/env python3
"""W1 · Joint Coconut + first_correct 标签（对齐 timing）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase18_common import (
    CAP, FINE_GRID, MIN_N_BEST, SEED, is_deployable_mvp, is_feasible,
    load_full_dataset, load_m2_head_state, load_splits, timed_run, write_phase18_result,
)
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from run_auto_submit_experiment import evaluate_policy, make_policies
from stop_head import calibrate_rich_threshold, evaluate_rich_stop, split_train_val_samples, train_joint_rich_stop_head


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        train_set, test_set = load_splits()
        full_set = load_full_dataset()
        train_sub, val_sub = split_train_val_samples(train_set, val_ratio=0.2, seed=43)
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)

        head, train_metrics = train_joint_rich_stop_head(
            model, tokenizer, train_sub, val_sub, cap=CAP, device=device, seed=42,
            predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt,
            eval_profile=profile, unfreeze_layers=1, epochs=args.epochs,
            coconut_lr=3e-6, head_lr=3e-4, early_stop_patience=4,
            init_head_state=load_m2_head_state(device), label_mode="first_correct",
        )
        ckpt = ROOT / "results" / "phase18" / "w1_joint_fc.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "head_state": head.state_dict(),
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "train_metrics": train_metrics,
            "label_mode": "first_correct",
        }, ckpt)

        thr, cal = calibrate_rich_threshold(
            head, model, tokenizer, val_sub, cap=CAP, min_n=MIN_N_BEST, device=device, seed=SEED,
            predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt,
            eval_profile=profile, optimize="balanced", thresholds=FINE_GRID,
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
            "label_mode": "first_correct",
            "calibration": cal,
            "threshold": thr,
            "test": test_row,
            "full_419": full_row,
            "feasible": is_feasible(full_row),
            "deployable_mvp": is_deployable_mvp(full_row),
            "auto_route_full_acc": auto["accuracy"],
            "insight": "Joint 解冻 Coconut 1 层 + first_correct 标签，直接优化停步时机。",
            "eval_split": "full_419",
            "sample_count": len(full_set),
            "device": str(device),
        }

    path = timed_run(run_body, "w1_joint_fc", "W1 · joint+fc", device=args.device)
    import json
    write_phase18_result("w1_joint_fc", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
