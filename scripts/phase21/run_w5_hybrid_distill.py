#!/usr/bin/env python3
"""W5 · hybrid_stop 教师蒸馏 v2：重训 head 学 soft-floor 停步标签。"""
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
    CAP, FINE_GRID, FIXED_3_ACC, SEED, is_deployable_mvp, is_feasible,
    load_full_dataset, load_m2_head_state, load_rich_head, load_splits, timed_run, write_phase21_result,
)
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import (
    build_rich_stop_examples_for_samples, calibrate_rich_threshold, evaluate_rich_stop,
    split_train_val_samples, train_rich_stop_head,
)

MIN_N_GRID = (2, 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        train_set, test_set = load_splits()
        full = load_full_dataset()
        train_sub, val_sub = split_train_val_samples(train_set, val_ratio=0.2, seed=43)
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
        init_state = load_m2_head_state(device)

        train_ex = build_rich_stop_examples_for_samples(
            model, tokenizer, train_sub, cap=CAP, device=device, seed=42,
            predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt,
            eval_profile=profile, label_mode="hybrid_stop",
        )
        val_ex = build_rich_stop_examples_for_samples(
            model, tokenizer, val_sub, cap=CAP, device=device, seed=43,
            predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt,
            eval_profile=profile, label_mode="hybrid_stop",
        )
        head, train_metrics = train_rich_stop_head(
            train_ex, val_ex, epochs=args.epochs, device=device, init_state=init_state,
        )
        ckpt = ROOT / "results" / "phase21" / "w5_hybrid_stop_head.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": head.state_dict(), "train_metrics": train_metrics, "label_mode": "hybrid_stop"}, ckpt)

        sweep = []
        best = None
        for min_n in MIN_N_GRID:
            thr, _ = calibrate_rich_threshold(
                head, model, tokenizer, val_sub, cap=CAP, min_n=min_n,
                thresholds=FINE_GRID, device=device, seed=SEED, predict_fn=pfn,
                expected_fn=expected_answer, build_prompt_fn=build_eval_prompt, eval_profile=profile,
                optimize="timing", min_accuracy=FIXED_3_ACC,
            )
            for split_name, samples in (("test", test_set), ("full_419", full)):
                row = evaluate_rich_stop(
                    head, model, tokenizer, samples, cap=CAP, min_n=min_n, threshold=thr,
                    device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
                    build_prompt_fn=build_eval_prompt, eval_profile=profile,
                )
                row["params"]["uses_oracle"] = False
                pt = {
                    "min_n": min_n, "threshold": thr, "split": split_name,
                    "accuracy": row["accuracy"], "stop_timing_acc": row.get("stop_timing_acc"),
                    "mean_stop_n": row.get("mean_stop_n"),
                    "feasible": is_feasible(row),
                    "deployable_mvp": is_deployable_mvp(row),
                }
                sweep.append(pt)
                if split_name == "full_419" and (best is None or (pt.get("stop_timing_acc") or 0, pt["accuracy"]) > ((best.get("stop_timing_acc") or 0), best["accuracy"])):
                    best = {**pt, "row": row}

        return {
            "sweep": sweep,
            "train_metrics": train_metrics,
            "best": best,
            "full_419": best,
            "test": next((p for p in sweep if p["split"] == "test" and p["min_n"] == best["min_n"]), {}),
            "feasible": best and best.get("feasible"),
            "deployable_mvp": best and best.get("deployable_mvp"),
            "insight": "hybrid_stop 蒸馏：教师含 BFS floor + fc 信息，推理仍纯 head；测标签范式能否突破 M2。",
            "mentor_brief": (
                f"W5 hybrid 蒸馏：min_n={best['min_n']} thr={best['threshold']} "
                f"timing {best.get('stop_timing_acc', 0):.1%} acc {best['accuracy']:.1%}。"
            ),
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "w5_hybrid_distill", "W5 · hybrid 蒸馏", device=args.device)
    write_phase21_result("w5_hybrid_distill", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
