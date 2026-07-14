#!/usr/bin/env python3
"""Q3 · K3 hybrid 蒸馏头 + min_n=3 推理扫描（不重训，只改推理约束）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase7"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase15_common import (
    CAP, FINE_GRID, MIN_N_BEST, SEED, is_deployable_mvp, is_feasible,
    load_k3_head_state, load_rich_head, load_splits, timed_run, write_phase15_result,
)
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from phase7._hybrid_eval import evaluate_hybrid
from run_adaptive_stop_experiment import predict_at_n
from run_auto_submit_experiment import evaluate_policy, make_policies
from stop_head import (
    build_rich_stop_examples_for_samples,
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
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
        policies = make_policies(cap=CAP)
        auto = evaluate_policy(model, tokenizer, test_set, policies["auto_route"], device, cap=CAP, eval_profile=profile)

        state = load_k3_head_state(device)
        trained_fresh = False
        if not state:
            train_sub, val_sub = split_train_val_samples(train_set, val_ratio=0.2, seed=43)
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
            head, train_metrics = train_rich_stop_head(train_ex, val_ex, epochs=args.epochs, device=device)
            ckpt = ROOT / "results" / "phase15" / "q3_hybrid_min3_head.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": head.state_dict(), "train_metrics": train_metrics}, ckpt)
            trained_fresh = True
        else:
            head = load_rich_head(device, state)
            train_metrics = None

        sweep = []
        for thr in FINE_GRID:
            row = evaluate_rich_stop(
                head, model, tokenizer, test_set, cap=CAP, min_n=MIN_N_BEST, threshold=thr,
                device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
                build_prompt_fn=build_eval_prompt, eval_profile=profile,
            )
            row["params"]["uses_oracle"] = False
            sweep.append({
                "min_n": MIN_N_BEST, "threshold": thr,
                "accuracy": row["accuracy"],
                "stop_timing_acc": row.get("stop_timing_acc"),
                "mean_stop_n": row.get("mean_stop_n"),
                "feasible": is_feasible(row),
                "deployable_mvp": is_deployable_mvp(row),
            })

        best_timing = max(sweep, key=lambda p: ((p["stop_timing_acc"] or 0), p["accuracy"]))
        best_mvp = max(sweep, key=lambda p: (p["deployable_mvp"], p["accuracy"], p["stop_timing_acc"] or 0))
        feasible_pts = [p for p in sweep if p["feasible"]]
        teacher = evaluate_hybrid(
            model, tokenizer, test_set, cap=CAP, min_n=MIN_N_BEST, device=device, seed=SEED, profile=profile,
        )
        return {
            "trained_fresh": trained_fresh,
            "train_metrics": train_metrics,
            "sweep": sweep,
            "feasible_points": feasible_pts,
            "feasible": bool(feasible_pts),
            "best_timing": best_timing,
            "best_deployable_mvp": best_mvp,
            "teacher_oracle": {
                "strategy": teacher.get("strategy"),
                "accuracy": teacher.get("accuracy"),
                "stop_timing_acc": teacher.get("stop_timing_acc"),
            },
            "auto_route_test_acc": auto["accuracy"],
            "insight": "K3 hybrid 头 + min_n=3 推理；teacher 上界对照，看蒸馏能否在 min_n=3 下抬 timing。",
            "eval_split": "test_40pct",
            "sample_count": len(test_set),
            "device": str(device),
        }

    path = timed_run(run_body, "q3_hybrid_min3_distill", "Q3 · hybrid+min3", device=args.device)
    import json
    write_phase15_result("q3_hybrid_min3_distill", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
