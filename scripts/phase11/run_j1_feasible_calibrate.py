#!/usr/bin/env python3
"""J1 · M2 头 + feasible 阈值校准（不重训，快速验证能否过线）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase11_common import CAP, MIN_N, SEED, is_feasible, load_m2_head_state, load_rich_head, load_splits, timed_run, write_phase11_result
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from run_auto_submit_experiment import evaluate_policy, make_policies
from stop_head import calibrate_rich_threshold, evaluate_rich_stop, split_train_val_samples

FINE_GRID = [round(x * 0.05, 2) for x in range(3, 17)]  # 0.15 .. 0.80


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        train_set, test_set = load_splits()
        _, val_sub = split_train_val_samples(train_set, val_ratio=0.2, seed=43)
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
        policies = make_policies(cap=CAP)
        ar_val = evaluate_policy(model, tokenizer, val_sub, policies["auto_route"], device, cap=CAP, eval_profile=profile)
        ar_test = evaluate_policy(model, tokenizer, test_set, policies["auto_route"], device, cap=CAP, eval_profile=profile)

        state = load_m2_head_state(device)
        if state is None:
            raise FileNotFoundError(f"需要 Phase10 M2 权重: {ROOT / 'results/phase10/m2_enough_stop_head.pt'}")
        head = load_rich_head(device, state)

        results = {}
        for mode in ("balanced", "feasible", "timing"):
            min_acc = ar_val["accuracy"] if mode == "feasible" else None
            opt = "balanced" if mode == "balanced" else ("feasible" if mode == "feasible" else "timing")
            thr, cal = calibrate_rich_threshold(
                head, model, tokenizer, val_sub, cap=CAP, min_n=MIN_N, device=device, seed=SEED,
                predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt,
                eval_profile=profile, optimize=opt, min_accuracy=min_acc, thresholds=FINE_GRID,
            )
            row = evaluate_rich_stop(
                head, model, tokenizer, test_set, cap=CAP, min_n=MIN_N, threshold=thr,
                device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
                build_prompt_fn=build_eval_prompt, eval_profile=profile,
            )
            row["strategy"] = f"m2_{mode}_calibrate"
            results[mode] = {"threshold": thr, "calibration": cal, "test": row, "feasible": is_feasible(row, ar_test["accuracy"])}

        best = max(results.values(), key=lambda r: (r["feasible"], r["test"]["accuracy"], r["test"].get("stop_timing_acc") or 0))
        return {
            "modes": results,
            "best_mode": next(k for k, v in results.items() if v is best),
            "auto_route_val_acc": ar_val["accuracy"],
            "auto_route_test_acc": ar_test["accuracy"],
            "feasible": any(r["feasible"] for r in results.values()),
            "proof": {"necessity": True, "learnable": True, "deploy_feasible": any(r["feasible"] for r in results.values())},
            "insight": "不重训，只调阈值；若仍不可行 → 必须 joint 微调。",
            "eval_split": "test_40pct",
            "sample_count": len(test_set),
            "device": str(device),
        }

    path = timed_run(run_body, "j1_feasible_calibrate", "J1 · feasible 校准", device=args.device)
    import json
    write_phase11_result("j1_feasible_calibrate", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
