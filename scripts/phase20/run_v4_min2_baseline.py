#!/usr/bin/env python3
"""V4 · min_n=2 基线扫描（M2 冻结头，验证理论 ceiling）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase20_common import (
    CAP, FINE_GRID, ROOT, SEED, is_deployable_mvp, is_feasible,
    load_full_dataset, load_m2_head_state, load_rich_head, load_splits, timed_run, write_phase20_result,
)
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import calibrate_rich_threshold, evaluate_rich_stop, split_train_val_samples


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

        thr, cal = calibrate_rich_threshold(
            head, model, tokenizer, val_sub, cap=CAP, min_n=2, device=device, seed=SEED,
            predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt,
            eval_profile=profile, optimize="timing", min_accuracy=0.863, thresholds=FINE_GRID,
        )
        full_row = evaluate_rich_stop(
            head, model, tokenizer, full_set, cap=CAP, min_n=2, threshold=thr,
            device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt, eval_profile=profile,
        )
        test_row = evaluate_rich_stop(
            head, model, tokenizer, test_set, cap=CAP, min_n=2, threshold=thr,
            device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt, eval_profile=profile,
        )
        full_row["params"]["uses_oracle"] = False
        test_row["params"]["uses_oracle"] = False

        ceiling = None
        cp = ROOT / "results" / "phase20" / "timing_ceiling_analysis.json"
        if cp.is_file():
            import json
            ceiling = (json.loads(cp.read_text()) or {}).get("ceilings", {}).get("2")
        return {
            "calibration": cal,
            "threshold": thr,
            "min_n": 2,
            "full_419": full_row,
            "test": test_row,
            "feasible": is_feasible(full_row),
            "deployable_mvp": is_deployable_mvp(full_row),
            "theoretical_ceiling_min2": ceiling,
            "insight": "min_n=2 理论 timing ceiling≈61%；对比 M2 实测验证瓶颈。",
            "sample_count": len(full_set),
            "device": str(device),
        }

    path = timed_run(run_body, "v4_min2_baseline", "V4 · min_n=2 基线", device=args.device)
    import json
    write_phase20_result("v4_min2_baseline", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
