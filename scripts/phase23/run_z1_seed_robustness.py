#!/usr/bin/env python3
"""Z1 · 定稿方案种子稳健性：structure_d / structure_d+M2 / knn。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase23_common import (
    CAP, FINE_GRID, MIN_N, ROBUST_SEEDS, SEED, eval_floor_m2, eval_structure_d,
    load_full_dataset, load_m2_head_state, load_rich_head, load_splits, stats, timed_run, write_phase23_result,
)
from boundary_budget import build_prompt_budget_labels, make_d4_knn_budget_fn, make_structure_budget_fn, train_d4_knn_bank
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import calibrate_rich_threshold, split_train_val_samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        train_set, _ = load_splits()
        full = load_full_dataset()
        train_sub, val_sub = split_train_val_samples(train_set, val_ratio=0.2, seed=43)
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
        head = load_rich_head(device, load_m2_head_state(device))
        struct_floor = make_structure_budget_fn(min_n=MIN_N, cap=CAP)
        train_rows = build_prompt_budget_labels(
            model, tokenizer, train_sub, cap=CAP, min_n=MIN_N, device=device, seed=42,
            predict_fn=pfn, expected_fn=expected_answer, eval_profile=profile,
        )
        knn_bank, _ = train_d4_knn_bank(train_rows, feature_key="joint_features")
        knn_floor = make_d4_knn_budget_fn(knn_bank, k=5, min_n=MIN_N, cap=CAP)
        thr, _ = calibrate_rich_threshold(
            head, model, tokenizer, val_sub, cap=CAP, min_n=MIN_N, thresholds=FINE_GRID,
            device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt, eval_profile=profile, optimize="accuracy",
        )

        report = {}
        seeds = list(ROBUST_SEEDS)
        for label, fn in (
            ("structure_d", lambda sd: eval_structure_d(model, tokenizer, full, device, sd, profile)),
            ("structure_d_m2", lambda sd: eval_floor_m2(head, model, tokenizer, full, device, sd, profile, thr, struct_floor, "structure_d_m2")),
            ("knn_m2", lambda sd: eval_floor_m2(head, model, tokenizer, full, device, sd, profile, 0.15, knn_floor, "knn_m2")),
        ):
            accs, eps1s = [], []
            rows = []
            for sd in seeds:
                row = fn(sd)
                accs.append(row["accuracy"])
                eps1s.append(row.get("timing_eps1") or 0)
                rows.append({"seed": sd, **row})
            report[label] = {"per_seed": rows, "acc_stats": stats(accs), "eps1_stats": stats(eps1s)}

        sd = report["structure_d"]["acc_stats"]
        em = report["structure_d_m2"]["eps1_stats"]
        return {
            "report": report,
            "seeds": seeds,
            "threshold_structure_m2": thr,
            "insight": "定稿三方案种子稳健性：acc/ε=1 方差是否可接受。",
            "mentor_brief": f"Z1 稳健性：structure_d acc μ={sd['mean']:.1%} σ={sd['stdev']:.3f}；structure_d+M2 ε=1 μ={em['mean']:.1%}。",
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "z1_seed_robustness", "Z1 · 种子稳健性", device=args.device)
    write_phase23_result("z1_seed_robustness", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
