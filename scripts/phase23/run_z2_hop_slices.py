#!/usr/bin/env python3
"""Z2 · 跳数切片：3跳/4跳 上定稿方案表现。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase23_common import (
    CAP, FINE_GRID, MIN_N, SEED, eval_floor_m2, eval_structure_d, filter_by_hop,
    is_deployable_mvp, is_eps_deployable, load_full_dataset, load_m2_head_state,
    load_rich_head, load_splits, timed_run, write_phase23_result,
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

        slices = []
        for hop in (3, 4):
            sub = filter_by_hop(full, hop)
            for label, row in (
                ("structure_d", eval_structure_d(model, tokenizer, sub, device, SEED, profile)),
                ("structure_d_m2", eval_floor_m2(head, model, tokenizer, sub, device, SEED, profile, thr, struct_floor, "structure_d_m2")),
                ("knn_m2", eval_floor_m2(head, model, tokenizer, sub, device, SEED, profile, 0.15, knn_floor, "knn_m2")),
            ):
                row["strategy"] = label
                row["hop"] = hop
                row["count"] = len(sub)
                row["deployable_mvp"] = is_deployable_mvp(row)
                row["eps_deployable"] = is_eps_deployable(row)
                slices.append(row)

        s4 = next(s for s in slices if s["hop"] == 4 and s["strategy"] == "structure_d")
        k4 = next(s for s in slices if s["hop"] == 4 and s["strategy"] == "knn_m2")
        return {
            "slices": slices,
            "insight": "P22 X5：4跳 structure 207/217 vs knn 198/217；切片验证定稿方案。",
            "mentor_brief": f"Z2 切片：4跳 structure_d {s4['accuracy']:.1%}；knn {k4['accuracy']:.1%}。",
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "z2_hop_slices", "Z2 · 跳数切片", device=args.device)
    write_phase23_result("z2_hop_slices", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
