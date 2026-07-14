#!/usr/bin/env python3
"""E4 · 3跳双路并行：hop=3 同时跑 struct+knn，答案不同则信 knn。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "phase25"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fallback_eval import run_knn_path, setup_fallback_stack
from _phase29_common import CAP, MIN_N, GAP_INDICES, timed_run, write_phase29_result
from boundary_budget import blind_depth
from evaluate_coconut import expected_answer
from phase23._phase23_common import is_deployable_mvp, load_full_dataset, timing_metrics
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import first_correct_step


@torch.no_grad()
def eval_hop3_dual_path(head, model, tokenizer, samples, *, device, seed, profile,
                        struct_floor, knn_floor, knn_thr, pfn):
    correct = dual_count = 0
    stop_ns, fcs = [], []
    gap_fixed = 0
    for idx, sample in enumerate(samples):
        expected = expected_answer(sample, profile)
        hop = blind_depth(sample)
        n0 = max(MIN_N, min(CAP, struct_floor(sample)))
        ps = predict_at_n(model, tokenizer, sample, n0, device, seed=seed + idx * 31, eval_profile=profile)
        if hop < 4:
            pk, nk, _, _ = run_knn_path(
                head, model, tokenizer, sample, device=device, seed=seed + idx * 31,
                profile=profile, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            )
            if ps != pk:
                dual_count += 1
                final, stop_n = pk, nk
            else:
                final, stop_n = ps, n0
        else:
            final, stop_n = ps, n0
        fc, _ = first_correct_step(
            model, tokenizer, sample, cap=CAP, device=device, seed=seed + idx * 31,
            predict_fn=pfn, expected_fn=expected_answer, eval_profile=profile,
        )
        if final == expected:
            correct += 1
            if idx in GAP_INDICES:
                gap_fixed += 1
        stop_ns.append(stop_n)
        fcs.append(fc)
    row = {
        "accuracy": round(correct / len(samples), 4),
        "total": len(samples),
        "dual_disagree_count": dual_count,
        "gap_fixed": gap_fixed,
        "mean_stop_n": round(sum(stop_ns) / len(samples), 2),
        "params": {"mode": "hop3_dual_path", "uses_oracle": False},
    }
    row.update(timing_metrics(stop_ns, fcs))
    row["deployable_mvp"] = is_deployable_mvp(row)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        full = load_full_dataset()
        head, struct_floor, knn_floor, knn_thr, pfn = setup_fallback_stack(model, tokenizer, device, profile)
        row = eval_hop3_dual_path(
            head, model, tokenizer, full, device=device, seed=99, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
        )
        return {
            "full_419": row,
            "gap_indices": list(GAP_INDICES),
            "mentor_brief": (
                f"E4 3跳双路：acc {row['accuracy']:.1%} "
                f"分歧 {row['dual_disagree_count']} gap_fixed {row['gap_fixed']}/3。"
            ),
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "e4_hop3_dual_path", "E4 · 3跳双路", device=args.device)
    write_phase29_result("e4_hop3_dual_path", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
