#!/usr/bin/env python3
"""X2 · 跨子集泛化：同一策略在不同 ProsQA 切片上的表现（OOD / 分布偏移）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase4_common import load_model_bundle, timed_run  # noqa: E402
from dataset_registry import (  # noqa: E402
    default_boundary_push_slice_ids,
    default_compare_slice_ids,
    load_slice,
)
from evaluate_coconut import expected_answer  # noqa: E402
from graph_utils import build_eval_prompt  # noqa: E402
from run_adaptive_stop_experiment import predict_at_n  # noqa: E402
from run_auto_submit_experiment import evaluate_policy, make_policies  # noqa: E402
from stop_head_tracks import evaluate_soft_floor_first_correct_stop  # noqa: E402

SLICE_GROUPS = {
    "core": default_compare_slice_ids(),
    "boundary_push": default_boundary_push_slice_ids(),
}


def eval_slice(model, tokenizer, device, profile, slice_id: str, max_samples, cap):
    meta, samples = load_slice(slice_id, max_samples=max_samples)
    policies = make_policies(cap=cap)
    out = {"slice_id": slice_id, "label": meta.get("label"), "count": len(samples), "strategies": {}}

    for name in ("fixed_3", "auto_route"):
        r = evaluate_policy(
            model, tokenizer, samples, policies[name], device, cap=cap, eval_profile=profile
        )
        out["strategies"][name] = {"accuracy": r["accuracy"]}

    sf = evaluate_soft_floor_first_correct_stop(
        model,
        tokenizer,
        samples,
        cap=cap,
        min_n=2,
        device=device,
        seed=42,
        predict_fn=lambda s, n, seed: predict_at_n(
            model, tokenizer, s, n, device, seed=seed, eval_profile=profile
        ),
        expected_fn=expected_answer,
        build_prompt_fn=build_eval_prompt,
        eval_profile=profile,
    )
    out["strategies"]["soft_floor_fc"] = {
        "accuracy": sf["accuracy"],
        "stop_timing_acc": sf.get("stop_timing_acc"),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=80)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--group", choices=list(SLICE_GROUPS), default="core")
    args = ap.parse_args()

    model, tokenizer, device, profile = load_model_bundle(args.device)
    slice_ids = SLICE_GROUPS[args.group]
    rows = [eval_slice(model, tokenizer, device, profile, sid, args.max_samples, args.cap) for sid in slice_ids]

    path = timed_run(
        lambda: {"slice_group": args.group, "slices": rows},
        f"x2_slice_generalization_{args.group}",
        f"X2 · 跨子集泛化 ({args.group})",
        device=args.device,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
