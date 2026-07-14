#!/usr/bin/env python3
"""Y2 · hop_split（T29）跨子集泛化，补 X2 仅测 soft_floor 的缺口。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase5_common import load_model_bundle, make_predict_fn, timed_run, write_phase5_result  # noqa: E402
from dataset_registry import (  # noqa: E402
    default_boundary_push_slice_ids,
    default_compare_slice_ids,
    load_slice,
)
from evaluate_coconut import expected_answer  # noqa: E402
from graph_utils import build_eval_prompt  # noqa: E402
from stop_head_tracks import evaluate_hop_split_first_correct_stop  # noqa: E402

SLICE_GROUPS = {
    "core": default_compare_slice_ids(),
    "boundary_push": default_boundary_push_slice_ids(),
}


def eval_slice_hop_split(model, tokenizer, device, profile, slice_id, max_samples, cap, min_n, predict_fn):
    meta, samples = load_slice(slice_id, max_samples=max_samples)
    hs = evaluate_hop_split_first_correct_stop(
        model,
        tokenizer,
        samples,
        cap=cap,
        min_n=min_n,
        split_depth=4,
        device=device,
        seed=42,
        predict_fn=predict_fn,
        expected_fn=expected_answer,
        build_prompt_fn=build_eval_prompt,
        eval_profile=profile,
    )
    return {
        "slice_id": slice_id,
        "label": meta.get("label"),
        "count": len(samples),
        "strategies": {
            "hop_split_fc": {
                "accuracy": hs["accuracy"],
                "stop_timing_acc": hs.get("stop_timing_acc"),
            }
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--group", choices=list(SLICE_GROUPS), default="core")
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--min-n", type=int, default=2)
    ap.add_argument("--max-samples", type=int, default=80)
    args = ap.parse_args()

    model, tokenizer, device, profile = load_model_bundle(args.device)
    predict_fn = make_predict_fn(model, tokenizer, device, profile)
    rows = [
        eval_slice_hop_split(
            model, tokenizer, device, profile, sid, args.max_samples, args.cap, args.min_n, predict_fn
        )
        for sid in SLICE_GROUPS[args.group]
    ]

    exp_id = f"y2_hop_split_slices_{args.group}"
    path = timed_run(
        lambda: {"slice_group": args.group, "slices": rows, "device": str(device)},
        exp_id,
        f"Y2 · hop_split 跨子集 ({args.group})",
        device=args.device,
    )
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    write_phase5_result(exp_id, data)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
