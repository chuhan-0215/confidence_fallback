#!/usr/bin/env python3
"""Y7 · soft_floor（T28）跨子集泛化，与 Y2 hop_split 对照。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase5"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase6_common import load_model_bundle, make_predict_fn, timed_run, write_phase6_result  # noqa: E402
from phase5.run_y2_hop_split_slices import SLICE_GROUPS  # noqa: E402
from dataset_registry import load_slice  # noqa: E402
from evaluate_coconut import expected_answer  # noqa: E402
from graph_utils import build_eval_prompt  # noqa: E402
from stop_head_tracks import evaluate_soft_floor_first_correct_stop  # noqa: E402


def eval_slice_soft_floor(model, tokenizer, device, profile, slice_id, max_samples, cap, min_n, predict_fn, seed):
    meta, samples = load_slice(slice_id, max_samples=max_samples)
    sf = evaluate_soft_floor_first_correct_stop(
        model,
        tokenizer,
        samples,
        cap=cap,
        min_n=min_n,
        device=device,
        seed=seed,
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
            "soft_floor_fc": {
                "accuracy": sf["accuracy"],
                "stop_timing_acc": sf.get("stop_timing_acc"),
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
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    model, tokenizer, device, profile = load_model_bundle(args.device)
    predict_fn = make_predict_fn(model, tokenizer, device, profile)
    rows = [
        eval_slice_soft_floor(
            model, tokenizer, device, profile, sid, args.max_samples, args.cap, args.min_n, predict_fn, args.seed
        )
        for sid in SLICE_GROUPS[args.group]
    ]

    exp_id = f"y7_soft_floor_slices_{args.group}"
    path = timed_run(
        lambda: {"slice_group": args.group, "slices": rows, "seed": args.seed, "device": str(device)},
        exp_id,
        f"Y7 · soft_floor 跨子集 ({args.group})",
        device=args.device,
    )
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    write_phase6_result(exp_id, data)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
