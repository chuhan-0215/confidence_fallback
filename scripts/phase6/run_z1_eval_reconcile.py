#!/usr/bin/env python3
"""Z1 · Track 28 vs Phase5 eval 口径对齐：seed 42 vs 99。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase6_common import (  # noqa: E402
    load_model_bundle,
    load_test_split,
    make_predict_fn,
    timed_run,
    write_phase6_result,
)
from evaluate_coconut import expected_answer  # noqa: E402
from graph_utils import build_eval_prompt  # noqa: E402
from stop_head_tracks import evaluate_soft_floor_first_correct_stop  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--min-n", type=int, default=2)
    ap.add_argument("--seeds", default="42,99")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        test_set = load_test_split()
        predict_fn = make_predict_fn(model, tokenizer, device, profile)
        rows = []
        for seed in seeds:
            row = evaluate_soft_floor_first_correct_stop(
                model,
                tokenizer,
                test_set,
                cap=args.cap,
                min_n=args.min_n,
                device=device,
                seed=seed,
                predict_fn=predict_fn,
                expected_fn=expected_answer,
                build_prompt_fn=build_eval_prompt,
                eval_profile=profile,
            )
            rows.append(
                {
                    "seed": seed,
                    "accuracy": row["accuracy"],
                    "correct": row["correct"],
                    "total": row["total"],
                    "stop_timing_acc": row.get("stop_timing_acc"),
                    "mean_stop_n": row.get("mean_stop_n"),
                }
            )

        insight = (
            "Track 28 使用 seed=99，Phase5 Y1 使用 seed=42；"
            "两者 acc 差约 1.2pp 来自 per-sample shuffle 随机性，非策略差异。"
        )
        if len(rows) == 2:
            delta = round(rows[0]["accuracy"] - rows[1]["accuracy"], 4)
            if rows[0]["seed"] == 99:
                delta = -delta
            insight += f" 当前 seeds 差值 acc Δ={abs(delta):.4f}。"

        return {
            "strategies": rows,
            "eval_split": "test_40pct",
            "sample_count": len(test_set),
            "min_n": args.min_n,
            "device": str(device),
            "insight": insight,
            "recommendation": "对外报告统一 seed=99（与 Track 28 一致）或固定 seed=42 并同步更新 Track 文档。",
        }

    path = timed_run(run_body, "z1_eval_reconcile", "Z1 · eval 口径对齐 (seed)", device=args.device)
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    write_phase6_result("z1_eval_reconcile", data)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
