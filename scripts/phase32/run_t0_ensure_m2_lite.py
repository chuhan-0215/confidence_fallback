#!/usr/bin/env python3
"""T0 · 开发机缺 M2 权重时，用精简训练生成 m2_enough_stop_head.pt（供 Phase32 使用）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase10"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase32_common import M2_HEAD, make_predict_fn, m2_head_ready, write_phase32_result  # noqa: E402
from _phase10_common import CAP, load_splits, timed_run  # noqa: E402
from evaluate_coconut import expected_answer  # noqa: E402
from graph_utils import build_eval_prompt  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402
from run_adaptive_stop_experiment import predict_at_n  # noqa: E402
from stop_head import (  # noqa: E402
    build_rich_stop_examples_for_samples,
    split_train_val_samples,
    train_rich_stop_head,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--max-train", type=int, default=80, help="训练子集上限（加速开发机）")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if m2_head_ready() and not args.force:
        print(f"M2 head already exists: {M2_HEAD}")
        return

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        train_set, _ = load_splits()
        train_sub, val_sub = split_train_val_samples(train_set, val_ratio=0.2, seed=43)
        train_sub = train_sub[: args.max_train]
        val_sub = val_sub[: max(20, args.max_train // 4)]
        pfn = make_predict_fn(model, tokenizer, device, profile)

        train_ex = build_rich_stop_examples_for_samples(
            model, tokenizer, train_sub, cap=CAP, device=device, seed=42,
            predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt,
            eval_profile=profile, label_mode="is_correct",
        )
        val_ex = build_rich_stop_examples_for_samples(
            model, tokenizer, val_sub, cap=CAP, device=device, seed=43,
            predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt,
            eval_profile=profile, label_mode="is_correct",
        )
        head, train_metrics = train_rich_stop_head(train_ex, val_ex, epochs=args.epochs, device=device)
        M2_HEAD.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": head.state_dict(), "train_metrics": train_metrics, "label_mode": "is_correct", "lite": True},
            M2_HEAD,
        )
        return {
            "checkpoint": str(M2_HEAD.relative_to(ROOT)),
            "train_samples": len(train_sub),
            "val_samples": len(val_sub),
            "epochs": args.epochs,
            "train_metrics": train_metrics,
            "note": "lite M2 for dev/Phase32; GPU 完整训练请跑 phase10/run_m2_learned_enough_stop.py",
        }

    path = timed_run(run_body, "t0_ensure_m2_lite", "T0 · 精简 M2 训练", device=args.device)
    write_phase32_result("t0_ensure_m2_lite", __import__("json").loads(path.read_text(encoding="utf-8")))
    print(f"Wrote M2 head to {M2_HEAD}")


if __name__ == "__main__":
    main()
