#!/usr/bin/env python3
"""X6 · Latent 反馈系数 α 表面：固定 n=d 与 n=3 下 α 网格对准确率的影响。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase4_common import timed_run  # noqa: E402
from coconut_feedback import apply_feedback_config  # noqa: E402
from evaluate_coconut import load_dataset, resolve_device  # noqa: E402
from run_auto_submit_experiment import blind_choice_depth, evaluate_policy, make_policies  # noqa: E402
from run_model_perturb_experiment import build_model  # noqa: E402
from run_experiment import ensure_checkpoint  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=120)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alphas", default="0.5,0.75,1.0,1.25,1.5,2.0")
    args = ap.parse_args()
    alphas = [float(x) for x in args.alphas.split(",")]

    config_path = ROOT / "configs" / "symbol-2layer-8head-768dim.json"
    checkpoint = ensure_checkpoint(ROOT / "checkpoints" / "checkpoint_300")
    device = resolve_device(args.device)
    base_state = torch.load(checkpoint, map_location="cpu")
    dataset = load_dataset(ROOT / "data" / "prosqa_test_graph_4_coconut.json", args.max_samples)

    def auto_route_alpha(alpha: float):
        def _policy(sample, cap):
            d = blind_choice_depth(sample)
            return {
                "n_latent": d,
                "feedback": {"latent_feedback_scale": alpha},
            }

        return _policy

    grid = []
    for alpha in alphas:
        model, tokenizer = build_model(
            checkpoint,
            config_path,
            device,
            latent_feedback_scale=alpha,
            base_state=base_state,
        )
        apply_feedback_config(model, {"latent_feedback_scale": alpha})
        r_fixed3 = evaluate_policy(
            model, tokenizer, dataset, make_policies()["fixed_3"], device
        )
        r_route = evaluate_policy(
            model, tokenizer, dataset, auto_route_alpha(alpha), device
        )
        grid.append(
            {
                "alpha": alpha,
                "fixed_3_acc": r_fixed3["accuracy"],
                "auto_route_acc": r_route["accuracy"],
            }
        )
        del model

    path = timed_run(
        lambda: {
            "alpha_grid": grid,
            "sample_count": len(dataset),
            "insight": "α≠1 改变 latent 写回强度；看边界是否随 α 平移",
        },
        "x6_latent_alpha_surface",
        "X6 · Latent 反馈 α 表面",
        device=args.device,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
