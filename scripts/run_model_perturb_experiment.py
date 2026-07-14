#!/usr/bin/env python3
"""固定 ProsQA 数据，只改模型数值（权重缩放 / latent 反馈系数），扫描边界是否移动。"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from boundary_detector import detect_max_accuracy  # noqa: E402
from dataset_registry import load_slice  # noqa: E402
from evaluate_coconut import load_coconut_model, resolve_device  # noqa: E402
from run_experiment import ensure_checkpoint, sweep_latent_steps  # noqa: E402
from coconut_feedback import apply_feedback_config  # noqa: E402


def load_base_state(checkpoint: Path) -> dict:
    return torch.load(checkpoint, map_location="cpu")


def build_model(
    checkpoint: Path,
    config_path: Path,
    device: torch.device,
    *,
    global_weight_scale: float = 1.0,
    latent_feedback_scale: float = 1.0,
    base_state: Optional[dict] = None,
):
    state = base_state
    if state is None:
        state = load_base_state(checkpoint)
    if global_weight_scale != 1.0:
        state = {k: v * global_weight_scale for k, v in state.items()}

    tokenizer_stub = None
    from transformers import AutoConfig, AutoModelForCausalLM

    sys.path.insert(0, str(ROOT / "model"))
    from coconut import Coconut  # noqa: E402
    from stokenizer import STokenizer  # noqa: E402

    tokenizer = STokenizer()
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    base = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(str(config_path)))
    model = Coconut(base, latent_id, start_id, end_id, tokenizer.eos_token_id)
    model.load_state_dict(state, strict=False)
    apply_feedback_config(model, {"latent_feedback_scale": latent_feedback_scale})
    model.eval()
    model.base_causallm.to(device)
    return model, tokenizer


def boundary_from_sweep(sweep: List[dict]) -> dict:
    xs = [r["n_latent"] for r in sweep]
    ys = [r["accuracy"] for r in sweep]
    det = detect_max_accuracy(xs, ys)
    acc_at = {r["n_latent"]: r["accuracy"] for r in sweep}
    mean_hops = None
    return {
        "boundary": det.boundary_x,
        "max_accuracy": max(ys) if ys else 0.0,
        "max_accuracy_at_steps": int(det.boundary_x) if det.boundary_x else None,
        "sweep": sweep,
        "acc_by_step": acc_at,
    }


def run_perturb_grid(
    slice_ids: List[str],
    perturbations: List[dict],
    max_samples: int,
    latent_min: int,
    latent_max: int,
    device_name: str = "cpu",
) -> dict:
    config_path = ROOT / "configs" / "symbol-2layer-8head-768dim.json"
    checkpoint = ensure_checkpoint(ROOT / "checkpoints" / "checkpoint_300")
    device = resolve_device(device_name)
    base_state = load_base_state(checkpoint)

    rows: List[dict] = []
    t0 = time.time()

    for pi, pert in enumerate(perturbations):
        g_scale = float(pert.get("global_weight_scale", 1.0))
        fb_scale = float(pert.get("latent_feedback_scale", 1.0))
        label = pert.get("label") or f"g{g_scale}_fb{fb_scale}"
        print(f"[perturb {pi + 1}/{len(perturbations)}] {label}", flush=True)

        model, tokenizer = build_model(
            checkpoint,
            config_path,
            device,
            global_weight_scale=g_scale,
            latent_feedback_scale=fb_scale,
            base_state=base_state,
        )

        for slice_id in slice_ids:
            meta, dataset = load_slice(slice_id, max_samples=max_samples)
            if not dataset:
                continue
            sweep = sweep_latent_steps(
                model,
                tokenizer,
                dataset,
                device,
                latent_min=latent_min,
                latent_max=latent_max,
            )
            bd = boundary_from_sweep(sweep)
            from graph_utils import reasoning_hops

            mean_hops = round(sum(reasoning_hops(s) for s in dataset) / len(dataset))
            acc_at_depth = bd["acc_by_step"].get(mean_hops)

            rows.append(
                {
                    "perturbation": copy.deepcopy(pert),
                    "perturbation_label": label,
                    "slice_id": slice_id,
                    "slice_label": meta["label"],
                    "mean_reasoning_hops": mean_hops,
                    "boundary": bd["boundary"],
                    "max_accuracy": bd["max_accuracy"],
                    "max_accuracy_at_steps": bd["max_accuracy_at_steps"],
                    "acc_at_depth": acc_at_depth,
                    "latent_sweep": sweep,
                }
            )
            print(
                f"  {slice_id}: boundary={bd['boundary']} max={bd['max_accuracy']:.3f} acc@d={acc_at_depth}",
                flush=True,
            )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return analyze_rows(rows, elapsed=time.time() - t0)


def analyze_rows(rows: List[dict], elapsed: float) -> dict:
    """归纳固定数据下「只改模型数值 → 边界怎么变」的规律。"""
    by_slice: Dict[str, List[dict]] = {}
    for r in rows:
        by_slice.setdefault(r["slice_id"], []).append(r)

    slice_laws = []
    for slice_id, group in by_slice.items():
        baseline = next(
            (g for g in group if g["perturbation"].get("latent_feedback_scale", 1.0) == 1.0
             and g["perturbation"].get("global_weight_scale", 1.0) == 1.0),
            group[0],
        )
        d = group[0]["mean_reasoning_hops"]
        fb_rows = sorted(
            [g for g in group if g["perturbation"].get("global_weight_scale", 1.0) == 1.0],
            key=lambda x: x["perturbation"].get("latent_feedback_scale", 1.0),
        )
        boundaries = [g["boundary"] for g in fb_rows if g["boundary"] is not None]
        acc_at_d = [g["acc_at_depth"] for g in fb_rows if g["acc_at_depth"] is not None]

        mono_boundary = all(
            boundaries[i] <= boundaries[i + 1]
            for i in range(len(boundaries) - 1)
        ) if len(boundaries) >= 2 else None

        slice_laws.append(
            {
                "slice_id": slice_id,
                "mean_hops": d,
                "baseline_boundary": baseline["boundary"],
                "baseline_acc_at_depth": baseline.get("acc_at_depth"),
                "feedback_scale_range": {
                    "min_fb": min(g["perturbation"]["latent_feedback_scale"] for g in fb_rows),
                    "max_fb": max(g["perturbation"]["latent_feedback_scale"] for g in fb_rows),
                    "boundaries": [
                        {
                            "latent_feedback_scale": g["perturbation"]["latent_feedback_scale"],
                            "boundary": g["boundary"],
                            "acc_at_depth": g["acc_at_depth"],
                            "max_accuracy": g["max_accuracy"],
                        }
                        for g in fb_rows
                    ],
                },
                "boundary_monotone_in_feedback": mono_boundary,
            }
        )

    unified = []
    all_fb = [g for g in rows if g["perturbation"].get("global_weight_scale", 1.0) == 1.0]
    if all_fb:
        shifted = sum(
            1 for g in all_fb
            if g["boundary"] != next(
                b["boundary"] for b in all_fb
                if b["slice_id"] == g["slice_id"]
                and b["perturbation"].get("latent_feedback_scale", 1.0) == 1.0
            )
        )
        unified.append(
            f"latent 反馈系数在 0.5–2.0 内扫描：{shifted}/{len(all_fb)} 组边界相对 baseline 发生变化。"
        )

    weight_rows = [g for g in rows if g["perturbation"].get("latent_feedback_scale", 1.0) == 1.0]
    degraded = sum(1 for g in weight_rows if (g["max_accuracy"] or 0) < 0.5)
    if weight_rows:
        unified.append(
            f"全局权重缩放：{degraded}/{len(weight_rows)} 组峰值准确率跌破 50%，"
            "大幅缩放主要破坏能力而非系统平移边界。"
        )

    return {
        "ok": True,
        "experiment_type": "model_perturb_boundary",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_sec": round(elapsed, 2),
        "rows": rows,
        "slice_laws": slice_laws,
        "unified_conclusion": " ".join(unified),
    }


def default_perturbations() -> List[dict]:
    out = [{"label": "baseline", "global_weight_scale": 1.0, "latent_feedback_scale": 1.0}]
    for fb in [0.5, 0.75, 1.25, 1.5, 2.0]:
        out.append({"label": f"fb×{fb}", "global_weight_scale": 1.0, "latent_feedback_scale": fb})
    for gw in [0.85, 0.95, 1.05, 1.15]:
        out.append({"label": f"weight×{gw}", "global_weight_scale": gw, "latent_feedback_scale": 1.0})
    return out


def main():
    parser = argparse.ArgumentParser(description="Model-only perturbation boundary experiment")
    parser.add_argument(
        "--slices",
        default="hops_3,hops_4,push_ext5_from3",
        help="Comma-separated slice ids (same data type, fixed ProsQA format)",
    )
    parser.add_argument("--max-samples", type=int, default=15)
    parser.add_argument("--latent-min", type=int, default=1)
    parser.add_argument("--latent-max", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        default=str(ROOT / "results" / "model_perturb_latest.json"),
    )
    args = parser.parse_args()

    slice_ids = [s.strip() for s in args.slices.split(",") if s.strip()]
    result = run_perturb_grid(
        slice_ids=slice_ids,
        perturbations=default_perturbations(),
        max_samples=args.max_samples,
        latent_min=args.latent_min,
        latent_max=args.latent_max,
        device_name=args.device,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"slice_laws": result["slice_laws"], "conclusion": result["unified_conclusion"]},
                     ensure_ascii=False, indent=2))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
