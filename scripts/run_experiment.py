#!/usr/bin/env python3
"""Automatic boundary sweep for Coconut continuous-thought efficiency.

Based on: Reasoning by Superposition (NeurIPS 2025)
https://arxiv.org/abs/2505.12514
Code: https://github.com/Ber666/reasoning-by-superposition
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from eval_profile import EvalProfile, parse_eval_profile  # noqa: E402
from boundary_detector import (  # noqa: E402
    detect_kneedle,
    detect_max_accuracy,
    detect_plateau_drop,
)
from boundary_analysis import analyze_boundary  # noqa: E402
from evaluate_coconut import (  # noqa: E402
    dataset_meta,
    evaluate_latent_steps,
    load_coconut_model,
    load_dataset,
    resolve_device,
)
from graph_utils import graph_diameter, reasoning_hops  # noqa: E402


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_config(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_checkpoint(checkpoint: Path) -> Path:
    if checkpoint.is_file():
        return checkpoint
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading Coconut checkpoint (hf-mirror)...")
    try:
        import os
        from huggingface_hub import hf_hub_download

        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        path = hf_hub_download(
            repo_id="Shibo-UCSD/coconut-theory",
            filename="checkpoint_300",
            local_dir=str(checkpoint.parent),
        )
        return Path(path)
    except Exception as exc:
        raise SystemExit(
            "Checkpoint missing. Run: python scripts/download_checkpoint.py\n" + str(exc)
        )


def sweep_latent_steps(
    model,
    tokenizer,
    dataset,
    device,
    latent_min: int,
    latent_max: int,
    progress_cb=None,
    eval_profile: EvalProfile | None = None,
):
    rows = []
    step_total = latent_max - latent_min + 1
    for i, n_latent in enumerate(range(latent_min, latent_max + 1)):
        def sample_cb(completed, sample_total, current_latent):
            if progress_cb:
                progress_cb(
                    i,
                    step_total,
                    {
                        "phase": "evaluating",
                        "n_latent": current_latent,
                        "sample_done": completed,
                        "sample_total": sample_total,
                    },
                )

        row = evaluate_latent_steps(
            model,
            tokenizer,
            dataset,
            n_latent,
            device,
            seed=42,
            sample_cb=sample_cb,
            eval_profile=eval_profile,
        )
        rows.append(row)
        if progress_cb:
            progress_cb(i + 1, step_total, row)
    return rows


def run_on_dataset(
    model,
    tokenizer,
    dataset,
    device,
    latent_min: int = 1,
    latent_max: int = 8,
    skip_hop_groups: bool = False,
    progress_cb=None,
    slice_meta: dict | None = None,
    eval_profile: EvalProfile | None = None,
) -> dict:
    """对已加载模型在单个数据集上完成 latent 扫描与边界分析。"""
    t0 = time.time()

    latent_rows = sweep_latent_steps(
        model,
        tokenizer,
        dataset,
        device,
        latent_min,
        latent_max,
        progress_cb,
        eval_profile=eval_profile,
    )

    xs = [r["n_latent"] for r in latent_rows]
    eff = [r["efficiency"] for r in latent_rows]
    acc = [r["accuracy"] for r in latent_rows]

    boundary_acc = detect_max_accuracy(xs, acc)
    boundary_eff = detect_plateau_drop(xs, eff)
    boundary_knee = detect_kneedle(xs, eff)
    peak_row = max(latent_rows, key=lambda r: (r["accuracy"], -r["n_latent"]))

    hop_groups = []
    if not skip_hop_groups:
        if progress_cb:
            progress_cb(len(latent_rows), len(latent_rows), {"phase": "hop_groups"})
        hop_groups = sweep_by_hop_groups(
            model, tokenizer, dataset, device, latent_max, eval_profile=eval_profile
        )

    if progress_cb:
        progress_cb(len(latent_rows), len(latent_rows), {"phase": "analyzing"})

    boundary_block = {
        "recommended_latent_steps": boundary_acc.boundary_x,
        "max_accuracy": peak_row["accuracy"],
        "max_accuracy_at_steps": peak_row["n_latent"],
        "max_accuracy_detail": boundary_acc.__dict__,
        "efficiency_plateau": boundary_eff.__dict__,
        "kneedle": boundary_knee.__dict__,
        "interpretation": (
            "边界取准确率最高的连续思维步数（并列时取更少步数）。"
            "超过该步数后准确率不再提升，额外计算不再带来收益。"
        ),
    }

    why_analysis = analyze_boundary(
        dataset, latent_rows, boundary_block, hop_groups or None
    )

    meta = dataset_meta(dataset)
    theoretical = {
        "mean_reasoning_hops": meta["mean_reasoning_hops"],
        "mean_root_target_distance": meta["mean_root_target_distance"],
        "paper_claim": "D continuous-thought steps suffice for diameter-D BFS (Zhu et al., NeurIPS 2025)",
        "graph_profile": why_analysis.get("dataset_graph_profile"),
    }

    result = {
        "ok": True,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_sec": round(time.time() - t0, 2),
        "dataset": meta,
        "theoretical": theoretical,
        "latent_sweep": latent_rows,
        "boundary": boundary_block,
        "why_analysis": why_analysis,
        "by_reasoning_hops": hop_groups,
    }
    if slice_meta:
        result["slice_meta"] = slice_meta
    if eval_profile is not None:
        result["eval_profile"] = eval_profile.to_dict()
    return result


def sweep_by_hop_groups(
    model, tokenizer, dataset, device, latent_max: int, eval_profile: EvalProfile | None = None
):
    groups = {}
    for sample in dataset:
        hops = reasoning_hops(sample)
        groups.setdefault(hops, []).append(sample)

    out = []
    for hops in sorted(groups):
        subset = groups[hops]
        latent_rows = sweep_latent_steps(
            model,
            tokenizer,
            subset,
            device,
            1,
            latent_max,
            eval_profile=eval_profile,
        )
        xs = [r["n_latent"] for r in latent_rows]
        eff = [r["efficiency"] for r in latent_rows]
        acc = [r["accuracy"] for r in latent_rows]
        b1 = detect_plateau_drop(xs, eff)
        b2 = detect_kneedle(xs, eff)
        b_acc = detect_max_accuracy(xs, acc)
        out.append(
            {
                "reasoning_hops": hops,
                "sample_count": len(subset),
                "mean_diameter": round(
                    sum(graph_diameter(s) for s in subset) / len(subset), 3
                ),
                "latent_sweep": latent_rows,
                "boundary_accuracy": b_acc.__dict__,
                "boundary_efficiency": {
                    "plateau_drop": b1.__dict__,
                    "kneedle": b2.__dict__,
                },
                "accuracy_at_max_latent": latent_rows[-1]["accuracy"] if latent_rows else 0,
            }
        )
    return out


def run_experiment(
    max_samples=None,
    latent_min: int = 1,
    latent_max: int = 8,
    device_name: str = "auto",
    skip_hop_groups: bool = False,
    progress_cb=None,
    status_file=None,
    data_path: Path | None = None,
) -> dict:
    data_path = data_path or (ROOT / "data" / "prosqa_test_graph_4_coconut.json")
    config_path = ROOT / "configs" / "symbol-2layer-8head-768dim.json"
    checkpoint = ROOT / "checkpoints" / "checkpoint_300"

    device = resolve_device(device_name)
    checkpoint = ensure_checkpoint(checkpoint)
    dataset = load_dataset(data_path, max_samples=max_samples)

    def report(done, total, row):
        if status_file:
            write_status(
                Path(status_file),
                {
                    "running": True,
                    "phase": "sweep",
                    "progress": {"done": done, "total": total, "last": row},
                },
            )
        if progress_cb:
            progress_cb(done, total, row)

    if status_file:
        write_status(
            Path(status_file),
            {
                "running": True,
                "phase": "loading_model",
                "progress": {"done": 0, "total": latent_max - latent_min + 1},
            },
        )

    t0 = time.time()
    model, tokenizer = load_coconut_model(checkpoint, config_path, device)

    report(0, latent_max - latent_min + 1, {"phase": "model_loaded"})

    core = run_on_dataset(
        model,
        tokenizer,
        dataset,
        device,
        latent_min=latent_min,
        latent_max=latent_max,
        skip_hop_groups=skip_hop_groups,
        progress_cb=report,
    )

    result = {
        **core,
        "duration_sec": round(time.time() - t0, 2),
        "device": str(device),
        "checkpoint": str(checkpoint),
        "data_path": str(data_path),
        "references": {
            "paper": "https://arxiv.org/abs/2505.12514",
            "code": "https://github.com/Ber666/reasoning-by-superposition",
        },
    }
    if status_file:
        write_status(
            Path(status_file),
            {
                "running": False,
                "phase": "done",
                "progress": {"done": 1, "total": 1},
                "boundary": result["boundary"],
            },
        )

    return result


def main():
    parser = argparse.ArgumentParser(description="Coconut boundary experiment")
    parser.add_argument("--config", default="", help="JSON config file")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--latent-min", type=int, default=1)
    parser.add_argument("--latent-max", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-hop-groups", action="store_true")
    parser.add_argument("--output", default=str(ROOT / "results" / "latest.json"))
    parser.add_argument("--status-file", default=str(ROOT / "results" / "status.json"))
    args = parser.parse_args()

    cfg = load_config(Path(args.config)) if args.config else {}
    max_samples = args.max_samples if args.max_samples is not None else cfg.get("max_samples", 80)

    def progress(done, total, row):
        print(f"[{done}/{total}] latent sweep:", row, flush=True)

    try:
        result = run_experiment(
            max_samples=max_samples,
            latent_min=cfg.get("latent_min", args.latent_min),
            latent_max=cfg.get("latent_max", args.latent_max),
            device_name=cfg.get("device", args.device),
            skip_hop_groups=cfg.get("skip_hop_groups", args.skip_hop_groups),
            progress_cb=progress,
            status_file=cfg.get("status_file", args.status_file),
        )
    except Exception as exc:
        sf = Path(cfg.get("status_file", args.status_file))
        write_status(sf, {"running": False, "phase": "error", "error": str(exc)})
        raise

    out = Path(cfg.get("output", args.output))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["boundary"], ensure_ascii=False, indent=2))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
