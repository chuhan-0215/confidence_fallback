#!/usr/bin/env python3
"""第九轮：latent 反馈 schedule / 残差写回 — 4 步以后性能是否还能平台。"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "model"))

from boundary_detector import detect_max_accuracy  # noqa: E402
from coconut import Coconut  # noqa: E402
from coconut_feedback import (  # noqa: E402
    apply_feedback_config,
    default_feedback_strategies,
    post_step4_metrics,
)
from dataset_registry import load_slice  # noqa: E402
from evaluate_coconut import resolve_device  # noqa: E402
from run_experiment import ensure_checkpoint, sweep_latent_steps  # noqa: E402
from stokenizer import STokenizer  # noqa: E402


def load_base_state(checkpoint: Path) -> dict:
    return torch.load(checkpoint, map_location="cpu")


def build_model(
    checkpoint: Path,
    config_path: Path,
    device: torch.device,
    feedback_cfg: dict,
    base_state: Optional[dict] = None,
):
    state = base_state if base_state is not None else load_base_state(checkpoint)
    tokenizer = STokenizer()
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    base = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(str(config_path)))
    model = Coconut(base, latent_id, start_id, end_id, tokenizer.eos_token_id)
    model.load_state_dict(state, strict=False)
    apply_feedback_config(model, feedback_cfg)
    model.eval()
    model.base_causallm.to(device)
    return model, tokenizer


def boundary_from_sweep(sweep: List[dict]) -> dict:
    xs = [r["n_latent"] for r in sweep]
    ys = [r["accuracy"] for r in sweep]
    det = detect_max_accuracy(xs, ys)
    return {
        "boundary": det.boundary_x,
        "max_accuracy": max(ys) if ys else 0.0,
        "max_accuracy_at_steps": int(det.boundary_x) if det.boundary_x else None,
    }


def run_feedback_schedule_experiment(
    slice_ids: List[str],
    strategies: List[dict],
    max_samples_by_slice: Dict[str, int],
    latent_min: int = 1,
    latent_max: int = 8,
    device_name: str = "cpu",
    progress_cb: Optional[Callable[[str, str, int, int], None]] = None,
) -> dict:
    config_path = ROOT / "configs" / "symbol-2layer-8head-768dim.json"
    checkpoint = ensure_checkpoint(ROOT / "checkpoints" / "checkpoint_300")
    device = resolve_device(device_name)
    base_state = load_base_state(checkpoint)

    rows: List[dict] = []
    t0 = time.time()

    for si, strategy in enumerate(strategies):
        sid = strategy.get("id") or f"strategy_{si}"
        label = strategy.get("label") or sid
        print(f"[strategy {si + 1}/{len(strategies)}] {label}", flush=True)

        cfg = {k: v for k, v in strategy.items() if k not in ("id", "label")}
        model, tokenizer = build_model(
            checkpoint, config_path, device, cfg, base_state=base_state
        )

        for sj, slice_id in enumerate(slice_ids):
            cap = max_samples_by_slice.get(slice_id)
            meta, dataset = load_slice(slice_id, max_samples=cap)
            if not dataset:
                continue

            if progress_cb:
                progress_cb(
                    label,
                    slice_id,
                    si * len(slice_ids) + sj + 1,
                    len(strategies) * len(slice_ids),
                )

            sweep = sweep_latent_steps(
                model,
                tokenizer,
                dataset,
                device,
                latent_min=latent_min,
                latent_max=latent_max,
            )
            bd = boundary_from_sweep(sweep)
            p4 = post_step4_metrics(sweep)

            rows.append(
                {
                    "strategy_id": sid,
                    "strategy_label": label,
                    "feedback": copy.deepcopy(cfg),
                    "slice_id": slice_id,
                    "slice_label": meta["label"],
                    "sample_count": len(dataset),
                    "boundary": bd["boundary"],
                    "max_accuracy": bd["max_accuracy"],
                    "max_accuracy_at_steps": bd["max_accuracy_at_steps"],
                    "post_step4": p4,
                    "latent_sweep": sweep,
                }
            )
            print(
                f"  {slice_id} n={len(dataset)}: @4={p4.get('acc_at_4')} "
                f"post4_min={p4.get('post4_min')} drop={p4.get('post4_drop_pp')}pp",
                flush=True,
            )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return analyze_feedback_rows(rows, elapsed=time.time() - t0)


def _feedback_strategy_score(row: dict, baseline_acc4: float | None = None) -> tuple:
    """Higher is better: keep acc@4, then maximize post4_drop_pp (least negative)."""
    drop = row.get("post4_drop_pp")
    acc4 = row.get("acc_at_4")
    if drop is None:
        return (0, -999.0, 0, 0.0)
    acc_ok = 1
    if baseline_acc4 is not None and acc4 is not None and acc4 < baseline_acc4 - 0.002:
        acc_ok = 0
    return (acc_ok, float(drop), 1 if row.get("post4_non_decreasing") else 0, float(acc4 or 0))


def analyze_feedback_rows(rows: List[dict], elapsed: float) -> dict:
    by_slice: Dict[str, List[dict]] = {}
    for r in rows:
        by_slice.setdefault(r["slice_id"], []).append(r)

    comparison_table = []
    insights: List[str] = []

    for slice_id, group in by_slice.items():
        baseline = next((g for g in group if g["strategy_id"] == "baseline"), group[0])
        b_p4 = baseline.get("post_step4") or {}
        b_drop = b_p4.get("post4_drop_pp")
        b_acc4 = b_p4.get("acc_at_4")

        table_rows = []
        for g in group:
            p4 = g.get("post_step4") or {}
            table_rows.append(
                {
                    "slice_id": slice_id,
                    "slice_label": g["slice_label"],
                    "strategy_id": g["strategy_id"],
                    "strategy_label": g["strategy_label"],
                    "sample_count": g["sample_count"],
                    "acc_at_4": p4.get("acc_at_4"),
                    "post4_min": p4.get("post4_min"),
                    "post4_drop_pp": p4.get("post4_drop_pp"),
                    "post4_range_pp": p4.get("post4_range_pp"),
                    "post4_non_decreasing": p4.get("post4_non_decreasing"),
                    "max_accuracy": g["max_accuracy"],
                    "boundary": g["boundary"],
                    "feedback": g["feedback"],
                }
            )
        comparison_table.extend(table_rows)

        best = max(table_rows, key=lambda r: _feedback_strategy_score(r, b_acc4))
        best_p4 = best

        if b_drop is not None and best_p4.get("post4_drop_pp") is not None:
            delta = round(b_drop - best_p4["post4_drop_pp"], 1)
            insights.append(
                f"{slice_id}：baseline 4 步后最多跌 {b_drop}pp；"
                f"最优「{best['strategy_label']}」4 步后最多跌 {best_p4['post4_drop_pp']}pp"
                f"（改善 {delta}pp）"
                + (
                    "，5–8 步不低于第 4 步。"
                    if best_p4.get("post4_non_decreasing")
                    else "。"
                )
            )

    full_rows = [r for r in comparison_table if r["slice_id"] == "full"]
    baseline_full = next((r for r in full_rows if r["strategy_id"] == "baseline"), None)
    b_acc4_full = (baseline_full or {}).get("acc_at_4")
    winner = None
    if full_rows:
        winner = max(full_rows, key=lambda r: _feedback_strategy_score(r, b_acc4_full))
    elif comparison_table:
        winner = max(comparison_table, key=lambda r: _feedback_strategy_score(r, None))

    recommendation = None
    if winner:
        recommendation = {
            "strategy_id": winner["strategy_id"],
            "strategy_label": winner["strategy_label"],
            "feedback": winner["feedback"],
            "acc_at_4": winner.get("acc_at_4"),
            "post4_drop_pp": winner.get("post4_drop_pp"),
            "post4_non_decreasing": winner.get("post4_non_decreasing"),
        }

    return {
        "ok": True,
        "experiment_type": "feedback_schedule_compare",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_sec": round(elapsed, 2),
        "rows": rows,
        "comparison": {
            "table": comparison_table,
            "insights": insights,
            "recommendation": recommendation,
        },
        "mechanism_note": (
            "scale 模式：写回 embedding = hidden × α（与第七轮 α 扫描相同）。"
            "residual 模式：写回 = (1-β)·原 embedding + β·hidden；β=0 时额外步不再改 embedding。"
        ),
    }


def parse_max_samples(raw: str, slice_ids: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not raw:
        return out
    for part in raw.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = int(v.strip())
    return out


def main():
    parser = argparse.ArgumentParser(description="Latent feedback schedule experiment (round 9)")
    parser.add_argument(
        "--slices",
        default="hops_3,hops_4,full",
        help="Comma-separated slice ids",
    )
    parser.add_argument(
        "--max-samples",
        default="hops_3=60,hops_4=60,full=419",
        help="Per-slice caps, e.g. hops_3=60,full=419",
    )
    parser.add_argument("--latent-min", type=int, default=1)
    parser.add_argument("--latent-max", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--strategies",
        default="",
        help="Comma-separated strategy ids (default: all built-in)",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "results" / "feedback_schedule_latest.json"),
    )
    args = parser.parse_args()

    slice_ids = [s.strip() for s in args.slices.split(",") if s.strip()]
    caps = parse_max_samples(args.max_samples, slice_ids)
    strategies = default_feedback_strategies()
    if args.strategies.strip():
        wanted = {s.strip() for s in args.strategies.split(",") if s.strip()}
        strategies = [s for s in strategies if s["id"] in wanted]

    result = run_feedback_schedule_experiment(
        slice_ids=slice_ids,
        strategies=strategies,
        max_samples_by_slice=caps,
        latent_min=args.latent_min,
        latent_max=args.latent_max,
        device_name=args.device,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["comparison"], ensure_ascii=False, indent=2))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
