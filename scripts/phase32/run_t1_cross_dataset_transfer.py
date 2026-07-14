#!/usr/bin/env python3
"""T1 · 通解跨数据集迁移：主路径 vs confidence_fallback(τ=0.48 冻结迁移)。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase25"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fallback_eval import eval_confidence_fallback, setup_fallback_stack  # noqa: E402
from _phase32_common import (  # noqa: E402
    TRANSFER_THR,
    eval_main_path,
    m2_head_ready,
    slice_ids_for_tier,
    write_phase32_result,
    write_status,
)
from boundary_budget import make_structure_budget_fn  # noqa: E402
from dataset_registry import load_slice  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402
from phase23._phase23_common import CAP, MIN_N  # noqa: E402


def rollup(rows: list[dict]) -> dict:
    """Phase 32 原始汇总逻辑（与 GPU 回传结果一致）。"""
    if not rows:
        return {}
    deltas = [r["delta_pp"] for r in rows if r.get("delta_pp") is not None]
    helps = sum(1 for d in deltas if d > 0.005)
    hurts = sum(1 for d in deltas if d < -0.005)
    by_cat: dict[str, list[float]] = {}
    for r in rows:
        by_cat.setdefault(r.get("category") or "unknown", []).append(r["delta_pp"])
    return {
        "slice_count": len(rows),
        "transfer_helps_count": helps,
        "transfer_hurts_count": hurts,
        "transfer_neutral_count": len(rows) - helps - hurts,
        "mean_delta_pp": round(sum(deltas) / len(deltas), 3) if deltas else None,
        "mean_main_acc": round(sum(r["main_acc"] for r in rows) / len(rows), 4),
        "mean_transfer_acc": round(
            sum(r["transfer_acc"] for r in rows if r.get("transfer_acc") is not None)
            / max(1, sum(1 for r in rows if r.get("transfer_acc") is not None)), 4,
        ),
        "by_category_mean_delta_pp": {
            k: round(sum(v) / len(v), 3) for k, v in sorted(by_cat.items())
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tier", choices=("smoke", "core", "all"), default="core")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--skip-fallback", action="store_true")
    args = ap.parse_args()

    if not args.skip_fallback and not m2_head_ready():
        raise SystemExit(
            f"缺少 {ROOT / 'results/phase10/m2_enough_stop_head.pt'}；"
            "请先运行: python3 scripts/phase32/run_t0_ensure_m2_lite.py --device cpu"
        )

    slice_ids = slice_ids_for_tier(args.tier)
    t0 = time.time()
    write_status({"running": True, "phase": "loading_model", "done": 0, "total": len(slice_ids)})

    model, tokenizer, device, profile = load_model_bundle(args.device)
    struct_floor = make_structure_budget_fn(min_n=MIN_N, cap=CAP)

    head = knn_floor = knn_thr = pfn = None
    if not args.skip_fallback:
        write_status({"running": True, "phase": "setup_fallback_stack", "done": 0, "total": len(slice_ids)})
        head, struct_floor_fb, knn_floor, knn_thr, pfn = setup_fallback_stack(model, tokenizer, device, profile)
        struct_floor = struct_floor_fb

    rows = []
    for i, sid in enumerate(slice_ids):
        meta, samples = load_slice(sid, max_samples=args.max_samples)
        write_status({
            "running": True, "phase": "evaluating", "done": i, "total": len(slice_ids),
            "current_slice": sid, "label": meta.get("label"),
        })
        main_row = eval_main_path(
            model, tokenizer, samples, device=device, seed=args.seed,
            profile=profile, struct_floor=struct_floor,
        )
        transfer_row = None
        if not args.skip_fallback and head is not None:
            transfer_row = eval_confidence_fallback(
                head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
                struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
                fallback_thr=TRANSFER_THR,
            )
        rows.append({
            "slice_id": sid,
            "label": meta.get("label"),
            "category": meta.get("category"),
            "construction": meta.get("construction"),
            "n_samples": len(samples),
            "main_acc": main_row["accuracy"],
            "transfer_acc": transfer_row["accuracy"] if transfer_row else None,
            "delta_pp": round((transfer_row["accuracy"] - main_row["accuracy"]) * 100, 2) if transfer_row else None,
            "fallback_rate": transfer_row.get("fallback_rate") if transfer_row else None,
            "transfer_helps": transfer_row and (transfer_row["accuracy"] - main_row["accuracy"]) * 100 > 0.5,
            "transfer_hurts": transfer_row and (transfer_row["accuracy"] - main_row["accuracy"]) * 100 < -0.5,
            "eval_profile": meta.get("eval_profile"),
        })
        delta_pp = rows[-1].get("delta_pp")
        print(
            f"[{i+1}/{len(slice_ids)}] {sid}: main={main_row['accuracy']:.1%}"
            + (f" transfer={transfer_row['accuracy']:.1%} Δ={delta_pp:+.2f}pp" if transfer_row else ""),
            flush=True,
        )

    summary = rollup(rows)
    payload = {
        "experiment_id": "t1_cross_dataset_transfer",
        "title": "T1 · 通解跨数据集迁移（τ=0.48 冻结）",
        "tier": args.tier,
        "max_samples": args.max_samples,
        "transfer_thr": TRANSFER_THR,
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "summary": summary,
        "slices": rows,
    }
    write_phase32_result("t1_cross_dataset_transfer", payload)
    write_status({"running": False, "phase": "done", "summary": summary})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
