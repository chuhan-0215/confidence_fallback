#!/usr/bin/env python3
"""F0 · AlienBench_v1 冻结通解迁移（τ=0.48，需 GPU）。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase25"))
sys.path.insert(0, str(ROOT / "scripts" / "phase32"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fallback_eval import eval_confidence_fallback, setup_fallback_stack  # noqa: E402
from _phase32_common import eval_main_path, m2_head_ready, write_status  # noqa: E402
from _phase45_common import (  # noqa: E402
    ALIEN_SLICE_IDS,
    TRANSFER_THR,
    rollup_transfer_rows,
    write_phase45_result,
)
from boundary_budget import make_structure_budget_fn  # noqa: E402
from dataset_registry import load_slice  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402
from phase23._phase23_common import CAP, MIN_N  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument(
        "--slices",
        nargs="*",
        default=ALIEN_SLICE_IDS,
        help="默认跑全部 5 个 alien 切片",
    )
    args = ap.parse_args()

    if not m2_head_ready():
        raise SystemExit(
            f"缺少 M2 head；请先运行 phase32/run_t0_ensure_m2_lite.py"
        )

    slice_ids = list(args.slices)
    t0 = time.time()
    write_status({
        "running": True,
        "phase": "phase45_f0_loading",
        "done": 0,
        "total": len(slice_ids),
    })

    model, tokenizer, device, profile = load_model_bundle(args.device)
    struct_floor = make_structure_budget_fn(min_n=MIN_N, cap=CAP)
    head, struct_floor_fb, knn_floor, knn_thr, pfn = setup_fallback_stack(
        model, tokenizer, device, profile
    )
    struct_floor = struct_floor_fb

    rows = []
    for i, sid in enumerate(slice_ids):
        meta, samples = load_slice(sid, max_samples=args.max_samples)
        write_status({
            "running": True,
            "phase": "phase45_f0_evaluating",
            "done": i,
            "total": len(slice_ids),
            "current_slice": sid,
        })
        main_row = eval_main_path(
            model, tokenizer, samples, device=device, seed=args.seed,
            profile=profile, struct_floor=struct_floor,
        )
        transfer_row = eval_confidence_fallback(
            head, model, tokenizer, samples, device=device, seed=args.seed,
            profile=profile, struct_floor=struct_floor,
            knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            fallback_thr=TRANSFER_THR,
        )
        delta = round((transfer_row["accuracy"] - main_row["accuracy"]) * 100, 2)
        rows.append({
            "slice_id": sid,
            "label": meta.get("label"),
            "category": meta.get("category"),
            "construction": meta.get("construction"),
            "topology": sid.replace("alien_", "") if sid != "alien_full" else "mixed",
            "n_samples": len(samples),
            "main_acc": main_row["accuracy"],
            "transfer_acc": transfer_row["accuracy"],
            "delta_pp": delta,
            "fallback_rate": transfer_row.get("fallback_rate"),
            "transfer_helps": delta > 0.5,
            "transfer_hurts": delta < -0.5,
            "eval_profile": meta.get("eval_profile"),
        })
        print(
            f"[{i+1}/{len(slice_ids)}] {sid}: main={main_row['accuracy']:.1%} "
            f"transfer={transfer_row['accuracy']:.1%} Δ={delta:+.2f}pp",
            flush=True,
        )

    summary = rollup_transfer_rows(rows)
    payload = {
        "experiment_id": "f0_alien_benchmark_transfer",
        "title": "F0 · AlienBench_v1 冻结通解迁移（τ=0.48）",
        "benchmark": "AlienBench_v1",
        "transfer_thr": TRANSFER_THR,
        "device": str(device),
        "seed": args.seed,
        "max_samples": args.max_samples,
        "duration_sec": round(time.time() - t0, 2),
        "summary": summary,
        "slices": rows,
        "alien_note": (
            "刻意非 ProsQA：星型/梯子/沙漏/宽树 + 模块命名；"
            "仍属 Coconut 图可达格式，非 GSM8K 等真外部 benchmark"
        ),
    }
    path = write_phase45_result("f0_alien_benchmark_transfer", payload)
    write_status({"running": False, "phase": "phase45_f0_done", "summary": summary})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
