#!/usr/bin/env python3
"""W3 · 可部署路由：construction 路由 vs main_acc 路由。"""
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

from _fallback_eval import setup_fallback_stack  # noqa: E402
from _phase35_common import (  # noqa: E402
    MAIN_ACC_ROUTER_THR,
    eval_construction_router,
    eval_main_acc_router,
    load_p34_best,
    m2_head_ready,
    unique_slice_ids,
    write_phase35_result,
)
from dataset_registry import load_slice  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402
from shared.eval_paths import eval_main_path, make_slice_row, rollup_slice_rows  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()
    if not m2_head_ready():
        raise SystemExit(f"缺少 M2 head: {ROOT / 'results/phase10/m2_enough_stop_head.pt'}")

    t_low, t_mid = load_p34_best()
    t0 = time.time()
    model, tokenizer, device, profile = load_model_bundle(args.device)
    head, struct_floor, knn_floor, knn_thr, pfn = setup_fallback_stack(model, tokenizer, device, profile)

    rows_main, rows_const = [], []
    for i, sid in enumerate(unique_slice_ids()):
        meta, samples = load_slice(sid)
        main_row = eval_main_path(
            model, tokenizer, samples, device=device, seed=args.seed,
            profile=profile, struct_floor=struct_floor,
        )
        prow_main = eval_main_acc_router(
            head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            t_low=t_low, t_mid=t_mid, main_thr=MAIN_ACC_ROUTER_THR,
        )
        prow_const = eval_construction_router(
            head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            t_low=t_low, t_mid=t_mid, meta=meta,
        )
        rows_main.append(make_slice_row(meta, samples, main_row, prow_main, policy_name="main_acc_router"))
        rows_const.append(make_slice_row(meta, samples, main_row, prow_const, policy_name="construction_router"))
        print(f"[{i+1}] {sid}", flush=True)

    payload = {
        "experiment_id": "w3_adaptive_router",
        "title": "W3 · 自适应路由",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "params": {"t_low": t_low, "t_mid": t_mid, "main_thr": MAIN_ACC_ROUTER_THR},
        "summaries": {
            "main_acc_router": rollup_slice_rows(rows_main),
            "construction_router": rollup_slice_rows(rows_const),
        },
        "slices": {"main_acc_router": rows_main, "construction_router": rows_const},
    }
    write_phase35_result("w3_adaptive_router", payload)
    print(json.dumps(payload["summaries"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
