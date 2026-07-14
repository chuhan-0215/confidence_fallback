#!/usr/bin/env python3
"""V2 · Tri-zone (T_low, T_mid) grid：val 选参 + test 跨集验证。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase25"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fallback_eval import setup_fallback_stack  # noqa: E402
from _phase34_common import (  # noqa: E402
    T_LOW_GRID,
    T_MID_GRID,
    eval_main_path,
    eval_tri_zone,
    make_slice_row,
    m2_head_ready,
    rollup_slice_rows,
    split_val_test,
    unique_slice_ids,
    write_phase34_result,
)
from dataset_registry import load_slice  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()
    if not m2_head_ready():
        raise SystemExit(f"缺少 M2 head: {ROOT / 'results/phase10/m2_enough_stop_head.pt'}")

    t0 = time.time()
    model, tokenizer, device, profile = load_model_bundle(args.device)
    head, struct_floor, knn_floor, knn_thr, pfn = setup_fallback_stack(model, tokenizer, device, profile)

    pooled_val: list = []
    test_by_slice: list[tuple[dict, list]] = []
    for sid in unique_slice_ids():
        meta, samples = load_slice(sid)
        val, test = split_val_test(samples, val_ratio=0.2, seed=43)
        pooled_val.extend(val)
        if test:
            test_by_slice.append((meta, test))

    grid_trials = []
    best = {"t_low": None, "t_mid": None, "val_acc": -1.0}
    for t_low in T_LOW_GRID:
        for t_mid in T_MID_GRID:
            if t_mid <= t_low:
                continue
            row = eval_tri_zone(
                head, model, tokenizer, pooled_val, device=device, seed=args.seed, profile=profile,
                struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
                t_low=t_low, t_mid=t_mid, hop4_only=False,
            )
            grid_trials.append({"t_low": t_low, "t_mid": t_mid, "val_acc": row["accuracy"]})
            if row["accuracy"] > best["val_acc"]:
                best = {"t_low": t_low, "t_mid": t_mid, "val_acc": row["accuracy"]}

    t_low, t_mid = best["t_low"], best["t_mid"]
    cross_rows = []
    for i, (meta, test) in enumerate(test_by_slice):
        main_row = eval_main_path(
            model, tokenizer, test, device=device, seed=args.seed,
            profile=profile, struct_floor=struct_floor,
        )
        prow = eval_tri_zone(
            head, model, tokenizer, test, device=device, seed=args.seed, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            t_low=t_low, t_mid=t_mid, hop4_only=False,
        )
        cross_rows.append(make_slice_row(meta, test, main_row, prow, policy_name="tri_zone_best"))
        print(f"[{i+1}/{len(test_by_slice)}] {meta.get('id')}", flush=True)

    summary = rollup_slice_rows(cross_rows)
    payload = {
        "experiment_id": "v2_tri_zone_sweep",
        "title": "V2 · Tri-zone 扫参",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "t_low_grid": list(T_LOW_GRID),
        "t_mid_grid": list(T_MID_GRID),
        "grid_trials": grid_trials,
        "best_val": best,
        "cross_summary": summary,
        "cross_slices": cross_rows,
    }
    write_phase34_result("v2_tri_zone_sweep", payload)
    print(json.dumps({"best": best, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
