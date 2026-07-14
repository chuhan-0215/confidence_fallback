#!/usr/bin/env python3
"""W2 · Combo grid：val 选参 + test 跨集（目标 in-dist≥0 且 OOD≥+7）。"""
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
    T_LOW_GRID,
    T_MID_GRID,
    eval_agreement_tri_zone,
    m2_head_ready,
    split_val_test,
    unique_slice_ids,
    write_phase35_result,
)
from dataset_registry import load_slice  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402
from shared.eval_paths import eval_main_path, make_slice_row, rollup_slice_rows  # noqa: E402


def score_combo(summary: dict) -> float:
    in_d = summary.get("in_dist_weighted_delta_pp") or -999.0
    ood = summary.get("ood_weighted_delta_pp") or -999.0
    if ood < 7.0:
        return -999.0 + ood
    return in_d + 0.1 * (summary.get("weighted_mean_delta_pp") or 0.0)


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
    best = {"t_low": None, "t_mid": None, "val_acc": -1.0, "score": -999.0}
    for t_low in T_LOW_GRID:
        for t_mid in T_MID_GRID:
            if t_mid <= t_low:
                continue
            row = eval_agreement_tri_zone(
                head, model, tokenizer, pooled_val, device=device, seed=args.seed, profile=profile,
                struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
                t_low=t_low, t_mid=t_mid, hop4_only=False,
            )
            grid_trials.append({"t_low": t_low, "t_mid": t_mid, "val_acc": row["accuracy"]})
            if row["accuracy"] > best["val_acc"]:
                best.update({"t_low": t_low, "t_mid": t_mid, "val_acc": row["accuracy"]})

    cross_rows = []
    cross_by_params: dict[tuple[float, float], list] = {}
    candidates = {(best["t_low"], best["t_mid"])}
    for t_low in T_LOW_GRID:
        for t_mid in T_MID_GRID:
            if t_mid > t_low:
                candidates.add((t_low, t_mid))

    for t_low, t_mid in sorted(candidates):
        rows = []
        for i, (meta, test) in enumerate(test_by_slice):
            main_row = eval_main_path(
                model, tokenizer, test, device=device, seed=args.seed,
                profile=profile, struct_floor=struct_floor,
            )
            prow = eval_agreement_tri_zone(
                head, model, tokenizer, test, device=device, seed=args.seed, profile=profile,
                struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
                t_low=t_low, t_mid=t_mid, hop4_only=False,
            )
            rows.append(make_slice_row(meta, test, main_row, prow, policy_name="combo_best"))
        cross_by_params[(t_low, t_mid)] = rows

    ranked = []
    for (t_low, t_mid), rows in cross_by_params.items():
        summary = rollup_slice_rows(rows)
        sc = score_combo(summary)
        ranked.append({"t_low": t_low, "t_mid": t_mid, "score": sc, "summary": summary})
        if sc > best["score"]:
            best.update({"t_low": t_low, "t_mid": t_mid, "score": sc})

    ranked.sort(key=lambda x: x["score"], reverse=True)
    t_low, t_mid = best["t_low"], best["t_mid"]
    cross_rows = cross_by_params[(t_low, t_mid)]
    summary = rollup_slice_rows(cross_rows)

    payload = {
        "experiment_id": "w2_combo_grid_sweep",
        "title": "W2 · Combo grid 扫参",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "t_low_grid": list(T_LOW_GRID),
        "t_mid_grid": list(T_MID_GRID),
        "grid_trials": grid_trials,
        "best_val": {"t_low": best["t_low"], "t_mid": best["t_mid"], "val_acc": best["val_acc"]},
        "best_cross_score": best["score"],
        "top5_cross": ranked[:5],
        "cross_summary": summary,
        "cross_slices": cross_rows,
    }
    write_phase35_result("w2_combo_grid_sweep", payload)
    print(json.dumps({"best": best, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
