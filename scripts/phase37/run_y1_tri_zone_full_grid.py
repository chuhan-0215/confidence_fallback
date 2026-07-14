#!/usr/bin/env python3
"""Y1 · tri_zone T_low 全量 53 切片扫参（禁止 test split）。"""
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
from _phase34_common import eval_tri_zone  # noqa: E402
from _phase37_common import (  # noqa: E402
    LOCKED_TRI_ZONE,
    T_LOW_FULL_GRID,
    T_MID_FIXED,
    dual_ok,
    m2_head_ready,
    score_summary,
    unique_slice_ids,
    write_phase37_result,
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
        raise SystemExit("缺少 M2 head")

    t0 = time.time()
    model, tokenizer, device, profile = load_model_bundle(args.device)
    head, sf, kf, kt, pfn = setup_fallback_stack(model, tokenizer, device, profile)

    ranked = []
    for t_low in T_LOW_FULL_GRID:
        rows = []
        for i, sid in enumerate(unique_slice_ids()):
            meta, samples = load_slice(sid)
            main_row = eval_main_path(model, tokenizer, samples, device=device, seed=args.seed, profile=profile, struct_floor=sf)
            prow = eval_tri_zone(
                head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
                struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, t_low=t_low, t_mid=T_MID_FIXED,
            )
            rows.append(make_slice_row(meta, samples, main_row, prow, policy_name="tri_zone"))
        summary = rollup_slice_rows(rows)
        entry = {"t_low": t_low, "t_mid": T_MID_FIXED, "summary": summary, "dual_ok": dual_ok(summary), "score": score_summary(summary)}
        ranked.append(entry)
        print(f"t_low={t_low} in_dist={summary.get('in_dist_weighted_delta_pp')} ood={summary.get('ood_weighted_delta_pp')}", flush=True)

    ranked.sort(key=lambda x: x["score"], reverse=True)
    locked = next((r for r in ranked if r["t_low"] == LOCKED_TRI_ZONE[0]), ranked[0])

    payload = {
        "experiment_id": "y1_tri_zone_full_grid",
        "title": "Y1 · tri_zone 全量 T_low 扫参",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "t_low_grid": list(T_LOW_FULL_GRID),
        "t_mid_fixed": T_MID_FIXED,
        "ranked": ranked,
        "locked_params": {"t_low": LOCKED_TRI_ZONE[0], "t_mid": LOCKED_TRI_ZONE[1], "summary": locked["summary"], "dual_ok": locked["dual_ok"]},
        "best": ranked[0],
    }
    write_phase37_result("y1_tri_zone_full_grid", payload)
    print(json.dumps({"best": ranked[0], "locked_dual_ok": locked["dual_ok"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
