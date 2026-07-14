#!/usr/bin/env python3
"""X1 · W2 赢家 combo (t_low=0.32) 全量 53 切片验证。"""
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
from _phase36_common import (  # noqa: E402
    P34_TRI_ZONE,
    W2_COMBO_BEST,
    eval_agreement_tri_zone,
    load_w2_best,
    m2_head_ready,
    unique_slice_ids,
    write_phase36_result,
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

    t_low, t_mid = load_w2_best()
    t0 = time.time()
    model, tokenizer, device, profile = load_model_bundle(args.device)
    head, sf, kf, kt, pfn = setup_fallback_stack(model, tokenizer, device, profile)

    rows_w2, rows_p34 = [], []
    for i, sid in enumerate(unique_slice_ids()):
        meta, samples = load_slice(sid)
        main_row = eval_main_path(model, tokenizer, samples, device=device, seed=args.seed, profile=profile, struct_floor=sf)
        rows_w2.append(make_slice_row(meta, samples, main_row, eval_agreement_tri_zone(
            head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
            struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, t_low=t_low, t_mid=t_mid,
        ), policy_name="combo_w2"))
        tl, tm = P34_TRI_ZONE
        rows_p34.append(make_slice_row(meta, samples, main_row, eval_agreement_tri_zone(
            head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
            struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, t_low=tl, t_mid=tm,
        ), policy_name="combo_p34"))
        print(f"[{i+1}] {sid}", flush=True)

    payload = {
        "experiment_id": "x1_w2_winner_full_slices",
        "title": "X1 · W2 combo 全量切片",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "w2_params": {"t_low": t_low, "t_mid": t_mid},
        "p34_params": {"t_low": P34_TRI_ZONE[0], "t_mid": P34_TRI_ZONE[1]},
        "summaries": {
            "combo_w2": rollup_slice_rows(rows_w2),
            "combo_p34": rollup_slice_rows(rows_p34),
        },
        "slices": {"combo_w2": rows_w2, "combo_p34": rows_p34},
    }
    write_phase36_result("x1_w2_winner_full_slices", payload)
    print(json.dumps(payload["summaries"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
