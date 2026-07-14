#!/usr/bin/env python3
"""X2 · 三策略对决：tri_zone / combo_w2 / category_router。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase25"))
sys.path.insert(0, str(ROOT / "scripts" / "phase34"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fallback_eval import setup_fallback_stack  # noqa: E402
from _phase34_common import eval_tri_zone  # noqa: E402
from _phase36_common import (  # noqa: E402
    P34_TRI_ZONE,
    eval_agreement_tri_zone,
    eval_category_router,
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

    w2_low, w2_mid = load_w2_best()
    t0 = time.time()
    model, tokenizer, device, profile = load_model_bundle(args.device)
    head, sf, kf, kt, pfn = setup_fallback_stack(model, tokenizer, device, profile)

    buckets = {k: [] for k in ("tri_zone", "combo_w2", "category_router")}
    for i, sid in enumerate(unique_slice_ids()):
        meta, samples = load_slice(sid)
        main_row = eval_main_path(model, tokenizer, samples, device=device, seed=args.seed, profile=profile, struct_floor=sf)
        tl, tm = P34_TRI_ZONE
        buckets["tri_zone"].append(make_slice_row(meta, samples, main_row, eval_tri_zone(
            head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
            struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, t_low=tl, t_mid=tm,
        ), policy_name="tri_zone"))
        buckets["combo_w2"].append(make_slice_row(meta, samples, main_row, eval_agreement_tri_zone(
            head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
            struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, t_low=w2_low, t_mid=w2_mid,
        ), policy_name="combo_w2"))
        buckets["category_router"].append(make_slice_row(meta, samples, main_row, eval_category_router(
            head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
            struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, meta=meta,
        ), policy_name="category_router"))
        print(f"[{i+1}] {sid}", flush=True)

    summaries = {k: rollup_slice_rows(v) for k, v in buckets.items()}
    payload = {
        "experiment_id": "x2_three_way_shootout",
        "title": "X2 · 三策略对决",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "params": {"w2": {"t_low": w2_low, "t_mid": w2_mid}, "p34": {"t_low": P34_TRI_ZONE[0], "t_mid": P34_TRI_ZONE[1]}},
        "summaries": summaries,
        "slices": buckets,
    }
    write_phase36_result("x2_three_way_shootout", payload)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
