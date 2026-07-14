#!/usr/bin/env python3
"""Y3 · hybrid slice 路由 vs 纯 tri_zone（按 construction 选策略）。"""
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
    HURT_SLICE_IDS,
    LOCKED_TRI_ZONE,
    dual_ok,
    eval_hybrid_slice_router,
    m2_head_ready,
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

    t_low, t_mid = LOCKED_TRI_ZONE
    t0 = time.time()
    model, tokenizer, device, profile = load_model_bundle(args.device)
    head, sf, kf, kt, pfn = setup_fallback_stack(model, tokenizer, device, profile)

    rows_tz, rows_hy = [], []
    hurt_compare = []
    for i, sid in enumerate(unique_slice_ids()):
        meta, samples = load_slice(sid)
        main_row = eval_main_path(model, tokenizer, samples, device=device, seed=args.seed, profile=profile, struct_floor=sf)
        tz = eval_tri_zone(
            head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
            struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, t_low=t_low, t_mid=t_mid,
        )
        hy = eval_hybrid_slice_router(
            head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
            struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, meta=meta,
        )
        rows_tz.append(make_slice_row(meta, samples, main_row, tz, policy_name="tri_zone"))
        rows_hy.append(make_slice_row(meta, samples, main_row, hy, policy_name="hybrid_router"))
        if sid in HURT_SLICE_IDS:
            hurt_compare.append({
                "slice_id": sid,
                "main_acc": main_row["accuracy"],
                "tri_zone_acc": tz["accuracy"],
                "hybrid_acc": hy["accuracy"],
                "tri_delta_pp": round((tz["accuracy"] - main_row["accuracy"]) * 100, 2),
                "hybrid_delta_pp": round((hy["accuracy"] - main_row["accuracy"]) * 100, 2),
                "hybrid_router": hy["params"].get("router"),
            })
        print(f"[{i+1}] {sid}", flush=True)

    sum_tz = rollup_slice_rows(rows_tz)
    sum_hy = rollup_slice_rows(rows_hy)
    payload = {
        "experiment_id": "y3_hybrid_slice_router",
        "title": "Y3 · hybrid slice 路由",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "params": {"t_low": t_low, "t_mid": t_mid},
        "summaries": {"tri_zone": sum_tz, "hybrid_router": sum_hy},
        "dual_ok": {"tri_zone": dual_ok(sum_tz), "hybrid_router": dual_ok(sum_hy)},
        "hurt_compare": hurt_compare,
        "slices": {"tri_zone": rows_tz, "hybrid_router": rows_hy},
    }
    write_phase37_result("y3_hybrid_slice_router", payload)
    print(json.dumps({"dual_ok": payload["dual_ok"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
