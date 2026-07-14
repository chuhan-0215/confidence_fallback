#!/usr/bin/env python3
"""E2 · v5 @99 canonical 增强档验证。"""
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
from _phase43_common import (  # noqa: E402
    CANONICAL_SEED,
    dual_ok,
    eval_hybrid_v4_router,
    eval_hybrid_v5_router,
    m2_head_ready,
    unique_slice_ids,
    write_phase43_result,
)
from dataset_registry import load_slice  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402
from shared.eval_paths import eval_main_path, make_slice_row, rollup_slice_rows  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=CANONICAL_SEED)
    args = ap.parse_args()
    if not m2_head_ready():
        raise SystemExit("缺少 M2 head")

    t0 = time.time()
    model, tokenizer, device, profile = load_model_bundle(args.device)
    head, sf, kf, kt, pfn = setup_fallback_stack(model, tokenizer, device, profile)

    rows_v4, rows_v5 = [], []
    for sid in unique_slice_ids():
        meta, samples = load_slice(sid)
        main_row = eval_main_path(model, tokenizer, samples, device=device, seed=args.seed, profile=profile, struct_floor=sf)
        v4 = eval_hybrid_v4_router(head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
                                   struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, meta=meta)
        v5 = eval_hybrid_v5_router(head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
                                   struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, meta=meta)
        rows_v4.append(make_slice_row(meta, samples, main_row, v4, policy_name="hybrid_v4"))
        rows_v5.append(make_slice_row(meta, samples, main_row, v5, policy_name="hybrid_v5"))
    s4 = rollup_slice_rows(rows_v4)
    s5 = rollup_slice_rows(rows_v5)

    upgrade = (
        dual_ok(s5)
        and (s5.get("weighted_mean_delta_pp") or 0) >= (s4.get("weighted_mean_delta_pp") or 0)
        and (s5.get("hurts_count") or 99) <= (s4.get("hurts_count") or 99)
    )

    payload = {
        "experiment_id": "e2_v5_canonical_enhanced",
        "title": "E2 · v5 @99 增强档",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "seed": args.seed,
        "hybrid_v4": {"summary": s4, "dual_ok": dual_ok(s4)},
        "hybrid_v5": {"summary": s5, "dual_ok": dual_ok(s5)},
        "delta_weighted_pp": round((s5.get("weighted_mean_delta_pp") or 0) - (s4.get("weighted_mean_delta_pp") or 0), 3),
        "delta_in_dist_pp": round((s5.get("in_dist_weighted_delta_pp") or 0) - (s4.get("in_dist_weighted_delta_pp") or 0), 3),
        "hurts_delta": (s4.get("hurts_count") or 0) - (s5.get("hurts_count") or 0),
        "recommend_enhanced_tier": upgrade,
        "ok": dual_ok(s5),
    }
    write_phase43_result("e2_v5_canonical_enhanced", payload)
    print(json.dumps({"upgrade": upgrade, "v5_delta": payload["delta_weighted_pp"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
