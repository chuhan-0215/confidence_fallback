#!/usr/bin/env python3
"""B2 · hybrid v3 vs v5 四 seed 对比。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase25"))
sys.path.insert(0, str(ROOT / "scripts" / "phase37"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fallback_eval import setup_fallback_stack  # noqa: E402
from _phase37_common import eval_hybrid_slice_router  # noqa: E402
from _phase40_common import (  # noqa: E402
    ROBUST_SEEDS,
    dual_ok,
    eval_hybrid_v2_router,
    eval_hybrid_v3_router,
    m2_head_ready,
    router_v3_rules_doc,
    unique_slice_ids,
    write_phase40_result,
)
from dataset_registry import load_slice  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402
from shared.eval_paths import eval_main_path, make_slice_row, rollup_slice_rows  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not m2_head_ready():
        raise SystemExit("缺少 M2 head")

    t0 = time.time()
    model, tokenizer, device, profile = load_model_bundle(args.device)
    head, sf, kf, kt, pfn = setup_fallback_stack(model, tokenizer, device, profile)

    by_seed = {}
    for seed in ROBUST_SEEDS:
        rows_v5, rows_v3, rows_v4 = [], [], []
        for sid in unique_slice_ids():
            meta, samples = load_slice(sid)
            main_row = eval_main_path(model, tokenizer, samples, device=device, seed=seed, profile=profile, struct_floor=sf)
            v5 = eval_hybrid_v2_router(head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
                                       struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, meta=meta)
            v3 = eval_hybrid_v3_router(head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
                                       struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, meta=meta)
            v4 = eval_hybrid_slice_router(head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
                                          struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, meta=meta)
            rows_v5.append(make_slice_row(meta, samples, main_row, v5, policy_name="hybrid_v5"))
            rows_v3.append(make_slice_row(meta, samples, main_row, v3, policy_name="hybrid_v3"))
            rows_v4.append(make_slice_row(meta, samples, main_row, v4, policy_name="hybrid_v4"))
        s5 = rollup_slice_rows(rows_v5)
        s3 = rollup_slice_rows(rows_v3)
        s4 = rollup_slice_rows(rows_v4)
        by_seed[str(seed)] = {
            "hybrid_v5": {"summary": s5, "dual_ok": dual_ok(s5)},
            "hybrid_v3": {"summary": s3, "dual_ok": dual_ok(s3)},
            "hybrid_v4": {"summary": s4, "dual_ok": dual_ok(s4)},
        }
        print(f"seed={seed} v5={dual_ok(s5)} v3={dual_ok(s3)} v3_in={s3.get('in_dist_weighted_delta_pp')} v3_ood={s3.get('ood_weighted_delta_pp')}", flush=True)

    payload = {
        "experiment_id": "b2_hybrid_v3_seed_robust",
        "title": "B2 · hybrid v3 vs v5 seed 稳健性",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "seeds": list(ROBUST_SEEDS),
        "router_v3_rules": router_v3_rules_doc(),
        "by_seed": by_seed,
        "v5_dual_ok_count": sum(1 for v in by_seed.values() if v["hybrid_v5"]["dual_ok"]),
        "v3_dual_ok_count": sum(1 for v in by_seed.values() if v["hybrid_v3"]["dual_ok"]),
        "v4_dual_ok_count": sum(1 for v in by_seed.values() if v["hybrid_v4"]["dual_ok"]),
    }
    write_phase40_result("b2_hybrid_v3_seed_robust", payload)
    print(json.dumps({"v5": payload["v5_dual_ok_count"], "v3": payload["v3_dual_ok_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
