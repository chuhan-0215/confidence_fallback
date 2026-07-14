#!/usr/bin/env python3
"""A2 · hybrid v2 路由 vs v4 × 四 seed。"""
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
from _phase39_common import ROBUST_SEEDS, dual_ok, eval_hybrid_v2_router, m2_head_ready, unique_slice_ids, write_phase39_result  # noqa: E402
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
        rows_v4, rows_v2 = [], []
        for sid in unique_slice_ids():
            meta, samples = load_slice(sid)
            main_row = eval_main_path(model, tokenizer, samples, device=device, seed=seed, profile=profile, struct_floor=sf)
            v4 = eval_hybrid_slice_router(head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
                                          struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, meta=meta)
            v2 = eval_hybrid_v2_router(head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
                                       struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, meta=meta)
            rows_v4.append(make_slice_row(meta, samples, main_row, v4, policy_name="hybrid_v4"))
            rows_v2.append(make_slice_row(meta, samples, main_row, v2, policy_name="hybrid_v2"))
        s4 = rollup_slice_rows(rows_v4)
        s2 = rollup_slice_rows(rows_v2)
        by_seed[str(seed)] = {
            "hybrid_v4": {"summary": s4, "dual_ok": dual_ok(s4)},
            "hybrid_v2": {"summary": s2, "dual_ok": dual_ok(s2)},
        }
        print(f"seed={seed} v4={dual_ok(s4)} v2={dual_ok(s2)} v2_ood={s2.get('ood_weighted_delta_pp')}", flush=True)

    payload = {
        "experiment_id": "a2_hybrid_v2_seed_robust",
        "title": "A2 · hybrid v2 vs v4 seed 稳健性",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "seeds": list(ROBUST_SEEDS),
        "router_v2_rules": __import__("_phase39_common", fromlist=["router_v2_rules_doc"]).router_v2_rules_doc(),
        "by_seed": by_seed,
        "v4_dual_ok_count": sum(1 for v in by_seed.values() if v["hybrid_v4"]["dual_ok"]),
        "v2_dual_ok_count": sum(1 for v in by_seed.values() if v["hybrid_v2"]["dual_ok"]),
    }
    write_phase39_result("a2_hybrid_v2_seed_robust", payload)
    print(json.dumps({"v4_ok": payload["v4_dual_ok_count"], "v2_ok": payload["v2_dual_ok_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
