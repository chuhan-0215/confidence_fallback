#!/usr/bin/env python3
"""D3 · v4/v5 pooled 边界 + 扩展 seed 抽检。"""
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
from _phase42_common import (  # noqa: E402
    CANONICAL_SEED,
    ROBUST_SEEDS,
    WORST_SEED,
    dual_ok,
    eval_hybrid_v4_router,
    eval_hybrid_v5_router,
    m2_head_ready,
    unique_slice_ids,
    write_phase42_result,
)
from dataset_registry import load_slice  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402
from shared.eval_paths import eval_main_path, rollup_slice_rows  # noqa: E402

EXTRA_SEEDS = (40, 41, 45, 100)


def pooled(fn, head, model, tokenizer, device, profile, sf, kf, kt, pfn, seeds):
    rows = []
    for sid in unique_slice_ids():
        meta, samples = load_slice(sid)
        mains, pols = [], []
        for seed in seeds:
            main_row = eval_main_path(model, tokenizer, samples, device=device, seed=seed, profile=profile, struct_floor=sf)
            prow = fn(head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
                      struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, meta=meta)
            mains.append(main_row["accuracy"])
            pols.append(prow["accuracy"])
        rows.append({
            "slice_id": sid, "category": meta.get("category"), "n_samples": len(samples),
            "main_acc": sum(mains) / len(mains), "policy_acc": sum(pols) / len(pols),
            "delta_pp": round((sum(pols) / len(pols) - sum(mains) / len(mains)) * 100, 2),
        })
    return rollup_slice_rows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not m2_head_ready():
        raise SystemExit("缺少 M2 head")

    t0 = time.time()
    model, tokenizer, device, profile = load_model_bundle(args.device)
    head, sf, kf, kt, pfn = setup_fallback_stack(model, tokenizer, device, profile)

    sum_v4 = pooled(eval_hybrid_v4_router, head, model, tokenizer, device, profile, sf, kf, kt, pfn, ROBUST_SEEDS)
    sum_v5 = pooled(eval_hybrid_v5_router, head, model, tokenizer, device, profile, sf, kf, kt, pfn, ROBUST_SEEDS)

    extra = {}
    for seed in EXTRA_SEEDS:
        rows = []
        for sid in unique_slice_ids():
            meta, samples = load_slice(sid)
            main_row = eval_main_path(model, tokenizer, samples, device=device, seed=seed, profile=profile, struct_floor=sf)
            v4 = eval_hybrid_v4_router(head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
                                       struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, meta=meta)
            rows.append({"slice_id": sid, "category": meta.get("category"), "n_samples": len(samples),
                         "main_acc": main_row["accuracy"], "policy_acc": v4["accuracy"],
                         "delta_pp": round((v4["accuracy"] - main_row["accuracy"]) * 100, 2)})
        s = rollup_slice_rows(rows)
        extra[str(seed)] = {"summary": s, "dual_ok": dual_ok(s)}

    canon_v4 = canon_v5 = False
    for label, fn in [("v4", eval_hybrid_v4_router), ("v5", eval_hybrid_v5_router)]:
        rows = []
        for sid in unique_slice_ids():
            meta, samples = load_slice(sid)
            main_row = eval_main_path(model, tokenizer, samples, device=device, seed=CANONICAL_SEED, profile=profile, struct_floor=sf)
            prow = fn(head, model, tokenizer, samples, device=device, seed=CANONICAL_SEED, profile=profile,
                      struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, meta=meta)
            rows.append({"slice_id": sid, "category": meta.get("category"), "n_samples": len(samples),
                         "main_acc": main_row["accuracy"], "policy_acc": prow["accuracy"],
                         "delta_pp": round((prow["accuracy"] - main_row["accuracy"]) * 100, 2)})
        s = rollup_slice_rows(rows)
        if label == "v4":
            canon_v4 = dual_ok(s)
        else:
            canon_v5 = dual_ok(s)

    bounds = {
        "hybrid_v4": {"canonical_dual_ok": canon_v4, "pooled_dual_ok": dual_ok(sum_v4), "pooled": sum_v4},
        "hybrid_v5": {"canonical_dual_ok": canon_v5, "pooled_dual_ok": dual_ok(sum_v5), "pooled": sum_v5},
    }

    payload = {
        "experiment_id": "d3_deploy_bounds_v5",
        "title": "D3 · v4/v5 边界 + 扩展 seed",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "deploy_bounds": bounds,
        "extra_seed_v4": extra,
        "extra_dual_ok_count": sum(1 for v in extra.values() if v["dual_ok"]),
    }
    write_phase42_result("d3_deploy_bounds_v5", payload)
    print(json.dumps(bounds, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
