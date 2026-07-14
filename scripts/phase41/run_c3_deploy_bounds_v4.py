#!/usr/bin/env python3
"""C3 · v3/v4 部署边界（canonical + pooled）。"""
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
from _phase41_common import (  # noqa: E402
    CANONICAL_SEED,
    ROBUST_SEEDS,
    WORST_SEED,
    dual_ok,
    eval_hybrid_v3_router,
    eval_hybrid_v4_router,
    m2_head_ready,
    unique_slice_ids,
    write_phase41_result,
)
from dataset_registry import load_slice  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402
from shared.eval_paths import eval_main_path, rollup_slice_rows  # noqa: E402


def pooled(eval_fn, head, model, tokenizer, device, profile, sf, kf, kt, pfn):
    rows = []
    for sid in unique_slice_ids():
        meta, samples = load_slice(sid)
        mains, pols = [], []
        for seed in ROBUST_SEEDS:
            main_row = eval_main_path(model, tokenizer, samples, device=device, seed=seed, profile=profile, struct_floor=sf)
            prow = eval_fn(head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
                           struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, meta=meta)
            mains.append(main_row["accuracy"])
            pols.append(prow["accuracy"])
        rows.append({
            "slice_id": sid, "category": meta.get("category"), "n_samples": len(samples),
            "main_acc": sum(mains) / len(mains), "policy_acc": sum(pols) / len(pols),
            "delta_pp": round((sum(pols) / len(pols) - sum(mains) / len(mains)) * 100, 2),
        })
    return rollup_slice_rows(rows)


def bounds(summary, *, canon_ok: bool, worst_in: float | None) -> dict:
    return {
        "canonical_dual_ok": canon_ok,
        "pooled_dual_ok": dual_ok(summary),
        "pooled_in_dist": summary.get("in_dist_weighted_delta_pp"),
        "pooled_ood": summary.get("ood_weighted_delta_pp"),
        "worst_seed_in_dist_min": worst_in,
        "deploy_ok_strict": canon_ok and dual_ok(summary),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not m2_head_ready():
        raise SystemExit("缺少 M2 head")

    t0 = time.time()
    model, tokenizer, device, profile = load_model_bundle(args.device)
    head, sf, kf, kt, pfn = setup_fallback_stack(model, tokenizer, device, profile)

    worst = {}
    canon = {}
    for label, fn in [("hybrid_v3", eval_hybrid_v3_router), ("hybrid_v4", eval_hybrid_v4_router)]:
        for seed in ROBUST_SEEDS:
            rows = []
            for sid in unique_slice_ids():
                meta, samples = load_slice(sid)
                main_row = eval_main_path(model, tokenizer, samples, device=device, seed=seed, profile=profile, struct_floor=sf)
                prow = fn(head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
                          struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, meta=meta)
                rows.append({
                    "slice_id": sid, "category": meta.get("category"), "n_samples": len(samples),
                    "main_acc": main_row["accuracy"], "policy_acc": prow["accuracy"],
                    "delta_pp": round((prow["accuracy"] - main_row["accuracy"]) * 100, 2),
                })
            s = rollup_slice_rows(rows)
            if seed == WORST_SEED:
                worst[label] = s.get("in_dist_weighted_delta_pp")
            if seed == CANONICAL_SEED:
                canon[label] = dual_ok(s)

    sum_v3 = pooled(eval_hybrid_v3_router, head, model, tokenizer, device, profile, sf, kf, kt, pfn)
    sum_v4 = pooled(eval_hybrid_v4_router, head, model, tokenizer, device, profile, sf, kf, kt, pfn)

    deploy_bounds = {
        "hybrid_v3": bounds(sum_v3, canon_ok=canon.get("hybrid_v3", False), worst_in=worst.get("hybrid_v3")),
        "hybrid_v4": bounds(sum_v4, canon_ok=canon.get("hybrid_v4", False), worst_in=worst.get("hybrid_v4")),
    }

    payload = {
        "experiment_id": "c3_deploy_bounds_v4",
        "title": "C3 · v3/v4 部署边界",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "pooled": {"hybrid_v3": sum_v3, "hybrid_v4": sum_v4},
        "deploy_bounds": deploy_bounds,
        "recommendation": "v3 pooled dual_ok 已证（P40 B3）；v4 若改善 OOD seed 则升级",
    }
    write_phase41_result("c3_deploy_bounds_v4", payload)
    print(json.dumps(deploy_bounds, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
