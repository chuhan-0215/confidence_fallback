#!/usr/bin/env python3
"""B3 · 部署边界指标（canonical + pooled + worst-seed）。"""
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
from _phase40_common import (  # noqa: E402
    CANONICAL_SEED,
    ROBUST_SEEDS,
    WORST_SEED,
    dual_ok,
    eval_hybrid_v2_router,
    eval_hybrid_v3_router,
    m2_head_ready,
    unique_slice_ids,
    write_phase40_result,
)
from dataset_registry import load_slice  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402
from shared.eval_paths import eval_main_path, rollup_slice_rows  # noqa: E402


def pooled_cross(eval_fn, head, model, tokenizer, device, profile, sf, kf, kt, pfn, seeds):
    pooled_rows = []
    for sid in unique_slice_ids():
        meta, samples = load_slice(sid)
        main_accs, pol_accs = [], []
        for seed in seeds:
            main_row = eval_main_path(model, tokenizer, samples, device=device, seed=seed, profile=profile, struct_floor=sf)
            prow = eval_fn(head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
                           struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, meta=meta)
            main_accs.append(main_row["accuracy"])
            pol_accs.append(prow["accuracy"])
        avg_main = sum(main_accs) / len(main_accs)
        avg_pol = sum(pol_accs) / len(pol_accs)
        pooled_rows.append({
            "slice_id": meta.get("slice_id") or meta.get("id"),
            "category": meta.get("category"),
            "n_samples": len(samples),
            "main_acc": round(avg_main, 4),
            "policy_acc": round(avg_pol, 4),
            "delta_pp": round((avg_pol - avg_main) * 100, 2),
        })
    return rollup_slice_rows(pooled_rows)


def deploy_bounds(summary: dict, *, canonical_ok: bool, worst_in_dist: float | None) -> dict:
    in_d = summary.get("in_dist_weighted_delta_pp")
    ood = summary.get("ood_weighted_delta_pp")
    return {
        "canonical_dual_ok": canonical_ok,
        "pooled_dual_ok": dual_ok(summary),
        "pooled_in_dist": in_d,
        "pooled_ood": ood,
        "worst_seed_in_dist_min": worst_in_dist,
        "deploy_ok_strict": canonical_ok and dual_ok(summary),
        "deploy_ok_documented": canonical_ok and (ood or 0) >= 7.0,
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

    worst_in = {}
    for label, fn in [("hybrid_v5", eval_hybrid_v2_router), ("hybrid_v3", eval_hybrid_v3_router)]:
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
                worst_in[label] = s.get("in_dist_weighted_delta_pp")

    sum_v5 = pooled_cross(eval_hybrid_v2_router, head, model, tokenizer, device, profile, sf, kf, kt, pfn, ROBUST_SEEDS)
    sum_v3 = pooled_cross(eval_hybrid_v3_router, head, model, tokenizer, device, profile, sf, kf, kt, pfn, ROBUST_SEEDS)

    # canonical @99 only
    rows99_v5, rows99_v3 = [], []
    for sid in unique_slice_ids():
        meta, samples = load_slice(sid)
        main_row = eval_main_path(model, tokenizer, samples, device=device, seed=CANONICAL_SEED, profile=profile, struct_floor=sf)
        v5 = eval_hybrid_v2_router(head, model, tokenizer, samples, device=device, seed=CANONICAL_SEED, profile=profile,
                                   struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, meta=meta)
        v3 = eval_hybrid_v3_router(head, model, tokenizer, samples, device=device, seed=CANONICAL_SEED, profile=profile,
                                   struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, meta=meta)
        rows99_v5.append({"slice_id": sid, "category": meta.get("category"), "n_samples": len(samples),
                          "main_acc": main_row["accuracy"], "policy_acc": v5["accuracy"],
                          "delta_pp": round((v5["accuracy"] - main_row["accuracy"]) * 100, 2)})
        rows99_v3.append({"slice_id": sid, "category": meta.get("category"), "n_samples": len(samples),
                          "main_acc": main_row["accuracy"], "policy_acc": v3["accuracy"],
                          "delta_pp": round((v3["accuracy"] - main_row["accuracy"]) * 100, 2)})
    s99_v5 = rollup_slice_rows(rows99_v5)
    s99_v3 = rollup_slice_rows(rows99_v3)

    bounds = {
        "hybrid_v5": deploy_bounds(sum_v5, canonical_ok=dual_ok(s99_v5), worst_in_dist=worst_in.get("hybrid_v5")),
        "hybrid_v3": deploy_bounds(sum_v3, canonical_ok=dual_ok(s99_v3), worst_in_dist=worst_in.get("hybrid_v3")),
    }

    payload = {
        "experiment_id": "b3_deploy_bounds",
        "title": "B3 · 部署边界指标",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "pooled": {"hybrid_v5": sum_v5, "hybrid_v3": sum_v3},
        "canonical_99": {"hybrid_v5": s99_v5, "hybrid_v3": s99_v3},
        "deploy_bounds": bounds,
        "recommendation": "canonical seed=99 dual_ok 为部署门槛；pooled 作参考；fixed_edges 不可用（P39 A1）",
    }
    write_phase40_result("b3_deploy_bounds", payload)
    print(json.dumps(bounds, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
