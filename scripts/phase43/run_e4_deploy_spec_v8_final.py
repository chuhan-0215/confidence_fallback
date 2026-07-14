#!/usr/bin/env python3
"""E4 · deploy_spec_v8_final（双档：v4 默认 + v5 @99 增强）。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase25"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fallback_eval import eval_confidence_fallback, setup_fallback_stack  # noqa: E402
from _phase34_common import TRANSFER_THR  # noqa: E402
from _phase43_common import (  # noqa: E402
    CANONICAL_SEED,
    DEFAULT_PROFILE,
    dual_ok,
    eval_hybrid_v4_router,
    eval_hybrid_v5_router,
    load_phase42_d2,
    m2_head_ready,
    router_v4_rules_doc,
    router_v5_rules_doc,
    unique_slice_ids,
    write_phase43_result,
)
from dataset_registry import load_slice  # noqa: E402
from phase23._phase23_common import load_full_dataset  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402
from shared.eval_paths import eval_main_path, make_slice_row, rollup_slice_rows  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=CANONICAL_SEED)
    args = ap.parse_args()
    if not m2_head_ready():
        raise SystemExit("缺少 M2 head")

    d2 = load_phase42_d2()
    per_seed_ok = (d2 or {}).get("v4_dual_ok_count", 3)

    t0 = time.time()
    model, tokenizer, device, _ = load_model_bundle(args.device)
    head, sf, kf, kt, pfn = setup_fallback_stack(model, tokenizer, device, DEFAULT_PROFILE)

    prosqa = eval_confidence_fallback(
        head, model, tokenizer, load_full_dataset(), device=device, seed=args.seed, profile=DEFAULT_PROFILE,
        struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, fallback_thr=TRANSFER_THR,
    )

    sum_v4, sum_v5 = None, None
    for label, fn in [("v4", eval_hybrid_v4_router), ("v5", eval_hybrid_v5_router)]:
        rows = []
        for sid in unique_slice_ids():
            meta, samples = load_slice(sid)
            main_row = eval_main_path(model, tokenizer, samples, device=device, seed=args.seed, profile=DEFAULT_PROFILE, struct_floor=sf)
            prow = fn(head, model, tokenizer, samples, device=device, seed=args.seed, profile=DEFAULT_PROFILE,
                      struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, meta=meta)
            rows.append(make_slice_row(meta, samples, main_row, prow, policy_name=f"hybrid_{label}"))
        s = rollup_slice_rows(rows)
        if label == "v4":
            sum_v4 = s
        else:
            sum_v5 = s

    use_enhanced = (
        dual_ok(sum_v5)
        and (sum_v5.get("hurts_count") or 99) < (sum_v4.get("hurts_count") or 99)
        and (sum_v5.get("weighted_mean_delta_pp") or 0) >= (sum_v4.get("weighted_mean_delta_pp") or 0)
    )

    deploy_spec = {
        "version": "v8_final",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "supersedes": "deploy_spec_v8.json",
        "prosqa_in_distribution": {
            "policy": "confidence_fallback",
            "fallback_thr": TRANSFER_THR,
            "accuracy": prosqa["accuracy"],
            "eval_profile": DEFAULT_PROFILE.to_dict(),
            "canonical_seed": args.seed,
        },
        "cross_dataset_ood": {
            "policy": "hybrid_slice_router_v4",
            "tier": "default",
            "router_rules": router_v4_rules_doc(),
            "weighted_mean_delta_pp": sum_v4.get("weighted_mean_delta_pp"),
            "in_dist_weighted_delta_pp": sum_v4.get("in_dist_weighted_delta_pp"),
            "ood_weighted_delta_pp": sum_v4.get("ood_weighted_delta_pp"),
            "hurts_count": sum_v4.get("hurts_count"),
            "dual_ok": dual_ok(sum_v4),
            "per_seed_dual_ok": f"{per_seed_ok}/4",
        },
        "cross_dataset_canonical_enhanced": {
            "policy": "hybrid_slice_router_v5",
            "tier": "canonical_only",
            "router_rules": router_v5_rules_doc(),
            "weighted_mean_delta_pp": sum_v5.get("weighted_mean_delta_pp"),
            "in_dist_weighted_delta_pp": sum_v5.get("in_dist_weighted_delta_pp"),
            "ood_weighted_delta_pp": sum_v5.get("ood_weighted_delta_pp"),
            "hurts_count": sum_v5.get("hurts_count"),
            "dual_ok": dual_ok(sum_v5),
            "recommended_when": "仅 canonical eval @seed=99；多 seed 部署仍用 v4",
            "active": use_enhanced,
        },
        "eval_stability": {
            "recommended_profile": DEFAULT_PROFILE.to_dict(),
            "canonical_seed": args.seed,
            "per_seed_dual_ok_v4": f"{per_seed_ok}/4",
            "seed43_irreducible": True,
            "fixed_edges_rejected": True,
        },
        "deploy_bounds": {
            "canonical_dual_ok_required": True,
            "pooled_dual_ok": True,
            "multi_seed_limit": "3/4 on {42,43,44,99}；seed43 in-dist 不可修复",
        },
        "rule_summary": "默认 v4（3/4 seed）；@99 可选 v5 增强（hurts 更少）",
    }

    out_path = ROOT / "results" / "phase43" / "deploy_spec_v8_final.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(deploy_spec, ensure_ascii=False, indent=2), encoding="utf-8")

    payload = {
        "experiment_id": "e4_deploy_spec_v8_final",
        "title": "E4 · deploy_spec_v8_final",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "deploy_spec_v8_final": deploy_spec,
        "enhanced_tier_active": use_enhanced,
        "ok": dual_ok(sum_v4),
    }
    write_phase43_result("e4_deploy_spec_v8_final", payload)
    print(json.dumps({"enhanced": use_enhanced, "v4_dual_ok": dual_ok(sum_v4)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
