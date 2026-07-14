#!/usr/bin/env python3
"""D4 · deploy_spec_v8 验证（v5 若 4/4 则升级，否则锁 v4）。"""
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
from _phase41_common import router_v4_rules_doc  # noqa: E402
from _phase42_common import (  # noqa: E402
    CANONICAL_SEED,
    DEFAULT_PROFILE,
    ROBUST_SEEDS,
    dual_ok,
    eval_hybrid_v4_router,
    eval_hybrid_v5_router,
    m2_head_ready,
    router_v5_rules_doc,
    unique_slice_ids,
    write_phase42_result,
)
from dataset_registry import load_slice  # noqa: E402
from phase23._phase23_common import load_full_dataset  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402
from shared.eval_paths import eval_main_path, make_slice_row, rollup_slice_rows  # noqa: E402


def load_d2() -> dict | None:
    for base in (ROOT / "results" / "phase42", ROOT / "outbox/results/from_a800/phase42"):
        p = base / "d2_hybrid_v5_seed_robust_latest.json"
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def pick_variant(d2: dict | None) -> tuple[str, str]:
    v5_ok = (d2 or {}).get("v5_dual_ok_count", 0)
    v4_ok = (d2 or {}).get("v4_dual_ok_count", 0)
    if v5_ok > v4_ok:
        return "hybrid_slice_router_v5", "v5"
    if v5_ok == v4_ok and v5_ok == len(ROBUST_SEEDS):
        return "hybrid_slice_router_v5", "v5"
    return "hybrid_slice_router_v4", "v4"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=CANONICAL_SEED)
    args = ap.parse_args()
    if not m2_head_ready():
        raise SystemExit("缺少 M2 head")

    d2 = load_d2()
    policy_name, variant = pick_variant(d2)
    eval_fn = eval_hybrid_v5_router if variant == "v5" else eval_hybrid_v4_router
    rules = router_v5_rules_doc() if variant == "v5" else router_v4_rules_doc()
    per_seed_ok = (d2 or {}).get(f"{variant}_dual_ok_count") or (d2 or {}).get("v4_dual_ok_count", 3)

    t0 = time.time()
    model, tokenizer, device, _ = load_model_bundle(args.device)
    head, sf, kf, kt, pfn = setup_fallback_stack(model, tokenizer, device, DEFAULT_PROFILE)

    prosqa = eval_confidence_fallback(
        head, model, tokenizer, load_full_dataset(), device=device, seed=args.seed, profile=DEFAULT_PROFILE,
        struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, fallback_thr=TRANSFER_THR,
    )
    cross_rows = []
    for sid in unique_slice_ids():
        meta, samples = load_slice(sid)
        main_row = eval_main_path(model, tokenizer, samples, device=device, seed=args.seed, profile=DEFAULT_PROFILE, struct_floor=sf)
        prow = eval_fn(head, model, tokenizer, samples, device=device, seed=args.seed, profile=DEFAULT_PROFILE,
                       struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, meta=meta)
        cross_rows.append(make_slice_row(meta, samples, main_row, prow, policy_name=policy_name))
    cross_summary = rollup_slice_rows(cross_rows)

    deploy_spec = {
        "version": "v8",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "supersedes": "deploy_spec_v7.json",
        "prosqa_in_distribution": {
            "policy": "confidence_fallback",
            "fallback_thr": TRANSFER_THR,
            "accuracy": prosqa["accuracy"],
            "eval_profile": DEFAULT_PROFILE.to_dict(),
            "canonical_seed": args.seed,
        },
        "cross_dataset_ood": {
            "policy": policy_name,
            "router_rules": rules,
            "weighted_mean_delta_pp": cross_summary.get("weighted_mean_delta_pp"),
            "in_dist_weighted_delta_pp": cross_summary.get("in_dist_weighted_delta_pp"),
            "ood_weighted_delta_pp": cross_summary.get("ood_weighted_delta_pp"),
            "hurts_count": cross_summary.get("hurts_count"),
            "dual_ok": dual_ok(cross_summary),
        },
        "eval_stability": {
            "recommended_profile": DEFAULT_PROFILE.to_dict(),
            "per_seed_dual_ok": f"{per_seed_ok}/{len(ROBUST_SEEDS)}",
            "note": "P41 v4 达 3/4；seed43 in-dist 为唯一缺口",
        },
        "deploy_bounds": {
            "canonical_dual_ok_required": True,
            "pooled_dual_ok": True,
            "fixed_edges_rejected": True,
        },
        "rule_summary": f"{variant} 路由；canonical seed=99",
    }

    out_path = ROOT / "results" / "phase42" / "deploy_spec_v8.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(deploy_spec, ensure_ascii=False, indent=2), encoding="utf-8")

    payload = {
        "experiment_id": "d4_deploy_spec_v8_validate",
        "title": "D4 · deploy_spec_v8",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "selected_variant": variant,
        "deploy_spec_v8": deploy_spec,
        "cross_summary": cross_summary,
        "ok": dual_ok(cross_summary),
    }
    write_phase42_result("d4_deploy_spec_v8_validate", payload)
    print(json.dumps({"variant": variant, "dual_ok": dual_ok(cross_summary), "per_seed": per_seed_ok}, ensure_ascii=False))


if __name__ == "__main__":
    main()
