#!/usr/bin/env python3
"""B4 · deploy_spec_v6 验证（v3 若优于 v5 则升级，否则锁定 v5 + 修正 eval 文档）。"""
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
from _phase39_common import router_v2_rules_doc  # noqa: E402
from _phase40_common import (  # noqa: E402
    CANONICAL_SEED,
    DEFAULT_PROFILE,
    dual_ok,
    eval_hybrid_v2_router,
    eval_hybrid_v3_router,
    m2_head_ready,
    router_v3_rules_doc,
    unique_slice_ids,
    write_phase40_result,
)
from dataset_registry import load_slice  # noqa: E402
from phase23._phase23_common import load_full_dataset  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402
from shared.eval_paths import eval_main_path, make_slice_row, rollup_slice_rows  # noqa: E402


def load_b2() -> dict | None:
    for base in (ROOT / "results" / "phase40", ROOT / "outbox/results/from_a800/phase40"):
        p = base / "b2_hybrid_v3_seed_robust_latest.json"
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def pick_router(b2: dict | None) -> tuple[str, str]:
    """Return (policy_name, variant) — v3 if wins on dual_ok count or canonical in-dist."""
    if not b2:
        return "hybrid_slice_router_v2", "v5"
    v3_ok = b2.get("v3_dual_ok_count", 0)
    v5_ok = b2.get("v5_dual_ok_count", 0)
    if v3_ok > v5_ok:
        return "hybrid_slice_router_v3", "v3"
    by99 = (b2.get("by_seed") or {}).get("99") or {}
    s3 = (by99.get("hybrid_v3") or {}).get("summary") or {}
    s5 = (by99.get("hybrid_v5") or {}).get("summary") or {}
    if dual_ok(s3) and (s3.get("in_dist_weighted_delta_pp") or 0) > (s5.get("in_dist_weighted_delta_pp") or 0):
        return "hybrid_slice_router_v3", "v3"
    return "hybrid_slice_router_v2", "v5"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=CANONICAL_SEED)
    args = ap.parse_args()
    if not m2_head_ready():
        raise SystemExit("缺少 M2 head")

    b2 = load_b2()
    policy_name, variant = pick_router(b2)
    eval_fn = eval_hybrid_v3_router if variant == "v3" else eval_hybrid_v2_router
    rules = router_v3_rules_doc() if variant == "v3" else router_v2_rules_doc()

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
        "version": "v6",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "supersedes": "deploy_spec_v5.json",
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
            "note": "跨集评估使用 default profile（coconut+random）+ canonical seed=99；fixed_edges 已证伪（P39 A1 0/4 dual_ok）",
            "seed_robustness": "仅 seed=99 保证 dual_ok；worst-seed 见 b1/b2",
        },
        "deploy_bounds": {
            "canonical_dual_ok_required": True,
            "pooled_dual_ok_reference_only": True,
            "fixed_edges_rejected": True,
        },
        "rule_summary": f"{'v3: v_diamond_5→skip' if variant == 'v3' else 'v5: hops_3→tri_zone; push_ext7→agreement'}；评估 seed=99 default profile",
    }

    out_path = ROOT / "results" / "phase40" / "deploy_spec_v6.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(deploy_spec, ensure_ascii=False, indent=2), encoding="utf-8")

    payload = {
        "experiment_id": "b4_deploy_spec_v6_validate",
        "title": "B4 · deploy_spec_v6",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "selected_variant": variant,
        "deploy_spec_v6": deploy_spec,
        "cross_summary": cross_summary,
        "ok": dual_ok(cross_summary),
    }
    write_phase40_result("b4_deploy_spec_v6_validate", payload)
    print(json.dumps({"variant": variant, "dual_ok": dual_ok(cross_summary)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
