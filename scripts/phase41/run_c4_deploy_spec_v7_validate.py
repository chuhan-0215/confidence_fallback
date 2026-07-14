#!/usr/bin/env python3
"""C4 · deploy_spec_v7 验证（锁定 v3 或 v4）。"""
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
from _phase41_common import (  # noqa: E402
    CANONICAL_SEED,
    DEFAULT_PROFILE,
    dual_ok,
    eval_hybrid_v3_router,
    eval_hybrid_v4_router,
    load_phase40_b3,
    m2_head_ready,
    pick_deploy_variant,
    router_v3_rules_doc,
    router_v4_rules_doc,
    unique_slice_ids,
    write_phase41_result,
)
from dataset_registry import load_slice  # noqa: E402
from phase23._phase23_common import load_full_dataset  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402
from shared.eval_paths import eval_main_path, make_slice_row, rollup_slice_rows  # noqa: E402


def load_c2() -> dict | None:
    for base in (ROOT / "results" / "phase41", ROOT / "outbox/results/from_a800/phase41"):
        p = base / "c2_hybrid_v4_seed_robust_latest.json"
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def load_c3() -> dict | None:
    for base in (ROOT / "results" / "phase41", ROOT / "outbox/results/from_a800/phase41"):
        p = base / "c3_deploy_bounds_v4_latest.json"
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def pick_variant(c2: dict | None, c3: dict | None, b3: dict | None) -> tuple[str, str]:
    v4_ok = (c2 or {}).get("v4_dual_ok_count", 0)
    v3_ok = (c2 or {}).get("v3_dual_ok_count", 0)
    if v4_ok > v3_ok:
        return "hybrid_slice_router_v4", "v4"
    bounds = (c3 or {}).get("deploy_bounds") or {}
    v4_strict = ((bounds.get("hybrid_v4") or {}).get("deploy_ok_strict"))
    v3_strict = ((bounds.get("hybrid_v3") or {}).get("deploy_ok_strict"))
    if v4_strict and not v3_strict:
        return "hybrid_slice_router_v4", "v4"
    if v3_strict or ((b3 or {}).get("deploy_bounds") or {}).get("hybrid_v3", {}).get("deploy_ok_strict"):
        return "hybrid_slice_router_v3", "v3"
    return pick_deploy_variant(b3, c2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=CANONICAL_SEED)
    args = ap.parse_args()
    if not m2_head_ready():
        raise SystemExit("缺少 M2 head")

    c2 = load_c2()
    c3 = load_c3()
    b3 = load_phase40_b3()
    policy_name, variant = pick_variant(c2, c3, b3)
    eval_fn = {"v3": eval_hybrid_v3_router, "v4": eval_hybrid_v4_router}.get(variant, eval_hybrid_v3_router)
    rules = {"v3": router_v3_rules_doc, "v4": router_v4_rules_doc}.get(variant, router_v3_rules_doc)()

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

    p40_bounds = (b3 or {}).get("deploy_bounds") or {}
    deploy_spec = {
        "version": "v7",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "supersedes": "deploy_spec_v6.json",
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
            "note": "canonical seed=99 dual_ok 为硬门槛；pooled dual_ok 为 v3 参考边界（P40 B3）",
            "per_seed_dual_ok": "1/4（42/43/44 仍失败 OOD 或 in-dist）",
        },
        "deploy_bounds": {
            "canonical_dual_ok_required": True,
            "pooled_dual_ok_v3_reference": p40_bounds.get("hybrid_v3"),
            "fixed_edges_rejected": True,
            "phase40_v6_correction": "v6 误锁 v5；v7 升级 v3/v4",
        },
        "rule_summary": f"{variant}: skip v_diamond_5" + ("; push_ext7 skip" if variant == "v4" else "") + "；seed=99",
    }

    out_path = ROOT / "results" / "phase41" / "deploy_spec_v7.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(deploy_spec, ensure_ascii=False, indent=2), encoding="utf-8")

    payload = {
        "experiment_id": "c4_deploy_spec_v7_validate",
        "title": "C4 · deploy_spec_v7",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "selected_variant": variant,
        "deploy_spec_v7": deploy_spec,
        "cross_summary": cross_summary,
        "ok": dual_ok(cross_summary),
    }
    write_phase41_result("c4_deploy_spec_v7_validate", payload)
    print(json.dumps({"variant": variant, "dual_ok": dual_ok(cross_summary)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
