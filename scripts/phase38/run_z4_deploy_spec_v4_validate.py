#!/usr/bin/env python3
"""Z4 · deploy_spec_v4 验证（hybrid_router 升级）。"""
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
sys.path.insert(0, str(ROOT / "scripts" / "phase37"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fallback_eval import eval_confidence_fallback, setup_fallback_stack  # noqa: E402
from _phase34_common import TRANSFER_THR  # noqa: E402
from _phase37_common import eval_hybrid_slice_router  # noqa: E402
from _phase38_common import CANONICAL_SEED, dual_ok, m2_head_ready, router_rules_doc, unique_slice_ids, write_phase38_result  # noqa: E402
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

    t0 = time.time()
    model, tokenizer, device, profile = load_model_bundle(args.device)
    head, sf, kf, kt, pfn = setup_fallback_stack(model, tokenizer, device, profile)

    prosqa = eval_confidence_fallback(
        head, model, tokenizer, load_full_dataset(), device=device, seed=args.seed, profile=profile,
        struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, fallback_thr=TRANSFER_THR,
    )
    cross_rows = []
    for sid in unique_slice_ids():
        meta, samples = load_slice(sid)
        main_row = eval_main_path(model, tokenizer, samples, device=device, seed=args.seed, profile=profile, struct_floor=sf)
        prow = eval_hybrid_slice_router(head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
                                        struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, meta=meta)
        cross_rows.append(make_slice_row(meta, samples, main_row, prow, policy_name="hybrid_router"))
    cross_summary = rollup_slice_rows(cross_rows)

    deploy_spec = {
        "version": "v4",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "supersedes": "deploy_spec_v3.json",
        "prosqa_in_distribution": {
            "policy": "confidence_fallback",
            "fallback_thr": TRANSFER_THR,
            "accuracy": prosqa["accuracy"],
            "fallback_rate": prosqa.get("fallback_rate"),
            "canonical_seed": args.seed,
        },
        "cross_dataset_ood": {
            "policy": "hybrid_slice_router",
            "router_rules": router_rules_doc(),
            "weighted_mean_delta_pp": cross_summary.get("weighted_mean_delta_pp"),
            "in_dist_weighted_delta_pp": cross_summary.get("in_dist_weighted_delta_pp"),
            "ood_weighted_delta_pp": cross_summary.get("ood_weighted_delta_pp"),
            "hurts_count": cross_summary.get("hurts_count"),
            "dual_ok": dual_ok(cross_summary),
        },
        "rule_summary": (
            "同源 ProsQA → τ=0.48 baseline；跨集 → hybrid 路由："
            "syn_chain_5_wide 跳过 transfer；push/mix/hops/diamond 用 agreement；其余 tri_zone"
        ),
        "seed_robustness_note": "canonical seed=99 dual_ok；其他 seed 见 z1_hybrid_seed_robust",
    }

    out_path = ROOT / "results" / "phase38" / "deploy_spec_v4.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(deploy_spec, ensure_ascii=False, indent=2), encoding="utf-8")

    payload = {
        "experiment_id": "z4_deploy_spec_v4_validate",
        "title": "Z4 · deploy_spec_v4",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "deploy_spec_v4": deploy_spec,
        "deploy_spec_path": str(out_path.relative_to(ROOT)),
        "cross_summary": cross_summary,
        "ok": dual_ok(cross_summary),
    }
    write_phase38_result("z4_deploy_spec_v4_validate", payload)
    print(json.dumps(deploy_spec, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
