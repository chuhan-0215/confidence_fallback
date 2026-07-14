#!/usr/bin/env python3
"""U5 · 统一分区策略：S1 hop4 / S2 surgical / S3 dual-threshold，full 419 + 跨集 rollup。"""
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

from _fallback_eval import eval_confidence_fallback, setup_fallback_stack  # noqa: E402
from _phase33_common import (  # noqa: E402
    TRANSFER_THR,
    eval_dual_threshold,
    eval_main_path,
    eval_surgical,
    make_slice_row,
    m2_head_ready,
    rollup_slice_rows,
    unique_slice_ids,
    write_phase33_result,
)
from dataset_registry import load_slice  # noqa: E402
from phase23._phase23_common import load_full_dataset  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402

POLICIES = (
    ("S1_hop4_only", lambda h, m, t, s, d, p, sf, kf, kt, pfn, seed, prof: eval_confidence_fallback(
        h, m, t, s, device=d, seed=seed, profile=prof, struct_floor=sf, knn_floor=kf,
        knn_thr=kt, pfn=pfn, fallback_thr=TRANSFER_THR, hop4_only=True,
    )),
    ("S2_surgical", lambda h, m, t, s, d, p, sf, kf, kt, pfn, seed, prof: eval_surgical(
        h, m, t, s, device=d, seed=seed, profile=prof, struct_floor=sf, knn_floor=kf,
        knn_thr=kt, pfn=pfn,
    )),
    ("S3_dual_threshold", lambda h, m, t, s, d, p, sf, kf, kt, pfn, seed, prof: eval_dual_threshold(
        h, m, t, s, device=d, seed=seed, profile=prof, struct_floor=sf, knn_floor=kf,
        knn_thr=kt, pfn=pfn, low_thr=TRANSFER_THR, high_thr=0.55,
    )),
    ("baseline", lambda h, m, t, s, d, p, sf, kf, kt, pfn, seed, prof: eval_confidence_fallback(
        h, m, t, s, device=d, seed=seed, profile=prof, struct_floor=sf, knn_floor=kf,
        knn_thr=kt, pfn=pfn, fallback_thr=TRANSFER_THR,
    )),
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()
    if not m2_head_ready():
        raise SystemExit(f"缺少 M2 head: {ROOT / 'results/phase10/m2_enough_stop_head.pt'}")

    t0 = time.time()
    model, tokenizer, device, profile = load_model_bundle(args.device)
    head, struct_floor, knn_floor, knn_thr, pfn = setup_fallback_stack(model, tokenizer, device, profile)

    full419 = load_full_dataset()
    full419_results = {}
    for pname, fn in POLICIES:
        full419_results[pname] = fn(
            head, model, tokenizer, full419, device, profile, struct_floor,
            knn_floor, knn_thr, pfn, args.seed, profile,
        )

    cross_rows: dict[str, list] = {pname: [] for pname, _ in POLICIES}
    for i, sid in enumerate(unique_slice_ids()):
        meta, samples = load_slice(sid)
        main_row = eval_main_path(
            model, tokenizer, samples, device=device, seed=args.seed,
            profile=profile, struct_floor=struct_floor,
        )
        for pname, fn in POLICIES:
            prow = fn(
                head, model, tokenizer, samples, device, profile, struct_floor,
                knn_floor, knn_thr, pfn, args.seed, profile,
            )
            cross_rows[pname].append(make_slice_row(meta, samples, main_row, prow, policy_name=pname))
        print(f"[{i+1}] {sid}", flush=True)

    cross_summaries = {p: rollup_slice_rows(rows) for p, rows in cross_rows.items()}
    best_cross = max(
        cross_summaries.items(),
        key=lambda kv: (kv[1].get("in_dist_weighted_delta_pp") or -999, kv[1].get("ood_weighted_delta_pp") or -999),
    )

    payload = {
        "experiment_id": "u5_unified_policy",
        "title": "U5 · 统一分区策略",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "full_419": {k: {"accuracy": v["accuracy"], "fallback_rate": v.get("fallback_rate"), "params": v.get("params")} for k, v in full419_results.items()},
        "cross_summaries": cross_summaries,
        "cross_slices_by_policy": cross_rows,
        "recommended_policy": best_cross[0],
        "recommendation_reason": "优先 in-distribution 加权 Δ，次优 OOD 加权 Δ",
    }
    write_phase33_result("u5_unified_policy", payload)
    print(json.dumps({"full_419": payload["full_419"], "best": best_cross[0]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
