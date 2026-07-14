#!/usr/bin/env python3
"""A3 · 多 seed 池化指标：逐切片四 seed 平均 Δ。"""
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
from _phase39_common import ROBUST_SEEDS, eval_hybrid_v2_router, m2_head_ready, unique_slice_ids, write_phase39_result  # noqa: E402
from dataset_registry import load_slice  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402
from shared.eval_paths import eval_main_path, make_slice_row, rollup_slice_rows  # noqa: E402


def pooled_cross(eval_fn, head, model, tokenizer, device, profile, sf, kf, kt, pfn, seeds, label):
    """每切片对多 seed 取平均 acc，再算 Δ。"""
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
        delta_pp = round((avg_pol - avg_main) * 100, 2)
        pooled_rows.append({
            "slice_id": meta.get("slice_id") or meta.get("id"),
            "label": meta.get("label"),
            "category": meta.get("category"),
            "n_samples": len(samples),
            "main_acc": round(avg_main, 4),
            "policy_acc": round(avg_pol, 4),
            "transfer_acc": round(avg_pol, 4),
            "delta_pp": delta_pp,
            "policy": label,
            "transfer_helps": delta_pp > 0.5,
            "transfer_hurts": delta_pp < -0.5,
        })
    return rollup_slice_rows(pooled_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not m2_head_ready():
        raise SystemExit("缺少 M2 head")

    t0 = time.time()
    model, tokenizer, device, profile = load_model_bundle(args.device)
    head, sf, kf, kt, pfn = setup_fallback_stack(model, tokenizer, device, profile)

    sum_v4 = pooled_cross(eval_hybrid_slice_router, head, model, tokenizer, device, profile, sf, kf, kt, pfn, ROBUST_SEEDS, "hybrid_v4_pooled")
    sum_v2 = pooled_cross(eval_hybrid_v2_router, head, model, tokenizer, device, profile, sf, kf, kt, pfn, ROBUST_SEEDS, "hybrid_v2_pooled")

    from _phase39_common import dual_ok
    payload = {
        "experiment_id": "a3_multiseed_pooled",
        "title": "A3 · 多 seed 池化跨集",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "seeds": list(ROBUST_SEEDS),
        "summaries": {"hybrid_v4_pooled": sum_v4, "hybrid_v2_pooled": sum_v2},
        "pooled_dual_ok": {
            "hybrid_v4": dual_ok(sum_v4),
            "hybrid_v2": dual_ok(sum_v2),
        },
    }
    write_phase39_result("a3_multiseed_pooled", payload)
    print(json.dumps(payload["pooled_dual_ok"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
