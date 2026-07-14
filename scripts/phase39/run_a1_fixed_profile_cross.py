#!/usr/bin/env python3
"""A1 · fixed_edges profile 跨集：消 seed 随机性测试。"""
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
from _phase37_common import LOCKED_TRI_ZONE, eval_hybrid_slice_router  # noqa: E402
from _phase39_common import (  # noqa: E402
    CANONICAL_SEED,
    DEFAULT_PROFILE,
    FIXED_PROFILE,
    ROBUST_SEEDS,
    dual_ok,
    m2_head_ready,
    unique_slice_ids,
    write_phase39_result,
)
from dataset_registry import load_slice  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402
from shared.eval_paths import eval_main_path, make_slice_row, rollup_slice_rows  # noqa: E402


def run_cross(head, model, tokenizer, device, profile, sf, kf, kt, pfn, seed, label):
    rows = []
    for sid in unique_slice_ids():
        meta, samples = load_slice(sid)
        main_row = eval_main_path(model, tokenizer, samples, device=device, seed=seed, profile=profile, struct_floor=sf)
        hy = eval_hybrid_slice_router(head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
                                      struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, meta=meta)
        rows.append(make_slice_row(meta, samples, main_row, hy, policy_name=label))
    return rollup_slice_rows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not m2_head_ready():
        raise SystemExit("缺少 M2 head")

    t0 = time.time()
    model, tokenizer, device, _ = load_model_bundle(args.device)
    head, sf, kf, kt, pfn = setup_fallback_stack(model, tokenizer, device, DEFAULT_PROFILE)

    results = {}
    for prof_name, prof in [("default", DEFAULT_PROFILE), ("fixed_edges", FIXED_PROFILE)]:
        by_seed = {}
        for seed in ROBUST_SEEDS:
            summary = run_cross(head, model, tokenizer, device, prof, sf, kf, kt, pfn, seed, prof_name)
            by_seed[str(seed)] = {"summary": summary, "dual_ok": dual_ok(summary)}
            print(f"{prof_name} seed={seed} dual_ok={dual_ok(summary)} ood={summary.get('ood_weighted_delta_pp')}", flush=True)
        results[prof_name] = {
            "profile": prof.to_dict(),
            "by_seed": by_seed,
            "dual_ok_count": sum(1 for v in by_seed.values() if v["dual_ok"]),
        }

    payload = {
        "experiment_id": "a1_fixed_profile_cross",
        "title": "A1 · fixed_edges profile 跨集",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "seeds": list(ROBUST_SEEDS),
        "results": results,
        "fixed_improves_robustness": (
            results["fixed_edges"]["dual_ok_count"] > results["default"]["dual_ok_count"]
        ),
    }
    write_phase39_result("a1_fixed_profile_cross", payload)
    print(json.dumps({k: v["dual_ok_count"] for k, v in results.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
