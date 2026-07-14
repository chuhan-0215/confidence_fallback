#!/usr/bin/env python3
"""Z1 · hybrid_router vs tri_zone 四 seed 稳健性对比。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase25"))
sys.path.insert(0, str(ROOT / "scripts" / "phase34"))
sys.path.insert(0, str(ROOT / "scripts" / "phase37"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fallback_eval import setup_fallback_stack  # noqa: E402
from _phase34_common import eval_tri_zone  # noqa: E402
from _phase37_common import LOCKED_TRI_ZONE, eval_hybrid_slice_router  # noqa: E402
from _phase38_common import ROBUST_SEEDS, dual_ok, m2_head_ready, unique_slice_ids, write_phase38_result  # noqa: E402
from dataset_registry import load_slice  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402
from shared.eval_paths import eval_main_path, make_slice_row, rollup_slice_rows  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not m2_head_ready():
        raise SystemExit("缺少 M2 head")

    t_low, t_mid = LOCKED_TRI_ZONE
    t0 = time.time()
    model, tokenizer, device, profile = load_model_bundle(args.device)
    head, sf, kf, kt, pfn = setup_fallback_stack(model, tokenizer, device, profile)

    by_seed = {}
    for seed in ROBUST_SEEDS:
        rows_tz, rows_hy = [], []
        for sid in unique_slice_ids():
            meta, samples = load_slice(sid)
            main_row = eval_main_path(model, tokenizer, samples, device=device, seed=seed, profile=profile, struct_floor=sf)
            tz = eval_tri_zone(head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
                               struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, t_low=t_low, t_mid=t_mid)
            hy = eval_hybrid_slice_router(head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
                                          struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, meta=meta)
            rows_tz.append(make_slice_row(meta, samples, main_row, tz, policy_name="tri_zone"))
            rows_hy.append(make_slice_row(meta, samples, main_row, hy, policy_name="hybrid_router"))
        sum_tz = rollup_slice_rows(rows_tz)
        sum_hy = rollup_slice_rows(rows_hy)
        by_seed[str(seed)] = {
            "tri_zone": {"summary": sum_tz, "dual_ok": dual_ok(sum_tz)},
            "hybrid_router": {"summary": sum_hy, "dual_ok": dual_ok(sum_hy)},
        }
        print(f"seed={seed} tz_dual={dual_ok(sum_tz)} hy_dual={dual_ok(sum_hy)} "
              f"hy_ood={sum_hy.get('ood_weighted_delta_pp')}", flush=True)

    tz_ok = sum(1 for v in by_seed.values() if v["tri_zone"]["dual_ok"])
    hy_ok = sum(1 for v in by_seed.values() if v["hybrid_router"]["dual_ok"])
    payload = {
        "experiment_id": "z1_hybrid_seed_robust",
        "title": "Z1 · hybrid vs tri_zone seed 稳健性",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "seeds": list(ROBUST_SEEDS),
        "params": {"t_low": t_low, "t_mid": t_mid},
        "by_seed": by_seed,
        "tri_zone_dual_ok_count": tz_ok,
        "hybrid_dual_ok_count": hy_ok,
        "hybrid_beats_tri_zone_seeds": sum(
            1 for v in by_seed.values()
            if (v["hybrid_router"]["summary"].get("weighted_mean_delta_pp") or 0)
            > (v["tri_zone"]["summary"].get("weighted_mean_delta_pp") or 0)
        ),
    }
    write_phase38_result("z1_hybrid_seed_robust", payload)
    print(json.dumps({"tz_ok": tz_ok, "hy_ok": hy_ok}, ensure_ascii=False))


if __name__ == "__main__":
    main()
