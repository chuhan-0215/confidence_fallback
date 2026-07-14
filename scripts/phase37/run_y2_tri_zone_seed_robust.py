#!/usr/bin/env python3
"""Y2 · tri_zone 定稿参数 seed 稳健性（42/43/44/99）。"""
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

from _fallback_eval import setup_fallback_stack  # noqa: E402
from _phase34_common import eval_tri_zone  # noqa: E402
from _phase37_common import (  # noqa: E402
    CHAMPION_SEEDS,
    LOCKED_TRI_ZONE,
    dual_ok,
    m2_head_ready,
    unique_slice_ids,
    write_phase37_result,
)
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
    for seed in CHAMPION_SEEDS:
        rows = []
        for sid in unique_slice_ids():
            meta, samples = load_slice(sid)
            main_row = eval_main_path(model, tokenizer, samples, device=device, seed=seed, profile=profile, struct_floor=sf)
            prow = eval_tri_zone(
                head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
                struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, t_low=t_low, t_mid=t_mid,
            )
            rows.append(make_slice_row(meta, samples, main_row, prow, policy_name="tri_zone"))
        summary = rollup_slice_rows(rows)
        by_seed[str(seed)] = {"summary": summary, "dual_ok": dual_ok(summary)}
        print(f"seed={seed} dual_ok={dual_ok(summary)} weighted={summary.get('weighted_mean_delta_pp')}", flush=True)

    payload = {
        "experiment_id": "y2_tri_zone_seed_robust",
        "title": "Y2 · tri_zone seed 稳健性",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "params": {"t_low": t_low, "t_mid": t_mid},
        "seeds": list(CHAMPION_SEEDS),
        "by_seed": by_seed,
        "all_dual_ok": all(v["dual_ok"] for v in by_seed.values()),
    }
    write_phase37_result("y2_tri_zone_seed_robust", payload)
    print(json.dumps({"all_dual_ok": payload["all_dual_ok"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
