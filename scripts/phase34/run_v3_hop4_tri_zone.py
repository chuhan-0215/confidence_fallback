#!/usr/bin/env python3
"""V3 · hop4_only + V2 最优 tri-zone，跨 53 切片。"""
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
from _phase34_common import (  # noqa: E402
    PHASE34_OUT,
    eval_main_path,
    eval_tri_zone,
    make_slice_row,
    m2_head_ready,
    rollup_slice_rows,
    unique_slice_ids,
    write_phase34_result,
)
from dataset_registry import load_slice  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402


def load_v2_best() -> tuple[float, float]:
    path = PHASE34_OUT / "v2_tri_zone_sweep_latest.json"
    if not path.is_file():
        return 0.38, 0.46
    data = json.loads(path.read_text(encoding="utf-8"))
    best = data.get("best_val") or {}
    return float(best.get("t_low", 0.38)), float(best.get("t_mid", 0.46))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--t-low", type=float, default=None)
    ap.add_argument("--t-mid", type=float, default=None)
    args = ap.parse_args()
    if not m2_head_ready():
        raise SystemExit(f"缺少 M2 head: {ROOT / 'results/phase10/m2_enough_stop_head.pt'}")

    t_low, t_mid = args.t_low, args.t_mid
    if t_low is None or t_mid is None:
        t_low, t_mid = load_v2_best()

    t0 = time.time()
    model, tokenizer, device, profile = load_model_bundle(args.device)
    head, struct_floor, knn_floor, knn_thr, pfn = setup_fallback_stack(model, tokenizer, device, profile)

    rows_plain, rows_hop4 = [], []
    for i, sid in enumerate(unique_slice_ids()):
        meta, samples = load_slice(sid)
        main_row = eval_main_path(
            model, tokenizer, samples, device=device, seed=args.seed,
            profile=profile, struct_floor=struct_floor,
        )
        for hop4, bucket in ((False, rows_plain), (True, rows_hop4)):
            prow = eval_tri_zone(
                head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
                struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
                t_low=t_low, t_mid=t_mid, hop4_only=hop4,
            )
            name = "hop4_tri_zone" if hop4 else "tri_zone"
            bucket.append(make_slice_row(meta, samples, main_row, prow, policy_name=name))
        print(f"[{i+1}] {sid}", flush=True)

    payload = {
        "experiment_id": "v3_hop4_tri_zone",
        "title": "V3 · hop4 + tri-zone",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "t_low": t_low,
        "t_mid": t_mid,
        "summaries": {
            "tri_zone": rollup_slice_rows(rows_plain),
            "hop4_tri_zone": rollup_slice_rows(rows_hop4),
        },
        "slices": {
            "tri_zone": rows_plain,
            "hop4_tri_zone": rows_hop4,
        },
    }
    write_phase34_result("v3_hop4_tri_zone", payload)
    print(json.dumps(payload["summaries"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
