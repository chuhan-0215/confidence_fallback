#!/usr/bin/env python3
"""V1 · Agreement lock 跨 53 切片。"""
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
    TRANSFER_THR,
    eval_agreement_lock,
    eval_main_path,
    make_slice_row,
    m2_head_ready,
    rollup_slice_rows,
    unique_slice_ids,
    write_phase34_result,
)
from dataset_registry import load_slice  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402


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

    variants = (
        ("agreement_lock", {"hop4_only": False}),
        ("agreement_hop4", {"hop4_only": True}),
    )
    all_rows: dict[str, list] = {n: [] for n, _ in variants}
    slice_ids = unique_slice_ids()

    for i, sid in enumerate(slice_ids):
        meta, samples = load_slice(sid)
        main_row = eval_main_path(
            model, tokenizer, samples, device=device, seed=args.seed,
            profile=profile, struct_floor=struct_floor,
        )
        for vname, vkwargs in variants:
            prow = eval_agreement_lock(
                head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
                struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
                fallback_thr=TRANSFER_THR, **vkwargs,
            )
            all_rows[vname].append(make_slice_row(meta, samples, main_row, prow, policy_name=vname))
        print(f"[{i+1}/{len(slice_ids)}] {sid}", flush=True)

    summaries = {v: rollup_slice_rows(rows) for v, rows in all_rows.items()}
    payload = {
        "experiment_id": "v1_agreement_lock",
        "title": "V1 · Agreement lock 跨集",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "summaries": summaries,
        "slices_by_variant": all_rows,
    }
    write_phase34_result("v1_agreement_lock", payload)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
