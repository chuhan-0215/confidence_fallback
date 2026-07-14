#!/usr/bin/env python3
"""U2 · P31 G1 手术刀死区跨 53 切片。"""
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
from _phase33_common import (  # noqa: E402
    eval_main_path,
    eval_surgical,
    make_slice_row,
    m2_head_ready,
    rollup_slice_rows,
    unique_slice_ids,
    write_phase33_result,
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
    rows = []
    for i, sid in enumerate(unique_slice_ids()):
        meta, samples = load_slice(sid)
        main_row = eval_main_path(
            model, tokenizer, samples, device=device, seed=args.seed,
            profile=profile, struct_floor=struct_floor,
        )
        srow = eval_surgical(
            head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
        )
        rows.append(make_slice_row(meta, samples, main_row, srow, policy_name="surgical"))
        print(f"[{i+1}] {sid}: main={main_row['accuracy']:.1%} surgical={srow['accuracy']:.1%}", flush=True)

    summary = rollup_slice_rows(rows)
    payload = {
        "experiment_id": "u2_surgical_cross_transfer",
        "title": "U2 · G1 手术刀死区跨集",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "summary": summary,
        "slices": rows,
    }
    write_phase33_result("u2_surgical_cross_transfer", payload)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
