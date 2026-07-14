#!/usr/bin/env python3
"""U1 · P25 变体跨 53 切片：baseline / hop4_only / arbitrate / hop4+arb。"""
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
    eval_main_path,
    make_slice_row,
    m2_head_ready,
    rollup_slice_rows,
    unique_slice_ids,
    write_phase33_result,
)
from dataset_registry import load_slice  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402

VARIANTS = (
    ("baseline", {"hop4_only": False, "answer_arbitrate": False}),
    ("hop4_only", {"hop4_only": True, "answer_arbitrate": False}),
    ("arbitrate", {"hop4_only": False, "answer_arbitrate": True}),
    ("hop4_arb", {"hop4_only": True, "answer_arbitrate": True}),
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

    all_variant_rows: dict[str, list] = {name: [] for name, _ in VARIANTS}
    slice_ids = unique_slice_ids()

    for i, sid in enumerate(slice_ids):
        meta, samples = load_slice(sid)
        main_row = eval_main_path(
            model, tokenizer, samples, device=device, seed=args.seed,
            profile=profile, struct_floor=struct_floor,
        )
        for vname, vkwargs in VARIANTS:
            prow = eval_confidence_fallback(
                head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
                struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
                fallback_thr=TRANSFER_THR, **vkwargs,
            )
            row = make_slice_row(meta, samples, main_row, prow, policy_name=vname)
            row["variant_params"] = vkwargs
            all_variant_rows[vname].append(row)
        print(f"[{i+1}/{len(slice_ids)}] {sid} done", flush=True)

    summaries = {v: rollup_slice_rows(rows) for v, rows in all_variant_rows.items()}
    payload = {
        "experiment_id": "u1_variant_cross_transfer",
        "title": "U1 · P25 变体跨集重放",
        "transfer_thr": TRANSFER_THR,
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "variants": list(all_variant_rows.keys()),
        "summaries": summaries,
        "slices_by_variant": all_variant_rows,
    }
    write_phase33_result("u1_variant_cross_transfer", payload)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
