#!/usr/bin/env python3
"""D1 · seed43 in-dist 法医（v4 @43 vs 99）。"""
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
from _phase42_common import (  # noqa: E402
    CANONICAL_SEED,
    DEFAULT_PROFILE,
    IN_DIST_CATS,
    WORST_SEED,
    eval_hybrid_v4_router,
    m2_head_ready,
    unique_slice_ids,
    write_phase42_result,
)
from dataset_registry import load_slice  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402
from shared.eval_paths import eval_main_path  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not m2_head_ready():
        raise SystemExit("缺少 M2 head")

    t0 = time.time()
    model, tokenizer, device, _ = load_model_bundle(args.device)
    head, sf, kf, kt, pfn = setup_fallback_stack(model, tokenizer, device, DEFAULT_PROFILE)

    flips, in_dist_neg = [], []
    for sid in unique_slice_ids():
        meta, samples = load_slice(sid)
        cat = meta.get("category")
        rows = {}
        for seed in (CANONICAL_SEED, WORST_SEED):
            main_row = eval_main_path(model, tokenizer, samples, device=device, seed=seed, profile=DEFAULT_PROFILE, struct_floor=sf)
            prow = eval_hybrid_v4_router(head, model, tokenizer, samples, device=device, seed=seed, profile=DEFAULT_PROFILE,
                                         struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, meta=meta)
            delta = round((prow["accuracy"] - main_row["accuracy"]) * 100, 2)
            rows[str(seed)] = {"delta_pp": delta, "hurts": delta < -0.5, "helps": delta > 0.5}
        gap = round(rows[str(CANONICAL_SEED)]["delta_pp"] - rows[str(WORST_SEED)]["delta_pp"], 2)
        flip = rows[str(CANONICAL_SEED)]["helps"] != rows[str(WORST_SEED)]["helps"] or rows[str(CANONICAL_SEED)]["hurts"] != rows[str(WORST_SEED)]["hurts"]
        entry = {"slice_id": sid, "category": cat, "seed_99": rows[str(CANONICAL_SEED)], "seed_43": rows[str(WORST_SEED)], "delta_pp_gap": gap, "flip": flip}
        if flip:
            flips.append(entry)
        if cat in IN_DIST_CATS and rows[str(WORST_SEED)]["delta_pp"] < 0:
            in_dist_neg.append(entry)

    flips.sort(key=lambda x: -abs(x["delta_pp_gap"]))
    in_dist_neg.sort(key=lambda x: x["seed_43"]["delta_pp"])

    payload = {
        "experiment_id": "d1_seed43_in_dist_forensic",
        "title": "D1 · seed43 in-dist 法医",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "router": "hybrid_v4",
        "flip_count": len(flips),
        "in_dist_neg_count": len(in_dist_neg),
        "flips": flips[:25],
        "in_dist_negatives": in_dist_neg[:15],
        "target_gap_pp": 0.268,
    }
    write_phase42_result("d1_seed43_in_dist_forensic", payload)
    print(json.dumps({"flips": len(flips), "in_dist_neg": len(in_dist_neg)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
