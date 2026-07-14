#!/usr/bin/env python3
"""B1 · worst-seed 逐切片审计（v5 @ seed 43 vs 99）。"""
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
from _phase40_common import (  # noqa: E402
    CANONICAL_SEED,
    DEFAULT_PROFILE,
    WORST_SEED,
    eval_hybrid_v2_router,
    m2_head_ready,
    unique_slice_ids,
    write_phase40_result,
)
from dataset_registry import load_slice  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402
from shared.eval_paths import eval_main_path, make_slice_row  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not m2_head_ready():
        raise SystemExit("缺少 M2 head")

    t0 = time.time()
    model, tokenizer, device, _ = load_model_bundle(args.device)
    head, sf, kf, kt, pfn = setup_fallback_stack(model, tokenizer, device, DEFAULT_PROFILE)

    flips, in_dist_hurts = [], []
    for sid in unique_slice_ids():
        meta, samples = load_slice(sid)
        rows = {}
        for seed in (CANONICAL_SEED, WORST_SEED):
            main_row = eval_main_path(model, tokenizer, samples, device=device, seed=seed, profile=DEFAULT_PROFILE, struct_floor=sf)
            prow = eval_hybrid_v2_router(head, model, tokenizer, samples, device=device, seed=seed, profile=DEFAULT_PROFILE,
                                         struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, meta=meta)
            delta = round((prow["accuracy"] - main_row["accuracy"]) * 100, 2)
            rows[str(seed)] = {"delta_pp": delta, "main_acc": main_row["accuracy"], "policy_acc": prow["accuracy"],
                               "hurts": delta < -0.5, "helps": delta > 0.5}
        gap = round(rows[str(CANONICAL_SEED)]["delta_pp"] - rows[str(WORST_SEED)]["delta_pp"], 2)
        flip = (rows[str(CANONICAL_SEED)]["helps"] != rows[str(WORST_SEED)]["helps"]) or (
            rows[str(CANONICAL_SEED)]["hurts"] != rows[str(WORST_SEED)]["hurts"]
        )
        entry = {"slice_id": sid, "category": meta.get("category"), "seed_99": rows[str(CANONICAL_SEED)],
                 "seed_43": rows[str(WORST_SEED)], "delta_pp_gap": gap, "flip": flip}
        if flip:
            flips.append(entry)
        if meta.get("category") in ("standard", "pattern") and rows[str(WORST_SEED)]["delta_pp"] < 0:
            in_dist_hurts.append(entry)

    flips.sort(key=lambda x: -abs(x["delta_pp_gap"]))
    payload = {
        "experiment_id": "b1_worst_seed_slice_audit",
        "title": "B1 · worst-seed 逐切片审计",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "router": "hybrid_v2",
        "seed_a": CANONICAL_SEED,
        "seed_b": WORST_SEED,
        "flip_count": len(flips),
        "in_dist_hurt_count": len(in_dist_hurts),
        "flips": flips[:30],
        "in_dist_hurts": in_dist_hurts[:15],
    }
    write_phase40_result("b1_worst_seed_slice_audit", payload)
    print(json.dumps({"flip_count": len(flips), "in_dist_hurts": len(in_dist_hurts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
