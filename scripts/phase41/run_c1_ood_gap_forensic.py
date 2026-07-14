#!/usr/bin/env python3
"""C1 · OOD 缺口法医（seed 42/44 vs 99 逐切片）。"""
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
from _phase41_common import (  # noqa: E402
    CANONICAL_SEED,
    DEFAULT_PROFILE,
    OOD_SEEDS,
    eval_hybrid_v3_router,
    m2_head_ready,
    unique_slice_ids,
    write_phase41_result,
)
from dataset_registry import load_slice  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402
from shared.eval_paths import eval_main_path  # noqa: E402

OOD_CATS = frozenset({"deep", "variant", "push_deep"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not m2_head_ready():
        raise SystemExit("缺少 M2 head")

    t0 = time.time()
    model, tokenizer, device, _ = load_model_bundle(args.device)
    head, sf, kf, kt, pfn = setup_fallback_stack(model, tokenizer, device, DEFAULT_PROFILE)

    gaps = []
    for sid in unique_slice_ids():
        meta, samples = load_slice(sid)
        cat = meta.get("category")
        if cat not in OOD_CATS:
            continue
        rows = {}
        for seed in (CANONICAL_SEED, *OOD_SEEDS):
            main_row = eval_main_path(model, tokenizer, samples, device=device, seed=seed, profile=DEFAULT_PROFILE, struct_floor=sf)
            prow = eval_hybrid_v3_router(head, model, tokenizer, samples, device=device, seed=seed, profile=DEFAULT_PROFILE,
                                         struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, meta=meta)
            delta = round((prow["accuracy"] - main_row["accuracy"]) * 100, 2)
            rows[str(seed)] = {"delta_pp": delta, "n": len(samples)}
        for ood_seed in OOD_SEEDS:
            gap = round(rows[str(CANONICAL_SEED)]["delta_pp"] - rows[str(ood_seed)]["delta_pp"], 2)
            if gap > 0.3:
                gaps.append({
                    "slice_id": sid, "category": cat, "ood_seed": ood_seed,
                    "delta_99": rows[str(CANONICAL_SEED)]["delta_pp"],
                    f"delta_{ood_seed}": rows[str(ood_seed)]["delta_pp"],
                    "gap_pp": gap, "n_samples": len(samples),
                })

    gaps.sort(key=lambda x: -x["gap_pp"])
    by_seed = {str(s): [g for g in gaps if g["ood_seed"] == s] for s in OOD_SEEDS}

    payload = {
        "experiment_id": "c1_ood_gap_forensic",
        "title": "C1 · OOD 缺口法医",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "router": "hybrid_v3",
        "canonical_seed": CANONICAL_SEED,
        "ood_seeds": list(OOD_SEEDS),
        "gap_count": len(gaps),
        "top_gaps": gaps[:25],
        "by_seed": {k: v[:12] for k, v in by_seed.items()},
    }
    write_phase41_result("c1_ood_gap_forensic", payload)
    print(json.dumps({"gap_count": len(gaps)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
