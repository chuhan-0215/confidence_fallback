#!/usr/bin/env python3
"""Z2 · seed 翻转法医：canonical(99) vs 最差 seed 逐切片对比。"""
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
from _phase37_common import eval_hybrid_slice_router  # noqa: E402
from _phase38_common import CANONICAL_SEED, ROBUST_SEEDS, m2_head_ready, unique_slice_ids, write_phase38_result  # noqa: E402
from dataset_registry import load_slice  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402
from shared.eval_paths import eval_main_path, make_slice_row  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed-a", type=int, default=CANONICAL_SEED)
    ap.add_argument("--seed-b", type=int, default=43)
    args = ap.parse_args()
    if not m2_head_ready():
        raise SystemExit("缺少 M2 head")

    t0 = time.time()
    model, tokenizer, device, profile = load_model_bundle(args.device)
    head, sf, kf, kt, pfn = setup_fallback_stack(model, tokenizer, device, profile)

    flips = []
    for sid in unique_slice_ids():
        meta, samples = load_slice(sid)
        rows = {}
        for seed in (args.seed_a, args.seed_b):
            main_row = eval_main_path(model, tokenizer, samples, device=device, seed=seed, profile=profile, struct_floor=sf)
            hy = eval_hybrid_slice_router(head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
                                          struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, meta=meta)
            row = make_slice_row(meta, samples, main_row, hy, policy_name="hybrid_router")
            rows[seed] = row
        da = rows[args.seed_a]["delta_pp"]
        db = rows[args.seed_b]["delta_pp"]
        if abs(da - db) > 1.0 or (rows[args.seed_a]["transfer_hurts"] != rows[args.seed_b]["transfer_hurts"]):
            flips.append({
                "slice_id": sid,
                "category": meta.get("category"),
                f"seed_{args.seed_a}": {"delta_pp": da, "hurts": rows[args.seed_a]["transfer_hurts"], "main_acc": rows[args.seed_a]["main_acc"]},
                f"seed_{args.seed_b}": {"delta_pp": db, "hurts": rows[args.seed_b]["transfer_hurts"], "main_acc": rows[args.seed_b]["main_acc"]},
                "delta_pp_gap": round(da - db, 2),
            })

    flips.sort(key=lambda x: abs(x["delta_pp_gap"]), reverse=True)
    payload = {
        "experiment_id": "z2_seed_flip_forensic",
        "title": "Z2 · seed 翻转法医",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "seed_a": args.seed_a,
        "seed_b": args.seed_b,
        "flip_count": len(flips),
        "flips": flips[:25],
        "ok": True,
    }
    write_phase38_result("z2_seed_flip_forensic", payload)
    print(json.dumps({"flip_count": len(flips), "top": flips[:5]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
