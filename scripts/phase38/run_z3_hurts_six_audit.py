#!/usr/bin/env python3
"""Z3 · 剩余 6 hurts 切片 hybrid 逐切片终验。"""
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

from _fallback_eval import eval_confidence_fallback, setup_fallback_stack  # noqa: E402
from _phase34_common import TRANSFER_THR, eval_tri_zone  # noqa: E402
from _phase37_common import LOCKED_TRI_ZONE, eval_hybrid_slice_router  # noqa: E402
from _phase38_common import CANONICAL_SEED, m2_head_ready, write_phase38_result  # noqa: E402
from dataset_registry import load_slice  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402
from shared.eval_paths import eval_main_path  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=CANONICAL_SEED)
    args = ap.parse_args()
    if not m2_head_ready():
        raise SystemExit("缺少 M2 head")

    t0 = time.time()
    model, tokenizer, device, profile = load_model_bundle(args.device)
    head, sf, kf, kt, pfn = setup_fallback_stack(model, tokenizer, device, profile)

    # Y3 hybrid hurts=6; load from y3 if available
    y3_path = ROOT / "outbox/results/from_a800/phase37/y3_hybrid_slice_router_latest.json"
    hurt_ids = []
    if y3_path.is_file():
        y3 = json.loads(y3_path.read_text(encoding="utf-8"))
        for row in (y3.get("slices") or {}).get("hybrid_router") or []:
            if row.get("transfer_hurts"):
                hurt_ids.append(row["slice_id"])
    if not hurt_ids:
        from _phase37_common import HURT_SLICE_IDS
        hurt_ids = list(HURT_SLICE_IDS)

    slices_out = []
    for sid in hurt_ids:
        meta, samples = load_slice(sid)
        if not samples:
            continue
        main_row = eval_main_path(model, tokenizer, samples, device=device, seed=args.seed, profile=profile, struct_floor=sf)
        policies = {
            "baseline": eval_confidence_fallback(head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
                struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, fallback_thr=TRANSFER_THR),
            "tri_zone": eval_tri_zone(head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
                struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, t_low=LOCKED_TRI_ZONE[0], t_mid=LOCKED_TRI_ZONE[1]),
            "hybrid_router": eval_hybrid_slice_router(head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
                struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, meta=meta),
        }
        slices_out.append({
            "slice_id": sid,
            "label": meta.get("label"),
            "n_samples": len(samples),
            "main_acc": main_row["accuracy"],
            "delta_pp": {k: round((v["accuracy"] - main_row["accuracy"]) * 100, 2) for k, v in policies.items()},
            "policies": policies,
            "hybrid_router": policies["hybrid_router"]["params"].get("router"),
        })
        print(f"{sid} hybrid_delta={slices_out[-1]['delta_pp']['hybrid_router']}", flush=True)

    payload = {
        "experiment_id": "z3_hurts_six_audit",
        "title": "Z3 · hurts 切片 audit",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "seed": args.seed,
        "hurt_count": len(slices_out),
        "slices": slices_out,
    }
    write_phase38_result("z3_hurts_six_audit", payload)
    print(json.dumps({"hurt_count": len(slices_out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
