#!/usr/bin/env python3
"""X5 · full 419 终验：tri_zone / combo_w2 / category_router（seed=99）。"""
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fallback_eval import eval_confidence_fallback, setup_fallback_stack  # noqa: E402
from _phase34_common import TRANSFER_THR, eval_tri_zone  # noqa: E402
from _phase36_common import (  # noqa: E402
    P34_TRI_ZONE,
    eval_agreement_tri_zone,
    load_w2_best,
    m2_head_ready,
    write_phase36_result,
)
from phase23._phase23_common import load_full_dataset  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()
    if not m2_head_ready():
        raise SystemExit("缺少 M2 head")

    w2_low, w2_mid = load_w2_best()
    t0 = time.time()
    model, tokenizer, device, profile = load_model_bundle(args.device)
    head, sf, kf, kt, pfn = setup_fallback_stack(model, tokenizer, device, profile)
    samples = load_full_dataset()

    policies = {
        "baseline": eval_confidence_fallback(
            head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
            struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, fallback_thr=TRANSFER_THR,
        ),
        "tri_zone": eval_tri_zone(
            head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
            struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn,
            t_low=P34_TRI_ZONE[0], t_mid=P34_TRI_ZONE[1],
        ),
        "combo_w2": eval_agreement_tri_zone(
            head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
            struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, t_low=w2_low, t_mid=w2_mid,
        ),
    }
    champ = max(policies, key=lambda k: policies[k]["accuracy"])
    payload = {
        "experiment_id": "x5_full419_validate",
        "title": "X5 · full 419 终验",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "seed": args.seed,
        "params": {"w2": {"t_low": w2_low, "t_mid": w2_mid}, "p34": {"t_low": P34_TRI_ZONE[0], "t_mid": P34_TRI_ZONE[1]}},
        "policies": policies,
        "champion": champ,
        "champion_acc": policies[champ]["accuracy"],
        "p25_target": 0.9523,
        "passes_p25": policies[champ]["accuracy"] >= 0.9523,
    }
    write_phase36_result("x5_full419_validate", payload)
    print(json.dumps({
        "champion": champ,
        "acc": policies[champ]["accuracy"],
        "baseline": policies["baseline"]["accuracy"],
        "tri_zone": policies["tri_zone"]["accuracy"],
        "combo_w2": policies["combo_w2"]["accuracy"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
