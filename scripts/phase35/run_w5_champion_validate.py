#!/usr/bin/env python3
"""W5 · champion full 419 验证（seed=99 对齐 P25）。"""
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
from _phase34_common import eval_agreement_lock, eval_tri_zone  # noqa: E402
from _phase35_common import (  # noqa: E402
    PHASE35_OUT,
    TRANSFER_THR,
    eval_agreement_tri_zone,
    load_p34_best,
    m2_head_ready,
    write_phase35_result,
)
from phase23._phase23_common import load_full_dataset  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402


def load_w2_best() -> tuple[float, float]:
    path = PHASE35_OUT / "w2_combo_grid_sweep_latest.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        top = (data.get("top5_cross") or [{}])[0]
        if top.get("t_low") is not None:
            return float(top["t_low"]), float(top["t_mid"])
    return load_p34_best()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()
    if not m2_head_ready():
        raise SystemExit(f"缺少 M2 head: {ROOT / 'results/phase10/m2_enough_stop_head.pt'}")

    t_low, t_mid = load_w2_best()
    t0 = time.time()
    model, tokenizer, device, profile = load_model_bundle(args.device)
    head, struct_floor, knn_floor, knn_thr, pfn = setup_fallback_stack(model, tokenizer, device, profile)
    samples = load_full_dataset()

    baseline = eval_confidence_fallback(
        head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
        struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
        fallback_thr=TRANSFER_THR,
    )
    policies = {
        "baseline": baseline,
        "agreement_lock": eval_agreement_lock(
            head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            fallback_thr=TRANSFER_THR, hop4_only=False,
        ),
        "tri_zone": eval_tri_zone(
            head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            t_low=t_low, t_mid=t_mid, hop4_only=False,
        ),
        "agreement_tri_zone": eval_agreement_tri_zone(
            head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            t_low=t_low, t_mid=t_mid, hop4_only=False,
        ),
        "combo_hop4": eval_agreement_tri_zone(
            head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            t_low=t_low, t_mid=t_mid, hop4_only=True,
        ),
    }

    champ_name = max(policies, key=lambda k: policies[k]["accuracy"])
    champ_acc = policies[champ_name]["accuracy"]
    payload = {
        "experiment_id": "w5_champion_validate",
        "title": "W5 · champion full 419（seed=99）",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "seed": args.seed,
        "params": {"t_low": t_low, "t_mid": t_mid},
        "policies": policies,
        "champion": champ_name,
        "champion_acc": champ_acc,
        "baseline_acc": baseline["accuracy"],
        "passes_95": champ_acc >= 0.9523,
        "baseline_matches_p25": baseline["accuracy"] >= 0.9520,
    }
    write_phase35_result("w5_champion_validate", payload)
    print(json.dumps({
        "champion": champ_name,
        "champion_acc": champ_acc,
        "baseline_acc": baseline["accuracy"],
        "passes_95": payload["passes_95"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
