#!/usr/bin/env python3
"""V5 · champion 策略 full 419 + 多 seed 验证。"""
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
from _phase34_common import (  # noqa: E402
    CHAMPION_SEEDS,
    PHASE34_OUT,
    TRANSFER_THR,
    eval_tri_zone,
    m2_head_ready,
    write_phase34_result,
)
from phase23._phase23_common import load_full_dataset  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402


def load_v3_params() -> tuple[float, float]:
    path = PHASE34_OUT / "v3_hop4_tri_zone_latest.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data.get("t_low", 0.38)), float(data.get("t_mid", 0.46))
    return 0.38, 0.46


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()
    if not m2_head_ready():
        raise SystemExit(f"缺少 M2 head: {ROOT / 'results/phase10/m2_enough_stop_head.pt'}")

    t_low, t_mid = load_v3_params()
    t0 = time.time()
    model, tokenizer, device, profile = load_model_bundle(args.device)
    head, struct_floor, knn_floor, knn_thr, pfn = setup_fallback_stack(model, tokenizer, device, profile)
    samples = load_full_dataset()

    by_seed = {}
    for seed in CHAMPION_SEEDS:
        baseline = eval_confidence_fallback(
            head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            fallback_thr=TRANSFER_THR,
        )
        champion = eval_tri_zone(
            head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            t_low=t_low, t_mid=t_mid, hop4_only=True,
        )
        by_seed[str(seed)] = {"baseline": baseline, "hop4_tri_zone": champion}
        print(f"seed={seed} baseline={baseline['accuracy']:.2%} champion={champion['accuracy']:.2%}", flush=True)

    champ_accs = [v["hop4_tri_zone"]["accuracy"] for v in by_seed.values()]
    payload = {
        "experiment_id": "v5_champion_validate",
        "title": "V5 · champion full 419 验证",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "t_low": t_low,
        "t_mid": t_mid,
        "seeds": list(CHAMPION_SEEDS),
        "by_seed": by_seed,
        "champion_mean_acc": round(sum(champ_accs) / len(champ_accs), 4),
        "passes_95": all(a >= 0.95 for a in champ_accs),
    }
    write_phase34_result("v5_champion_validate", payload)
    print(json.dumps({"champion_mean_acc": payload["champion_mean_acc"], "passes_95": payload["passes_95"]}))


if __name__ == "__main__":
    main()
