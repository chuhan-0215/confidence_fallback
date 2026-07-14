#!/usr/bin/env python3
"""W6 · Phase 21 汇总 + mentor_brief。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase21_common import load_json, timed_run, write_phase21_result


def _pick_full(d: dict, key: str = "full_419"):
    if isinstance(d.get(key), dict) and d[key].get("accuracy") is not None:
        return d[key]
    if isinstance(d.get("best"), dict):
        return d["best"]
    for r in d.get("results") or d.get("configs") or d.get("sweep") or []:
        if r.get("split") == "full_419" or r.get("split") is None:
            if r.get("accuracy") is not None:
                return r
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        exps = [
            ("w1_epsilon_timing", "ε-timing"),
            ("w2_meta_budget", "元预算"),
            ("w3_hop_conditional", "跳数条件化"),
            ("w4_stability_gate", "稳定性门控"),
            ("w5_hybrid_distill", "hybrid 蒸馏"),
        ]
        candidates = []
        for eid, label in exps:
            d = load_json(f"phase21/{eid}_latest.json")
            if not d:
                continue
            full = _pick_full(d)
            if eid == "w1_epsilon_timing":
                full = next((c for c in d.get("configs", []) if c.get("split") == "full_419" and c.get("min_n") == 3), full)
            if eid == "w2_meta_budget":
                full = d.get("best_deployable_mvp") or d.get("best_acc") or full
            acc = full.get("accuracy")
            timing = full.get("stop_timing_acc") or full.get("timing_strict")
            if acc is None:
                continue
            candidates.append({
                "id": eid,
                "label": label,
                "accuracy": acc,
                "timing": timing,
                "timing_eps1": full.get("timing_eps1"),
                "mean_n": full.get("mean_stop_n") or full.get("mean_n"),
                "feasible": full.get("feasible") or d.get("feasible"),
                "deployable_mvp": full.get("deployable_mvp") or d.get("deployable_mvp"),
                "paradigm": label,
            })

        p20 = load_json("phase20/v5_proof_rollup_latest.json")
        p17 = load_json("phase17/s1_corrected_final_latest.json")
        bl20 = (p20.get("best") or {}).get("timing")
        knn_acc = ((p17.get("deployable_mvp") or {}).get("accuracy"))
        best_timing = max(candidates, key=lambda c: ((c.get("timing") or 0), c.get("accuracy") or 0), default={})
        best_acc = max(candidates, key=lambda c: c.get("accuracy") or 0, default={})
        best_mvp = max(candidates, key=lambda c: (c.get("deployable_mvp"), c.get("accuracy") or 0), default={})
        best_eps1 = max(candidates, key=lambda c: (c.get("timing_eps1") or 0), default={})

        brief = (
            f"Phase21 通用范式：最优严格 timing {best_timing.get('id')} {best_timing.get('timing', 0):.1%}；"
            f"最优 acc {best_acc.get('id')} {best_acc.get('accuracy', 0):.1%}；"
            f"ε=1 最高 {best_eps1.get('id')} {best_eps1.get('timing_eps1', 0):.1%}；"
            f"Phase20 {bl20:.1%} knn {knn_acc:.1%}。"
        )

        return {
            "candidates": candidates,
            "best_timing": best_timing,
            "best_acc": best_acc,
            "best_deployable_mvp": best_mvp,
            "best_eps1": best_eps1,
            "baseline_phase20_timing": bl20,
            "baseline_knn_acc": knn_acc,
            "feasible_any": any(c.get("feasible") for c in candidates),
            "deployable_mvp_any": any(c.get("deployable_mvp") for c in candidates),
            "mentor_brief": brief,
            "insight": "Phase21 换范式：元预算/跳数条件化/稳定性/ε-timing，不再只调 min_n 阈值。",
            "device": args.device,
        }

    path = timed_run(run_body, "w6_proof_rollup", "W6 · 汇总", device=args.device)
    write_phase21_result("w6_proof_rollup", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
