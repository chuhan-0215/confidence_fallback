#!/usr/bin/env python3
"""D4 · Phase 28 rollup + 与 P25/P26/P27 对照。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase28_common import load_json, timed_run, write_phase28_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        snippets = []
        for eid in ("d1_gap3_diagnosis", "d2_hop_split_thr", "d3_disagree_reroute"):
            d = load_json(f"phase28/{eid}_latest.json")
            if not d:
                continue
            if eid == "d1_gap3_diagnosis":
                snippets.append({"id": eid, "not_triggered": d.get("not_triggered_count"), "cases": d.get("cases")})
            else:
                full = d.get("full_419") or d.get("best") or {}
                snippets.append({"id": eid, "accuracy": full.get("accuracy")})
        p27 = load_json("phase27/c7_alt_paradigm_rollup_latest.json")
        p25 = 0.9523
        best_p28 = max((s for s in snippets if s.get("accuracy")), key=lambda s: s["accuracy"], default={})
        best_p27 = (p27 or {}).get("best_alternative") or {}
        overall_best = max(
            [x for x in (
                {"source": "p25", "acc": p25},
                {"source": best_p28.get("id", ""), "acc": best_p28.get("accuracy", 0)},
                {"source": best_p27.get("id", ""), "acc": best_p27.get("accuracy", 0)},
            ) if x["acc"]],
            key=lambda x: x["acc"],
            default={"source": "p25", "acc": p25},
        )
        return {
            "snippets": snippets,
            "phase27_best": best_p27,
            "overall_best": overall_best,
            "union_ceiling": 0.9594,
            "mentor_brief": (
                f"P28 rollup：最优 {overall_best['source']} {overall_best['acc']:.1%}；"
                f"上界 95.94%。"
            ),
            "device": args.device,
        }

    path = timed_run(run_body, "d4_phase28_rollup", "D4 · P28 rollup", device=args.device)
    write_phase28_result("d4_phase28_rollup", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
