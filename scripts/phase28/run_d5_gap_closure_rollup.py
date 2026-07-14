#!/usr/bin/env python3
"""D5 · Phase 28 rollup + 缺口闭环判定。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase28_common import GAP_INDICES, load_json, timed_run, write_phase28_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        snippets = []
        for eid in ("d1_gap3_diagnosis", "d2_mid_conf_zone", "d3_hop_hybrid", "d4_always_compare"):
            d = load_json(f"phase28/{eid}_latest.json")
            if not d:
                continue
            if eid == "d1_gap3_diagnosis":
                snippets.append({"id": eid, "not_triggered": d.get("not_triggered_count"), "cases": d.get("cases")})
            else:
                full = d.get("full_419") or d.get("best") or {}
                snippets.append({
                    "id": eid,
                    "accuracy": full.get("accuracy"),
                    "gap_hit": full.get("gap_hit"),
                })
        p27 = load_json("phase27/c7_alt_paradigm_rollup_latest.json")
        p25 = 0.9523
        best = max(
            (s for s in snippets if s.get("accuracy")),
            key=lambda s: s["accuracy"],
            default={},
        )
        gap_closed = (best.get("gap_hit") or 0) >= 3
        beat = (best.get("accuracy") or 0) > p25
        return {
            "snippets": snippets,
            "gap_indices": list(GAP_INDICES),
            "best_p28": best,
            "champion_p25": p25,
            "p27_best": (p27 or {}).get("best_alternative", {}).get("accuracy"),
            "beat_champion": beat,
            "gap_closed": gap_closed,
            "union_ceiling": 0.9594,
            "mentor_brief": (
                f"P28：最优 {best.get('id', '—')} {best.get('accuracy', 0):.1%} "
                f"gap_hit {best.get('gap_hit', '—')}/3；{'超冠军' if beat else '未超'}。"
            ),
            "device": args.device,
        }

    path = timed_run(run_body, "d5_gap_closure_rollup", "D5 · 闭环", device=args.device)
    write_phase28_result("d5_gap_closure_rollup", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
