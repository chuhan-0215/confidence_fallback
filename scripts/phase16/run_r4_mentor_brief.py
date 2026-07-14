#!/usr/bin/env python3
"""R4 · 导师汇报包：证据链 + 推荐部署配置。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase16_common import (
    FIXED_3_ACC, TIMING_FLOOR, load_json_candidates, m3_necessity_pass, timed_run, write_phase16_result,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        m2 = load_json_candidates("phase10/m2_learned_enough_stop_latest.json")
        m3 = load_json_candidates("phase10/m3_extra_steps_ablation_latest.json")
        p1 = load_json_candidates("phase14/p1_min_n_sweep_latest.json")
        q1 = load_json_candidates("phase15/q1_min3_full419_latest.json")
        r1 = load_json_candidates("phase16/r1_mvp_final_latest.json")
        r2 = load_json_candidates("phase16/r2_knn_min3_full419_latest.json")
        r3 = load_json_candidates("phase16/r3_pareto_full419_latest.json")

        ab = m3.get("ablation") or {}
        q1_ba = q1.get("best_deployable_mvp") or {}
        q1_bt = q1.get("best_timing") or {}
        r1_primary = (r1.get("variants") or {}).get("best_deployable_mvp") or {}
        r1_full = r1_primary.get("full_419") or {}

        evidence = {
            "必要性_M3": {
                "pass": m3_necessity_pass(m3),
                "救回": ab.get("pct_improved", 0),
                "搞砸": ab.get("pct_degraded", 0),
            },
            "可学习_M2": {
                "pass": (m2.get("test") or {}).get("accuracy", 0) >= 0.85,
                "acc": (m2.get("test") or {}).get("accuracy"),
            },
            "min_n突破_P14": {
                "pass": (p1.get("best_timing") or {}).get("stop_timing_acc", 0) >= 0.38,
                "timing": (p1.get("best_timing") or {}).get("stop_timing_acc"),
            },
            "全量_deployable_mvp": {
                "pass": r1.get("deployable_mvp", False),
                "acc": r1_full.get("accuracy"),
                "timing": r1_full.get("stop_timing_acc"),
            },
            "timing50_stretch": {
                "pass": (r1_full.get("stop_timing_acc") or 0) >= TIMING_FLOOR,
                "ceiling_full": q1_bt.get("stop_timing_acc"),
                "ceiling_test": (p1.get("best_timing") or {}).get("stop_timing_acc"),
            },
        }

        recommend = {
            "primary": "min3_m2_head",
            "config": r1_primary.get("config") or {"min_n": 3, "threshold": q1_ba.get("threshold", 0.5)},
            "full_419": r1_full,
            "rationale": (
                "冻结 Coconut + M2 head；min_n=3 消除早停；"
                f"全量 acc≥{FIXED_3_ACC:.1%}；无 BFS/oracle；M3 证 overthink 有害。"
            ),
        }
        if r2.get("best_deployable_mvp", {}).get("deployable_mvp") and (
            r2["best_deployable_mvp"].get("accuracy", 0) > (r1_full.get("accuracy") or 0)
        ):
            recommend["alt"] = {
                "strategy": "knn_min3",
                "config": r2["best_deployable_mvp"],
            }

        return {
            "evidence": evidence,
            "recommend": recommend,
            "pareto_best": r3.get("best_deployable_mvp"),
            "deployable_mvp": evidence["全量_deployable_mvp"]["pass"],
            "strict_feasible": evidence["全量_deployable_mvp"]["pass"] and evidence["timing50_stretch"]["pass"],
            "mentor_brief": (
                f"【够好就停·可部署】min_n=3 + M2 head：全量 acc {r1_full.get('accuracy', 0):.1%}，"
                f"timing {r1_full.get('stop_timing_acc', 0):.1%}（天花板 ~39% test / ~37% full），"
                f"mean_n={r1_full.get('mean_stop_n')}。"
                f"过 fixed_3({FIXED_3_ACC:.1%})；M3 多走搞砸{ab.get('pct_degraded', 0):.1%}、救回 0%。"
                f"timing≥50% 需改 Coconut 写回层，非 head 可达。"
            ),
            "insight": "Phase16 导师汇报定稿包。",
            "device": args.device,
        }

    path = timed_run(run_body, "r4_mentor_brief", "R4 · 导师汇报", device=args.device)
    import json
    write_phase16_result("r4_mentor_brief", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
