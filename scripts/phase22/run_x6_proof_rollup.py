#!/usr/bin/env python3
"""X6 · Phase 22 汇总 + 部署定稿 mentor_brief。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase22_common import load_json, timed_run, write_phase22_result


def _full(d: dict):
    return d.get("full_419") or d.get("best_acc") or d.get("best") or {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        exps = [
            ("x1_structure_m2", "二段式"),
            ("x2_hop_split_budget", "跳数分治"),
            ("x3_epsilon_stop_train", "ε-stop"),
            ("x4_eps_deploy_pareto", "ε-Pareto"),
            ("x5_complement_audit", "互补"),
        ]
        snippets = []
        for eid, label in exps:
            d = load_json(f"phase22/{eid}_latest.json")
            if not d:
                continue
            if eid == "x5_complement_audit":
                snippets.append({
                    "id": eid, "label": label,
                    "union_acc": d.get("union_acc"),
                    "structure_d_acc": d.get("structure_d_acc"),
                    "knn_m2_acc": d.get("knn_m2_acc"),
                })
            elif eid == "x4_eps_deploy_pareto":
                snippets.append({
                    "id": eid, "label": label,
                    "best_acc": d.get("best_acc"),
                    "best_eps1": d.get("best_eps1"),
                    "eps_deployable_any": d.get("eps_deployable_any"),
                })
            else:
                full = _full(d)
                snippets.append({
                    "id": eid, "label": label,
                    "accuracy": full.get("accuracy"),
                    "stop_timing_acc": full.get("stop_timing_acc"),
                    "timing_eps1": full.get("timing_eps1"),
                    "deployable_mvp": full.get("deployable_mvp") or d.get("deployable_mvp"),
                    "eps_deployable": full.get("eps_deployable") or d.get("eps_deployable"),
                })

        p21 = load_json("phase21/w6_proof_rollup_latest.json")
        p17 = load_json("phase17/s1_corrected_final_latest.json")
        x4 = load_json("phase22/x4_eps_deploy_pareto_latest.json")

        best_acc = (x4 or {}).get("best_acc") or (p21 or {}).get("best_acc") or {}
        best_eps = (x4 or {}).get("best_eps1") or (p21 or {}).get("best_eps1") or {}
        knn = (p17.get("deployable_mvp") or {})

        brief = (
            f"Phase22 部署定稿：acc 冠军 {best_acc.get('id', 'structure_d')} {best_acc.get('accuracy', 0):.1%}；"
            f"ε=1 最高 {best_eps.get('id', 'm2')} {best_eps.get('timing_eps1', 0):.1%}；"
            f"knn 基线 {knn.get('accuracy', 0.926):.1%}。"
        )

        return {
            "snippets": snippets,
            "best_acc": best_acc,
            "best_eps1": best_eps,
            "baseline_knn": knn,
            "baseline_p21_structure_d": (p21.get("best_acc") or {}),
            "feasible_any": any(s.get("feasible") for s in snippets if isinstance(s.get("feasible"), bool)),
            "eps_deployable_any": (x4 or {}).get("eps_deployable_any") or (p21 or {}).get("best_eps1", {}).get("timing_eps1", 0) >= 0.5,
            "deploy_recommendation": {
                "acc_primary": best_acc.get("id", "structure_d"),
                "acc_value": best_acc.get("accuracy"),
                "eps_primary": best_eps.get("id"),
                "eps_value": best_eps.get("timing_eps1"),
                "legacy": "knn_min3_full",
                "legacy_acc": knn.get("accuracy"),
            },
            "mentor_brief": brief,
            "insight": "Phase22：在 P21 收获上定稿——structure_d acc 线 + ε-timing stretch 线。",
            "device": args.device,
        }

    path = timed_run(run_body, "x6_proof_rollup", "X6 · 汇总", device=args.device)
    write_phase22_result("x6_proof_rollup", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
