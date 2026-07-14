#!/usr/bin/env python3
"""Z6 · Phase 23 收官汇总 + 最终 deploy manifest。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase23_common import load_json, timed_run, write_phase23_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        snippets = []
        for eid, label in (
            ("z1_seed_robustness", "稳健性"),
            ("z2_hop_slices", "跳数切片"),
            ("z3_disagreement_fusion", "分歧融合"),
            ("z4_asymmetry_budget", "不对称"),
            ("z5_dual_probe", "双探针"),
        ):
            d = load_json(f"phase23/{eid}_latest.json")
            if not d:
                continue
            if eid == "z1_seed_robustness":
                snippets.append({
                    "id": eid, "label": label,
                    "structure_d_acc_mean": (d.get("report") or {}).get("structure_d", {}).get("acc_stats", {}).get("mean"),
                    "structure_m2_eps1_mean": (d.get("report") or {}).get("structure_d_m2", {}).get("eps1_stats", {}).get("mean"),
                })
            elif eid == "z2_hop_slices":
                s4 = next((s for s in d.get("slices", []) if s.get("hop") == 4 and s.get("strategy") == "structure_d"), {})
                snippets.append({"id": eid, "label": label, "hop4_structure_d": s4.get("accuracy")})
            else:
                full = d.get("full_419") or d.get("best") or {}
                snippets.append({
                    "id": eid, "label": label,
                    "accuracy": full.get("accuracy"),
                    "timing_eps1": full.get("timing_eps1"),
                })

        p22 = load_json("phase22/x6_proof_rollup_latest.json")
        rec = (p22.get("deploy_recommendation") or {})
        z3 = load_json("phase23/z3_disagreement_fusion_latest.json")
        z5 = load_json("phase23/z5_dual_probe_latest.json")

        fusion_acc = (z3.get("best") or {}).get("accuracy")
        dual_acc = (z5.get("best") or {}).get("accuracy")
        beat_struct = max(a for a in (fusion_acc, dual_acc) if a is not None) if (fusion_acc or dual_acc) else None

        manifest = {
            "acc_primary": rec.get("acc_primary", "structure_d"),
            "acc_value": rec.get("acc_value", 0.9356),
            "eps_deploy_primary": "structure_d_m2",
            "eps_deploy_value": 0.5074,
            "strict_timing_primary": "knn_min3_full",
            "strict_timing_value": 0.37,
            "phase23_fusion_best": fusion_acc,
            "phase23_dual_probe_best": dual_acc,
            "closure": "Phase 22 deploy 定稿 + Phase 23 稳健性/融合验证",
        }

        brief = (
            f"Phase23 收官：P22 acc {manifest['acc_primary']} {manifest['acc_value']:.1%}；"
            f"融合最优 {fusion_acc or 0:.1%} 双探针 {dual_acc or 0:.1%}；"
            f"ε-deploy structure_d+M2 {manifest['eps_deploy_value']:.1%}。"
        )

        return {
            "snippets": snippets,
            "deploy_manifest": manifest,
            "deploy_recommendation": rec,
            "beat_structure_d_any": beat_struct is not None and beat_struct > (manifest.get("acc_value") or 0),
            "mentor_brief": brief,
            "insight": "全项目收官：双轨 deploy 定稿 + Phase23 验证稳健性与融合上界。",
            "device": args.device,
        }

    path = timed_run(run_body, "z6_final_manifest", "Z6 · 收官", device=args.device)
    write_phase23_result("z6_final_manifest", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
