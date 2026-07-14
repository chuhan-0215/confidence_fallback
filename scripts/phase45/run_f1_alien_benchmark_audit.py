#!/usr/bin/env python3
"""F1 · AlienBench 数据集描述 + F0 结果审计（无需 GPU）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase45_common import (  # noqa: E402
    ALIEN_SLICE_IDS,
    PHASE45_OUT,
    rollup_transfer_rows,
    write_phase45_result,
)
from dataset_registry import list_alien_slices, load_slice  # noqa: E402
from graph_utils import graph_diameter, reasoning_hops, root_to_target_distance  # noqa: E402


def _topology_stats(slice_id: str) -> dict:
    _, rows = load_slice(slice_id)
    hops = [reasoning_hops(r) for r in rows]
    diams = [graph_diameter(r) for r in rows]
    dists = [root_to_target_distance(r) for r in rows]
    topologies = [r.get("_meta", {}).get("topology") for r in rows]
    return {
        "n": len(rows),
        "reasoning_hops_min": min(hops) if hops else None,
        "reasoning_hops_max": max(hops) if hops else None,
        "reasoning_hops_mean": round(sum(hops) / len(hops), 2) if hops else None,
        "diameter_min": min(diams) if diams else None,
        "diameter_max": max(diams) if diams else None,
        "bfs_dist_min": min(dists) if dists else None,
        "bfs_dist_max": max(dists) if dists else None,
        "topologies": sorted(set(t for t in topologies if t)),
    }


def main() -> None:
    readme_path = ROOT / "data" / "alien" / "README.json"
    readme = json.loads(readme_path.read_text(encoding="utf-8")) if readme_path.is_file() else {}

    catalog = list_alien_slices()
    stats = {s["id"]: _topology_stats(s["id"]) for s in catalog}

    f0_path = PHASE45_OUT / "f0_alien_benchmark_transfer_latest.json"
    f0 = json.loads(f0_path.read_text(encoding="utf-8")) if f0_path.is_file() else None
    f0_rows = (f0 or {}).get("slices") or []
    f0_summary = (f0 or {}).get("summary") or rollup_transfer_rows(f0_rows) if f0_rows else None

    hurt = [r for r in f0_rows if r.get("transfer_hurts")]
    help_ = [r for r in f0_rows if r.get("transfer_helps")]

    conclusions = [
        "AlienBench_v1：80 题、4 种非 ProsQA 拓扑、模块式命名（Alpha/Hub/Node 等）",
        "与 Phase44 OOD 代理区别：不再用 gibberish 链名，图结构为星/梯/沙漏/树",
        "仍共享 Coconut JSON schema；不等于 GSM8K / HotpotQA 等真外部 benchmark",
    ]
    if f0_summary:
        conclusions.append(
            f"F0 冻结 τ=0.48：helps {f0_summary.get('transfer_helps_count', 0)} / "
            f"hurts {f0_summary.get('transfer_hurts_count', 0)}，"
            f"mean Δ {f0_summary.get('mean_delta_pp')} pp"
        )
        if hurt:
            conclusions.append(
                "hurt 切片建议：tri_zone → skip → τ sweep（复用 Phase44 E1 流程）"
            )
    else:
        conclusions.append("F0 尚未跑：请在 A800 执行 run_f0_alien_benchmark_transfer.py")

    payload = {
        "experiment_id": "f1_alien_benchmark_audit",
        "title": "F1 · AlienBench 题库审计",
        "benchmark_readme": readme,
        "slice_catalog": catalog,
        "topology_stats": stats,
        "alien_slice_ids": ALIEN_SLICE_IDS,
        "f0_available": f0 is not None,
        "f0_summary": f0_summary,
        "f0_rows": f0_rows,
        "hurt_slices": [{"slice_id": r["slice_id"], "delta_pp": r["delta_pp"]} for r in hurt],
        "help_slices": [{"slice_id": r["slice_id"], "delta_pp": r["delta_pp"]} for r in help_],
        "conclusions": conclusions,
        "ok": True,
    }
    path = write_phase45_result("f1_alien_benchmark_audit", payload)
    print(f"Wrote {path}")
    for line in conclusions:
        print(f"  · {line}")


if __name__ == "__main__":
    main()
