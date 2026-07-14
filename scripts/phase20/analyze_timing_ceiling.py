#!/usr/bin/env python3
"""CPU · 从 X4 gap 分布推算 min_n 约束下的 timing 理论上界。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_x4() -> dict:
    for base in (ROOT / "results/from_a800", ROOT / "outbox/results/from_a800"):
        p = base / "run_20260630_x4_depth_mismatch/x4_depth_mismatch_latest.json"
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
        p2 = base / "results/from_a800/phase4/x4_depth_mismatch_latest.json"
        if p2.is_file():
            return json.loads(p2.read_text(encoding="utf-8"))
    raise FileNotFoundError("x4_depth_mismatch_latest.json not found")


def gap_to_fc(gap: int, d: int) -> int | None:
    if gap == "never":
        return None
    return d + int(gap)


def main() -> None:
    x4 = load_x4()
    dist = x4["mismatch_fc_minus_blind_depth"]
    by_hop = x4["by_reasoning_hops"]

    fc_counts: dict[int, int] = {}
    never = int(dist.get("never", 0))
    for hop_str, hop_dist in by_hop.items():
        d = int(hop_str)
        for gap_str, cnt in hop_dist.items():
            if gap_str == "never":
                continue
            fc = d + int(gap_str)
            fc_counts[fc] = fc_counts.get(fc, 0) + cnt

    total_fc = sum(fc_counts.values())
    print("=== X4 first_correct 分布（全量 419）===\n")
    for fc in sorted(fc_counts):
        pct = 100 * fc_counts[fc] / total_fc
        print(f"  fc={fc}: {fc_counts[fc]:4d} ({pct:5.1f}%)")
    print(f"  never: {never}")
    print(f"  total with fc: {total_fc}\n")

    print("=== min_n 约束下的 timing 理论上界（假设完美停步）===\n")
    for min_n in (1, 2, 3, 4):
        hittable = sum(c for fc, c in fc_counts.items() if fc >= min_n)
        ceiling = hittable / total_fc if total_fc else 0
        blocked = total_fc - hittable
        print(f"  min_n={min_n}: ceiling={ceiling:.1%}  "
              f"(blocked={blocked} 题 fc<{min_n})")

    print("\n=== 与 Phase16 实测对比 ===\n")
    benchmarks = [
        ("Phase16 min3 thr=0.35", 0.3701, 3),
        ("Phase16 min3 thr=0.5", 0.3284, 3),
        ("Phase18 W1 thr=0.15", 0.3652, 3),
    ]
    ceiling3 = sum(c for fc, c in fc_counts.items() if fc >= 3) / total_fc
    for name, timing, min_n in benchmarks:
        util = timing / ceiling3 if ceiling3 else 0
        print(f"  {name}: timing={timing:.1%}  "
              f"utilization vs ceiling(min_n={min_n})={util:.1%}")

    out = {
        "fc_distribution": fc_counts,
        "never_correct": never,
        "total_with_fc": total_fc,
        "ceilings": {
            str(min_n): round(
                sum(c for fc, c in fc_counts.items() if fc >= min_n) / total_fc, 4
            )
            for min_n in (1, 2, 3, 4)
        },
        "insight": "min_n=3 理论 timing 上限约 44%；当前 37% 已达上限 84%。",
    }
    out_dir = ROOT / "results" / "phase20"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "timing_ceiling_analysis.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
