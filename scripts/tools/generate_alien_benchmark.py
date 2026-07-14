#!/usr/bin/env python3
"""生成「陌生题库」AlienBench — 与 ProsQA 不同的拓扑、命名、跳数分布。"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "alien"

# 刻意不用 ProsQA 式造词；用模块/层级命名
ENTITY_POOL = [
    "Pilot", "Engine", "Sensor", "Relay", "Core", "Buffer", "Gateway", "Node",
    "Module", "Unit", "Port", "Bus", "Cache", "Frame", "Signal", "Layer",
    "Axis", "Cell", "Grid", "Hub", "Link", "Matrix", "Sector", "Zone",
]


def _pick_names(rng: random.Random, n: int) -> list[str]:
    pool = ENTITY_POOL[:]
    rng.shuffle(pool)
    names = ["Alpha"]
    while len(names) < n:
        cand = pool.pop() if pool else f"X{len(names)}"
        if cand not in names:
            names.append(cand)
    return names


def _make_sample(
    *,
    rng: random.Random,
    topology: str,
    edges: list[list[int]],
    steps: list[str],
    target: int,
    neg_target: int,
    sample_id: int,
) -> dict:
    n_nodes = max(max(u, v) for u, v in edges) + 1
    symbols = _pick_names(rng, n_nodes)
    root_name = symbols[0]
    tgt_name = symbols[target]
    neg_name = symbols[neg_target]
    question = (
        f"{root_name} connects through subsystem graph. "
        f"Is {root_name} linked to {tgt_name} or {neg_name}?"
    )
    answer = f"{root_name} is linked to {tgt_name}."
    return {
        "question": question,
        "answer": answer,
        "steps": steps,
        "idx_to_symbol": symbols,
        "edges": edges,
        "root": 0,
        "target": target,
        "neg_target": neg_target,
        "neighbor_k": {},
        "_alien": True,
        "_meta": {
            "sample_id": sample_id,
            "topology": topology,
            "reasoning_hops": len(steps),
            "benchmark": "AlienBench_v1",
        },
    }


def _star_spoke(rng: random.Random, sid: int, chain_len: int) -> dict:
    """星型枢纽 + 长链：BFS 深度与 steps 长度可不一致。"""
    n = chain_len + 3
    edges = [[0, 1]]
    steps = [f"Alpha reaches Hub."]
    for i in range(1, chain_len):
        edges.append([i, i + 1])
        steps.append(f"Hub layer {i} forwards to layer {i + 1}.")
    target = chain_len
    for j in range(chain_len + 1, n):
        edges.append([1, j])
    neg = chain_len + 1
    steps.append(f"Layer {chain_len} connects to Target.")
    return _make_sample(
        rng=rng, topology="star", edges=edges, steps=steps,
        target=target, neg_target=neg, sample_id=sid,
    )


def _ladder(rng: random.Random, sid: int, height: int) -> dict:
    """双轨梯子 + 横档：图宽，非 ProsQA 链式。"""
    edges = []
    steps = ["Alpha enters rail A."]
    for i in range(height):
        edges.append([i, i + 1])
        edges.append([height + 1 + i, height + 2 + i])
        steps.append(f"Rung {i + 1} on rail A advances.")
        if i < height:
            edges.append([i + 1, height + 2 + i])
    target = height
    neg = 2 * height + 2
    edges.append([height + 1, neg])
    steps.append("Target reached at rail A end.")
    return _make_sample(
        rng=rng, topology="ladder", edges=edges, steps=steps,
        target=target, neg_target=neg, sample_id=sid,
    )


def _hourglass(rng: random.Random, sid: int, depth: int) -> dict:
    """沙漏：分叉再汇合再分叉。"""
    mid = depth
    edges = [[0, 1], [0, 2], [1, mid], [2, mid]]
    steps = ["Alpha splits to path U and path L.", "Paths merge at Mid."]
    for i in range(mid, mid + depth - 1):
        edges.extend([[i, i + 1], [i, i + depth]])
        steps.append(f"Mid layer {i - mid + 1} expands.")
    target = mid + depth - 1
    neg = mid + depth
    edges.append([mid + depth - 2, neg])
    steps.append("Target on upper branch.")
    return _make_sample(
        rng=rng, topology="hourglass", edges=edges, steps=steps,
        target=target, neg_target=neg, sample_id=sid,
    )


def _wide_tree(rng: random.Random, sid: int, depth: int) -> dict:
    """宽树：每层 2 分支，深度 depth。"""
    edges = []
    steps = ["Alpha at tree root."]
    node = 0
    next_id = 1
    frontier = [0]
    layers = [0]
    for d in range(depth):
        new_frontier = []
        for parent in frontier:
            for _ in range(2):
                edges.append([parent, next_id])
                new_frontier.append(next_id)
                next_id += 1
        frontier = new_frontier
        layers.append(frontier[-1])
        steps.append(f"Tree depth {d + 1} branch.")
    target = layers[-1]
    neg = layers[-2] if len(layers) > 1 else 1
    return _make_sample(
        rng=rng, topology="tree", edges=edges, steps=steps,
        target=target, neg_target=neg, sample_id=sid,
    )


def generate_all(seed: int = 20260713, per_topology: int = 20) -> list[dict]:
    rng = random.Random(seed)
    samples: list[dict] = []
    sid = 0
    for _ in range(per_topology):
        samples.append(_star_spoke(rng, sid, chain_len=rng.choice([4, 5, 6, 7])))
        sid += 1
    for _ in range(per_topology):
        samples.append(_ladder(rng, sid, height=rng.choice([3, 4, 5])))
        sid += 1
    for _ in range(per_topology):
        samples.append(_hourglass(rng, sid, depth=rng.choice([2, 3])))
        sid += 1
    for _ in range(per_topology):
        samples.append(_wide_tree(rng, sid, depth=rng.choice([3, 4, 5])))
        sid += 1
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260713)
    ap.add_argument("--per-topology", type=int, default=20)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_samples = generate_all(seed=args.seed, per_topology=args.per_topology)

    (OUT_DIR / "alien_benchmark.json").write_text(
        json.dumps(all_samples, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    by_topo: dict[str, list] = {}
    for s in all_samples:
        by_topo.setdefault(s["_meta"]["topology"], []).append(s)

    for topo, rows in by_topo.items():
        path = OUT_DIR / f"alien_{topo}.json"
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {path} ({len(rows)} samples)")

    meta = {
        "benchmark": "AlienBench_v1",
        "total": len(all_samples),
        "per_topology": {k: len(v) for k, v in by_topo.items()},
        "seed": args.seed,
        "note": "非 ProsQA 拓扑/命名；用于通解外推真·陌生集验证",
    }
    (OUT_DIR / "README.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Wrote {OUT_DIR / 'alien_benchmark.json'} ({len(all_samples)} total)")


if __name__ == "__main__":
    main()
