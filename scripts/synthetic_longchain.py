#!/usr/bin/env python3
"""合成 5–6 跳 ProsQA 风格图样本，用于探测边界是否上移到 5–6 步。"""

from __future__ import annotations

import argparse
import json
import random
import string
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

SYLLABLES = [
    "rom", "ter", "pus", "imp", "lum", "zor", "vex", "quim", "brim", "storp",
    "wum", "yum", "jel", "hil", "shum", "ger", "lor", "zor", "dump", "scrom",
]


def _rand_name(rng: random.Random, used: set) -> str:
    for _ in range(200):
        a, b, c = rng.sample(SYLLABLES, 3)
        name = (a + b + c).capitalize()
        if name not in used:
            used.add(name)
            return name
    return "Type" + "".join(rng.choice(string.ascii_lowercase) for _ in range(5))


def _rand_person(rng: random.Random, used: set) -> str:
    for _ in range(100):
        name = rng.choice(["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Henry"])
        if name not in used:
            used.add(name)
            return name
    return "Person" + str(rng.randint(100, 999))


def make_chain_sample(
    rng: random.Random,
    chain_hops: int,
    *,
    branch_depth: int = 0,
    branch_count: int = 0,
    sample_id: int = 0,
) -> dict:
    """构造单链推理样本。

    chain_hops: 推理链长度（= steps 数 = 根到目标 BFS 距离，纯链时）
    branch_depth: 在链的哪一层岔出干扰边（0=无）
    branch_count: 干扰分支数量
    """
    if chain_hops < 3:
        raise ValueError("chain_hops must be >= 3")

    used: set = set()
    person = _rand_person(rng, used)
    types = [_rand_name(rng, used) for _ in range(chain_hops + 2)]

    # 0=person, 1..chain_hops=链上类型, chain_hops+1=neg
    root = 0
    target = chain_hops
    neg_target = chain_hops + 1
    symbols = [person] + types

    steps: List[str] = [f"{person} is a {types[0]}."]
    for i in range(chain_hops - 1):
        steps.append(f"Every {types[i]} is a {types[i + 1]}.")
    answer_type = types[chain_hops - 1]
    steps[-1] = f"Every {types[chain_hops - 2]} is a {answer_type}."
    # 最后一步应落到 target 类型名
    if chain_hops >= 2:
        steps[-1] = f"Every {types[chain_hops - 2]} is a {types[chain_hops - 1]}."

    # 修正：target 节点对应 types[chain_hops-1] 即链末端
    target = chain_hops
    answer = f"{person} is a {symbols[target]}."

    edges = [[root, 1]]
    for i in range(1, chain_hops):
        edges.append([i, i + 1])

    # 干扰：从某层岔出死胡同，拉高图直径但不改变最短路径长度
    next_node = chain_hops + 2
    for b in range(branch_count):
        parent = max(1, min(branch_depth, chain_hops - 1)) if branch_depth else rng.randint(1, max(1, chain_hops - 1))
        dead_types = [_rand_name(rng, used) for _ in range(rng.randint(1, 2))]
        prev = parent
        for dt in dead_types:
            if next_node >= len(symbols):
                symbols.append(dt)
            else:
                symbols[next_node] = dt
            edges.append([prev, next_node])
            prev = next_node
            next_node += 1

    # neg 类型：与 root 不连通或连到错误类
    edges.append([1, neg_target])

    question_parts = list(steps) + [
        f"Every {symbols[neg_target]} is a {symbols[1]}.",
        f"Is {person} a {symbols[target]} or {symbols[neg_target]}?",
    ]
    question = " ".join(question_parts)

    return {
        "question": question,
        "answer": answer,
        "steps": steps,
        "idx_to_symbol": symbols,
        "edges": edges,
        "root": root,
        "target": target,
        "neg_target": neg_target,
        "neighbor_k": {},
        "_synthetic": True,
        "_meta": {
            "sample_id": sample_id,
            "chain_hops": chain_hops,
            "branch_depth": branch_depth,
            "branch_count": branch_count,
        },
    }


def generate_dataset(
    *,
    chain_hops: int,
    count: int,
    seed: int = 42,
    branch_depth: int = 0,
    branch_count: int = 0,
    tag: str = "",
) -> List[dict]:
    rng = random.Random(seed)
    return [
        make_chain_sample(
            rng,
            chain_hops,
            branch_depth=branch_depth,
            branch_count=branch_count,
            sample_id=i,
        )
        for i in range(count)
    ]


def write_bundle(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic 5–6 hop ProsQA graphs")
    parser.add_argument("--output-dir", default=str(DATA_DIR))
    args = parser.parse_args()
    out = Path(args.output_dir)

    bundles = [
        ("synthetic_chain_5hop.json", generate_dataset(chain_hops=5, count=60, seed=42)),
        ("synthetic_chain_6hop.json", generate_dataset(chain_hops=6, count=60, seed=43)),
        ("synthetic_chain_5hop_wide.json", generate_dataset(
            chain_hops=5, count=60, seed=44, branch_depth=3, branch_count=2
        )),
        ("synthetic_chain_6hop_wide.json", generate_dataset(
            chain_hops=6, count=60, seed=45, branch_depth=4, branch_count=3
        )),
        ("synthetic_mixed_56hop.json", (
            generate_dataset(chain_hops=5, count=30, seed=46)
            + generate_dataset(chain_hops=6, count=30, seed=47)
        )),
    ]

    for name, rows in bundles:
        p = out / name
        write_bundle(p, rows)
        print(f"Wrote {len(rows)} samples -> {p}")


if __name__ == "__main__":
    main()
