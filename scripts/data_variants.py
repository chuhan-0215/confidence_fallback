#!/usr/bin/env python3
"""多种数据构造变体：链式 / 扩展真实 ProsQA / 树 / 菱形 / 稠密噪声图。"""

from __future__ import annotations

import copy
import json
import random
import string
from pathlib import Path
from typing import List, Optional, Set

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "variants"
MASTER = ROOT / "data" / "prosqa_test_graph_4_coconut.json"

SYLLABLES = [
    "rom", "ter", "pus", "imp", "lum", "zor", "vex", "quim", "brim", "storp",
    "wum", "yum", "jel", "hil", "shum", "ger", "lor", "dump", "scrom", "brimp",
]


def _rand_type(rng: random.Random, used: Set[str]) -> str:
    for _ in range(200):
        a, b, c = rng.sample(SYLLABLES, 3)
        name = (a + b + c).capitalize()
        if name not in used:
            used.add(name)
            return name
    name = "Type" + "".join(rng.choice(string.ascii_lowercase) for _ in range(5))
    used.add(name)
    return name


def _rand_person(rng: random.Random, used: Set[str]) -> str:
    pool = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Henry", "Tom", "Max"]
    for _ in range(50):
        name = rng.choice(pool)
        if name not in used:
            used.add(name)
            return name
    name = "Person" + str(rng.randint(100, 999))
    used.add(name)
    return name


def make_pure_chain(
    rng: random.Random,
    chain_hops: int,
    *,
    sample_id: int = 0,
    noise_edges: Optional[List[List[int]]] = None,
) -> dict:
    """纯链 + 可选噪声边（不改变最短路径长度）。"""
    if chain_hops < 3:
        raise ValueError("chain_hops >= 3")

    used: Set[str] = set()
    person = _rand_person(rng, used)
    types = [_rand_type(rng, used) for _ in range(chain_hops + 1)]

    root = 0
    target = chain_hops
    neg_target = chain_hops + 1
    symbols = [person] + types + [_rand_type(rng, used)]

    steps = [f"{person} is a {types[0]}."]
    for i in range(chain_hops - 1):
        steps.append(f"Every {types[i]} is a {types[i + 1]}.")

    edges = [[root, 1]]
    for i in range(1, chain_hops):
        edges.append([i, i + 1])
    edges.append([1, neg_target])

    if noise_edges:
        max_node = max(max(u, v) for u, v in edges)
        for u, v in noise_edges:
            if u <= max_node and v <= max_node:
                edges.append([u, v])

    answer = f"{person} is a {symbols[target]}."
    question = " ".join(steps + [f"Is {person} a {symbols[target]} or {symbols[neg_target]}?"])

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
        "_variant": "pure_chain",
        "_meta": {"sample_id": sample_id, "chain_hops": chain_hops},
    }


def make_dense_chain(
    rng: random.Random,
    chain_hops: int,
    template: dict,
    *,
    sample_id: int = 0,
) -> dict:
    """主链 + 侧枝死胡同（不改变根→目标最短路径）。"""
    base = make_pure_chain(rng, chain_hops, sample_id=sample_id)
    symbols = list(base["idx_to_symbol"])
    edges = [list(e) for e in base["edges"]]
    next_node = len(symbols)
    branch_count = min(3, max(1, len(template.get("edges") or []) // 15))

    for _ in range(branch_count):
        parent = rng.randint(1, max(1, chain_hops - 1))
        depth = rng.randint(1, 2)
        prev = parent
        for _ in range(depth):
            used = set(symbols)
            dt = _rand_type(rng, used)
            symbols.append(dt)
            edges.append([prev, next_node])
            prev = next_node
            next_node += 1

    base["idx_to_symbol"] = symbols
    base["edges"] = edges
    base["_variant"] = "dense_chain"
    base["_meta"]["noise_branches"] = branch_count
    return base


def extend_prosqa_chain(sample: dict, target_hops: int, rng: random.Random) -> Optional[dict]:
    """在真实 ProsQA 样本末尾延长推理链（保留原图 + 新节点）。"""
    from graph_utils import reasoning_hops

    current = reasoning_hops(sample)
    if current >= target_hops:
        return None

    s = copy.deepcopy(sample)
    symbols = list(s["idx_to_symbol"])
    edges = [list(e) for e in s["edges"]]
    steps = list(s["steps"])
    target_idx = int(s["target"])
    next_node = len(symbols)

    for _ in range(target_hops - current):
        used = set(symbols)
        new_type = _rand_type(rng, used)
        old_type = symbols[target_idx]
        symbols.append(new_type)
        edges.append([target_idx, next_node])
        steps.append(f"Every {old_type} is a {new_type}.")
        target_idx = next_node
        next_node += 1

    person = symbols[s["root"]]
    s["idx_to_symbol"] = symbols
    s["edges"] = edges
    s["steps"] = steps
    s["target"] = target_idx
    s["answer"] = f"{person} is a {symbols[target_idx]}."
    s["question"] = s.get("question", "") + " " + " ".join(steps[current:])
    s["_variant"] = "prosqa_extend"
    s["_meta"] = {"extended_from_hops": current, "target_hops": target_hops}
    return s


def make_tree_sample(rng: random.Random, depth: int, *, sample_id: int = 0) -> dict:
    """二叉树式推理：根沿左子链到目标，右子链为干扰。"""
    if depth < 3:
        raise ValueError("depth >= 3")

    used: Set[str] = set()
    person = _rand_person(rng, used)
    # 左链 depth 层 + 每层右分支 1 跳
    left_types = [_rand_type(rng, used) for _ in range(depth)]
    right_types = [_rand_type(rng, used) for _ in range(depth - 1)]
    neg_type = _rand_type(rng, used)

    symbols = [person] + left_types + right_types + [neg_type]
    root = 0
    target = depth
    neg_target = len(symbols) - 1

    steps = [f"{person} is a {left_types[0]}."]
    for i in range(depth - 1):
        steps.append(f"Every {left_types[i]} is a {left_types[i + 1]}.")

    edges = [[root, 1]]
    for i in range(1, depth):
        edges.append([i, i + 1])
    # 右分支：从中间层岔出
    r_base = depth + 1
    for i in range(depth - 1):
        parent = 1 + i
        rnode = r_base + i
        edges.append([parent, rnode])
    edges.append([1, neg_target])

    answer = f"{person} is a {left_types[-1]}."
    return {
        "question": " ".join(steps + [f"Is {person} a {left_types[-1]} or {neg_type}?"]),
        "answer": answer,
        "steps": steps,
        "idx_to_symbol": symbols,
        "edges": edges,
        "root": root,
        "target": target,
        "neg_target": neg_target,
        "neighbor_k": {},
        "_variant": "tree",
        "_meta": {"sample_id": sample_id, "depth": depth},
    }


def make_diamond_sample(rng: random.Random, path_len: int, *, sample_id: int = 0) -> dict:
    """菱形：两条等长路径汇合到目标，检验是否需要更多 latent 步。"""
    if path_len < 3:
        raise ValueError("path_len >= 3")

    used: Set[str] = set()
    person = _rand_person(rng, used)
    upper = [_rand_type(rng, used) for _ in range(path_len - 1)]
    lower = [_rand_type(rng, used) for _ in range(path_len - 1)]
    goal = _rand_type(rng, used)
    neg_type = _rand_type(rng, used)

    symbols = [person] + upper + lower + [goal, neg_type]
    root = 0
    target = len(symbols) - 2
    neg_target = len(symbols) - 1

    steps = [f"{person} is a {upper[0]}.", f"{person} is a {lower[0]}."]
    for i in range(path_len - 2):
        steps.append(f"Every {upper[i]} is a {upper[i + 1]}.")
        steps.append(f"Every {lower[i]} is a {lower[i + 1]}.")
    steps.append(f"Every {upper[-1]} is a {goal}.")
    steps.append(f"Every {lower[-1]} is a {goal}.")

    u_start, l_start = 1, 1 + len(upper)
    edges = [[root, u_start], [root, l_start]]
    for i in range(path_len - 2):
        edges.append([u_start + i, u_start + i + 1])
        edges.append([l_start + i, l_start + i + 1])
    edges.append([u_start + path_len - 2, target])
    edges.append([l_start + path_len - 2, target])
    edges.append([u_start, neg_target])

    answer = f"{person} is a {goal}."
    return {
        "question": " ".join(steps + [f"Is {person} a {goal} or {neg_type}?"]),
        "answer": answer,
        "steps": steps,
        "idx_to_symbol": symbols,
        "edges": edges,
        "root": root,
        "target": target,
        "neg_target": neg_target,
        "neighbor_k": {},
        "_variant": "diamond",
        "_meta": {"sample_id": sample_id, "path_len": path_len},
    }


def generate_extend_bundle(
    master: List[dict],
    target_hops: int,
    count: int,
    seed: int,
    *,
    max_node_id: int = 30,
) -> List[dict]:
    from graph_utils import reasoning_hops

    rng = random.Random(seed)
    pool = [s for s in master if reasoning_hops(s) in (3, 4)]
    rng.shuffle(pool)
    out = []
    pi = 0
    while len(out) < count and pi < len(pool) * 5:
        src = pool[pi % len(pool)]
        pi += 1
        ext = extend_prosqa_chain(src, target_hops, rng)
        if ext and _max_node_id(ext) <= max_node_id:
            ext["_idx"] = len(out)
            out.append(ext)
    return out


def _max_node_id(sample: dict) -> int:
    edges = sample.get("edges") or []
    if not edges:
        return 0
    return max(max(u, v) for u, v in edges)


def generate_extend_bundle_from_base_hops(
    master: List[dict],
    base_hops: int,
    target_hops: int,
    count: int,
    seed: int,
    *,
    max_node_id: int = 30,
) -> List[dict]:
    """只从指定原始跳数的 ProsQA 样本延长，保证同质深度。"""
    from graph_utils import reasoning_hops

    if target_hops <= base_hops:
        raise ValueError("target_hops must exceed base_hops")

    rng = random.Random(seed)
    pool = [s for s in master if reasoning_hops(s) == base_hops]
    if not pool:
        raise ValueError(f"no samples with base_hops={base_hops}")

    rng.shuffle(pool)
    out = []
    pi = 0
    while len(out) < count and pi < len(pool) * 8:
        src = pool[pi % len(pool)]
        pi += 1
        ext = extend_prosqa_chain(src, target_hops, rng)
        if ext and _max_node_id(ext) <= max_node_id:
            meta = dict(ext.get("_meta") or {})
            meta["base_hops"] = base_hops
            ext["_meta"] = meta
            ext["_idx"] = len(out)
            out.append(ext)
    return out


def write_bundle(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} -> {path}")


def generate_all(out_dir: Path = DATA_DIR) -> None:
    master = json.loads(MASTER.read_text(encoding="utf-8"))
    template = next(s for s in master if len(s.get("steps") or []) == 4)

    bundles = [
        ("chain_3hop.json", [make_pure_chain(random.Random(100 + i), 3, sample_id=i) for i in range(50)]),
        ("chain_4hop.json", [make_pure_chain(random.Random(200 + i), 4, sample_id=i) for i in range(50)]),
        ("chain_5hop_dense.json", [
            make_dense_chain(random.Random(300 + i), 5, template, sample_id=i) for i in range(50)
        ]),
        ("chain_6hop_dense.json", [
            make_dense_chain(random.Random(400 + i), 6, template, sample_id=i) for i in range(50)
        ]),
        ("chain_7hop_dense.json", [
            make_dense_chain(random.Random(500 + i), 7, template, sample_id=i) for i in range(50)
        ]),
        ("prosqa_extend_5hop.json", generate_extend_bundle(master, 5, 50, seed=600)),
        ("prosqa_extend_6hop.json", generate_extend_bundle(master, 6, 50, seed=601)),
        ("prosqa_extend_7hop.json", generate_extend_bundle(master, 7, 50, seed=602)),
        ("tree_5hop.json", [make_tree_sample(random.Random(700 + i), 5, sample_id=i) for i in range(50)]),
        ("diamond_5hop.json", [make_diamond_sample(random.Random(800 + i), 5, sample_id=i) for i in range(50)]),
    ]

    for name, rows in bundles:
        write_bundle(out_dir / name, rows)


def generate_boundary_push_bundles(out_dir: Path = DATA_DIR) -> None:
    """边界上推实验：同质基线跳数 → 5/6 跳延长。"""
    master = json.loads(MASTER.read_text(encoding="utf-8"))
    ext5_from4 = generate_extend_bundle_from_base_hops(master, 4, 5, 50, seed=610)
    ext6_from4 = generate_extend_bundle_from_base_hops(master, 4, 6, 50, seed=611)
    ext5_from3 = generate_extend_bundle_from_base_hops(master, 3, 5, 50, seed=612)
    ext6_from3 = generate_extend_bundle_from_base_hops(master, 3, 6, 50, seed=613)
    ext7_from3 = generate_extend_bundle_from_base_hops(master, 3, 7, 50, seed=614)
    ext8_from3 = generate_extend_bundle_from_base_hops(master, 3, 8, 50, seed=615)
    ext7_mixed = generate_extend_bundle(master, 7, 50, seed=616)
    mix_56_from4 = ext5_from4[:30] + ext6_from4[:30]
    for i, row in enumerate(mix_56_from4):
        row["_idx"] = i

    bundles = [
        ("prosqa_extend_5hop_from4.json", ext5_from4),
        ("prosqa_extend_6hop_from4.json", ext6_from4),
        ("prosqa_extend_5hop_from3.json", ext5_from3),
        ("prosqa_extend_6hop_from3.json", ext6_from3),
        ("prosqa_extend_7hop_from3.json", ext7_from3),
        ("prosqa_extend_8hop_from3.json", ext8_from3),
        ("prosqa_extend_7hop_mixed.json", ext7_mixed),
        ("prosqa_mix_56_from4.json", mix_56_from4),
    ]
    for name, rows in bundles:
        write_bundle(out_dir / name, rows)


if __name__ == "__main__":
    generate_all()
    generate_boundary_push_bundles()
