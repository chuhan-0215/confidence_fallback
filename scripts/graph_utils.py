"""Graph helpers for ProsQA-style reachability samples."""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple


def build_question_prefix(sample: dict, rng, *, choice_order: str = "random", shuffle_edges: bool = True) -> str:
    edges = sample["edges"][:]
    if shuffle_edges:
        rng.shuffle(edges)
    question = (
        "<eos> "
        + "|".join(f" {u} {v} " for u, v in edges).strip()
        + " [Q] "
    )
    if choice_order == "target_first":
        question += f"{sample['target']} {sample['neg_target']}"
    elif choice_order == "neg_first":
        question += f"{sample['neg_target']} {sample['target']}"
    elif rng.random() < 0.5:
        question += f"{sample['target']} {sample['neg_target']}"
    else:
        question += f"{sample['neg_target']} {sample['target']}"
    question += f" [R] {sample['root']}"
    return question


def build_eval_prompt(
    sample: dict,
    n_latent: int,
    seed: int,
    *,
    choice_order: str = "random",
    shuffle_edges: bool = True,
) -> str:
    import random

    rng = random.Random(seed)
    prefix = build_question_prefix(
        sample, rng, choice_order=choice_order, shuffle_edges=shuffle_edges
    )
    n_latent = max(0, int(n_latent))
    return prefix + (" <|latent|>" * n_latent) + " [A] "


def bfs_distances(edges: List[List[int]], root: int, n_nodes: int) -> Dict[int, int]:
    adj: List[List[int]] = [[] for _ in range(n_nodes)]
    for u, v in edges:
        adj[u].append(v)
    dist = {root: 0}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def reasoning_hops(sample: dict) -> int:
    return len(sample.get("steps") or [])


def root_to_target_distance(sample: dict) -> int:
    n_nodes = len(sample.get("idx_to_symbol") or [])
    dist = bfs_distances(sample["edges"], sample["root"], n_nodes)
    target = sample["target"]
    return dist.get(target, reasoning_hops(sample))


def graph_diameter(sample: dict) -> int:
    n_nodes = len(sample.get("idx_to_symbol") or [])
    dist = bfs_distances(sample["edges"], sample["root"], n_nodes)
    return max(dist.values()) if dist else 0
