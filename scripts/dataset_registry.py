#!/usr/bin/env python3
"""ProsQA 多切片数据集注册表 — 从同一 JSON 派生多种对比子集。"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional

from graph_utils import graph_diameter, reasoning_hops, root_to_target_distance

ROOT = Path(__file__).resolve().parent.parent
MASTER_PATH = ROOT / "data" / "prosqa_test_graph_4_coconut.json"

from eval_profile import parse_eval_profile, profile_label  # noqa: E402

FilterFn = Callable[[dict], bool]

_FILTERS: Dict[str, FilterFn] = {
    "hops_3": lambda s: reasoning_hops(s) == 3,
    "hops_4": lambda s: reasoning_hops(s) == 4,
    "diameter_3": lambda s: graph_diameter(s) == 3,
    "diameter_4": lambda s: graph_diameter(s) == 4,
    "diameter_5": lambda s: graph_diameter(s) == 5,
    "diameter_6": lambda s: graph_diameter(s) == 6,
    "diameter_wide": lambda s: graph_diameter(s) >= 5,
    "distance_3": lambda s: root_to_target_distance(s) == 3,
    "distance_4": lambda s: root_to_target_distance(s) == 4,
    "hop3_diam3": lambda s: reasoning_hops(s) == 3 and graph_diameter(s) == 3,
    "hop3_diam4": lambda s: reasoning_hops(s) == 3 and graph_diameter(s) == 4,
    "hop4_diam4": lambda s: reasoning_hops(s) == 4 and graph_diameter(s) == 4,
    "hop4_diam5": lambda s: reasoning_hops(s) == 4 and graph_diameter(s) >= 5,
}

_MIX_SEED = 2026
_MIX_CAP = 80


def _mixed_hop_ratio(master: List[dict], ratio_4hop: float, cap: int = _MIX_CAP) -> List[dict]:
    """按 4 跳占比混合 3/4 跳样本（固定 seed，可复现）。"""
    hop3 = [s for s in master if reasoning_hops(s) == 3]
    hop4 = [s for s in master if reasoning_hops(s) == 4]
    n = min(cap, len(hop3) + len(hop4))
    n4 = min(len(hop4), max(0, round(n * ratio_4hop)))
    n3 = min(len(hop3), n - n4)
    if n3 + n4 < n:
        n4 = min(len(hop4), n - n3)
    rng = random.Random(_MIX_SEED + int(ratio_4hop * 100))
    picked = rng.sample(hop3, n3) + rng.sample(hop4, n4)
    rng.shuffle(picked)
    return picked


from dataset_slice_specs import (  # noqa: E402
    ALIEN_SLICE_IDS,
    ALIEN_SLICE_SPECS,
    BOUNDARY_PUSH_DEEP_SLICE_IDS,
    BOUNDARY_PUSH_DEEP_SLICE_SPECS,
    BOUNDARY_PUSH_SLICE_IDS,
    BOUNDARY_PUSH_SLICE_SPECS,
    DEEP_SLICE_IDS,
    DEEP_SLICE_SPECS,
    PATTERN_SLICE_IDS,
    PATTERN_SLICE_SPECS,
    SLICE_SPECS,
    VARIANT_SLICE_IDS,
    VARIANT_SLICE_SPECS,
)

_ALL_SPECS: Optional[List[dict]] = None


def all_slice_specs() -> List[dict]:
    global _ALL_SPECS
    if _ALL_SPECS is None:
        raw = (
            SLICE_SPECS
            + DEEP_SLICE_SPECS
            + VARIANT_SLICE_SPECS
            + BOUNDARY_PUSH_SLICE_SPECS
            + BOUNDARY_PUSH_DEEP_SLICE_SPECS
            + PATTERN_SLICE_SPECS
            + ALIEN_SLICE_SPECS
        )
        seen: set[str] = set()
        deduped: List[dict] = []
        for spec in raw:
            if spec["id"] in seen:
                continue
            seen.add(spec["id"])
            deduped.append(spec)
        _ALL_SPECS = deduped
    return _ALL_SPECS


_RANDOM_CACHE: Dict[str, List[int]] = {}


def load_master() -> List[dict]:
    return json.loads(MASTER_PATH.read_text(encoding="utf-8"))


def _spec_by_id(slice_id: str) -> dict:
    spec = next((s for s in all_slice_specs() if s["id"] == slice_id), None)
    if spec is None:
        raise ValueError(f"unknown slice_id: {slice_id}")
    return spec


def _load_source_rows(spec: dict) -> List[dict]:
    rel = spec.get("source")
    if rel:
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(f"slice data missing: {path}")
        return json.loads(path.read_text(encoding="utf-8"))
    return _filter_rows(spec, load_master())


def list_slices(
    deep: bool = False,
    variant: bool = False,
    pattern: bool = False,
    push: bool = False,
    push_deep: bool = False,
    alien: bool = False,
) -> List[dict]:
    if alien:
        specs = ALIEN_SLICE_SPECS
    elif push_deep:
        specs = BOUNDARY_PUSH_DEEP_SLICE_SPECS
    elif pattern:
        specs = PATTERN_SLICE_SPECS
    elif push:
        specs = BOUNDARY_PUSH_SLICE_SPECS
    elif variant:
        specs = VARIANT_SLICE_SPECS
    elif deep:
        specs = DEEP_SLICE_SPECS
    else:
        specs = SLICE_SPECS
    master = load_master()
    out = []
    for spec in specs:
        if spec.get("source"):
            rows = _load_source_rows(spec)
            available = len(rows)
        else:
            rows = _filter_rows(spec, master)
            available = len(rows)
        profile = parse_eval_profile(spec.get("eval_profile"))
        out.append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "description": spec["description"],
                "available_count": available,
                "default_cap": spec.get("default_cap"),
                "category": (
                    "alien"
                    if spec["id"] in ALIEN_SLICE_IDS
                    else (
                    "push_deep"
                    if spec["id"] in BOUNDARY_PUSH_DEEP_SLICE_IDS
                    else (
                    "pattern"
                    if spec["id"] in PATTERN_SLICE_IDS
                    else (
                        "push"
                        if spec["id"] in BOUNDARY_PUSH_SLICE_IDS
                        else (
                            "variant"
                            if spec["id"] in VARIANT_SLICE_IDS
                            else ("deep" if spec["id"] in DEEP_SLICE_IDS else "standard")
                        )
                    )
                    )
                    )
                ),
                "construction": spec.get("construction"),
                "eval_profile": profile.to_dict(),
                "supervision_label": profile_label(profile),
            }
        )
    return out


def list_variant_slices() -> List[dict]:
    return list_slices(variant=True)


def list_deep_slices() -> List[dict]:
    return list_slices(deep=True)


def _random_indices(key: str, n: int, cap: int) -> List[int]:
    if key not in _RANDOM_CACHE:
        rng = random.Random(42 if key == "random_a" else 7)
        _RANDOM_CACHE[key] = sorted(rng.sample(range(n), min(cap, n)))
    return _RANDOM_CACHE[key]


def list_pattern_slices() -> List[dict]:
    return list_slices(pattern=True)


def _filter_rows(spec: dict, master: List[dict]) -> List[dict]:
    fid = spec["filter_name"]
    if fid is None:
        return list(master)
    mix_ratios = {
        "mix_0_4": 0.0,
        "mix_25_4": 0.25,
        "mix_50_4": 0.5,
        "mix_75_4": 0.75,
        "mix_100_4": 1.0,
    }
    if fid in mix_ratios:
        cap = spec.get("default_cap") or _MIX_CAP
        return _mixed_hop_ratio(master, mix_ratios[fid], cap)
    if fid in _FILTERS:
        return [s for s in master if _FILTERS[fid](s)]
    if fid == "random_a":
        idx = _random_indices("random_a", len(master), spec.get("default_cap") or 80)
        return [master[i] for i in idx]
    if fid == "random_b":
        idx = _random_indices("random_b", len(master), spec.get("default_cap") or 80)
        return [master[i] for i in idx]
    if fid == "first_half":
        half = len(master) // 2
        return master[:half]
    if fid == "second_half":
        half = len(master) // 2
        return master[half:]
    raise ValueError(f"unknown filter: {fid}")


def load_slice(
    slice_id: str,
    max_samples: Optional[int] = None,
    master: Optional[List[dict]] = None,
) -> tuple[dict, List[dict]]:
    spec = _spec_by_id(slice_id)

    if spec.get("source"):
        rows = list(_load_source_rows(spec))
        source_label = spec["source"]
    else:
        if master is None:
            master = load_master()
        rows = _filter_rows(spec, master)
        source_label = str(MASTER_PATH.relative_to(ROOT))

    cap = max_samples
    if cap is None:
        cap = spec.get("default_cap")
    if cap is not None and len(rows) > cap:
        rows = rows[:cap]

    out_rows = []
    for i, row in enumerate(rows):
        item = dict(row)
        item["_idx"] = i
        out_rows.append(item)

    if spec.get("source"):
        available = len(_load_source_rows(spec))
    else:
        available = len(_filter_rows(spec, master or load_master()))

    profile = parse_eval_profile(spec.get("eval_profile"))
    meta = {
        "slice_id": slice_id,
        "label": spec["label"],
        "description": spec["description"],
        "source": source_label,
        "available_in_slice": available,
        "used_count": len(out_rows),
        "category": (
            "alien"
            if spec["id"] in ALIEN_SLICE_IDS
            else (
            "push_deep"
            if spec["id"] in BOUNDARY_PUSH_DEEP_SLICE_IDS
            else (
            "pattern"
            if spec["id"] in PATTERN_SLICE_IDS
            else (
                "push"
                if spec["id"] in BOUNDARY_PUSH_SLICE_IDS
                else (
                    "variant"
                    if spec["id"] in VARIANT_SLICE_IDS
                    else ("deep" if spec["id"] in DEEP_SLICE_IDS else "standard")
                )
            )
            )
            )
        ),
        "construction": spec.get("construction"),
        "eval_profile": profile.to_dict(),
        "supervision_label": profile_label(profile),
        "pattern_axis": spec.get("pattern_axis"),
        "ratio_4hop": spec.get("ratio_4hop"),
    }
    return meta, out_rows


def default_variant_compare_slice_ids() -> List[str]:
    return [s["id"] for s in VARIANT_SLICE_SPECS]


def default_compare_slice_ids() -> List[str]:
    return [s["id"] for s in SLICE_SPECS]


def default_pattern_compare_slice_ids() -> List[str]:
    return [s["id"] for s in PATTERN_SLICE_SPECS]


def default_deep_compare_slice_ids() -> List[str]:
    return [s["id"] for s in DEEP_SLICE_SPECS]


def default_boundary_push_slice_ids() -> List[str]:
    return [s["id"] for s in BOUNDARY_PUSH_SLICE_SPECS]


def default_boundary_push_deep_slice_ids() -> List[str]:
    return [s["id"] for s in BOUNDARY_PUSH_DEEP_SLICE_SPECS]


def list_alien_slices() -> List[dict]:
    return list_slices(alien=True)


def default_alien_slice_ids() -> List[str]:
    return [s["id"] for s in ALIEN_SLICE_SPECS]


def list_boundary_push_slices() -> List[dict]:
    return list_slices(push=True)
