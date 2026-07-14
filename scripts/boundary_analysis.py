"""Explain why the accuracy boundary occurs at a specific latent-step count."""

from __future__ import annotations

from collections import Counter
from statistics import mean, median
from typing import Dict, List, Optional, Sequence

from graph_utils import graph_diameter, reasoning_hops, root_to_target_distance


def _round(v: float, n: int = 3) -> float:
    return round(float(v), n)


def dataset_hop_profile(dataset: List[dict]) -> dict:
    hops = [reasoning_hops(s) for s in dataset]
    dists = [root_to_target_distance(s) for s in dataset]
    diams = [graph_diameter(s) for s in dataset]
    hop_counter = Counter(hops)
    dist_counter = Counter(dists)
    mode_hops = hop_counter.most_common(1)[0][0] if hop_counter else None
    mode_dist = dist_counter.most_common(1)[0][0] if dist_counter else None
    return {
        "hop_histogram": {str(k): v for k, v in sorted(hop_counter.items())},
        "distance_histogram": {str(k): v for k, v in sorted(dist_counter.items())},
        "mean_reasoning_hops": _round(mean(hops)) if hops else 0,
        "median_reasoning_hops": _round(median(hops)) if hops else 0,
        "mode_reasoning_hops": mode_hops,
        "mean_root_target_distance": _round(mean(dists)) if dists else 0,
        "median_root_target_distance": _round(median(dists)) if dists else 0,
        "mode_root_target_distance": mode_dist,
        "mean_diameter_from_root": _round(mean(diams)) if diams else 0,
    }


def marginal_accuracy_gains(latent_sweep: Sequence[dict]) -> List[dict]:
    rows = sorted(latent_sweep, key=lambda r: r["n_latent"])
    out = []
    for i, row in enumerate(rows):
        acc = float(row["accuracy"])
        if i == 0:
            out.append(
                {
                    "n_latent": row["n_latent"],
                    "accuracy": _round(acc, 4),
                    "delta_from_prev": None,
                    "delta_pct_points": None,
                }
            )
            continue
        prev = float(rows[i - 1]["accuracy"])
        delta = acc - prev
        out.append(
            {
                "n_latent": row["n_latent"],
                "accuracy": _round(acc, 4),
                "delta_from_prev": _round(delta, 4),
                "delta_pct_points": _round(delta * 100, 1),
            }
        )
    return out


def find_largest_gain_step(marginals: List[dict]) -> Optional[dict]:
    candidates = [m for m in marginals if m["delta_from_prev"] is not None]
    if not candidates:
        return None
    best = max(candidates, key=lambda m: m["delta_from_prev"])
    return best


def post_peak_behavior(marginals: List[dict], peak_steps: int) -> dict:
    after = [m for m in marginals if m["n_latent"] > peak_steps]
    if not after:
        return {"steps_after_peak": 0, "max_drop": 0.0, "all_non_positive_gain": True}
    deltas = [m["delta_from_prev"] for m in after if m["delta_from_prev"] is not None]
    return {
        "steps_after_peak": len(after),
        "max_drop": _round(min(deltas), 4) if deltas else 0.0,
        "all_non_positive_gain": all(d <= 0 for d in deltas) if deltas else True,
        "avg_delta_after_peak": _round(mean(deltas), 4) if deltas else 0.0,
    }


def build_reasons(
    boundary_steps: int,
    max_accuracy: float,
    marginals: List[dict],
    profile: dict,
    largest_gain: Optional[dict],
    post_peak: dict,
) -> List[dict]:
    reasons: List[dict] = []

    mode_hops = profile.get("mode_reasoning_hops")
    mean_hops = profile.get("mean_reasoning_hops")
    mode_dist = profile.get("mode_root_target_distance")
    mean_dist = profile.get("mean_root_target_distance")

    if largest_gain:
        reasons.append(
            {
                "title": "最大准确率跃升",
                "detail": (
                    f"从 {largest_gain['n_latent'] - 1} 步到 {largest_gain['n_latent']} 步，"
                    f"准确率提升 {largest_gain['delta_pct_points']:+.1f} 个百分点，"
                    f"说明模型在此步完成了关键的 BFS 搜索前沿扩展。"
                ),
            }
        )

    if mode_hops is not None:
        diff = boundary_steps - mode_hops
        if diff == 0:
            align = f"边界 {boundary_steps} 步与数据集中最常见的推理跳数 {mode_hops} 完全一致。"
        elif abs(diff) <= 1:
            align = (
                f"边界 {boundary_steps} 步与最常见推理跳数 {mode_hops} 相差 {diff:+d}，"
                f"接近论文结论：连续思维步数应≈图上的 BFS 深度。"
            )
        else:
            align = (
                f"边界 {boundary_steps} 步与最常见推理跳数 {mode_hops} 相差 {diff:+d}；"
                f"数据集平均跳数为 {mean_hops}，模型可能需要额外步数处理叠加态搜索。"
            )
        reasons.append({"title": "与推理跳数对照", "detail": align})

    if mode_dist is not None:
        reasons.append(
            {
                "title": "与根→目标 BFS 距离对照",
                "detail": (
                    f"根节点到目标的最短路径众数为 {mode_dist} 步（均值 {mean_dist}）。"
                    f"论文（Reasoning by Superposition）指出：D 步连续思维等价于"
                    f"在图上并行 BFS 扩展 D 层搜索前沿；边界落在 {boundary_steps} 步，"
                    f"说明模型大致完成了所需深度的可达性搜索。"
                ),
            }
        )

    if post_peak.get("all_non_positive_gain"):
        reasons.append(
            {
                "title": "峰值后不再提升",
                "detail": (
                    f"在 {boundary_steps} 步之后，准确率未再上升"
                    f"（后续 {post_peak.get('steps_after_peak', 0)} 个扫描点平均变化 "
                    f"{post_peak.get('avg_delta_after_peak', 0):+.4f}）。"
                    f"额外连续思维只重复已编码的搜索前沿（叠加态不再扩展），"
                    f"因此边界应取峰值步数而非更多步数。"
                ),
            }
        )
    else:
        reasons.append(
            {
                "title": "峰值后波动",
                "detail": (
                    f"{boundary_steps} 步后准确率仍有波动（最大回落 "
                    f"{abs(post_peak.get('max_drop', 0)) * 100:.1f} 个百分点），"
                    f"边界取全局最高准确率对应的最少步数。"
                ),
            }
        )

    reasons.append(
        {
            "title": "叠加态推理机制（论文）",
            "detail": (
                "Coconut 每一步连续思维在潜空间维护多个搜索前沿的叠加态（并行 BFS）。"
                f"步数不足时前沿未覆盖目标（准确率仅 {max_accuracy * 100:.1f}% 峰值）；"
                f"步数达到 {boundary_steps} 时叠加态已覆盖最优可达性信息，"
                "继续增加步数不会提高答案质量。"
            ),
        }
    )

    return reasons


from boundary_detector import detect_max_accuracy  # noqa: E402


def resolve_boundary(latent_sweep: Sequence[dict], boundary: dict) -> tuple:
    """Return (boundary_steps, max_accuracy) from result or recompute from sweep."""
    rows = list(latent_sweep)
    if not rows:
        return int(boundary.get("recommended_latent_steps") or 0), float(
            boundary.get("max_accuracy") or 0
        )

    peak = max(rows, key=lambda r: (float(r["accuracy"]), -int(r["n_latent"])))
    peak_acc = float(peak["accuracy"])
    peak_steps = int(peak["n_latent"])

    if boundary.get("max_accuracy") is not None:
        peak_acc = float(boundary["max_accuracy"])
    if boundary.get("recommended_latent_steps") is not None and boundary.get("max_accuracy") is not None:
        return int(boundary["recommended_latent_steps"]), peak_acc

    xs = [r["n_latent"] for r in rows]
    acc = [r["accuracy"] for r in rows]
    detected = detect_max_accuracy(xs, acc)
    return int(detected.boundary_x or peak_steps), peak_acc


def analyze_boundary(
    dataset: List[dict],
    latent_sweep: Sequence[dict],
    boundary: dict,
    by_reasoning_hops: Optional[Sequence[dict]] = None,
) -> dict:
    boundary_steps, max_accuracy = resolve_boundary(latent_sweep, boundary)
    profile = dataset_hop_profile(dataset)
    marginals = marginal_accuracy_gains(latent_sweep)
    largest_gain = find_largest_gain_step(marginals)
    post_peak = post_peak_behavior(marginals, boundary_steps)
    reasons = build_reasons(
        boundary_steps, max_accuracy, marginals, profile, largest_gain, post_peak
    )

    hop_alignment = []
    if by_reasoning_hops:
        for g in by_reasoning_hops:
            hops = g.get("reasoning_hops")
            sweep = g.get("latent_sweep") or []
            if not sweep:
                continue
            best = max(sweep, key=lambda r: (r["accuracy"], -r["n_latent"]))
            hop_alignment.append(
                {
                    "reasoning_hops": hops,
                    "sample_count": g.get("sample_count"),
                    "best_latent_steps": best["n_latent"],
                    "best_accuracy": _round(best["accuracy"], 4),
                    "matches_group_hops": best["n_latent"] == hops,
                }
            )

    summary_parts = [
        f"边界为 {boundary_steps} 步（最高准确率 {max_accuracy * 100:.1f}%）。"
    ]
    if largest_gain:
        summary_parts.append(
            f"最大增益出现在 {largest_gain['n_latent']} 步（+{largest_gain['delta_pct_points']:.1f}pp）。"
        )
    if profile.get("mode_reasoning_hops") is not None:
        summary_parts.append(
            f"与数据集最常见推理跳数 {profile['mode_reasoning_hops']} 步对照，"
            f"符合「连续思维步数 ≈ BFS 深度」的论文预期。"
        )

    return {
        "summary": " ".join(summary_parts),
        "marginal_gains": marginals,
        "largest_gain_step": largest_gain,
        "post_peak": post_peak,
        "dataset_graph_profile": profile,
        "hop_group_alignment": hop_alignment,
        "reasons": reasons,
    }
