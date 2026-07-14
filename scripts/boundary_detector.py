"""Detect efficiency cliff / boundary from sweep curves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass
class BoundaryResult:
    boundary_x: Optional[float]
    method: str
    confidence: float
    note: str


def _as_floats(xs: Sequence, ys: Sequence) -> Tuple[List[float], List[float]]:
    return [float(x) for x in xs], [float(y) for y in ys]


def detect_plateau_drop(
    xs: Sequence,
    efficiencies: Sequence,
    drop_threshold: float = 0.015,
    plateau_ratio: float = 0.35,
) -> BoundaryResult:
    """Find x where efficiency drops sharply, then changes little afterward.

    Matches the paper intuition: after enough continuous-thought steps (≈ graph
    diameter D), accuracy saturates while compute keeps growing, so
    efficiency = accuracy / steps falls and then plateaus.
    """
    x_vals, y_vals = _as_floats(xs, efficiencies)
    if len(x_vals) < 3:
        return BoundaryResult(None, "plateau_drop", 0.0, "not enough points")

    drops = [y_vals[i - 1] - y_vals[i] for i in range(1, len(y_vals))]
    max_drop = max(drops) if drops else 0.0
    if max_drop <= 0:
        return BoundaryResult(None, "plateau_drop", 0.0, "efficiency never decreases")

    abs_threshold = max(drop_threshold, max_drop * 0.25)

    for i, drop in enumerate(drops):
        if drop < abs_threshold:
            continue
        tail = drops[i + 1 :]
        if not tail:
            return BoundaryResult(
                x_vals[i + 1],
                "plateau_drop",
                min(1.0, drop / max_drop),
                "last point is cliff",
            )
        tail_mean = sum(abs(d) for d in tail) / len(tail)
        if tail_mean <= max(abs_threshold * plateau_ratio, drop * plateau_ratio):
            return BoundaryResult(
                x_vals[i + 1],
                "plateau_drop",
                min(1.0, drop / max_drop),
                "sharp drop followed by plateau",
            )

    # Kneedle fallback on decreasing segment
    knee = detect_kneedle(x_vals, y_vals)
    if knee.boundary_x is not None:
        return knee

    max_i = drops.index(max_drop)
    return BoundaryResult(
        x_vals[max_i + 1],
        "max_drop",
        0.5,
        "largest single-step efficiency drop",
    )


def detect_kneedle(xs: Sequence, ys: Sequence) -> BoundaryResult:
    """Kneedle-style elbow on (x, efficiency)."""
    x_vals, y_vals = _as_floats(xs, ys)
    if len(x_vals) < 3:
        return BoundaryResult(None, "kneedle", 0.0, "not enough points")

    x0, y0 = x_vals[0], y_vals[0]
    x1, y1 = x_vals[-1], y_vals[-1]
    span = (x1 - x0) or 1.0

    best_idx = 0
    best_dist = -1.0
    for i, (x, y) in enumerate(zip(x_vals, y_vals)):
        # distance from point to line through endpoints (normalized by x span)
        nx = (x - x0) / span
        ny = y - y0
        # line direction
        dx = 1.0
        dy = (y1 - y0) / span if span else 0.0
        # perpendicular distance
        dist = abs(dy * nx - dx * ny) / ((dx * dx + dy * dy) ** 0.5 or 1.0)
        if dist > best_dist:
            best_dist = dist
            best_idx = i

    if best_idx == 0 or best_idx == len(x_vals) - 1:
        return BoundaryResult(None, "kneedle", 0.0, "elbow at endpoint")

    return BoundaryResult(
        x_vals[best_idx],
        "kneedle",
        min(1.0, best_dist / (max(y_vals) - min(y_vals) + 1e-9)),
        "maximum curvature elbow",
    )


def detect_max_accuracy(
    xs: Sequence,
    accuracies: Sequence,
) -> BoundaryResult:
    """Pick latent steps where accuracy peaks; tie-break toward fewer steps."""
    x_vals, y_vals = _as_floats(xs, accuracies)
    if not x_vals:
        return BoundaryResult(None, "max_accuracy", 0.0, "not enough points")

    best_acc = max(y_vals)
    candidates = [x for x, y in zip(x_vals, y_vals) if y >= best_acc - 1e-9]
    boundary = min(candidates)
    return BoundaryResult(
        boundary,
        "max_accuracy",
        1.0,
        f"accuracy peak {best_acc:.4f} at {int(boundary)} step(s)",
    )


def aggregate_boundary(points: Iterable[BoundaryResult]) -> Optional[float]:
    vals = [p.boundary_x for p in points if p.boundary_x is not None]
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]
