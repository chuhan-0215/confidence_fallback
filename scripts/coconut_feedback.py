"""Apply latent feedback configuration to a Coconut model instance."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def apply_feedback_config(model, cfg: Optional[Dict[str, Any]] = None) -> None:
    cfg = cfg or {}
    model.latent_feedback_scale = float(cfg.get("latent_feedback_scale", 1.0))
    schedule = cfg.get("latent_feedback_schedule")
    model.latent_feedback_schedule = list(schedule) if schedule else None
    model.latent_feedback_mode = str(cfg.get("latent_feedback_mode", "scale"))


def default_feedback_strategies() -> List[dict]:
    """Inference-only knobs for post-step-4 plateau experiments."""
    return [
        {
            "id": "baseline",
            "label": "baseline · α=1 恒定",
            "latent_feedback_scale": 1.0,
        },
        {
            "id": "fb075",
            "label": "全局 α=0.75",
            "latent_feedback_scale": 0.75,
        },
        {
            "id": "decay_after4",
            "label": "4 步后 scale 衰减",
            "latent_feedback_scale": 1.0,
            "latent_feedback_schedule": [1, 1, 1, 1, 0.5, 0.5, 0.25, 0.25],
        },
        {
            "id": "zero_after4",
            "label": "4 步后停写回 (scale=0)",
            "latent_feedback_scale": 1.0,
            "latent_feedback_schedule": [1, 1, 1, 1, 0, 0, 0, 0],
        },
        {
            "id": "residual_decay4",
            "label": "4 步后残差衰减",
            "latent_feedback_mode": "residual",
            "latent_feedback_schedule": [1, 1, 1, 1, 0.5, 0.5, 0.25, 0.25],
        },
        {
            "id": "residual_zero4",
            "label": "4 步后残差停写 (β=0)",
            "latent_feedback_mode": "residual",
            "latent_feedback_schedule": [1, 1, 1, 1, 0, 0, 0, 0],
        },
    ]


def post_step4_metrics(sweep: List[dict]) -> dict:
    acc = {int(r["n_latent"]): float(r["accuracy"]) for r in sweep}
    a4 = acc.get(4)
    post_vals = [acc[i] for i in range(5, 9) if i in acc]
    if a4 is None or not post_vals:
        return {
            "acc_at_4": a4,
            "post4_min": None,
            "post4_max": None,
            "post4_drop_pp": None,
            "post4_range_pp": None,
            "post4_non_decreasing": None,
        }
    post_min = min(post_vals)
    post_max = max(post_vals)
    return {
        "acc_at_4": round(a4, 4),
        "post4_min": round(post_min, 4),
        "post4_max": round(post_max, 4),
        "post4_drop_pp": round((post_min - a4) * 100, 1),
        "post4_range_pp": round((post_max - post_min) * 100, 1),
        "post4_non_decreasing": post_min >= a4 - 1e-9,
    }
