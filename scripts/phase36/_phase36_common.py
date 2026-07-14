"""Phase 36 · Dual Deploy Lock（W2 赢家全量验证 + category 路由）。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase25"))
sys.path.insert(0, str(ROOT / "scripts" / "phase34"))
sys.path.insert(0, str(ROOT / "scripts" / "phase35"))

from _phase34_common import (  # noqa: E402
    HURT_SLICE_IDS,
    TRANSFER_THR,
    eval_agreement_lock,
    eval_tri_zone,
    m2_head_ready,
    unique_slice_ids,
)
from _phase35_common import eval_agreement_tri_zone  # noqa: E402
from shared.eval_paths import eval_main_path, make_slice_row, rollup_slice_rows  # noqa: E402

PHASE36_OUT = ROOT / "results" / "phase36"
P34_TRI_ZONE = (0.40, 0.48)
W2_COMBO_BEST = (0.32, 0.48)
AGREEMENT_CATEGORIES = frozenset({"standard", "pattern"})
RESCUE_CATEGORIES = frozenset({"deep", "variant"})


def write_phase36_result(eid: str, payload: dict) -> Path:
    from shared.phase_io import write_phase_result
    return write_phase_result(36, eid, payload)


def load_w2_best() -> tuple[float, float]:
    import json
    path = ROOT / "outbox/results/from_a800/phase35/w2_combo_grid_sweep_latest.json"
    if not path.is_file():
        path = ROOT / "results/phase35/w2_combo_grid_sweep_latest.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        top = (data.get("top5_cross") or [{}])[0]
        if top.get("t_low") is not None:
            return float(top["t_low"]), float(top["t_mid"])
    return W2_COMBO_BEST


def eval_category_router(
    head, model, tokenizer, samples, *, device, seed, profile,
    struct_floor, knn_floor, knn_thr, pfn, meta: dict,
    rescue_params: tuple[float, float] = W2_COMBO_BEST,
    safe_params: tuple[float, float] = P34_TRI_ZONE,
):
    """可部署路由：deep/variant → combo(W2)；diamond → agreement；其余 → tri_zone。"""
    category = (meta.get("category") or "").lower()
    construction = (meta.get("construction") or "").lower()
    if category in RESCUE_CATEGORIES:
        return eval_agreement_tri_zone(
            head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            t_low=rescue_params[0], t_mid=rescue_params[1], hop4_only=False,
        )
    if construction == "diamond" or category in AGREEMENT_CATEGORIES:
        row = eval_agreement_lock(
            head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            fallback_thr=TRANSFER_THR, hop4_only=False,
        )
        row["params"]["router"] = "agreement_lock"
        return row
    return eval_tri_zone(
        head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
        struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
        t_low=safe_params[0], t_mid=safe_params[1], hop4_only=False,
    )
