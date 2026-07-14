"""Phase 44 · 置信度通解零样本外推 + 适配改进。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

PHASE44_OUT = ROOT / "results" / "phase44"
PHASE32_T1 = ROOT / "results" / "phase32" / "t1_cross_dataset_transfer_latest.json"
TRANSFER_THR = 0.48

# 与 ProsQA 原题差异最大：纯合成链 / 宽链 / symbol 监督
OOD_PROXY_SLICES = [
    "syn_chain_5",
    "syn_chain_6",
    "syn_chain_5_wide",
    "syn_chain_6_wide",
    "syn_mixed_56",
    "v_chain_6_symbol",
    "v_diamond_5",
    "hops_3",
]

TAU_SWEEP = [0.35, 0.40, 0.44, 0.48, 0.52, 0.55, 0.60, 0.99]  # 0.99 ≈ 永不回退


def write_phase44_result(eid: str, payload: dict) -> Path:
    from datetime import datetime, timezone
    PHASE44_OUT.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    payload = {**payload, "finished_at": payload.get("finished_at") or ts}
    for name in (f"{eid}_latest.json", f"{eid}.json"):
        p = PHASE44_OUT / name
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return PHASE44_OUT / f"{eid}_latest.json"


def load_phase32_t1() -> dict | None:
    if not PHASE32_T1.is_file():
        return None
    return json.loads(PHASE32_T1.read_text(encoding="utf-8"))


def slice_rows_by_ids(t1: dict, slice_ids: list[str]) -> list[dict]:
    by_id = {r["slice_id"]: r for r in t1.get("slices") or []}
    return [by_id[sid] for sid in slice_ids if sid in by_id]


def rollup_transfer_rows(rows: list[dict]) -> dict:
    if not rows:
        return {}
    deltas = [r["delta_pp"] for r in rows]
    helps = sum(1 for d in deltas if d > 0.5)
    hurts = sum(1 for d in deltas if d < -0.5)
    return {
        "slice_count": len(rows),
        "transfer_helps_count": helps,
        "transfer_hurts_count": hurts,
        "mean_delta_pp": round(sum(deltas) / len(deltas), 3),
        "mean_main_acc": round(sum(r["main_acc"] for r in rows) / len(rows), 4),
        "mean_transfer_acc": round(sum(r["transfer_acc"] for r in rows) / len(rows), 4),
        "mean_fallback_rate": round(sum(r.get("fallback_rate") or 0 for r in rows) / len(rows), 4),
    }


__all__ = [
    "OOD_PROXY_SLICES",
    "PHASE32_T1",
    "PHASE44_OUT",
    "ROOT",
    "TAU_SWEEP",
    "TRANSFER_THR",
    "load_phase32_t1",
    "rollup_transfer_rows",
    "slice_rows_by_ids",
    "write_phase44_result",
]
