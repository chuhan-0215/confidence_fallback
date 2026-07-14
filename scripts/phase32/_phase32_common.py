"""Phase 32 · 通解跨数据集迁移验证（confidence_fallback transfer）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase25"))
sys.path.insert(0, str(ROOT / "scripts" / "phase23"))
sys.path.insert(0, str(ROOT / "model"))

from phase23._phase23_common import M2_HEAD  # noqa: E402
from phase4._phase4_common import utc_now  # noqa: E402
from shared.eval_paths import eval_main_path, rollup_slice_rows  # noqa: E402
from shared.predict_fn import make_predict_fn  # noqa: E402
from shared.predict_fn import make_predict_fn  # noqa: E402

PHASE32_OUT = ROOT / "results" / "phase32"
TRANSFER_THR = 0.48


def write_phase32_result(eid: str, payload: dict) -> Path:
    from shared.phase_io import write_phase_result
    return write_phase_result(32, eid, payload)


def write_status(status: dict) -> None:
    path = PHASE32_OUT / "cross_dataset_status.json"
    status["updated_at"] = utc_now()
    path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def m2_head_ready() -> bool:
    return M2_HEAD.is_file()


def slice_ids_for_tier(tier: str) -> list[str]:
    from dataset_registry import all_slice_specs

    specs = all_slice_specs()
    if tier == "smoke":
        return ["full", "hops_3", "hops_4", "v_chain_5_dense", "push_ext5_from3"]
    if tier == "core":
        return [
            "full", "hops_3", "hops_4",
            "diameter_4", "diameter_wide",
            "syn_chain_5", "syn_chain_6", "syn_mixed_56",
            "v_chain_3", "v_chain_4", "v_chain_6_dense", "v_extend_5", "v_extend_6",
            "v_tree_5", "v_diamond_5", "v_chain_6_symbol",
            "push_ext5_from3", "push_ext6_from4", "push_ext7_from3",
            "mix_50_4", "hop3_diam3", "hop4_diam4", "hop4_diam5",
        ]
    return [s["id"] for s in specs]
