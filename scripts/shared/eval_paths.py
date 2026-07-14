"""Main-path evaluation and cross-slice rollup helpers."""
from __future__ import annotations

from collections import defaultdict


def eval_main_path(model, tokenizer, samples, *, device, seed, profile, struct_floor):
    from evaluate_coconut import expected_answer
    from run_adaptive_stop_experiment import predict_at_n

    correct = 0
    for idx, sample in enumerate(samples):
        n0 = struct_floor(sample)
        pred = predict_at_n(
            model, tokenizer, sample, n0, device,
            seed=seed + idx * 31, eval_profile=profile,
        )
        if pred == expected_answer(sample, profile):
            correct += 1
    total = len(samples)
    return {
        "accuracy": round(correct / total, 4) if total else 0.0,
        "correct": correct,
        "total": total,
        "mode": "main_path_only",
    }


def make_slice_row(meta: dict, samples: list, main_row: dict, policy_row: dict, *, policy_name: str) -> dict:
    delta_pp = round((policy_row["accuracy"] - main_row["accuracy"]) * 100, 2)
    return {
        "slice_id": meta.get("slice_id") or meta.get("id"),
        "label": meta.get("label"),
        "category": meta.get("category"),
        "construction": meta.get("construction"),
        "n_samples": len(samples),
        "main_acc": main_row["accuracy"],
        "policy_acc": policy_row["accuracy"],
        "transfer_acc": policy_row["accuracy"],
        "delta_pp": delta_pp,
        "fallback_rate": policy_row.get("fallback_rate"),
        "policy": policy_name,
        "transfer_helps": delta_pp > 0.5,
        "transfer_hurts": delta_pp < -0.5,
    }


def rollup_slice_rows(rows: list[dict], *, main_key: str = "main_acc") -> dict:
    if not rows:
        return {}
    deltas = [r["delta_pp"] for r in rows if r.get("delta_pp") is not None]
    helps = sum(1 for d in deltas if d > 0.5)
    hurts = sum(1 for d in deltas if d < -0.5)
    by_cat: dict[str, list[float]] = defaultdict(list)
    w_delta = w_n = w_ind_delta = w_ind_n = w_ood_delta = w_ood_n = 0.0
    for r in rows:
        by_cat[r.get("category") or "unknown"].append(r["delta_pp"])
        n = r.get("n_samples") or 1
        d = r.get("delta_pp") or 0.0
        w_delta += d * n
        w_n += n
        if (r.get(main_key) or 0) >= 0.85:
            w_ind_delta += d * n
            w_ind_n += n
        else:
            w_ood_delta += d * n
            w_ood_n += n
    return {
        "slice_count": len(rows),
        "helps_count": helps,
        "hurts_count": hurts,
        "transfer_helps_count": helps,
        "transfer_hurts_count": hurts,
        "neutral_count": len(rows) - helps - hurts,
        "transfer_neutral_count": len(rows) - helps - hurts,
        "mean_delta_pp": round(sum(deltas) / len(deltas), 3) if deltas else None,
        "weighted_mean_delta_pp": round(w_delta / w_n, 3) if w_n else None,
        "mean_main_acc": round(sum(r[main_key] for r in rows) / len(rows), 4) if rows else None,
        "in_dist_weighted_delta_pp": round(w_ind_delta / w_ind_n, 3) if w_ind_n else None,
        "ood_weighted_delta_pp": round(w_ood_delta / w_ood_n, 3) if w_ood_n else None,
        "by_category_mean_delta_pp": {
            k: round(sum(v) / len(v), 3) for k, v in sorted(by_cat.items())
        },
    }
