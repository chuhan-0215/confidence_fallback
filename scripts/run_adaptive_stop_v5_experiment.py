#!/usr/bin/env python3
"""实验十五：Coconut 联合微调 + Rich stop head（is_correct + 平衡校准）。

相对实验十四：
  - 解冻 Coconut 最后 N 层，与 stop head 联合训练
  - 标签仍为 is_correct，阈值仍用 balanced 校准
  - 评估 joint_correctness_stop 与 joint_stable_correctness_stop

可行判定：test 上 best 学习策略 acc ≥ fixed_3 且 timing ≥ 50%
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "model"))

from evaluate_coconut import expected_answer, load_coconut_model, load_dataset, resolve_device  # noqa: E402
from eval_profile import parse_eval_profile  # noqa: E402
from graph_utils import build_eval_prompt  # noqa: E402
from run_adaptive_stop_experiment import (  # noqa: E402
    evaluate_oracle_first_correct,
    evaluate_stable_stop,
    predict_at_n,
)
from run_auto_submit_experiment import evaluate_policy, make_policies, write_status  # noqa: E402
from stop_head import (  # noqa: E402
    calibrate_rich_threshold,
    evaluate_rich_stop,
    evaluate_streak_gated_stop,
    split_dataset,
    split_train_val_samples,
    train_joint_rich_stop_head,
)


def run_adaptive_stop_v5_experiment(
    dataset_path: Path,
    *,
    max_samples: Optional[int] = None,
    cap: int = 8,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    joint_epochs: int = 3,
    unfreeze_layers: int = 2,
    stop_min_n: int = 2,
    device_name: str = "cpu",
    progress_cb=None,
    status_file: Optional[Path] = None,
    stop_head_path: Optional[Path] = None,
    coconut_path: Optional[Path] = None,
) -> dict:
    from run_experiment import ensure_checkpoint  # noqa: E402

    config_path = ROOT / "configs" / "symbol-2layer-8head-768dim.json"
    checkpoint = ensure_checkpoint(ROOT / "checkpoints" / "checkpoint_300")
    device = resolve_device(device_name)
    profile = parse_eval_profile(None)

    dataset = load_dataset(dataset_path, max_samples=max_samples)
    train_set, test_set = split_dataset(dataset, train_ratio=train_ratio)
    train_sub, val_sub = split_train_val_samples(train_set, val_ratio=val_ratio, seed=43)
    policies = make_policies(cap=cap)
    stop_head_path = stop_head_path or (ROOT / "results" / "stop_head_v5.pt")
    coconut_path = coconut_path or (ROOT / "results" / "coconut_joint_v5.pt")

    total_phases = 6
    t0 = time.time()

    def phase_status(phase: str, done: int, label: str, **extra):
        write_status(
            status_file,
            {
                "running": True,
                "phase": phase,
                "params": {"experiment_type": "adaptive_stop_v5_compare"},
                "progress": {"done": done, "total": total_phases, "label": label},
                **extra,
            },
        )

    phase_status("loading_model", 0, "加载 Coconut 模型")
    model, tokenizer = load_coconut_model(checkpoint, config_path, device)

    phase_status("joint_training", 1, f"联合微调 · epoch 0/{joint_epochs}")

    def joint_cb(epoch, total_epochs, sample_done, sample_total):
        if progress_cb:
            progress_cb("joint_train", sample_done, sample_total)
        phase_status(
            "joint_training",
            1,
            f"联合微调 · epoch {epoch}/{total_epochs} · 样本 {sample_done}/{sample_total}",
        )

    head, train_metrics = train_joint_rich_stop_head(
        model,
        tokenizer,
        train_sub,
        val_sub,
        cap=cap,
        device=device,
        seed=42,
        predict_fn=predict_at_n,
        expected_fn=expected_answer,
        build_prompt_fn=build_eval_prompt,
        eval_profile=profile,
        unfreeze_layers=unfreeze_layers,
        epochs=joint_epochs,
        progress_cb=joint_cb,
    )

    phase_status("calibrating_threshold", 2, "val 平衡阈值校准")
    calibrated_threshold, calibration = calibrate_rich_threshold(
        head,
        model,
        tokenizer,
        val_sub,
        cap=cap,
        min_n=stop_min_n,
        device=device,
        seed=77,
        predict_fn=predict_at_n,
        expected_fn=expected_answer,
        build_prompt_fn=build_eval_prompt,
        eval_profile=profile,
        optimize="balanced",
    )

    stop_head_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "train_metrics": train_metrics,
            "calibration": calibration,
            "label_mode": "is_correct",
            "calibrated_threshold": calibrated_threshold,
            "joint_finetune": True,
            "unfreeze_layers": unfreeze_layers,
        },
        stop_head_path,
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "unfreeze_layers": unfreeze_layers,
            "base_checkpoint": str(checkpoint),
        },
        coconut_path,
    )

    rows: List[dict] = []
    eval_strategies = [
        ("joint_correctness_stop", "correct"),
        ("joint_stable_correctness_stop", "stable_correct"),
        ("stable_stop", "stable"),
        ("fixed_3", "baseline"),
        ("auto_route", "baseline"),
        ("oracle_first_correct", "oracle"),
    ]

    for step_idx, (sid, kind) in enumerate(eval_strategies):
        phase_status("evaluating", 3 + step_idx, f"test · {sid}", partial_strategies=rows)
        print(f"[exp15] test {sid} …", flush=True)

        def sample_cb(done, total, _sid=sid):
            if progress_cb:
                progress_cb(_sid, done, total)

        if kind == "correct":
            row = evaluate_rich_stop(
                head,
                model,
                tokenizer,
                test_set,
                cap=cap,
                min_n=stop_min_n,
                threshold=calibrated_threshold,
                device=device,
                seed=99,
                predict_fn=predict_at_n,
                expected_fn=expected_answer,
                build_prompt_fn=build_eval_prompt,
                eval_profile=profile,
                progress_cb=sample_cb,
            )
            row["strategy_label"] = (
                f"joint_correctness_stop · 联合微调 · thr={calibrated_threshold:.2f}"
            )
            row["params"]["label_mode"] = "is_correct"
            row["params"]["joint_finetune"] = True
        elif kind == "stable_correct":
            row = evaluate_streak_gated_stop(
                head,
                model,
                tokenizer,
                test_set,
                cap=cap,
                min_n=stop_min_n,
                threshold=calibrated_threshold,
                device=device,
                seed=99,
                predict_fn=predict_at_n,
                expected_fn=expected_answer,
                build_prompt_fn=build_eval_prompt,
                eval_profile=profile,
                progress_cb=sample_cb,
            )
            row["strategy_label"] = "joint_stable_correctness_stop · 联合微调 ∧ 稳定"
            row["params"]["joint_finetune"] = True
        elif kind == "stable":
            row = evaluate_stable_stop(
                model,
                tokenizer,
                test_set,
                device,
                cap=cap,
                progress_cb=sample_cb,
                eval_profile=profile,
            )
            row["strategy_label"] = "stable_stop"
        elif kind == "oracle":
            row = evaluate_oracle_first_correct(
                model,
                tokenizer,
                test_set,
                device,
                cap=cap,
                progress_cb=sample_cb,
                eval_profile=profile,
            )
            row["strategy_label"] = "oracle_first_correct"
        else:
            row = evaluate_policy(
                model,
                tokenizer,
                test_set,
                policies[sid],
                device,
                cap=cap,
                progress_cb=sample_cb,
            )
            row["strategy_label"] = sid

        row["strategy_id"] = sid
        row["eval_split"] = "test"
        rows.append(row)
        print(
            f"  acc={row['accuracy']*100:.1f}% timing={row.get('stop_timing_acc')}",
            flush=True,
        )

    by_id = {r["strategy_id"]: r for r in rows}
    joint = by_id.get("joint_correctness_stop")
    joint_stable = by_id.get("joint_stable_correctness_stop")
    fixed3 = by_id.get("fixed_3")
    auto = by_id.get("auto_route")
    oracle = by_id.get("oracle_first_correct")
    stable = by_id.get("stable_stop")

    best_learned = joint
    if joint_stable and joint:
        j_score = joint["accuracy"] * 5 + (joint.get("stop_timing_acc") or 0) * 3
        s_score = joint_stable["accuracy"] * 5 + (joint_stable.get("stop_timing_acc") or 0) * 3
        if s_score > j_score + 0.02:
            best_learned = joint_stable

    timing = best_learned.get("stop_timing_acc") if best_learned else None
    feasible = bool(
        best_learned
        and fixed3
        and best_learned["accuracy"] >= fixed3["accuracy"] - 0.005
        and timing is not None
        and timing >= 0.5
    )

    insights = [
        "实验十四：correctness 88.7% / timing 31.1%，acc 已超 fixed_3 但 timing 未达标。",
        f"实验十五 joint {joint['accuracy']*100:.1f}% · stable∧joint {joint_stable['accuracy']*100:.1f}%"
        f" vs fixed_3 {fixed3['accuracy']*100:.1f}% · timing {((timing or 0)*100):.1f}%"
        + (" → **可行**" if feasible else " → **仍不可行**"),
        f"解冻最后 {unfreeze_layers} 层 · 联合 epoch={train_metrics.get('epochs_ran')} · thr={calibrated_threshold:.2f}。",
    ]
    if not feasible:
        insights.append("若仍不达标：可试更多解冻层、answer-aware 辅助 loss，或 RL 式停步策略。")

    summary = {
        "trainable_stop_feasible": feasible,
        "best_learned_strategy": best_learned["strategy_id"] if best_learned else None,
        "train_size": len(train_set),
        "val_size": len(val_sub),
        "test_size": len(test_set),
        "calibrated_threshold": calibrated_threshold,
        "calibration": calibration,
        "joint_train_metrics": train_metrics,
        "unfreeze_layers": unfreeze_layers,
        "joint_correctness_stop_accuracy": joint["accuracy"] if joint else None,
        "joint_correctness_stop_timing_acc": joint.get("stop_timing_acc") if joint else None,
        "joint_stable_correctness_stop_accuracy": joint_stable["accuracy"] if joint_stable else None,
        "joint_stable_correctness_stop_timing_acc": joint_stable.get("stop_timing_acc") if joint_stable else None,
        "stable_stop_accuracy": stable["accuracy"] if stable else None,
        "fixed_3_accuracy": fixed3["accuracy"] if fixed3 else None,
        "auto_route_accuracy": auto["accuracy"] if auto else None,
        "oracle_first_correct_accuracy": oracle["accuracy"] if oracle else None,
        "prior_exp14_correctness_stop_acc": 0.8869,
        "prior_exp14_correctness_stop_timing": 0.311,
    }

    payload = {
        "ok": True,
        "experiment_type": "adaptive_stop_v5_compare",
        "experiment_id": 15,
        "title": "实验十五 · Coconut 联合微调 stop head",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_sec": round(time.time() - t0, 2),
        "dataset": str(dataset_path.relative_to(ROOT)),
        "sample_count": len(dataset),
        "cap": cap,
        "device": device_name,
        "method_note": (
            f"解冻 Coconut 最后 {unfreeze_layers} 层，与 RichStopHead 联合训练（is_correct）；"
            "阈值 balanced 校准。"
        ),
        "summary": summary,
        "insights": insights,
        "strategies": rows,
        "stop_head_checkpoint": str(stop_head_path.relative_to(ROOT)),
        "coconut_joint_checkpoint": str(coconut_path.relative_to(ROOT)),
    }

    write_status(
        status_file,
        {
            "running": False,
            "phase": "done",
            "params": {"experiment_type": "adaptive_stop_v5_compare"},
            "progress": {"done": total_phases, "total": total_phases, "label": "完成"},
            "summary": summary,
            "error": None,
        },
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="实验十五")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "prosqa_test_graph_4_coconut.json")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--joint-epochs", type=int, default=3)
    parser.add_argument("--unfreeze-layers", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "adaptive_stop_v5_latest.json")
    parser.add_argument("--status-file", type=Path, default=ROOT / "results" / "adaptive_stop_v5_status.json")
    parser.add_argument("--stop-head-out", type=Path, default=ROOT / "results" / "stop_head_v5.pt")
    parser.add_argument("--coconut-out", type=Path, default=ROOT / "results" / "coconut_joint_v5.pt")
    args = parser.parse_args()

    payload = run_adaptive_stop_v5_experiment(
        args.dataset,
        max_samples=args.max_samples if args.max_samples > 0 else None,
        joint_epochs=args.joint_epochs,
        unfreeze_layers=args.unfreeze_layers,
        device_name=args.device,
        status_file=args.status_file,
        stop_head_path=args.stop_head_out,
        coconut_path=args.coconut_out,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
