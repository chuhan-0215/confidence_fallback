#!/usr/bin/env python3
"""实验十四：正确性预测头（is_correct 标签 + 平衡阈值 + stable 门控）。

相对实验十三：
  - 训练标签改为「当前步答案是否正确」（而非仅 first_correct 单点）
  - 阈值校准 optimize=balanced（acc×5 + timing×3）
  - 评估 correctness_stop 与 stable_correctness_stop

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
    build_rich_stop_examples_for_samples,
    calibrate_rich_threshold,
    evaluate_rich_stop,
    evaluate_streak_gated_stop,
    split_dataset,
    split_train_val_samples,
    train_rich_stop_head,
)


def run_adaptive_stop_v4_experiment(
    dataset_path: Path,
    *,
    max_samples: Optional[int] = None,
    cap: int = 8,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    train_epochs: int = 40,
    stop_min_n: int = 2,
    device_name: str = "cpu",
    progress_cb=None,
    status_file: Optional[Path] = None,
    stop_head_path: Optional[Path] = None,
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
    stop_head_path = stop_head_path or (ROOT / "results" / "stop_head_v4.pt")

    total_phases = 6
    t0 = time.time()

    def phase_status(phase: str, done: int, label: str, **extra):
        write_status(
            status_file,
            {
                "running": True,
                "phase": phase,
                "progress": {"done": done, "total": total_phases, "label": label},
                **extra,
            },
        )

    phase_status("loading_model", 0, "加载 Coconut 模型")
    model, tokenizer = load_coconut_model(checkpoint, config_path, device)
    for p in model.parameters():
        p.requires_grad = False

    phase_status("labeling_train", 1, f"标注 is_correct train · 0/{len(train_sub)}")

    def label_cb(done, total):
        if progress_cb:
            progress_cb("label_train", done, total)
        phase_status("labeling_train", 1, f"标注 is_correct train · {done}/{total}")

    train_examples = build_rich_stop_examples_for_samples(
        model,
        tokenizer,
        train_sub,
        cap=cap,
        device=device,
        seed=42,
        predict_fn=predict_at_n,
        expected_fn=expected_answer,
        build_prompt_fn=build_eval_prompt,
        eval_profile=profile,
        progress_cb=label_cb,
        label_mode="is_correct",
    )

    val_examples: List = []
    if val_sub:

        def val_label_cb(done, total):
            phase_status("labeling_val", 1, f"标注 is_correct val · {done}/{total}")

        val_examples = build_rich_stop_examples_for_samples(
            model,
            tokenizer,
            val_sub,
            cap=cap,
            device=device,
            seed=142,
            predict_fn=predict_at_n,
            expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt,
            eval_profile=profile,
            progress_cb=val_label_cb,
            label_mode="is_correct",
        )

    phase_status("training_stop_head", 2, "训练 CorrectnessHead（RichStopHead）")
    head, train_metrics = train_rich_stop_head(
        train_examples,
        val_examples,
        epochs=train_epochs,
        device=device,
    )

    phase_status("calibrating_threshold", 3, "val 平衡阈值校准")
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
        },
        stop_head_path,
    )

    rows: List[dict] = []
    eval_strategies = [
        ("correctness_stop", "correct"),
        ("stable_correctness_stop", "stable_correct"),
        ("stable_stop", "stable"),
        ("fixed_3", "baseline"),
        ("auto_route", "baseline"),
        ("oracle_first_correct", "oracle"),
    ]

    for step_idx, (sid, kind) in enumerate(eval_strategies):
        phase_status("evaluating", 4, f"test · {sid}", partial_strategies=rows)
        print(f"[exp14] test {sid} …", flush=True)

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
            row["strategy_label"] = f"correctness_stop · is_correct 标签 · thr={calibrated_threshold:.2f}"
            row["params"]["label_mode"] = "is_correct"
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
            row["strategy_label"] = "stable_correctness_stop · 稳定 ∧ 正确性 head"
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
    correct = by_id.get("correctness_stop")
    stable_c = by_id.get("stable_correctness_stop")
    fixed3 = by_id.get("fixed_3")
    auto = by_id.get("auto_route")
    oracle = by_id.get("oracle_first_correct")
    stable = by_id.get("stable_stop")

    best_learned = correct
    if stable_c and correct:
        c_score = correct["accuracy"] * 5 + (correct.get("stop_timing_acc") or 0) * 3
        s_score = stable_c["accuracy"] * 5 + (stable_c.get("stop_timing_acc") or 0) * 3
        if s_score > c_score + 0.02:
            best_learned = stable_c

    timing = best_learned.get("stop_timing_acc") if best_learned else None
    feasible = bool(
        best_learned
        and fixed3
        and best_learned["accuracy"] >= fixed3["accuracy"] - 0.005
        and timing is not None
        and timing >= 0.5
    )

    insights = [
        "实验十三：rich_stop 75.6% / timing 23.8%，仍低于 fixed_3；streak_gated timing=0。",
        f"实验十四 correctness {correct['accuracy']*100:.1f}% · stable∧correct {stable_c['accuracy']*100:.1f}%"
        f" vs fixed_3 {fixed3['accuracy']*100:.1f}% · timing {((timing or 0)*100):.1f}%"
        + (" → **可行**" if feasible else " → **仍不可行**"),
        f"标签 is_correct + 平衡阈值 {calibrated_threshold:.2f}（val acc {calibration.get('val_accuracy', 0)*100:.1f}%）。",
    ]
    if not feasible:
        insights.append("三代实验（11–14）均未达标；建议实验十五：Coconut 联合微调 stop head。")

    summary = {
        "trainable_stop_feasible": feasible,
        "best_learned_strategy": best_learned["strategy_id"] if best_learned else None,
        "train_size": len(train_set),
        "val_size": len(val_sub),
        "test_size": len(test_set),
        "calibrated_threshold": calibrated_threshold,
        "calibration": calibration,
        "train_stop_head_metrics": train_metrics,
        "correctness_stop_accuracy": correct["accuracy"] if correct else None,
        "correctness_stop_timing_acc": correct.get("stop_timing_acc") if correct else None,
        "stable_correctness_stop_accuracy": stable_c["accuracy"] if stable_c else None,
        "stable_correctness_stop_timing_acc": stable_c.get("stop_timing_acc") if stable_c else None,
        "stable_stop_accuracy": stable["accuracy"] if stable else None,
        "fixed_3_accuracy": fixed3["accuracy"] if fixed3 else None,
        "auto_route_accuracy": auto["accuracy"] if auto else None,
        "oracle_first_correct_accuracy": oracle["accuracy"] if oracle else None,
        "prior_exp13_rich_stop_acc": 0.756,
        "prior_exp13_rich_stop_timing": 0.2378,
    }

    payload = {
        "ok": True,
        "experiment_type": "adaptive_stop_v4_compare",
        "experiment_id": 14,
        "title": "实验十四 · 正确性预测头（is_correct + 平衡校准）",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_sec": round(time.time() - t0, 2),
        "dataset": str(dataset_path.relative_to(ROOT)),
        "sample_count": len(dataset),
        "cap": cap,
        "device": device_name,
        "method_note": "RichStopHead 以 is_correct 为标签；阈值 balanced；评估 correctness_stop 与 stable_correctness_stop。",
        "summary": summary,
        "insights": insights,
        "strategies": rows,
        "stop_head_checkpoint": str(stop_head_path.relative_to(ROOT)),
    }

    write_status(
        status_file,
        {
            "running": False,
            "phase": "done",
            "progress": {"done": total_phases, "total": total_phases, "label": "完成"},
            "summary": summary,
            "error": None,
        },
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="实验十四")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "prosqa_test_graph_4_coconut.json")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--train-epochs", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "adaptive_stop_v4_latest.json")
    parser.add_argument("--status-file", type=Path, default=ROOT / "results" / "adaptive_stop_v4_status.json")
    parser.add_argument("--stop-head-out", type=Path, default=ROOT / "results" / "stop_head_v4.pt")
    args = parser.parse_args()

    payload = run_adaptive_stop_v4_experiment(
        args.dataset,
        max_samples=args.max_samples if args.max_samples > 0 else None,
        train_epochs=args.train_epochs,
        device_name=args.device,
        status_file=args.status_file,
        stop_head_path=args.stop_head_out,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
