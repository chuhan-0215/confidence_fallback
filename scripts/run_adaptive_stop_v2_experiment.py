#!/usr/bin/env python3
"""实验十二：改进 stop head（正则 + 验证集阈值校准 + 混合停步）。

相对实验十一：
  - train 再切 80/20 作 val，dropout + weight decay + early stopping 防过拟合
  - 在 val 上网格搜索 stop 阈值（优化 timing + accuracy）
  - test 上评估 trained_stop_v2、hybrid_stop（head ∧ 答案稳定）
  - 可行判定：test 上 best(trained_stop_v2, hybrid_stop) acc ≥ fixed_3 且 timing ≥ 50%
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "model"))

from coconut_feedback import apply_feedback_config  # noqa: E402
from evaluate_coconut import (  # noqa: E402
    decode_answer,
    expected_answer,
    load_coconut_model,
    load_dataset,
    resolve_device,
)
from eval_profile import EvalProfile, parse_eval_profile  # noqa: E402
from graph_utils import build_eval_prompt  # noqa: E402
from run_adaptive_stop_experiment import (  # noqa: E402
    evaluate_oracle_first_correct,
    evaluate_stable_stop,
    predict_at_n,
)
from run_auto_submit_experiment import (  # noqa: E402
    evaluate_policy,
    make_policies,
    write_status,
)
from stop_head import (  # noqa: E402
    build_stop_examples_for_samples,
    calibrate_stop_threshold,
    evaluate_hybrid_stop,
    evaluate_trained_stop,
    split_dataset,
    split_train_val_samples,
    train_stop_head_v2,
)

PolicyFn = Callable[[dict, int], dict]


def run_adaptive_stop_v2_experiment(
    dataset_path: Path,
    *,
    max_samples: Optional[int] = None,
    cap: int = 8,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    train_epochs: int = 50,
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
    stop_head_path = stop_head_path or (ROOT / "results" / "stop_head_v2.pt")

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

    phase_status("labeling_train", 1, f"标注 train · 0/{len(train_sub)}")

    def label_cb(done, total):
        if progress_cb:
            progress_cb("label_train", done, total)
        phase_status("labeling_train", 1, f"标注 train · {done}/{total}")

    train_examples = build_stop_examples_for_samples(
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
    )

    val_examples: List = []
    if val_sub:
        phase_status("labeling_val", 1, f"标注 val · 0/{len(val_sub)}")

        def val_label_cb(done, total):
            phase_status("labeling_val", 1, f"标注 val · {done}/{total}")

        val_examples = build_stop_examples_for_samples(
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
        )

    phase_status(
        "training_stop_head",
        2,
        f"训练 stop head v2 · {len(train_examples)} 行 · val {len(val_examples)} 行",
    )
    head, train_metrics = train_stop_head_v2(
        train_examples,
        val_examples,
        epochs=train_epochs,
        device=device,
    )

    phase_status("calibrating_threshold", 3, f"val 阈值校准 · {len(val_sub)} 题")
    calibrated_threshold, calibration = calibrate_stop_threshold(
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
    )

    stop_head_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "train_metrics": train_metrics,
            "calibration": calibration,
            "cap": cap,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "train_size": len(train_set),
            "test_size": len(test_set),
            "calibrated_threshold": calibrated_threshold,
        },
        stop_head_path,
    )

    rows: List[dict] = []
    eval_strategies = [
        ("trained_stop_v2", "trained_v2"),
        ("hybrid_stop", "hybrid"),
        ("stable_stop", "stable"),
        ("fixed_3", "baseline"),
        ("fixed_4", "baseline"),
        ("auto_route", "baseline"),
        ("oracle_first_correct", "oracle"),
    ]

    for step_idx, (sid, kind) in enumerate(eval_strategies):
        phase_status(
            "evaluating",
            4,
            f"test 评估 · {sid} · 0/{len(test_set)}",
            partial_strategies=rows,
        )
        print(f"[exp12] test {sid} …", flush=True)

        def sample_cb(done, total, _sid=sid):
            if progress_cb:
                progress_cb(_sid, done, total)
            phase_status(
                "evaluating",
                4 + step_idx / len(eval_strategies),
                f"test 评估 · {_sid} · {done}/{total}",
                partial_strategies=rows,
            )

        if kind == "trained_v2":
            row = evaluate_trained_stop(
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
                f"trained_stop_v2 · dropout+早停+阈值={calibrated_threshold:.2f}（test）"
            )
            row["params"]["calibrated"] = True
        elif kind == "hybrid":
            row = evaluate_hybrid_stop(
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
            row["strategy_label"] = "hybrid_stop · head ∧ 答案稳定（test）"
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
            row["strategy_label"] = "stable_stop · 答案稳定（无训练对照）"
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
            row["strategy_label"] = "oracle_first_correct · 首次答对即停（上界）"
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
            f"  acc={row['accuracy']*100:.1f}%"
            + (f" timing={row.get('stop_timing_acc')}" if row.get("stop_timing_acc") is not None else "")
            + (f" mean_stop={row.get('mean_stop_n')}" if row.get("mean_stop_n") is not None else ""),
            flush=True,
        )

    by_id = {r["strategy_id"]: r for r in rows}
    trained = by_id.get("trained_stop_v2")
    hybrid = by_id.get("hybrid_stop")
    fixed3 = by_id.get("fixed_3")
    auto = by_id.get("auto_route")
    oracle = by_id.get("oracle_first_correct")
    stable = by_id.get("stable_stop")

    best_learned = trained
    if hybrid and trained:
        if hybrid["accuracy"] > trained["accuracy"] + 0.005:
            best_learned = hybrid
        elif (
            abs(hybrid["accuracy"] - trained["accuracy"]) <= 0.005
            and (hybrid.get("stop_timing_acc") or 0) > (trained.get("stop_timing_acc") or 0)
        ):
            best_learned = hybrid

    gap_fixed = None
    gap_auto = None
    if best_learned and fixed3:
        gap_fixed = round((best_learned["accuracy"] - fixed3["accuracy"]) * 100, 1)
    if best_learned and auto:
        gap_auto = round((best_learned["accuracy"] - auto["accuracy"]) * 100, 1)

    timing = best_learned.get("stop_timing_acc") if best_learned else None
    feasible = bool(
        best_learned
        and fixed3
        and best_learned["accuracy"] >= fixed3["accuracy"] - 0.005
        and timing is not None
        and timing >= 0.5
    )

    insights = []
    if feasible and best_learned and fixed3:
        insights.append(
            f"{best_learned['strategy_id']} 在 test 准确率 {best_learned['accuracy']*100:.1f}%"
            f"（fixed_3 {fixed3['accuracy']*100:.1f}%），停步时机 {timing*100:.1f}%"
            f" → **改进后训练自停可行**。"
        )
    elif best_learned and fixed3:
        insights.append(
            f"trained_stop_v2 {trained['accuracy']*100:.1f}% · hybrid {hybrid['accuracy']*100:.1f}%"
            f" vs fixed_3 {fixed3['accuracy']*100:.1f}%"
            f"，最佳停步时机 {((timing or 0)*100):.1f}%"
            f" → **改进后仍不能认定可行**，可考虑联合微调 Coconut。"
        )
    if trained and hybrid:
        insights.append(
            f"阈值校准 {calibrated_threshold:.2f}（val timing "
            f"{((calibration.get('val_stop_timing_acc') or 0)*100):.1f}%）；"
            f"hybrid 相对 v2 {'提升' if hybrid['accuracy'] > trained['accuracy'] + 0.005 else '无显著'}准确率。"
        )
    if best_learned and oracle:
        insights.append(
            f"距 oracle 上界 {oracle['accuracy']*100:.1f}% 差 "
            f"{round((oracle['accuracy'] - best_learned['accuracy']) * 100, 1)} pp。"
        )
    if best_learned and stable:
        insights.append(
            f"对比 stable_stop {stable['accuracy']*100:.1f}%："
            f"最佳学习策略 {'有' if best_learned['accuracy'] > stable['accuracy'] + 0.005 else '无'}明显增益。"
        )
    if auto and best_learned:
        insights.append(
            f"对比 auto_route {auto['accuracy']*100:.1f}%："
            f"改进自停 {'接近' if best_learned['accuracy'] >= auto['accuracy'] - 0.02 else '仍低于'}结构路由。"
        )

    summary = {
        "trainable_stop_feasible": feasible,
        "best_learned_strategy": best_learned["strategy_id"] if best_learned else None,
        "train_size": len(train_set),
        "val_size": len(val_sub),
        "test_size": len(test_set),
        "calibrated_threshold": calibrated_threshold,
        "calibration": calibration,
        "train_stop_head_metrics": train_metrics,
        "trained_stop_v2_accuracy": trained["accuracy"] if trained else None,
        "trained_stop_v2_timing_acc": trained.get("stop_timing_acc") if trained else None,
        "trained_stop_v2_mean_n": trained.get("mean_stop_n") if trained else None,
        "hybrid_stop_accuracy": hybrid["accuracy"] if hybrid else None,
        "hybrid_stop_timing_acc": hybrid.get("stop_timing_acc") if hybrid else None,
        "hybrid_stop_mean_n": hybrid.get("mean_stop_n") if hybrid else None,
        "stable_stop_accuracy": stable["accuracy"] if stable else None,
        "fixed_3_accuracy": fixed3["accuracy"] if fixed3 else None,
        "auto_route_accuracy": auto["accuracy"] if auto else None,
        "oracle_first_correct_accuracy": oracle["accuracy"] if oracle else None,
        "gap_best_to_fixed_3_pp": gap_fixed,
        "gap_best_to_auto_route_pp": gap_auto,
    }

    payload = {
        "ok": True,
        "experiment_type": "adaptive_stop_v2_compare",
        "experiment_id": 12,
        "title": "实验十二 · 改进 stop head（正则 + 阈值校准 + 混合停步）",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_sec": round(time.time() - t0, 2),
        "dataset": str(dataset_path.relative_to(ROOT)),
        "sample_count": len(dataset),
        "cap": cap,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "device": device_name,
        "method_note": (
            "train 80/20 切 val；dropout+weight decay+早停训 stop head；"
            "val 上网格搜索阈值；test 评估 trained_stop_v2 与 hybrid_stop。"
            f"可行判定：best 学习策略 acc ≥ fixed_3 且 timing ≥ 50%。"
        ),
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
    parser = argparse.ArgumentParser(description="实验十二：改进 stop head")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data" / "prosqa_test_graph_4_coconut.json",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 = 全量 419")
    parser.add_argument("--cap", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--train-epochs", type=int, default=50)
    parser.add_argument("--stop-min-n", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "adaptive_stop_v2_latest.json",
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=ROOT / "results" / "adaptive_stop_v2_status.json",
    )
    parser.add_argument(
        "--stop-head-out",
        type=Path,
        default=ROOT / "results" / "stop_head_v2.pt",
    )
    args = parser.parse_args()

    payload = run_adaptive_stop_v2_experiment(
        args.dataset,
        max_samples=args.max_samples if args.max_samples > 0 else None,
        cap=args.cap,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        train_epochs=args.train_epochs,
        stop_min_n=args.stop_min_n,
        device_name=args.device,
        status_file=args.status_file,
        stop_head_path=args.stop_head_out,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
