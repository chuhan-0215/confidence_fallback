#!/usr/bin/env python3
"""实验十三：Rich stop head（答案 + 稳定性特征 + focal loss）。

假设（基于实验十一、十二失败分析）：
  - 仅靠 latent hidden 不足以判断「该停」；推理时可观测的答案与稳定 streak 应一并输入
  - 阈值校准应优先优化 val 准确率（实验十二 timing×2+acc 选阈导致 test 更差）
  - streak_gated：答案稳定后由 head 放行，避免 hybrid 同-step AND 死锁

可行判定：test 上 best(rich_stop, streak_gated) acc ≥ fixed_3 且 timing ≥ 50%
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

from evaluate_coconut import load_coconut_model, load_dataset, resolve_device  # noqa: E402
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
from evaluate_coconut import expected_answer  # noqa: E402


def run_adaptive_stop_v3_experiment(
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
    stop_head_path = stop_head_path or (ROOT / "results" / "stop_head_v3.pt")

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

    phase_status("labeling_train", 1, f"标注 rich train · 0/{len(train_sub)}")

    def label_cb(done, total):
        if progress_cb:
            progress_cb("label_train", done, total)
        phase_status("labeling_train", 1, f"标注 rich train · {done}/{total}")

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
    )

    val_examples: List = []
    if val_sub:
        def val_label_cb(done, total):
            phase_status("labeling_val", 1, f"标注 rich val · {done}/{total}")

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
        )

    phase_status("training_stop_head", 2, f"训练 RichStopHead · {len(train_examples)} 行")
    head, train_metrics = train_rich_stop_head(
        train_examples,
        val_examples,
        epochs=train_epochs,
        device=device,
    )

    phase_status("calibrating_threshold", 3, f"val 阈值校准（优先 acc）· {len(val_sub)} 题")
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
        optimize="accuracy",
    )

    stop_head_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "train_metrics": train_metrics,
            "calibration": calibration,
            "cap": cap,
            "calibrated_threshold": calibrated_threshold,
        },
        stop_head_path,
    )

    rows: List[dict] = []
    eval_strategies = [
        ("rich_stop", "rich"),
        ("streak_gated_stop", "gated"),
        ("stable_stop", "stable"),
        ("fixed_3", "baseline"),
        ("auto_route", "baseline"),
        ("oracle_first_correct", "oracle"),
    ]

    for step_idx, (sid, kind) in enumerate(eval_strategies):
        phase_status("evaluating", 4, f"test · {sid} · 0/{len(test_set)}", partial_strategies=rows)
        print(f"[exp13] test {sid} …", flush=True)

        def sample_cb(done, total, _sid=sid):
            if progress_cb:
                progress_cb(_sid, done, total)
            phase_status(
                "evaluating",
                4 + step_idx / len(eval_strategies),
                f"test · {_sid} · {done}/{total}",
                partial_strategies=rows,
            )

        if kind == "rich":
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
            row["strategy_label"] = f"rich_stop · 答案+streak+hidden · thr={calibrated_threshold:.2f}"
        elif kind == "gated":
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
            row["strategy_label"] = "streak_gated_stop · 稳定后 head 放行"
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
            row["strategy_label"] = "stable_stop · 对照"
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
            row["strategy_label"] = "oracle_first_correct · 上界"
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
            + (f" timing={row.get('stop_timing_acc')}" if row.get("stop_timing_acc") is not None else ""),
            flush=True,
        )

    by_id = {r["strategy_id"]: r for r in rows}
    rich = by_id.get("rich_stop")
    gated = by_id.get("streak_gated_stop")
    fixed3 = by_id.get("fixed_3")
    auto = by_id.get("auto_route")
    oracle = by_id.get("oracle_first_correct")
    stable = by_id.get("stable_stop")

    best_learned = rich
    if gated and rich:
        rich_score = rich["accuracy"] * 10 + (rich.get("stop_timing_acc") or 0)
        gated_score = gated["accuracy"] * 10 + (gated.get("stop_timing_acc") or 0)
        if gated_score > rich_score + 0.01:
            best_learned = gated

    timing = best_learned.get("stop_timing_acc") if best_learned else None
    feasible = bool(
        best_learned
        and fixed3
        and best_learned["accuracy"] >= fixed3["accuracy"] - 0.005
        and timing is not None
        and timing >= 0.5
    )

    gap_fixed = round((best_learned["accuracy"] - fixed3["accuracy"]) * 100, 1) if best_learned and fixed3 else None

    insights = [
        "实验十一/十二结论：纯 hidden stop head 泛化不足；十二 hybrid 同-step AND 导致 timing=0。",
        f"实验十三 RichStopHead：rich {rich['accuracy']*100:.1f}% · gated {gated['accuracy']*100:.1f}%"
        f" vs fixed_3 {fixed3['accuracy']*100:.1f}%"
        f" · 最佳 timing {((timing or 0)*100):.1f}%"
        + (f" → **训练自停可行**" if feasible else " → **仍不能认定可行**"),
        f"阈值 {calibrated_threshold:.2f}（val 优先 acc 校准）；train stop recall "
        f"{train_metrics.get('train_stop_recall', 0)*100:.1f}%。",
    ]
    if best_learned and stable:
        insights.append(
            f"对比 stable_stop {stable['accuracy']*100:.1f}%："
            f"{'超过' if best_learned['accuracy'] > stable['accuracy'] + 0.005 else '未超过'}无训练基线。"
        )
    if best_learned and oracle:
        insights.append(f"距 oracle {oracle['accuracy']*100:.1f}% 差 {round((oracle['accuracy']-best_learned['accuracy'])*100,1)} pp。")
    if not feasible:
        insights.append(
            "若仍失败：下一步应尝试 Coconut 联合微调（stop head 梯度回传 latent）或扩大标注数据。"
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
        "rich_stop_accuracy": rich["accuracy"] if rich else None,
        "rich_stop_timing_acc": rich.get("stop_timing_acc") if rich else None,
        "streak_gated_stop_accuracy": gated["accuracy"] if gated else None,
        "streak_gated_stop_timing_acc": gated.get("stop_timing_acc") if gated else None,
        "stable_stop_accuracy": stable["accuracy"] if stable else None,
        "fixed_3_accuracy": fixed3["accuracy"] if fixed3 else None,
        "auto_route_accuracy": auto["accuracy"] if auto else None,
        "oracle_first_correct_accuracy": oracle["accuracy"] if oracle else None,
        "gap_best_to_fixed_3_pp": gap_fixed,
        "prior_exp11_trained_stop_acc": 0.7619,
        "prior_exp12_hybrid_stop_acc": 0.7798,
    }

    payload = {
        "ok": True,
        "experiment_type": "adaptive_stop_v3_compare",
        "experiment_id": 13,
        "title": "实验十三 · Rich stop head（答案+稳定性+focal）",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_sec": round(time.time() - t0, 2),
        "dataset": str(dataset_path.relative_to(ROOT)),
        "sample_count": len(dataset),
        "cap": cap,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "device": device_name,
        "method_note": (
            "RichStopHead：hidden+step+answer_bucket+streak+changed；focal loss+早停；"
            "val 阈值优先 acc；评估 rich_stop 与 streak_gated_stop。"
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
    parser = argparse.ArgumentParser(description="实验十三：Rich stop head")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "prosqa_test_graph_4_coconut.json")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--cap", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--train-epochs", type=int, default=40)
    parser.add_argument("--stop-min-n", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "adaptive_stop_v3_latest.json")
    parser.add_argument("--status-file", type=Path, default=ROOT / "results" / "adaptive_stop_v3_status.json")
    parser.add_argument("--stop-head-out", type=Path, default=ROOT / "results" / "stop_head_v3.pt")
    args = parser.parse_args()

    payload = run_adaptive_stop_v3_experiment(
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


if __name__ == "__main__":
    main()
