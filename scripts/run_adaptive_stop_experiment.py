#!/usr/bin/env python3
"""实验十一：训练 stop head，验证模型能否学会「何时停止」。

流程：
  1. 将 ProsQA 按 60/40 划分 train / test（test 不参与训练）
  2. 在 train 上：每题扫 n=1..cap，用「首次答对步」作 stop 标签，提取 latent hidden
  3. 训练轻量 LatentStopHead（冻结 Coconut）
  4. 在 test 上评估 trained_stop vs fixed_3 / auto_route / oracle 上界

结论字段 trainable_stop_feasible：test 准确率不低于 fixed_3 且停步时机准确率 ≥ 50%。
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
from run_auto_submit_experiment import (  # noqa: E402
    evaluate_policy,
    make_policies,
    write_status,
)
from stop_head import (  # noqa: E402
    build_stop_examples_for_samples,
    evaluate_trained_stop,
    split_dataset,
    train_stop_head,
)

PolicyFn = Callable[[dict, int], dict]


def predict_at_n(
    model,
    tokenizer,
    sample: dict,
    n_latent: int,
    device: torch.device,
    *,
    seed: int,
    eval_profile: EvalProfile | None = None,
) -> str:
    profile = eval_profile or parse_eval_profile(None)
    shuffle_edges = profile.prompt_mode != "fixed_edges"
    prompt = build_eval_prompt(
        sample,
        n_latent,
        seed=seed,
        choice_order=profile.choice_order,
        shuffle_edges=shuffle_edges,
    )
    input_ids = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=False)],
        device=device,
    )
    outputs = model.generate(
        input_ids,
        attention_mask=None,
        max_new_tokens=profile.max_new_tokens,
    )
    pred = decode_answer(tokenizer, outputs)
    if profile.answer_mode.startswith("symbol"):
        pred = pred.lower()
    return pred


def evaluate_stable_stop(
    model,
    tokenizer,
    dataset: List[dict],
    device: torch.device,
    *,
    cap: int = 8,
    min_n: int = 3,
    patience: int = 2,
    seed: int = 42,
    eval_profile: EvalProfile | None = None,
    progress_cb=None,
) -> dict:
    profile = eval_profile or parse_eval_profile(None)
    correct = 0
    total = 0
    stop_hist: Dict[str, int] = {}
    stop_sum = 0

    apply_feedback_config(model, {"latent_feedback_scale": 1.0})

    with torch.no_grad():
        for idx, sample in enumerate(dataset):
            expected = expected_answer(sample, profile)
            recent: List[str] = []
            stop_n = cap
            final_pred = ""

            for n in range(1, cap + 1):
                pred = predict_at_n(
                    model,
                    tokenizer,
                    sample,
                    n,
                    device,
                    seed=seed + idx * 17 + n,
                    eval_profile=profile,
                )
                recent.append(pred)
                final_pred = pred
                stop_n = n
                if n >= min_n and len(recent) >= patience:
                    window = recent[-patience:]
                    if all(p == window[0] and p for p in window):
                        break

            total += 1
            if final_pred == expected:
                correct += 1
            stop_sum += stop_n
            stop_hist[str(stop_n)] = stop_hist.get(str(stop_n), 0) + 1
            if progress_cb:
                progress_cb(idx + 1, len(dataset))

    acc = correct / total if total else 0.0
    return {
        "accuracy": round(acc, 4),
        "correct": correct,
        "total": total,
        "mean_stop_n": round(stop_sum / total, 2) if total else 0.0,
        "stop_n_histogram": stop_hist,
        "params": {"min_n": min_n, "patience": patience, "cap": cap},
    }


def run_adaptive_stop_experiment(
    dataset_path: Path,
    *,
    max_samples: Optional[int] = None,
    cap: int = 8,
    train_ratio: float = 0.6,
    train_epochs: int = 25,
    stop_threshold: float = 0.5,
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
    policies = make_policies(cap=cap)
    stop_head_path = stop_head_path or (ROOT / "results" / "stop_head.pt")

    total_phases = 5
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

    phase_status("labeling_train", 1, f"标注 train 停步标签 · 0/{len(train_set)}")

    def label_cb(done, total):
        if progress_cb:
            progress_cb("label_train", done, total)
        phase_status("labeling_train", 1, f"标注 train 停步标签 · {done}/{total}")

    train_examples = build_stop_examples_for_samples(
        model,
        tokenizer,
        train_set,
        cap=cap,
        device=device,
        seed=42,
        predict_fn=predict_at_n,
        expected_fn=expected_answer,
        build_prompt_fn=build_eval_prompt,
        eval_profile=profile,
        progress_cb=label_cb,
    )

    phase_status("training_stop_head", 2, f"训练 stop head · {len(train_examples)} 行")
    head, train_metrics = train_stop_head(
        train_examples,
        epochs=train_epochs,
        device=device,
    )
    stop_head_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "train_metrics": train_metrics,
            "cap": cap,
            "train_ratio": train_ratio,
            "train_size": len(train_set),
            "test_size": len(test_set),
        },
        stop_head_path,
    )

    rows: List[dict] = []
    eval_strategies = [
        ("trained_stop", "trained"),
        ("stable_stop", "stable"),
        ("fixed_3", "baseline"),
        ("fixed_4", "baseline"),
        ("auto_route", "baseline"),
        ("oracle_first_correct", "oracle"),
    ]

    for step_idx, (sid, kind) in enumerate(eval_strategies):
        phase_status(
            "evaluating",
            3,
            f"test 评估 · {sid} · 0/{len(test_set)}",
            partial_strategies=rows,
        )
        print(f"[exp11] test {sid} …", flush=True)

        def sample_cb(done, total, _sid=sid):
            if progress_cb:
                progress_cb(_sid, done, total)
            phase_status(
                "evaluating",
                3 + step_idx / len(eval_strategies),
                f"test 评估 · {_sid} · {done}/{total}",
                partial_strategies=rows,
            )

        if kind == "trained":
            row = evaluate_trained_stop(
                head,
                model,
                tokenizer,
                test_set,
                cap=cap,
                min_n=stop_min_n,
                threshold=stop_threshold,
                device=device,
                seed=99,
                predict_fn=predict_at_n,
                expected_fn=expected_answer,
                build_prompt_fn=build_eval_prompt,
                eval_profile=profile,
                progress_cb=sample_cb,
            )
            row["strategy_label"] = "trained_stop · 训练 stop head（test 集）"
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
    trained = by_id.get("trained_stop")
    fixed3 = by_id.get("fixed_3")
    auto = by_id.get("auto_route")
    oracle = by_id.get("oracle_first_correct")
    stable = by_id.get("stable_stop")

    gap_fixed = None
    gap_auto = None
    if trained and fixed3:
        gap_fixed = round((trained["accuracy"] - fixed3["accuracy"]) * 100, 1)
    if trained and auto:
        gap_auto = round((trained["accuracy"] - auto["accuracy"]) * 100, 1)

    timing = trained.get("stop_timing_acc") if trained else None
    feasible = bool(
        trained
        and fixed3
        and trained["accuracy"] >= fixed3["accuracy"] - 0.005
        and timing is not None
        and timing >= 0.5
    )

    insights = []
    if feasible:
        insights.append(
            f"trained_stop 在未见过的 test 集准确率 {trained['accuracy']*100:.1f}%"
            f"（fixed_3 {fixed3['accuracy']*100:.1f}%），停步时机 {timing*100:.1f}%"
            f" → **训练自停可行**（Coconut 冻结，仅训 stop head）。"
        )
    elif trained and fixed3:
        insights.append(
            f"trained_stop {trained['accuracy']*100:.1f}% vs fixed_3 {fixed3['accuracy']*100:.1f}%"
            f"，停步时机 {((timing or 0)*100):.1f}%"
            f" → **尚不能认定训练自停可行**，需更多数据/结构或联合微调。"
        )
    if trained and oracle:
        insights.append(
            f"距 oracle_first_correct 上界 {oracle['accuracy']*100:.1f}% 差 "
            f"{round((oracle['accuracy'] - trained['accuracy']) * 100, 1)} pp。"
        )
    if trained and stable:
        insights.append(
            f"对比无训练 stable_stop {stable['accuracy']*100:.1f}%："
            f"训练 stop head {'有' if trained['accuracy'] > stable['accuracy'] + 0.005 else '无'}明显增益。"
        )
    if auto and trained:
        insights.append(
            f"对比 auto_route {auto['accuracy']*100:.1f}%（读图结构）："
            f"训练自停 {'接近' if trained['accuracy'] >= auto['accuracy'] - 0.02 else '仍低于'}结构路由。"
        )

    summary = {
        "trainable_stop_feasible": feasible,
        "train_size": len(train_set),
        "test_size": len(test_set),
        "train_stop_head_metrics": train_metrics,
        "trained_stop_accuracy": trained["accuracy"] if trained else None,
        "trained_stop_timing_acc": timing,
        "trained_stop_mean_n": trained.get("mean_stop_n") if trained else None,
        "stable_stop_accuracy": stable["accuracy"] if stable else None,
        "fixed_3_accuracy": fixed3["accuracy"] if fixed3 else None,
        "auto_route_accuracy": auto["accuracy"] if auto else None,
        "oracle_first_correct_accuracy": oracle["accuracy"] if oracle else None,
        "gap_trained_to_fixed_3_pp": gap_fixed,
        "gap_trained_to_auto_route_pp": gap_auto,
    }

    payload = {
        "ok": True,
        "experiment_type": "adaptive_stop_compare",
        "experiment_id": 11,
        "title": "实验十一 · 训练 stop head（能否学会自停）",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_sec": round(time.time() - t0, 2),
        "dataset": str(dataset_path.relative_to(ROOT)),
        "sample_count": len(dataset),
        "cap": cap,
        "train_ratio": train_ratio,
        "device": device_name,
        "method_note": (
            "train 集：首次答对步作 stop 标签 + latent hidden；"
            "训 LatentStopHead（冻结 Coconut）；test 集评估 trained_stop。"
            f"可行判定：test acc ≥ fixed_3 且 stop_timing ≥ 50%。"
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


def evaluate_oracle_first_correct(
    model,
    tokenizer,
    dataset: List[dict],
    device: torch.device,
    *,
    cap: int = 8,
    seed: int = 42,
    eval_profile: EvalProfile | None = None,
    progress_cb=None,
) -> dict:
    profile = eval_profile or parse_eval_profile(None)
    correct = 0
    total = 0
    stop_sum = 0
    stop_hist: Dict[str, int] = {}

    apply_feedback_config(model, {"latent_feedback_scale": 1.0})

    with torch.no_grad():
        for idx, sample in enumerate(dataset):
            expected = expected_answer(sample, profile)
            stop_n = cap
            final_pred = ""
            for n in range(1, cap + 1):
                pred = predict_at_n(
                    model,
                    tokenizer,
                    sample,
                    n,
                    device,
                    seed=seed + idx * 17 + n,
                    eval_profile=profile,
                )
                final_pred = pred
                stop_n = n
                if pred == expected:
                    break
            total += 1
            if final_pred == expected:
                correct += 1
            stop_sum += stop_n
            stop_hist[str(stop_n)] = stop_hist.get(str(stop_n), 0) + 1
            if progress_cb:
                progress_cb(idx + 1, len(dataset))

    acc = correct / total if total else 0.0
    return {
        "accuracy": round(acc, 4),
        "correct": correct,
        "total": total,
        "mean_stop_n": round(stop_sum / total, 2) if total else 0.0,
        "stop_n_histogram": stop_hist,
        "params": {"cap": cap, "oracle_labeled": True},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="实验十一：训练 stop head 验证自停")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data" / "prosqa_test_graph_4_coconut.json",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 = 全量 419")
    parser.add_argument("--cap", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--train-epochs", type=int, default=25)
    parser.add_argument("--stop-threshold", type=float, default=0.5)
    parser.add_argument("--stop-min-n", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "adaptive_stop_latest.json",
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=ROOT / "results" / "adaptive_stop_status.json",
    )
    parser.add_argument(
        "--stop-head-out",
        type=Path,
        default=ROOT / "results" / "stop_head.pt",
    )
    args = parser.parse_args()

    payload = run_adaptive_stop_experiment(
        args.dataset,
        max_samples=args.max_samples if args.max_samples > 0 else None,
        cap=args.cap,
        train_ratio=args.train_ratio,
        train_epochs=args.train_epochs,
        stop_threshold=args.stop_threshold,
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
