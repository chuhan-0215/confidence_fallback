#!/usr/bin/env python3
"""Parallel adaptive-stop tracks · 实验十六–二十。

  16 · 结构深度 + 表示收敛（BFS d + cos OR stable）
  17 · auto_route + Δ 修正头（{-1,0,+1}）
  18 · first_correct 标签 + timing 优先校准
  19 · 收敛 OR 稳定（无结构深度门槛，OR 非 AND）
  20 · BFS 深度下界 + 冻结 v4 is_correct 头（无再训练）
  21 · BFS 深度下界 + Exp15 联合模型与 v5 头（无再训练）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

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
from stop_head import split_dataset, split_train_val_samples  # noqa: E402
from stop_head_tracks import (  # noqa: E402
    build_delta_examples,
    calibrate_heuristic_stop,
    calibrate_route_gated_autoroute_cap_threshold,
    calibrate_route_gated_threshold,
    evaluate_convergence_or_stable_stop,
    evaluate_delta_route_stop,
    evaluate_first_correct_capped_stop,
    evaluate_first_correct_gated_stop,
    evaluate_first_correct_gated_with_breakdown,
    evaluate_hop_plus_one_capped_stop,
    evaluate_hop_hybrid_fc_v4_stop,
    evaluate_hop_split_first_correct_stop,
    evaluate_route_gated_autoroute_cap_stop,
    evaluate_route_gated_correctness_stop,
    evaluate_soft_floor_first_correct_stop,
    evaluate_structure_convergence_stop,
    evaluate_rich_or_conv_stop,
    evaluate_rich_or_multi_stop,
    evaluate_rich_or_stable_stop,
    run_track_first_correct_timing,
    run_track_online_rich_combo,
    train_delta_stop_head,
)

TRACK_META = {
    16: {
        "experiment_type": "adaptive_stop_v16_compare",
        "title": "实验十六 · 结构深度 + 表示收敛",
        "method_note": "n≥blind_depth 且（hidden 余弦收敛 OR 答案稳定）— 无训练。",
        "direction": "结构通解 + 收敛检测",
    },
    17: {
        "experiment_type": "adaptive_stop_v17_compare",
        "title": "实验十七 · BFS 深度 + Δ 修正头",
        "method_note": "auto_route 估 d，Rich 头预测 Δ∈{-1,0,+1} 修正停步。",
        "direction": "结构路由 + 学习残差",
    },
    18: {
        "experiment_type": "adaptive_stop_v18_compare",
        "title": "实验十八 · first_correct + timing 优先校准",
        "method_note": "标签改回首次答对步；val 阈值 optimize=timing。",
        "direction": "直接优化停步时机",
    },
    19: {
        "experiment_type": "adaptive_stop_v19_compare",
        "title": "实验十九 · 收敛 OR 稳定（非 AND）",
        "method_note": "n≥min_n 且（hidden 收敛 OR 答案稳定）— 避免门控归零。",
        "direction": "轨迹收敛信号",
    },
    20: {
        "experiment_type": "adaptive_stop_v20_compare",
        "title": "实验二十 · BFS 下界 + v4 正确性头",
        "method_note": "n≥blind_depth 后由冻结 Exp14 head 决定停步；val balanced 校准阈值。",
        "direction": "结构路由 + 学习停步组合",
    },
    21: {
        "experiment_type": "adaptive_stop_v21_compare",
        "title": "实验二十一 · BFS 下界 + v5 联合头",
        "method_note": "加载 Exp15 联合微调 Coconut + stop_head_v5；n≥blind_depth 后停步；val balanced。",
        "direction": "Exp15 后续 · 结构下界补 acc",
    },
    22: {
        "experiment_type": "adaptive_stop_v22_compare",
        "title": "实验二十二 · BFS 下界 + v4 头 · timing 优先",
        "method_note": "同实验二十，但 val 阈值 optimize=timing，直接优化停步时机。",
        "direction": "Exp20 后续 · timing 校准",
    },
    23: {
        "experiment_type": "adaptive_stop_v23_compare",
        "title": "实验二十三 · v4 头 + auto_route 上界",
        "method_note": "n≥d 且 head 可早停，但不超过 auto_route 深度 d；val balanced。",
        "direction": "acc 与 timing 折中",
    },
    24: {
        "experiment_type": "adaptive_stop_v24_compare",
        "title": "实验二十四 · first_correct 结构门控",
        "method_note": "n≥d 时若在 d 之后首次答对则停，否则停在 d；无训练。",
        "direction": "结构 + 首次答对启发式",
    },
    25: {
        "experiment_type": "adaptive_stop_v25_compare",
        "title": "实验二十五 · first_correct + auto_route 上界",
        "method_note": "实验二十四 + min(first_correct,d)，禁止超过 BFS 深度继续走。",
        "direction": "Exp24·cap d",
    },
    26: {
        "experiment_type": "adaptive_stop_v26_compare",
        "title": "实验二十六 · 分跳数诊断（同二十四）",
        "method_note": "算法同实验二十四，额外输出 3 跳 / 4 跳子集 acc 与 timing。",
        "direction": "按跳数拆分评估",
    },
    27: {
        "experiment_type": "adaptive_stop_v27_compare",
        "title": "实验二十七 · 4 跳允许 d+1",
        "method_note": "3 跳 capped 于 d；4 跳 capped 于 d+1，兼顾 timing 与晚熟样本。",
        "direction": "分跳数 cap 策略",
    },
    28: {
        "experiment_type": "adaptive_stop_v28_compare",
        "title": "实验二十八 · soft floor 首次答对",
        "method_note": "取消 fc≥d 门槛：fc≥min_n 即停，否则停 d；针对 4 跳早答对样本。",
        "direction": "放宽结构下界",
    },
    29: {
        "experiment_type": "adaptive_stop_v29_compare",
        "title": "实验二十九 · 分跳数 soft/strict",
        "method_note": "d<4 用实验二十四 strict；d≥4 用 soft floor（fc≥min_n）。",
        "direction": "4跳专项 soft floor",
    },
    30: {
        "experiment_type": "adaptive_stop_v30_compare",
        "title": "实验三十 · 3跳 fc + 4跳 v4 头",
        "method_note": "d<4 实验二十四；d≥4 实验二十 route_gated v4 head · val balanced。",
        "direction": "分跳数启发式+学习",
    },
    31: {
        "experiment_type": "adaptive_stop_v31_compare",
        "title": "实验三十一 · 结构预算 · 单次推理",
        "method_note": "通解 A：n=blind_depth(d)，每题仅 1 次 forward；不逐步试步数。",
        "direction": "结构→预算·单次",
    },
    32: {
        "experiment_type": "adaptive_stop_v32_compare",
        "title": "实验三十二 · 图特征预算 MLP",
        "method_note": "train 用 Exp28 teacher 标 optimal_n；图特征 MLP 预测步数；test 单次 forward。",
        "direction": "学习预算·单次",
    },
    33: {
        "experiment_type": "adaptive_stop_v33_compare",
        "title": "实验三十三 · 图特征 Δ 残差",
        "method_note": "预测 Δ∈{-1,0,+1}，n=d+Δ；只用图特征、不用 latent；test 单次 forward。",
        "direction": "结构+残差·单次",
    },
    34: {
        "experiment_type": "adaptive_stop_v34_compare",
        "title": "实验三十四 · 完美预算上界",
        "method_note": "test 用 Exp28 teacher 标 optimal_n（预测上界）；每题仍只 1 次 forward。",
        "direction": "单次推理上界",
    },
    35: {
        "experiment_type": "adaptive_stop_v35_compare",
        "title": "实验三十五 · 完美预算上界（修正）",
        "method_note": "修复 Exp34 seed 不一致；teacher 标注与 test forward 同 seed。",
        "direction": "单次上界·修正",
    },
    36: {
        "experiment_type": "adaptive_stop_v36_compare",
        "title": "实验三十六 · train 查表预算",
        "method_note": "train 上 Exp28 teacher → 每 d 取 mode(optimal_n)；test 单次 forward。",
        "direction": "可部署查表",
    },
    37: {
        "experiment_type": "adaptive_stop_v37_compare",
        "title": "实验三十七 · 4跳固定3步",
        "method_note": "通解启发式：d≥4 → n=3，d=3 → n=3；无训练，单次 forward。",
        "direction": "结构规则·单次",
    },
    38: {
        "experiment_type": "adaptive_stop_v38_compare",
        "title": "实验三十八 · 4跳 d-1",
        "method_note": "d≥4 → n=d-1，d=3 → n=d；无训练，单次 forward。",
        "direction": "结构残差·单次",
    },
    39: {
        "experiment_type": "adaptive_stop_v39_compare",
        "title": "实验三十九 · d4 二分类 MLP",
        "method_note": "d≥4 专用 MLP 预测 n∈{3,4}；d<4 用 n=d；train Exp28 teacher。",
        "direction": "4跳专项学习",
    },
    40: {
        "experiment_type": "adaptive_stop_v40_compare",
        "title": "实验四十 · (d,不对称)查表",
        "method_note": "train 查表 key=(d,候选深度是否对称)；test 单次 forward。",
        "direction": "细粒度查表",
    },
    41: {
        "experiment_type": "adaptive_stop_v41_compare",
        "title": "实验四十一 · 不对称→3步",
        "method_note": "d≥4 且两候选深度不同→n=3，否则 n=d；无训练规则。",
        "direction": "结构不对称规则",
    },
    42: {
        "experiment_type": "adaptive_stop_v42_compare",
        "title": "实验四十二 · d4 完美预算上界",
        "method_note": "d<4 用 n=d；d≥4 用 Exp28 teacher optimal_n；测 d4 单独修正增益。",
        "direction": "d4 上界·单次",
    },
    43: {
        "experiment_type": "adaptive_stop_v43_compare",
        "title": "实验四十三 · d4 阈值校准",
        "method_note": "同 Exp39 二分类 MLP，val 上校准 P(n=3) 阈值；test 单次 forward。",
        "direction": "d4 阈值校准",
    },
    44: {
        "experiment_type": "adaptive_stop_v44_compare",
        "title": "实验四十四 · d4 三分类 MLP",
        "method_note": "d≥4 预测 optimal_n∈{2,3,4}；d<4 用 n=d；train Exp28 teacher。",
        "direction": "d4 三分类",
    },
    45: {
        "experiment_type": "adaptive_stop_v45_compare",
        "title": "实验四十五 · d3 二分类 + d4 结构",
        "method_note": "d=3 用 rich 特征预测 {2,3}；其余 n=d；deployable 单次 forward。",
        "direction": "d3 专项学习",
    },
    46: {
        "experiment_type": "adaptive_stop_v46_compare",
        "title": "实验四十六 · d4 kNN 预算",
        "method_note": "d≥4 用 train rich 特征 kNN 投票；d<4 用 n=d；deployable 单次 forward。",
        "direction": "d4 kNN",
    },
    47: {
        "experiment_type": "adaptive_stop_v47_compare",
        "title": "实验四十七 · Coconut 前缀预算",
        "method_note": "Coconut n=0 前缀 hidden + 图特征 → d≥4 预算；模型自预测边界。",
        "direction": "模型自预测·前缀",
    },
    48: {
        "experiment_type": "adaptive_stop_v48_compare",
        "title": "实验四十八 · Coconut 前缀 Δ",
        "method_note": "Coconut 前缀 hidden 预测 Δ∈{-1,0,+1}，n=d+Δ；全深度模型自预测。",
        "direction": "模型自预测·Δ",
    },
    49: {
        "experiment_type": "adaptive_stop_v49_compare",
        "title": "实验四十九 · 前缀阈值（标签）",
        "method_note": "Exp47 MLP + val 上校准 P(n=3) 阈值（优化标签准确率）。",
        "direction": "前缀·阈值校准",
    },
    50: {
        "experiment_type": "adaptive_stop_v50_compare",
        "title": "实验五十 · 前缀阈值（答题）",
        "method_note": "Exp47 MLP + val 上网格搜索阈值，直接优化 d≥4 答题准确率。",
        "direction": "前缀·acc校准",
    },
    51: {
        "experiment_type": "adaptive_stop_v51_compare",
        "title": "实验五十一 · 前缀+kNN 集成",
        "method_note": "Coconut 前缀与 rich kNN 均投票 n=3 才走 3 步，否则 n=4。",
        "direction": "前缀·集成",
    },
    52: {
        "experiment_type": "adaptive_stop_v52_compare",
        "title": "实验五十二 · 前缀保守回退",
        "method_note": "P(n=3)<阈值时回退 n=d（非 n=4）；val 上优化答题 acc。",
        "direction": "前缀·保守回退",
    },
    53: {
        "experiment_type": "adaptive_stop_v53_compare",
        "title": "实验五十三 · 在线 head∨稳定",
        "method_note": "逐步推理：RichStopHead 或答案稳定 streak 即停；无 BFS、无 upfront 预算。",
        "direction": "在线自停·head∨stable",
    },
    54: {
        "experiment_type": "adaptive_stop_v54_compare",
        "title": "实验五十四 · 在线 head∨收敛",
        "method_note": "逐步推理：RichStopHead 或 hidden 余弦收敛即停；无 BFS。",
        "direction": "在线自停·head∨conv",
    },
    55: {
        "experiment_type": "adaptive_stop_v55_compare",
        "title": "实验五十五 · 在线三重 OR",
        "method_note": "逐步推理：head∨stable∨conv 任一满足即停；无 BFS，纯模型轨迹信号。",
        "direction": "在线自停·三重OR",
    },
}


def _feasible(best_row, fixed3) -> bool:
    if not best_row or not fixed3:
        return False
    timing = best_row.get("stop_timing_acc")
    return (
        best_row["accuracy"] >= fixed3["accuracy"] - 0.005
        and timing is not None
        and timing >= 0.5
    )


def run_adaptive_stop_track_experiment(
    track: int,
    dataset_path: Path,
    *,
    max_samples: Optional[int] = None,
    cap: int = 8,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    train_epochs: int = 30,
    stop_min_n: int = 2,
    device_name: str = "cpu",
    progress_cb=None,
    status_file: Optional[Path] = None,
) -> dict:
    from run_experiment import ensure_checkpoint  # noqa: E402

    if track not in TRACK_META:
        raise ValueError(f"unknown track {track}")

    meta = TRACK_META[track]
    config_path = ROOT / "configs" / "symbol-2layer-8head-768dim.json"
    if track == 21:
        joint_ckpt = ROOT / "results" / "coconut_joint_v5.pt"
        if not joint_ckpt.is_file():
            raise FileNotFoundError(f"missing joint coconut checkpoint: {joint_ckpt}")
        checkpoint = joint_ckpt
    else:
        checkpoint = ensure_checkpoint(ROOT / "checkpoints" / "checkpoint_300")
    device = resolve_device(device_name)
    profile = parse_eval_profile(None)

    dataset = load_dataset(dataset_path, max_samples=max_samples)
    train_set, test_set = split_dataset(dataset, train_ratio=train_ratio)
    train_sub, val_sub = split_train_val_samples(train_set, val_ratio=val_ratio, seed=43)
    policies = make_policies(cap=cap)

    total_phases = 5
    t0 = time.time()

    def phase_status(phase: str, done: int, label: str, **extra):
        write_status(
            status_file,
            {
                "running": True,
                "phase": phase,
                "params": {
                    "experiment_type": meta["experiment_type"],
                    "track": track,
                },
                "progress": {"done": done, "total": total_phases, "label": label},
                **extra,
            },
        )

    phase_status("loading_model", 0, "加载 Coconut 模型" + (" · joint_v5" if track == 21 else ""))
    model, tokenizer = load_coconut_model(checkpoint, config_path, device)
    for p in model.parameters():
        p.requires_grad = False

    rows: List[dict] = []
    train_metrics = None
    calibration = None
    checkpoint_note = None
    hop_breakdown = None

    if track == 16:
        phase_status("calibrating", 1, "val 网格搜索 cos × patience")
        best_params, calibration = calibrate_heuristic_stop(
            evaluate_structure_convergence_stop,
            model,
            tokenizer,
            val_sub,
            cap=cap,
            min_n=stop_min_n,
            cos_grid=[0.88, 0.92, 0.95, 0.98],
            patience_grid=[2, 3],
            device=device,
            seed=77,
            predict_fn=predict_at_n,
            expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt,
            eval_profile=profile,
            optimize="balanced",
        )
        phase_status("evaluating", 2, "test · structure_convergence_stop")

        def sample_cb(done, total):
            if progress_cb:
                progress_cb("structure_convergence_stop", done, total)

        main_row = evaluate_structure_convergence_stop(
            model,
            tokenizer,
            test_set,
            cap=cap,
            min_n=stop_min_n,
            cos_threshold=best_params["cos_threshold"],
            patience=best_params["patience"],
            device=device,
            seed=99,
            predict_fn=predict_at_n,
            expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt,
            eval_profile=profile,
            progress_cb=sample_cb,
        )
        rows.append(main_row)

    elif track == 17:
        phase_status("labeling_train", 1, f"构建 Δ 样本 · 0/{len(train_sub)}")

        def delta_cb(done, total):
            phase_status("labeling_train", 1, f"构建 Δ 样本 · {done}/{total}")

        train_delta = build_delta_examples(
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
            progress_cb=delta_cb,
        )
        val_delta = build_delta_examples(
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
        ) if val_sub else []
        phase_status("training", 2, "训练 DeltaStopHead")
        head, train_metrics = train_delta_stop_head(
            train_delta,
            val_delta,
            device=device,
            epochs=train_epochs,
        )
        ckpt = ROOT / "results" / f"stop_head_v{track}.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": head.state_dict(), "train_metrics": train_metrics}, ckpt)
        checkpoint_note = str(ckpt.relative_to(ROOT))
        phase_status("evaluating", 3, "test · delta_route_stop")

        def sample_cb(done, total):
            if progress_cb:
                progress_cb("delta_route_stop", done, total)

        main_row = evaluate_delta_route_stop(
            head,
            model,
            tokenizer,
            test_set,
            cap=cap,
            min_n=stop_min_n,
            device=device,
            seed=99,
            predict_fn=predict_at_n,
            expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt,
            eval_profile=profile,
            progress_cb=sample_cb,
        )
        rows.append(main_row)

    elif track == 18:

        def status_cb(phase, done, label):
            phase_status(phase, done, label)

        def sample_cb(done, total):
            if progress_cb:
                progress_cb("first_correct_timing_stop", done, total)

        head, train_metrics, calibration, thr, main_row = run_track_first_correct_timing(
            model,
            tokenizer,
            train_sub,
            val_sub,
            test_set,
            cap=cap,
            stop_min_n=stop_min_n,
            device=device,
            profile=profile,
            predict_fn=predict_at_n,
            expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt,
            train_epochs=train_epochs,
            status_cb=status_cb,
            progress_cb=sample_cb,
        )
        ckpt = ROOT / "results" / f"stop_head_v{track}.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": head.state_dict(),
                "train_metrics": train_metrics,
                "calibration": calibration,
                "calibrated_threshold": thr,
            },
            ckpt,
        )
        checkpoint_note = str(ckpt.relative_to(ROOT))
        rows.append(main_row)

    elif track == 19:
        phase_status("calibrating", 1, "val 网格搜索 conv OR stable")
        best_params, calibration = calibrate_heuristic_stop(
            evaluate_convergence_or_stable_stop,
            model,
            tokenizer,
            val_sub,
            cap=cap,
            min_n=stop_min_n,
            cos_grid=[0.90, 0.94, 0.97, 0.99],
            patience_grid=[2, 3, 4],
            device=device,
            seed=77,
            predict_fn=predict_at_n,
            expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt,
            eval_profile=profile,
            optimize="balanced",
        )
        phase_status("evaluating", 2, "test · convergence_or_stable_stop")

        def sample_cb(done, total):
            if progress_cb:
                progress_cb("convergence_or_stable_stop", done, total)

        main_row = evaluate_convergence_or_stable_stop(
            model,
            tokenizer,
            test_set,
            cap=cap,
            min_n=stop_min_n,
            cos_threshold=best_params["cos_threshold"],
            patience=best_params["patience"],
            device=device,
            seed=99,
            predict_fn=predict_at_n,
            expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt,
            eval_profile=profile,
            progress_cb=sample_cb,
        )
        rows.append(main_row)

    elif track in (20, 22):
        from stop_head import RichStopHead  # noqa: E402

        optimize = "balanced" if track == 20 else "timing"
        v4_ckpt = ROOT / "results" / "stop_head_v4.pt"
        if not v4_ckpt.is_file():
            raise FileNotFoundError(f"missing v4 head checkpoint: {v4_ckpt}")
        phase_status("loading_head", 1, "加载 Exp14 stop_head_v4.pt")
        head = RichStopHead(hidden_dim=768, max_steps=cap).to(device)
        payload_ckpt = torch.load(v4_ckpt, map_location=device)
        head.load_state_dict(payload_ckpt["state_dict"])
        head.eval()
        checkpoint_note = str(v4_ckpt.relative_to(ROOT))

        phase_status("calibrating", 2, f"val {optimize} 阈值 · route_gated")
        calibrated_threshold, calibration = calibrate_route_gated_threshold(
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
            optimize=optimize,
        )
        phase_status("evaluating", 3, "test · route_gated stop")

        def sample_cb(done, total):
            if progress_cb:
                progress_cb("route_gated_stop", done, total)

        main_row = evaluate_route_gated_correctness_stop(
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
        if track == 22:
            main_row["strategy_id"] = "route_gated_timing_stop"
            main_row["strategy_label"] = (
                f"route_gated_timing · n≥blind_depth ∧ head · thr={calibrated_threshold:.2f} · optimize=timing"
            )
            main_row["params"]["optimize"] = "timing"
        rows.append(main_row)

    elif track == 21:
        from stop_head import RichStopHead  # noqa: E402

        v5_ckpt = ROOT / "results" / "stop_head_v5.pt"
        if not v5_ckpt.is_file():
            raise FileNotFoundError(f"missing v5 head checkpoint: {v5_ckpt}")
        phase_status("loading_head", 1, "加载 Exp15 stop_head_v5.pt")
        head = RichStopHead(hidden_dim=768, max_steps=cap).to(device)
        payload_ckpt = torch.load(v5_ckpt, map_location=device)
        head.load_state_dict(payload_ckpt["state_dict"])
        head.eval()
        checkpoint_note = str(v5_ckpt.relative_to(ROOT))

        phase_status("calibrating", 2, "val balanced 阈值 · route_gated · joint")
        calibrated_threshold, calibration = calibrate_route_gated_threshold(
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
        phase_status("evaluating", 3, "test · route_gated_joint_stop")

        def sample_cb(done, total):
            if progress_cb:
                progress_cb("route_gated_joint_stop", done, total)

        main_row = evaluate_route_gated_correctness_stop(
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
        main_row["strategy_id"] = "route_gated_joint_stop"
        main_row["strategy_label"] = (
            f"route_gated_joint · n≥blind_depth ∧ v5 head · thr={calibrated_threshold:.2f}"
        )
        main_row["params"]["head"] = "rich_v5"
        main_row["params"]["coconut"] = "joint_v5"
        rows.append(main_row)

    elif track == 23:
        from stop_head import RichStopHead  # noqa: E402

        v4_ckpt = ROOT / "results" / "stop_head_v4.pt"
        if not v4_ckpt.is_file():
            raise FileNotFoundError(f"missing v4 head checkpoint: {v4_ckpt}")
        phase_status("loading_head", 1, "加载 Exp14 stop_head_v4.pt")
        head = RichStopHead(hidden_dim=768, max_steps=cap).to(device)
        payload_ckpt = torch.load(v4_ckpt, map_location=device)
        head.load_state_dict(payload_ckpt["state_dict"])
        head.eval()
        checkpoint_note = str(v4_ckpt.relative_to(ROOT))

        phase_status("calibrating", 2, "val balanced · route_gated + auto_route cap")
        calibrated_threshold, calibration = calibrate_route_gated_autoroute_cap_threshold(
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
        phase_status("evaluating", 3, "test · route_gated_autoroute_cap_stop")

        def sample_cb23(done, total):
            if progress_cb:
                progress_cb("route_gated_autoroute_cap_stop", done, total)

        main_row = evaluate_route_gated_autoroute_cap_stop(
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
            progress_cb=sample_cb23,
        )
        rows.append(main_row)

    elif track == 24:
        phase_status("evaluating", 2, "test · first_correct_gated_stop")

        def sample_cb24(done, total):
            if progress_cb:
                progress_cb("first_correct_gated_stop", done, total)

        main_row = evaluate_first_correct_gated_stop(
            model,
            tokenizer,
            test_set,
            cap=cap,
            min_n=stop_min_n,
            device=device,
            seed=99,
            predict_fn=predict_at_n,
            expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt,
            eval_profile=profile,
            progress_cb=sample_cb24,
        )
        rows.append(main_row)

    elif track == 25:
        phase_status("evaluating", 2, "test · first_correct_capped_stop")

        def sample_cb25(done, total):
            if progress_cb:
                progress_cb("first_correct_capped_stop", done, total)

        main_row = evaluate_first_correct_capped_stop(
            model,
            tokenizer,
            test_set,
            cap=cap,
            min_n=stop_min_n,
            device=device,
            seed=99,
            predict_fn=predict_at_n,
            expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt,
            eval_profile=profile,
            progress_cb=sample_cb25,
        )
        rows.append(main_row)

    elif track == 26:
        phase_status("evaluating", 2, "test · first_correct_gated + hop breakdown")

        def sample_cb26(done, total):
            if progress_cb:
                progress_cb("first_correct_gated_hop_report", done, total)

        main_row, hop_breakdown = evaluate_first_correct_gated_with_breakdown(
            model,
            tokenizer,
            test_set,
            cap=cap,
            min_n=stop_min_n,
            device=device,
            seed=99,
            predict_fn=predict_at_n,
            expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt,
            eval_profile=profile,
            progress_cb=sample_cb26,
        )
        rows.append(main_row)

    elif track == 27:
        phase_status("evaluating", 2, "test · hop_plus_one_capped_stop")

        def sample_cb27(done, total):
            if progress_cb:
                progress_cb("hop_plus_one_capped_stop", done, total)

        main_row = evaluate_hop_plus_one_capped_stop(
            model,
            tokenizer,
            test_set,
            cap=cap,
            min_n=stop_min_n,
            device=device,
            seed=99,
            predict_fn=predict_at_n,
            expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt,
            eval_profile=profile,
            progress_cb=sample_cb27,
        )
        rows.append(main_row)

    elif track == 28:
        phase_status("evaluating", 2, "test · soft_floor_first_correct_stop")

        def sample_cb28(done, total):
            if progress_cb:
                progress_cb("soft_floor_first_correct_stop", done, total)

        main_row = evaluate_soft_floor_first_correct_stop(
            model,
            tokenizer,
            test_set,
            cap=cap,
            min_n=stop_min_n,
            device=device,
            seed=99,
            predict_fn=predict_at_n,
            expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt,
            eval_profile=profile,
            progress_cb=sample_cb28,
        )
        rows.append(main_row)

    elif track == 29:
        phase_status("evaluating", 2, "test · hop_split_first_correct_stop")

        def sample_cb29(done, total):
            if progress_cb:
                progress_cb("hop_split_first_correct_stop", done, total)

        main_row = evaluate_hop_split_first_correct_stop(
            model,
            tokenizer,
            test_set,
            cap=cap,
            min_n=stop_min_n,
            split_depth=4,
            device=device,
            seed=99,
            predict_fn=predict_at_n,
            expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt,
            eval_profile=profile,
            progress_cb=sample_cb29,
        )
        rows.append(main_row)

    elif track == 30:
        from stop_head import RichStopHead  # noqa: E402

        v4_ckpt = ROOT / "results" / "stop_head_v4.pt"
        if not v4_ckpt.is_file():
            raise FileNotFoundError(f"missing v4 head checkpoint: {v4_ckpt}")
        phase_status("loading_head", 1, "加载 Exp14 stop_head_v4.pt · hop hybrid")
        head = RichStopHead(hidden_dim=768, max_steps=cap).to(device)
        payload_ckpt = torch.load(v4_ckpt, map_location=device)
        head.load_state_dict(payload_ckpt["state_dict"])
        head.eval()
        checkpoint_note = str(v4_ckpt.relative_to(ROOT))

        phase_status("calibrating", 2, "val balanced · hop_hybrid fc+v4")
        calibrated_threshold, calibration = calibrate_route_gated_threshold(
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
        phase_status("evaluating", 3, "test · hop_hybrid_fc_v4_stop")

        def sample_cb30(done, total):
            if progress_cb:
                progress_cb("hop_hybrid_fc_v4_stop", done, total)

        main_row = evaluate_hop_hybrid_fc_v4_stop(
            head,
            model,
            tokenizer,
            test_set,
            cap=cap,
            min_n=stop_min_n,
            split_depth=4,
            threshold=calibrated_threshold,
            device=device,
            seed=99,
            predict_fn=predict_at_n,
            expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt,
            eval_profile=profile,
            progress_cb=sample_cb30,
        )
        rows.append(main_row)

    elif track in (31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52):
        from boundary_budget import (  # noqa: E402
            build_prompt_budget_labels,
            build_soft_floor_budget_labels,
            calibrate_d4_binary_threshold,
            calibrate_prompt_d4_threshold,
            calibrate_prompt_d4_threshold_by_accuracy,
            evaluate_upfront_budget_stop,
            make_asymmetry_lookup_budget_fn,
            make_asymmetry_rule_budget_fn,
            make_d3_hybrid_budget_fn,
            make_d4_binary_budget_fn,
            make_d4_early_budget_fn,
            make_d4_knn_budget_fn,
            make_d4_multiclass_budget_fn,
            make_d4_rich_binary_budget_fn,
            make_d4_teacher_hybrid_budget_fn,
            make_d4_threshold_budget_fn,
            make_d_minus_one_budget_fn,
            make_delta_mlp_budget_fn,
            make_lookup_budget_fn,
            make_mlp_budget_fn,
            make_oracle_teacher_budget_fn,
            make_prompt_d4_budget_fn,
            make_prompt_d4_threshold_budget_fn,
            make_prompt_delta_budget_fn,
            make_prompt_knn_ensemble_budget_fn,
            make_structure_budget_fn,
            train_asymmetry_lookup_table,
            train_d3_binary_mlp,
            train_d4_binary_mlp,
            train_d4_knn_bank,
            train_d4_multiclass_mlp,
            train_d4_weighted_binary_mlp,
            train_graph_budget_mlp,
            train_graph_delta_mlp,
            train_lookup_budget_table,
            train_prompt_d4_binary_mlp,
            train_prompt_delta_mlp,
        )

        budget_fn = None
        extra_params = {"single_forward": True, "inference_probes": 1}
        eval_seed = 99

        if track == 31:
            phase_status("evaluating", 2, "test · upfront structure budget · single forward")
            budget_fn = make_structure_budget_fn(min_n=stop_min_n, cap=cap)
            strategy_id = "upfront_structure_budget"
            strategy_label = "upfront · n=blind_depth · 1×forward"

        elif track == 32:
            def label_cb(done, total):
                if progress_cb:
                    progress_cb("label_train", done, total)

            phase_status("labeling", 1, "train · Exp28 teacher labels")
            train_rows = build_soft_floor_budget_labels(
                model, tokenizer, train_sub,
                cap=cap, min_n=stop_min_n, device=device, seed=77,
                predict_fn=predict_at_n, expected_fn=expected_answer, eval_profile=profile,
                progress_cb=label_cb,
            )
            phase_status("labeling", 2, "val · Exp28 teacher labels")
            val_rows = build_soft_floor_budget_labels(
                model, tokenizer, val_sub,
                cap=cap, min_n=stop_min_n, device=device, seed=78,
                predict_fn=predict_at_n, expected_fn=expected_answer, eval_profile=profile,
            )
            phase_status("training", 3, "train · graph budget MLP")
            budget_head, train_metrics = train_graph_budget_mlp(
                train_rows, val_rows, min_n=stop_min_n, cap=cap, epochs=train_epochs,
            )
            ckpt_path = ROOT / "results" / "boundary_budget_v32.pt"
            torch.save({"state_dict": budget_head.state_dict(), "min_n": stop_min_n, "cap": cap}, ckpt_path)
            checkpoint_note = str(ckpt_path.relative_to(ROOT))
            budget_fn = make_mlp_budget_fn(
                budget_head, min_n=stop_min_n, cap=cap, device=device
            )
            strategy_id = "upfront_graph_mlp_budget"
            strategy_label = "upfront · graph MLP budget · 1×forward"
            extra_params["predictor"] = "graph_mlp"

        elif track == 33:
            phase_status("labeling", 1, "train · teacher Δ labels")
            train_rows = build_soft_floor_budget_labels(
                model, tokenizer, train_sub,
                cap=cap, min_n=stop_min_n, device=device, seed=77,
                predict_fn=predict_at_n, expected_fn=expected_answer, eval_profile=profile,
            )
            phase_status("labeling", 2, "val · teacher Δ labels")
            val_rows = build_soft_floor_budget_labels(
                model, tokenizer, val_sub,
                cap=cap, min_n=stop_min_n, device=device, seed=78,
                predict_fn=predict_at_n, expected_fn=expected_answer, eval_profile=profile,
            )
            phase_status("training", 3, "train · graph Δ MLP")
            delta_head, train_metrics = train_graph_delta_mlp(
                train_rows, val_rows, epochs=train_epochs,
            )
            ckpt_path = ROOT / "results" / "boundary_delta_v33.pt"
            torch.save({"state_dict": delta_head.state_dict(), "min_n": stop_min_n, "cap": cap}, ckpt_path)
            checkpoint_note = str(ckpt_path.relative_to(ROOT))
            budget_fn = make_delta_mlp_budget_fn(
                delta_head, min_n=stop_min_n, cap=cap, device=device
            )
            strategy_id = "upfront_graph_delta_budget"
            strategy_label = "upfront · graph Δ on d · 1×forward"
            extra_params["predictor"] = "graph_delta_mlp"

        elif track == 34:
            phase_status("labeling", 2, "test · perfect teacher budget labels")
            test_rows = build_soft_floor_budget_labels(
                model, tokenizer, test_set,
                cap=cap, min_n=stop_min_n, device=device, seed=79,
                predict_fn=predict_at_n, expected_fn=expected_answer, eval_profile=profile,
            )
            teacher_map = {
                int(test_set[i].get("_idx", i)): row["optimal_n"]
                for i, row in enumerate(test_rows)
            }
            budget_fn = make_oracle_teacher_budget_fn(
                teacher_map,
                min_n=stop_min_n,
                cap=cap,
                fallback=make_structure_budget_fn(min_n=stop_min_n, cap=cap),
            )
            strategy_id = "upfront_teacher_budget"
            strategy_label = "upfront · perfect Exp28 budget · 1×forward"
            extra_params["predictor"] = "oracle_teacher"
            extra_params["note"] = "upper bound: perfect budget prediction, not deployable"

        elif track == 35:
            phase_status("labeling", 2, "test · teacher labels (seed matched)")
            test_rows = build_soft_floor_budget_labels(
                model, tokenizer, test_set,
                cap=cap, min_n=stop_min_n, device=device, seed=eval_seed,
                predict_fn=predict_at_n, expected_fn=expected_answer, eval_profile=profile,
            )
            teacher_map = {
                int(test_set[i].get("_idx", i)): row["optimal_n"]
                for i, row in enumerate(test_rows)
            }
            budget_fn = make_oracle_teacher_budget_fn(
                teacher_map,
                min_n=stop_min_n,
                cap=cap,
                fallback=make_structure_budget_fn(min_n=stop_min_n, cap=cap),
            )
            strategy_id = "upfront_teacher_budget_fixed"
            strategy_label = "upfront · teacher budget · seed一致 · 1×forward"
            extra_params["predictor"] = "oracle_teacher_seed_fixed"
            extra_params["label_seed"] = eval_seed
            extra_params["note"] = "true single-forward ceiling for Exp28 budget"

        elif track == 36:
            phase_status("labeling", 1, "train · teacher labels for lookup")
            train_rows = build_soft_floor_budget_labels(
                model, tokenizer, train_sub,
                cap=cap, min_n=stop_min_n, device=device, seed=77,
                predict_fn=predict_at_n, expected_fn=expected_answer, eval_profile=profile,
            )
            lookup_table, lookup_meta = train_lookup_budget_table(
                train_rows, min_n=stop_min_n, cap=cap
            )
            train_metrics = lookup_meta
            calibration = {"lookup_table": lookup_meta["lookup_table"]}
            budget_fn = make_lookup_budget_fn(
                lookup_table, min_n=stop_min_n, cap=cap
            )
            strategy_id = "upfront_lookup_budget"
            strategy_label = "upfront · train lookup by d · 1×forward"
            extra_params["predictor"] = "lookup_table"

        elif track == 37:
            phase_status("evaluating", 2, "test · d>=4→3 upfront rule")
            budget_fn = make_d4_early_budget_fn(min_n=stop_min_n, cap=cap)
            strategy_id = "upfront_d4_to_3_budget"
            strategy_label = "upfront · d≥4→3 · 1×forward"
            extra_params["predictor"] = "rule_d4_to_3"

        elif track == 38:
            phase_status("evaluating", 2, "test · d>=4→d-1 upfront rule")
            budget_fn = make_d_minus_one_budget_fn(min_n=stop_min_n, cap=cap)
            strategy_id = "upfront_d_minus_one_budget"
            strategy_label = "upfront · d≥4→d-1 · 1×forward"
            extra_params["predictor"] = "rule_d_minus_one"

        elif track == 39:
            phase_status("labeling", 1, "train · teacher labels for d4 binary")
            train_rows = build_soft_floor_budget_labels(
                model, tokenizer, train_sub,
                cap=cap, min_n=stop_min_n, device=device, seed=77,
                predict_fn=predict_at_n, expected_fn=expected_answer, eval_profile=profile,
            )
            val_rows = build_soft_floor_budget_labels(
                model, tokenizer, val_sub,
                cap=cap, min_n=stop_min_n, device=device, seed=78,
                predict_fn=predict_at_n, expected_fn=expected_answer, eval_profile=profile,
            )
            phase_status("training", 3, "train · d4 binary MLP (3 vs 4)")
            d4_head, train_metrics = train_d4_binary_mlp(train_rows, val_rows, epochs=train_epochs)
            ckpt_path = ROOT / "results" / "boundary_d4_binary_v39.pt"
            torch.save({"state_dict": d4_head.state_dict()}, ckpt_path)
            checkpoint_note = str(ckpt_path.relative_to(ROOT))
            budget_fn = make_d4_binary_budget_fn(
                d4_head, min_n=stop_min_n, cap=cap, device=device
            )
            strategy_id = "upfront_d4_binary_budget"
            strategy_label = "upfront · d≥4 MLP 3|4 · 1×forward"
            extra_params["predictor"] = "d4_binary_mlp"

        elif track == 40:
            phase_status("labeling", 1, "train · teacher for asymmetry lookup")
            train_rows = build_soft_floor_budget_labels(
                model, tokenizer, train_sub,
                cap=cap, min_n=stop_min_n, device=device, seed=77,
                predict_fn=predict_at_n, expected_fn=expected_answer, eval_profile=profile,
            )
            asym_table, asym_meta = train_asymmetry_lookup_table(
                train_rows, min_n=stop_min_n, cap=cap
            )
            train_metrics = asym_meta
            calibration = {"lookup_table": asym_meta["lookup_table"]}
            budget_fn = make_asymmetry_lookup_budget_fn(
                asym_table, min_n=stop_min_n, cap=cap
            )
            strategy_id = "upfront_asymmetry_lookup"
            strategy_label = "upfront · (d,asym) lookup · 1×forward"
            extra_params["predictor"] = "asymmetry_lookup"

        elif track == 41:
            phase_status("evaluating", 2, "test · asymmetry rule budget")
            budget_fn = make_asymmetry_rule_budget_fn(min_n=stop_min_n, cap=cap)
            strategy_id = "upfront_asymmetry_rule"
            strategy_label = "upfront · d≥4∧asym→3 · 1×forward"
            extra_params["predictor"] = "asymmetry_rule"

        elif track == 42:
            phase_status("labeling", 2, "test · d4 teacher labels (hybrid upper bound)")
            test_rows = build_soft_floor_budget_labels(
                model, tokenizer, test_set,
                cap=cap, min_n=stop_min_n, device=device, seed=eval_seed,
                predict_fn=predict_at_n, expected_fn=expected_answer, eval_profile=profile,
            )
            teacher_map_d4 = {
                int(test_set[i].get("_idx", i)): row["optimal_n"]
                for i, row in enumerate(test_rows)
                if row["blind_depth"] >= 4
            }
            budget_fn = make_d4_teacher_hybrid_budget_fn(
                teacher_map_d4, min_n=stop_min_n, cap=cap
            )
            strategy_id = "upfront_d4_teacher_hybrid"
            strategy_label = "upfront · d<4:n=d · d≥4:teacher · 1×forward"
            extra_params["predictor"] = "d4_teacher_hybrid"
            extra_params["note"] = "upper bound: perfect d>=4 budget only"

        elif track == 43:
            phase_status("labeling", 1, "train · teacher labels for d4 threshold")
            train_rows = build_soft_floor_budget_labels(
                model, tokenizer, train_sub,
                cap=cap, min_n=stop_min_n, device=device, seed=77,
                predict_fn=predict_at_n, expected_fn=expected_answer, eval_profile=profile,
            )
            val_rows = build_soft_floor_budget_labels(
                model, tokenizer, val_sub,
                cap=cap, min_n=stop_min_n, device=device, seed=78,
                predict_fn=predict_at_n, expected_fn=expected_answer, eval_profile=profile,
            )
            phase_status("training", 3, "train · d4 binary MLP + val threshold")
            d4_head, train_metrics = train_d4_binary_mlp(train_rows, val_rows, epochs=train_epochs)
            threshold, cal_meta = calibrate_d4_binary_threshold(d4_head, val_rows)
            calibration = cal_meta
            train_metrics = {**(train_metrics or {}), **cal_meta}
            ckpt_path = ROOT / "results" / "boundary_d4_binary_v43.pt"
            torch.save({"state_dict": d4_head.state_dict(), "threshold": threshold}, ckpt_path)
            checkpoint_note = str(ckpt_path.relative_to(ROOT))
            budget_fn = make_d4_threshold_budget_fn(
                d4_head, threshold=threshold, min_n=stop_min_n, cap=cap, device=device
            )
            strategy_id = "upfront_d4_threshold_budget"
            strategy_label = f"upfront · d≥4 thr={threshold:.2f} · 1×forward"
            extra_params["predictor"] = "d4_binary_threshold"
            extra_params["threshold"] = threshold

        elif track == 44:
            phase_status("labeling", 1, "train · teacher labels for d4 multiclass")
            train_rows = build_soft_floor_budget_labels(
                model, tokenizer, train_sub,
                cap=cap, min_n=stop_min_n, device=device, seed=77,
                predict_fn=predict_at_n, expected_fn=expected_answer, eval_profile=profile,
            )
            val_rows = build_soft_floor_budget_labels(
                model, tokenizer, val_sub,
                cap=cap, min_n=stop_min_n, device=device, seed=78,
                predict_fn=predict_at_n, expected_fn=expected_answer, eval_profile=profile,
            )
            phase_status("training", 3, "train · d4 multiclass MLP {2,3,4}")
            d4_multi, train_metrics = train_d4_multiclass_mlp(
                train_rows, val_rows, epochs=train_epochs
            )
            ckpt_path = ROOT / "results" / "boundary_d4_multi_v44.pt"
            torch.save({"state_dict": d4_multi.state_dict()}, ckpt_path)
            checkpoint_note = str(ckpt_path.relative_to(ROOT))
            budget_fn = make_d4_multiclass_budget_fn(
                d4_multi, min_n=stop_min_n, cap=cap, device=device
            )
            strategy_id = "upfront_d4_multiclass_budget"
            strategy_label = "upfront · d≥4 MLP {2,3,4} · 1×forward"
            extra_params["predictor"] = "d4_multiclass_mlp"

        elif track == 45:
            phase_status("labeling", 1, "train · teacher labels for d3 binary")
            train_rows = build_soft_floor_budget_labels(
                model, tokenizer, train_sub,
                cap=cap, min_n=stop_min_n, device=device, seed=77,
                predict_fn=predict_at_n, expected_fn=expected_answer, eval_profile=profile,
            )
            val_rows = build_soft_floor_budget_labels(
                model, tokenizer, val_sub,
                cap=cap, min_n=stop_min_n, device=device, seed=78,
                predict_fn=predict_at_n, expected_fn=expected_answer, eval_profile=profile,
            )
            phase_status("training", 3, "train · d3 rich binary {2,3}")
            d3_head, train_metrics = train_d3_binary_mlp(train_rows, val_rows, epochs=train_epochs)
            ckpt_path = ROOT / "results" / "boundary_d3_binary_v45.pt"
            torch.save({"state_dict": d3_head.state_dict()}, ckpt_path)
            checkpoint_note = str(ckpt_path.relative_to(ROOT))
            budget_fn = make_d3_hybrid_budget_fn(
                d3_head, min_n=stop_min_n, cap=cap, device=device
            )
            strategy_id = "upfront_d3_hybrid_budget"
            strategy_label = "upfront · d=3→{2,3} · else n=d · 1×forward"
            extra_params["predictor"] = "d3_rich_binary"

        elif track == 46:
            phase_status("labeling", 1, "train · teacher labels for d4 kNN")
            train_rows = build_soft_floor_budget_labels(
                model, tokenizer, train_sub,
                cap=cap, min_n=stop_min_n, device=device, seed=77,
                predict_fn=predict_at_n, expected_fn=expected_answer, eval_profile=profile,
            )
            val_rows = build_soft_floor_budget_labels(
                model, tokenizer, val_sub,
                cap=cap, min_n=stop_min_n, device=device, seed=78,
                predict_fn=predict_at_n, expected_fn=expected_answer, eval_profile=profile,
            )
            phase_status("training", 3, "build · d4 kNN bank (k=9)")
            knn_bank, knn_meta = train_d4_knn_bank(train_rows)
            train_metrics = knn_meta
            calibration = {"k": 9, "label_hist": knn_meta.get("label_hist")}
            budget_fn = make_d4_knn_budget_fn(
                knn_bank, k=9, min_n=stop_min_n, cap=cap
            )
            strategy_id = "upfront_d4_knn_budget"
            strategy_label = "upfront · d≥4 kNN rich · 1×forward"
            extra_params["predictor"] = "d4_knn_rich"

        elif track == 47:
            def prompt_label_cb(done, total):
                if progress_cb:
                    progress_cb("label_train_prompt", done, total)

            phase_status("labeling", 1, "train · prompt hidden + teacher")
            train_rows = build_prompt_budget_labels(
                model, tokenizer, train_sub,
                cap=cap, min_n=stop_min_n, device=device, seed=77,
                predict_fn=predict_at_n, expected_fn=expected_answer, eval_profile=profile,
                progress_cb=prompt_label_cb,
            )
            phase_status("labeling", 2, "val · prompt hidden + teacher")
            val_rows = build_prompt_budget_labels(
                model, tokenizer, val_sub,
                cap=cap, min_n=stop_min_n, device=device, seed=78,
                predict_fn=predict_at_n, expected_fn=expected_answer, eval_profile=profile,
            )
            phase_status("training", 3, "train · Coconut prefix d4 binary")
            prompt_head, train_metrics = train_prompt_d4_binary_mlp(
                train_rows, val_rows, epochs=train_epochs
            )
            ckpt_path = ROOT / "results" / "boundary_prompt_d4_v47.pt"
            torch.save({"state_dict": prompt_head.state_dict()}, ckpt_path)
            checkpoint_note = str(ckpt_path.relative_to(ROOT))
            budget_fn = make_prompt_d4_budget_fn(
                prompt_head, model, tokenizer,
                min_n=stop_min_n, cap=cap, device=device, eval_profile=profile, seed_base=eval_seed,
            )
            strategy_id = "upfront_prompt_d4_budget"
            strategy_label = "upfront · Coconut prefix · d≥4 · model-self"
            extra_params["predictor"] = "coconut_prefix_d4"
            extra_params["model_self_budget"] = True
            extra_params["single_forward"] = False
            extra_params["inference_probes"] = 2
            extra_params["note"] = "prefix encode + 1 answer forward; budget from Coconut hidden"

        elif track == 48:
            phase_status("labeling", 1, "train · prompt hidden + teacher Δ")
            train_rows = build_prompt_budget_labels(
                model, tokenizer, train_sub,
                cap=cap, min_n=stop_min_n, device=device, seed=77,
                predict_fn=predict_at_n, expected_fn=expected_answer, eval_profile=profile,
            )
            val_rows = build_prompt_budget_labels(
                model, tokenizer, val_sub,
                cap=cap, min_n=stop_min_n, device=device, seed=78,
                predict_fn=predict_at_n, expected_fn=expected_answer, eval_profile=profile,
            )
            phase_status("training", 3, "train · Coconut prefix Δ MLP")
            delta_head, train_metrics = train_prompt_delta_mlp(
                train_rows, val_rows, epochs=train_epochs
            )
            ckpt_path = ROOT / "results" / "boundary_prompt_delta_v48.pt"
            torch.save({"state_dict": delta_head.state_dict()}, ckpt_path)
            checkpoint_note = str(ckpt_path.relative_to(ROOT))
            budget_fn = make_prompt_delta_budget_fn(
                delta_head, model, tokenizer,
                min_n=stop_min_n, cap=cap, device=device, eval_profile=profile, seed_base=eval_seed,
            )
            strategy_id = "upfront_prompt_delta_budget"
            strategy_label = "upfront · Coconut prefix Δ · model-self"
            extra_params["predictor"] = "coconut_prefix_delta"
            extra_params["model_self_budget"] = True
            extra_params["single_forward"] = False
            extra_params["inference_probes"] = 2
            extra_params["note"] = "prefix encode + 1 answer forward; Δ from Coconut hidden"

        elif track in (49, 50, 51, 52):
            def prompt_label_cb(done, total):
                if progress_cb:
                    progress_cb("label_train_prompt", done, total)

            phase_status("labeling", 1, "train · prompt hidden + teacher")
            train_rows = build_prompt_budget_labels(
                model, tokenizer, train_sub,
                cap=cap, min_n=stop_min_n, device=device, seed=77,
                predict_fn=predict_at_n, expected_fn=expected_answer, eval_profile=profile,
                progress_cb=prompt_label_cb,
            )
            phase_status("labeling", 2, "val · prompt hidden + teacher")
            val_rows = build_prompt_budget_labels(
                model, tokenizer, val_sub,
                cap=cap, min_n=stop_min_n, device=device, seed=78,
                predict_fn=predict_at_n, expected_fn=expected_answer, eval_profile=profile,
            )
            phase_status("training", 3, "train · Coconut prefix d4 MLP")
            prompt_head, train_metrics = train_prompt_d4_binary_mlp(
                train_rows, val_rows, epochs=train_epochs
            )
            ckpt_path = ROOT / "results" / f"boundary_prompt_d4_v{track}.pt"
            torch.save({"state_dict": prompt_head.state_dict()}, ckpt_path)
            checkpoint_note = str(ckpt_path.relative_to(ROOT))
            extra_params["predictor"] = f"coconut_prefix_d4_v{track}"
            extra_params["model_self_budget"] = True
            extra_params["single_forward"] = False
            extra_params["inference_probes"] = 2

            if track == 49:
                phase_status("training", 3, "calibrate · prompt threshold (labels)")
                threshold, cal_meta = calibrate_prompt_d4_threshold(prompt_head, val_rows)
                calibration = cal_meta
                train_metrics = {**(train_metrics or {}), **cal_meta}
                budget_fn = make_prompt_d4_threshold_budget_fn(
                    prompt_head, model, tokenizer,
                    threshold=threshold, min_n=stop_min_n, cap=cap,
                    device=device, eval_profile=profile, seed_base=eval_seed,
                )
                strategy_id = "upfront_prompt_d4_threshold"
                strategy_label = f"upfront · prefix thr={threshold:.2f} · labels"

            elif track == 50:
                phase_status("training", 3, "calibrate · prompt threshold (val acc)")
                threshold, cal_meta = calibrate_prompt_d4_threshold_by_accuracy(
                    prompt_head, val_rows, val_sub,
                    model, tokenizer,
                    min_n=stop_min_n, cap=cap, device=device, eval_profile=profile,
                    predict_fn=predict_at_n, expected_fn=expected_answer, seed=78,
                    conservative=False,
                )
                calibration = cal_meta
                train_metrics = {**(train_metrics or {}), **cal_meta}
                budget_fn = make_prompt_d4_threshold_budget_fn(
                    prompt_head, model, tokenizer,
                    threshold=threshold, min_n=stop_min_n, cap=cap,
                    device=device, eval_profile=profile, seed_base=eval_seed,
                )
                strategy_id = "upfront_prompt_d4_threshold_acc"
                strategy_label = f"upfront · prefix thr={threshold:.2f} · val acc"

            elif track == 51:
                phase_status("labeling", 2, "train · rich labels for kNN bank")
                sf_train = build_soft_floor_budget_labels(
                    model, tokenizer, train_sub,
                    cap=cap, min_n=stop_min_n, device=device, seed=77,
                    predict_fn=predict_at_n, expected_fn=expected_answer, eval_profile=profile,
                )
                knn_bank, knn_meta = train_d4_knn_bank(sf_train)
                calibration = {"k": 9, "knn": knn_meta}
                train_metrics = {**(train_metrics or {}), "knn": knn_meta}
                budget_fn = make_prompt_knn_ensemble_budget_fn(
                    prompt_head, knn_bank, model, tokenizer,
                    min_n=stop_min_n, cap=cap, device=device, eval_profile=profile,
                    seed_base=eval_seed, k=9,
                )
                strategy_id = "upfront_prompt_knn_ensemble"
                strategy_label = "upfront · prefix∧kNN→3 · model-self"

            elif track == 52:
                phase_status("training", 3, "calibrate · conservative default-to-d")
                threshold, cal_meta = calibrate_prompt_d4_threshold_by_accuracy(
                    prompt_head, val_rows, val_sub,
                    model, tokenizer,
                    min_n=stop_min_n, cap=cap, device=device, eval_profile=profile,
                    predict_fn=predict_at_n, expected_fn=expected_answer, seed=78,
                    conservative=True,
                )
                calibration = cal_meta
                train_metrics = {**(train_metrics or {}), **cal_meta}
                budget_fn = make_prompt_d4_threshold_budget_fn(
                    prompt_head, model, tokenizer,
                    threshold=threshold, min_n=stop_min_n, cap=cap,
                    device=device, eval_profile=profile, seed_base=eval_seed,
                    conservative=True,
                )
                strategy_id = "upfront_prompt_d4_conservative"
                strategy_label = f"upfront · prefix thr={threshold:.2f} · fallback d"
                extra_params["note"] = "if P(n=3)<thr use n=d not n=4"

        phase_status("evaluating", 4, f"test · {strategy_id}")

        def sample_cb_upfront(done, total):
            if progress_cb:
                progress_cb(strategy_id, done, total)

        main_row = evaluate_upfront_budget_stop(
            model,
            tokenizer,
            test_set,
            budget_fn,
            strategy_id=strategy_id,
            strategy_label=strategy_label,
            cap=cap,
            min_n=stop_min_n,
            device=device,
            seed=eval_seed,
            predict_fn=predict_at_n,
            expected_fn=expected_answer,
            eval_profile=profile,
            progress_cb=sample_cb_upfront,
            extra_params=extra_params,
        )
        rows.append(main_row)

    elif track in (53, 54, 55):
        online_specs = {
            53: (
                evaluate_rich_or_stable_stop,
                {
                    "threshold": [0.35, 0.45, 0.55, 0.65],
                    "patience": [2, 3, 4],
                },
            ),
            54: (
                evaluate_rich_or_conv_stop,
                {
                    "threshold": [0.35, 0.45, 0.55, 0.65],
                    "cos_threshold": [0.88, 0.92, 0.95, 0.98],
                },
            ),
            55: (
                evaluate_rich_or_multi_stop,
                {
                    "threshold": [0.4, 0.5, 0.6],
                    "patience": [2, 3],
                    "cos_threshold": [0.90, 0.94, 0.97],
                },
            ),
        }
        eval_fn, param_grid = online_specs[track]

        def status_cb(phase, done, label):
            phase_status(phase, done, label)

        def sample_cb(done, total):
            if progress_cb:
                progress_cb(eval_fn.__name__, done, total)

        head, train_metrics, calibration, best_params, main_row = run_track_online_rich_combo(
            model,
            tokenizer,
            train_sub,
            val_sub,
            test_set,
            track=track,
            eval_fn=eval_fn,
            param_grid=param_grid,
            cap=cap,
            stop_min_n=stop_min_n,
            device=device,
            profile=profile,
            predict_fn=predict_at_n,
            expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt,
            train_epochs=train_epochs,
            optimize="balanced",
            status_cb=status_cb,
            progress_cb=sample_cb,
        )
        ckpt = ROOT / "results" / f"stop_head_v{track}.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": head.state_dict(),
                "train_metrics": train_metrics,
                "calibration": calibration,
                "best_params": best_params,
            },
            ckpt,
        )
        checkpoint_note = str(ckpt.relative_to(ROOT))
        rows.append(main_row)

    else:
        raise ValueError(f"track {track} has no evaluation branch")

    baseline_ids = [
        ("stable_stop", lambda: evaluate_stable_stop(
            model, tokenizer, test_set, device, cap=cap, eval_profile=profile
        )),
        ("fixed_3", lambda: evaluate_policy(
            model, tokenizer, test_set, policies["fixed_3"], device, cap=cap
        )),
        ("auto_route", lambda: evaluate_policy(
            model, tokenizer, test_set, policies["auto_route"], device, cap=cap
        )),
        ("oracle_first_correct", lambda: evaluate_oracle_first_correct(
            model, tokenizer, test_set, device, cap=cap, eval_profile=profile
        )),
    ]
    for sid, fn in baseline_ids:
        phase_status("evaluating", 3, f"test · {sid}", partial_strategies=rows)
        row = fn()
        row["strategy_id"] = sid
        row["strategy_label"] = sid
        row["eval_split"] = "test"
        rows.append(row)
        print(f"[exp{track}] {sid} acc={row['accuracy']*100:.1f}%", flush=True)

    by_id = {r["strategy_id"]: r for r in rows}
    main = rows[0]
    fixed3 = by_id.get("fixed_3")
    auto = by_id.get("auto_route")
    oracle = by_id.get("oracle_first_correct")
    feasible = _feasible(main, fixed3)
    timing = main.get("stop_timing_acc")

    insights = [
        f"并行线 {track}（{meta['direction']}）· 主策略 {main['strategy_id']}",
        f"acc {main['accuracy']*100:.1f}% · timing {((timing or 0)*100):.1f}%"
        f" vs fixed_3 {fixed3['accuracy']*100:.1f}% · auto_route {auto['accuracy']*100:.1f}%"
        + (" → **可行**" if feasible else " → 仍待改进"),
        meta["method_note"],
    ]
    if hop_breakdown:
        for item in hop_breakdown.get("by_blind_depth", []):
            acc_pct = (item.get("accuracy") or 0) * 100
            timing_pct = (item.get("timing_acc") or 0) * 100
            insights.append(
                f"d={item['blind_depth']} · n={item['total']} · acc {acc_pct:.1f}% · timing {timing_pct:.1f}%"
            )

    summary = {
        "track": track,
        "direction": meta["direction"],
        "trainable_stop_feasible": feasible,
        "best_learned_strategy": main["strategy_id"],
        "main_strategy_accuracy": main["accuracy"],
        "main_strategy_timing_acc": timing,
        "fixed_3_accuracy": fixed3["accuracy"] if fixed3 else None,
        "auto_route_accuracy": auto["accuracy"] if auto else None,
        "oracle_first_correct_accuracy": oracle["accuracy"] if oracle else None,
        "calibration": calibration,
        "train_metrics": train_metrics,
        "prior_exp14_correctness_acc": 0.8869,
        "prior_exp14_correctness_timing": 0.311,
        "single_forward": bool(main.get("params", {}).get("single_forward")),
        "inference_probes": main.get("params", {}).get("inference_probes"),
    }
    if hop_breakdown:
        summary["hop_breakdown"] = hop_breakdown

    payload = {
        "ok": True,
        "experiment_type": meta["experiment_type"],
        "experiment_id": track,
        "title": meta["title"],
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_sec": round(time.time() - t0, 2),
        "dataset": str(dataset_path.relative_to(ROOT)),
        "sample_count": len(dataset),
        "cap": cap,
        "device": device_name,
        "method_note": meta["method_note"],
        "summary": summary,
        "insights": insights,
        "strategies": rows,
        "stop_head_checkpoint": checkpoint_note,
    }

    write_status(
        status_file,
        {
            "running": False,
            "phase": "done",
            "params": {"experiment_type": meta["experiment_type"], "track": track},
            "progress": {"done": total_phases, "total": total_phases, "label": "完成"},
            "summary": summary,
            "error": None,
        },
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel adaptive-stop tracks 16–34")
    parser.add_argument(
        "--track",
        type=int,
        required=True,
        choices=list(range(16, 56)),
    )
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "prosqa_test_graph_4_coconut.json")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--train-epochs", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output = ROOT / "results" / f"adaptive_stop_v{args.track}_latest.json"
    status_file = ROOT / "results" / f"adaptive_stop_v{args.track}_status.json"

    payload = run_adaptive_stop_track_experiment(
        args.track,
        args.dataset,
        max_samples=args.max_samples if args.max_samples > 0 else None,
        train_epochs=args.train_epochs,
        device_name=args.device,
        status_file=status_file,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
