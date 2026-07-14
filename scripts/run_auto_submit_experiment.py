#!/usr/bin/env python3
"""实验十：无标签自动配参（结构估 d + schedule）vs 固定步数 baseline。

对比策略（均可在拿到新数据后直接配置，无需扫准确率曲线）：
  - fixed_3 / fixed_4：全局固定 latent 步数
  - fallback_zero4：C=8 + schedule [1,1,1,1,0,0,0,0]（结构未知兜底）
  - auto_route：每题 n = max(BFS(root→c1), BFS(root→c2))，α=1
  - auto_route_zero：每题 C=8，schedule 在 d 步后 α=0（通解主方案）
  - oracle_hop：每题 n = len(steps)（结构金标准，非答案标签）

另报告 oracle_fixed：同一子集上「单一固定 n」的最高准确率（需标签扫步，作上界对照）。
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
from graph_utils import bfs_distances, build_eval_prompt, reasoning_hops  # noqa: E402
from run_experiment import ensure_checkpoint, sweep_latent_steps  # noqa: E402

PolicyFn = Callable[[dict, int], dict]


def blind_choice_depth(sample: dict) -> int:
    """从题面两个候选节点估 d，不使用 target 是否为正确答案。"""
    n_nodes = len(sample.get("idx_to_symbol") or [])
    dist = bfs_distances(sample["edges"], sample["root"], n_nodes)
    c1 = int(sample["target"])
    c2 = int(sample["neg_target"])
    return max(dist.get(c1, 0), dist.get(c2, 0))


def schedule_zero_after(d: int, cap: int = 8) -> List[float]:
    d = max(1, min(int(d), cap))
    return [1.0] * d + [0.0] * (cap - d)


def make_policies(cap: int = 8) -> Dict[str, PolicyFn]:
    def fixed(n: int, feedback: Optional[dict] = None) -> PolicyFn:
        fb = feedback or {"latent_feedback_scale": 1.0}

        def _policy(_sample: dict, _cap: int) -> dict:
            return {"n_latent": n, "feedback": fb}

        return _policy

    def auto_route(_sample: dict, _cap: int) -> dict:
        d = blind_choice_depth(_sample)
        return {"n_latent": d, "feedback": {"latent_feedback_scale": 1.0}}

    def auto_route_zero(sample: dict, c: int) -> dict:
        d = blind_choice_depth(sample)
        return {
            "n_latent": c,
            "feedback": {
                "latent_feedback_scale": 1.0,
                "latent_feedback_schedule": schedule_zero_after(d, c),
            },
        }

    def oracle_hop(sample: dict, _cap: int) -> dict:
        d = reasoning_hops(sample)
        return {"n_latent": max(1, d), "feedback": {"latent_feedback_scale": 1.0}}

    return {
        "fixed_3": fixed(3),
        "fixed_4": fixed(4),
        "fallback_zero4": fixed(
            cap,
            {
                "latent_feedback_scale": 1.0,
                "latent_feedback_schedule": schedule_zero_after(4, cap),
            },
        ),
        "auto_route": auto_route,
        "auto_route_zero": lambda s, c: auto_route_zero(s, c),
        "oracle_hop": oracle_hop,
    }


def evaluate_policy(
    model,
    tokenizer,
    dataset: List[dict],
    policy: PolicyFn,
    device: torch.device,
    *,
    cap: int = 8,
    seed: int = 42,
    eval_profile: EvalProfile | None = None,
    progress_cb=None,
) -> dict:
    profile = eval_profile or parse_eval_profile(None)
    shuffle_edges = profile.prompt_mode != "fixed_edges"
    correct = 0
    total = 0
    depth_hist: Dict[str, int] = {}
    n_hist: Dict[str, int] = {}

    with torch.no_grad():
        for idx, sample in enumerate(dataset):
            cfg = policy(sample, cap)
            n_latent = int(cfg["n_latent"])
            apply_feedback_config(model, cfg.get("feedback") or {})

            prompt = build_eval_prompt(
                sample,
                n_latent,
                seed=seed + idx,
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
            expected = expected_answer(sample, profile)
            total += 1
            if pred == expected:
                correct += 1

            d = blind_choice_depth(sample)
            depth_hist[str(d)] = depth_hist.get(str(d), 0) + 1
            n_hist[str(n_latent)] = n_hist.get(str(n_latent), 0) + 1
            if progress_cb:
                progress_cb(idx + 1, len(dataset))

    acc = correct / total if total else 0.0
    return {
        "accuracy": round(acc, 4),
        "correct": correct,
        "total": total,
        "blind_depth_histogram": depth_hist,
        "n_latent_histogram": n_hist,
    }


def oracle_fixed_upper_bound(
    model,
    tokenizer,
    dataset: List[dict],
    device: torch.device,
    *,
    latent_min: int = 1,
    latent_max: int = 8,
    eval_profile: EvalProfile | None = None,
    sweep_progress_cb=None,
) -> dict:
    def step_cb(i, step_total, row):
        if sweep_progress_cb:
            sweep_progress_cb(i, step_total, row or {})

    sweep = sweep_latent_steps(
        model,
        tokenizer,
        dataset,
        device,
        latent_min=latent_min,
        latent_max=latent_max,
        eval_profile=eval_profile,
        progress_cb=step_cb if sweep_progress_cb else None,
    )
    best = max(sweep, key=lambda r: (r["accuracy"], -r["n_latent"]))
    return {
        "best_n_latent": best["n_latent"],
        "accuracy": best["accuracy"],
        "note": "需标签扫步，仅作上界对照",
        "latent_sweep": sweep,
    }


def write_status(path: Optional[Path], payload: dict) -> None:
    if not path:
        return
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_auto_submit_experiment(
    dataset_path: Path,
    *,
    max_samples: Optional[int] = None,
    cap: int = 8,
    device_name: str = "cpu",
    strategy_ids: Optional[List[str]] = None,
    include_oracle_fixed: bool = True,
    oracle_only: bool = False,
    existing_strategies: Optional[List[dict]] = None,
    progress_cb=None,
    status_file: Optional[Path] = None,
) -> dict:
    config_path = ROOT / "configs" / "symbol-2layer-8head-768dim.json"
    checkpoint = ensure_checkpoint(ROOT / "checkpoints" / "checkpoint_300")
    device = resolve_device(device_name)
    dataset = load_dataset(dataset_path, max_samples=max_samples)

    policies = make_policies(cap=cap)
    ids = strategy_ids or list(policies.keys())
    t0 = time.time()
    total_steps = (0 if oracle_only else len(ids)) + (1 if include_oracle_fixed else 0)
    step_done = 0

    write_status(
        status_file,
        {
            "running": True,
            "phase": "loading_model",
            "progress": {"done": 0, "total": max(total_steps, 1), "label": "加载模型"},
            "params": {
                "max_samples": max_samples,
                "cap": cap,
                "device": device_name,
                "oracle_only": oracle_only,
            },
        },
    )

    model, tokenizer = load_coconut_model(checkpoint, config_path, device)
    apply_feedback_config(model, {"latent_feedback_scale": 1.0})

    rows: List[dict] = list(existing_strategies or []) if oracle_only else []
    if not oracle_only:
        for i, sid in enumerate(ids):
            if sid not in policies:
                continue
            print(f"[{i + 1}/{len(ids)}] {sid} …", flush=True)
            write_status(
                status_file,
                {
                    "running": True,
                    "phase": "evaluating",
                    "progress": {
                        "done": step_done,
                        "total": max(total_steps, 1),
                        "label": f"策略 {sid} · 0/{len(dataset)}",
                        "strategy_id": sid,
                    },
                    "partial_strategies": rows,
                },
            )

            def sample_cb(done, total, _sid=sid):
                if progress_cb:
                    progress_cb(_sid, done, total)
                write_status(
                    status_file,
                    {
                        "running": True,
                        "phase": "evaluating",
                        "progress": {
                            "done": step_done,
                            "total": max(total_steps, 1),
                            "label": f"策略 {_sid} · {done}/{total}",
                            "strategy_id": _sid,
                        },
                        "partial_strategies": rows,
                    },
                )

            row = evaluate_policy(
                model,
                tokenizer,
                dataset,
                policies[sid],
                device,
                cap=cap,
                progress_cb=sample_cb,
            )
            row["strategy_id"] = sid
            rows.append(row)
            step_done += 1
            print(f"  acc={row['accuracy']*100:.1f}% ({row['correct']}/{row['total']})", flush=True)
            write_status(
                status_file,
                {
                    "running": True,
                    "phase": "evaluating",
                    "progress": {
                        "done": step_done,
                        "total": max(total_steps, 1),
                        "label": f"策略 {sid} 完成 · {row['accuracy']*100:.1f}%",
                        "strategy_id": sid,
                    },
                    "partial_strategies": rows,
                },
            )

    oracle = None
    if include_oracle_fixed:
        print("oracle_fixed sweep …", flush=True)
        write_status(
            status_file,
            {
                "running": True,
                "phase": "oracle_sweep",
                "progress": {
                    "done": step_done,
                    "total": max(total_steps, 1),
                    "label": "oracle 上界扫步 1–8",
                },
                "partial_strategies": rows,
            },
        )
        apply_feedback_config(model, {"latent_feedback_scale": 1.0})

        def oracle_step_cb(step_i, step_total, row):
            write_status(
                status_file,
                {
                    "running": True,
                    "phase": "oracle_sweep",
                    "progress": {
                        "done": step_done,
                        "total": max(total_steps, 1),
                        "strategies_done": len(rows),
                        "strategies_total": len(rows) if oracle_only else len(ids),
                        "oracle_step": step_i,
                        "oracle_step_total": step_total,
                        "label": (
                            f"oracle 扫步 {row.get('n_latent', step_i)}/{step_total}"
                            f" · acc {float(row.get('accuracy', 0)) * 100:.1f}%"
                            if row
                            else f"oracle 扫步 {step_i}/{step_total}"
                        ),
                    },
                    "partial_strategies": rows,
                },
            )

        oracle = oracle_fixed_upper_bound(
            model,
            tokenizer,
            dataset,
            device,
            sweep_progress_cb=oracle_step_cb,
        )
        step_done += 1

    auto_route = next((r for r in rows if r["strategy_id"] == "auto_route"), None)
    auto_zero = next((r for r in rows if r["strategy_id"] == "auto_route_zero"), None)
    fixed3 = next((r for r in rows if r["strategy_id"] == "fixed_3"), None)
    best_row = max(rows, key=lambda r: r["accuracy"]) if rows else None
    blind = auto_route or auto_zero

    summary = {
        "auto_route_accuracy": auto_route["accuracy"] if auto_route else None,
        "auto_route_zero_accuracy": auto_zero["accuracy"] if auto_zero else None,
        "fixed_3_accuracy": fixed3["accuracy"] if fixed3 else None,
        "best_blind_strategy": best_row["strategy_id"] if best_row else None,
        "best_blind_accuracy": best_row["accuracy"] if best_row else None,
        "oracle_fixed_best_n": oracle["best_n_latent"] if oracle else None,
        "oracle_fixed_accuracy": oracle["accuracy"] if oracle else None,
        "gap_to_oracle_pp": round(
            ((oracle["accuracy"] if oracle else 0) - (blind["accuracy"] if blind else 0)) * 100,
            1,
        )
        if blind and oracle
        else None,
    }

    payload = {
        "ok": True,
        "experiment_type": "auto_submit_compare",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_sec": round(time.time() - t0, 2),
        "dataset": str(dataset_path.relative_to(ROOT)),
        "sample_count": len(dataset),
        "cap": cap,
        "device": device_name,
        "summary": summary,
        "strategies": rows,
        "oracle_fixed": oracle,
    }
    write_status(
        status_file,
        {
            "running": False,
            "phase": "done",
            "progress": {"done": step_done, "total": max(total_steps, 1), "label": "完成"},
            "summary": summary,
            "error": None,
        },
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="无标签自动配参 vs baseline 对照实验")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data" / "prosqa_test_graph_4_coconut.json",
    )
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--cap", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "auto_submit_latest.json",
    )
    parser.add_argument("--no-oracle-sweep", action="store_true")
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument(
        "--status-file",
        type=Path,
        default=ROOT / "results" / "auto_submit_status.json",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="已有 partial/full 结果 JSON，仅补跑 oracle 扫步",
    )
    args = parser.parse_args()

    existing = None
    oracle_only = args.oracle_only
    if args.resume_from and args.resume_from.is_file():
        prev = json.loads(args.resume_from.read_text(encoding="utf-8"))
        existing = prev.get("strategies") or []
        oracle_only = True

    payload = run_auto_submit_experiment(
        args.dataset,
        max_samples=args.max_samples,
        cap=args.cap,
        device_name=args.device,
        include_oracle_fixed=not args.no_oracle_sweep,
        oracle_only=oracle_only,
        existing_strategies=existing,
        status_file=args.status_file,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
