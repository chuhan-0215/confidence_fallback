"""Parallel adaptive-stop tracks (experiments 16–20)."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from stop_head import (
    RichStopHead,
    _rich_step_features,
    build_rich_stop_examples_for_samples,
    calibrate_rich_threshold,
    encode_answer_bucket,
    evaluate_rich_stop,
    extract_latent_hidden,
    first_correct_step,
    split_train_val_samples,
    train_rich_stop_head,
)


def blind_depth(sample: dict) -> int:
    from run_auto_submit_experiment import blind_choice_depth

    return blind_choice_depth(sample)


def _timing_row(
    *,
    strategy_id: str,
    strategy_label: str,
    correct: int,
    total: int,
    stop_sum: int,
    stop_hist: Dict[str, int],
    timing_hits: int,
    timing_total: int,
    params: dict,
) -> dict:
    acc = correct / total if total else 0.0
    return {
        "strategy_id": strategy_id,
        "strategy_label": strategy_label,
        "accuracy": round(acc, 4),
        "correct": correct,
        "total": total,
        "mean_stop_n": round(stop_sum / total, 2) if total else 0.0,
        "stop_n_histogram": stop_hist,
        "stop_timing_acc": round(timing_hits / timing_total, 4) if timing_total else None,
        "stop_timing_hits": timing_hits,
        "stop_timing_total": timing_total,
        "params": params,
        "eval_split": "test",
    }


@torch.no_grad()
def evaluate_structure_convergence_stop(
    model,
    tokenizer,
    samples: Sequence[dict],
    *,
    cap: int,
    min_n: int,
    cos_threshold: float,
    patience: int,
    device: torch.device,
    seed: int,
    predict_fn,
    expected_fn,
    build_prompt_fn,
    eval_profile,
    progress_cb=None,
) -> dict:
    """Stop when n >= blind_depth and (hidden converged OR answer stable)."""
    correct = total = stop_sum = timing_hits = timing_total = 0
    stop_hist: Dict[str, int] = {}

    for idx, sample in enumerate(samples):
        expected = expected_fn(sample, eval_profile)
        d = blind_depth(sample)
        first_correct, preds_by_step = first_correct_step(
            model,
            tokenizer,
            sample,
            cap=cap,
            device=device,
            seed=seed + idx * 31,
            predict_fn=predict_fn,
            expected_fn=expected_fn,
            eval_profile=eval_profile,
        )
        prev_hidden = None
        prev_pred = ""
        streak = 0
        stop_n = cap
        final_pred = preds_by_step.get(cap, "")

        for n in range(1, cap + 1):
            final_pred = preds_by_step[n]
            _, streak, _ = _rich_step_features(final_pred, prev_pred, streak)
            prev_pred = final_pred
            prompt = build_prompt_fn(
                sample,
                n,
                seed=seed + idx * 31 + n,
                choice_order=eval_profile.choice_order,
                shuffle_edges=eval_profile.prompt_mode != "fixed_edges",
            )
            input_ids = torch.tensor(
                [tokenizer.encode(prompt, add_special_tokens=False)],
                device=device,
            )
            hidden = extract_latent_hidden(model, input_ids, pass_idx=n - 1).to(device)
            cos_sim = 0.0
            if prev_hidden is not None:
                cos_sim = float(F.cosine_similarity(hidden, prev_hidden, dim=0).item())
            prev_hidden = hidden.clone()
            stop_n = n
            stable = streak >= patience and n >= min_n
            converged = cos_sim >= cos_threshold and n >= max(min_n, d)
            if n >= max(min_n, d) and (converged or stable):
                break

        total += 1
        if final_pred == expected:
            correct += 1
        stop_sum += stop_n
        stop_hist[str(stop_n)] = stop_hist.get(str(stop_n), 0) + 1
        if first_correct is not None:
            timing_total += 1
            if stop_n == first_correct:
                timing_hits += 1
        if progress_cb:
            progress_cb(idx + 1, len(samples))

    return _timing_row(
        strategy_id="structure_convergence_stop",
        strategy_label=(
            f"structure_convergence · d=BFS · cos≥{cos_threshold:.2f} OR stable≥{patience}"
        ),
        correct=correct,
        total=total,
        stop_sum=stop_sum,
        stop_hist=stop_hist,
        timing_hits=timing_hits,
        timing_total=timing_total,
        params={
            "min_n": min_n,
            "cos_threshold": cos_threshold,
            "patience": patience,
            "cap": cap,
            "uses_blind_depth": True,
        },
    )


@torch.no_grad()
def evaluate_convergence_or_stable_stop(
    model,
    tokenizer,
    samples: Sequence[dict],
    *,
    cap: int,
    min_n: int,
    cos_threshold: float,
    patience: int,
    device: torch.device,
    seed: int,
    predict_fn,
    expected_fn,
    build_prompt_fn,
    eval_profile,
    progress_cb=None,
) -> dict:
    """Stop when n >= min_n and (hidden converged OR answer stable) — OR not AND."""
    correct = total = stop_sum = timing_hits = timing_total = 0
    stop_hist: Dict[str, int] = {}

    for idx, sample in enumerate(samples):
        expected = expected_fn(sample, eval_profile)
        first_correct, preds_by_step = first_correct_step(
            model,
            tokenizer,
            sample,
            cap=cap,
            device=device,
            seed=seed + idx * 31,
            predict_fn=predict_fn,
            expected_fn=expected_fn,
            eval_profile=eval_profile,
        )
        prev_hidden = None
        prev_pred = ""
        streak = 0
        stop_n = cap
        final_pred = preds_by_step.get(cap, "")

        for n in range(1, cap + 1):
            final_pred = preds_by_step[n]
            _, streak, _ = _rich_step_features(final_pred, prev_pred, streak)
            prev_pred = final_pred
            prompt = build_prompt_fn(
                sample,
                n,
                seed=seed + idx * 31 + n,
                choice_order=eval_profile.choice_order,
                shuffle_edges=eval_profile.prompt_mode != "fixed_edges",
            )
            input_ids = torch.tensor(
                [tokenizer.encode(prompt, add_special_tokens=False)],
                device=device,
            )
            hidden = extract_latent_hidden(model, input_ids, pass_idx=n - 1).to(device)
            cos_sim = 0.0
            if prev_hidden is not None:
                cos_sim = float(F.cosine_similarity(hidden, prev_hidden, dim=0).item())
            prev_hidden = hidden.clone()
            stop_n = n
            if n >= min_n and (cos_sim >= cos_threshold or streak >= patience):
                break

        total += 1
        if final_pred == expected:
            correct += 1
        stop_sum += stop_n
        stop_hist[str(stop_n)] = stop_hist.get(str(stop_n), 0) + 1
        if first_correct is not None:
            timing_total += 1
            if stop_n == first_correct:
                timing_hits += 1
        if progress_cb:
            progress_cb(idx + 1, len(samples))

    return _timing_row(
        strategy_id="convergence_or_stable_stop",
        strategy_label=f"conv OR stable · cos≥{cos_threshold:.2f} / streak≥{patience}",
        correct=correct,
        total=total,
        stop_sum=stop_sum,
        stop_hist=stop_hist,
        timing_hits=timing_hits,
        timing_total=timing_total,
        params={
            "min_n": min_n,
            "cos_threshold": cos_threshold,
            "patience": patience,
            "cap": cap,
            "mode": "or",
        },
    )


def calibrate_heuristic_stop(
    eval_fn,
    model,
    tokenizer,
    val_samples: Sequence[dict],
    *,
    cap: int,
    min_n: int,
    cos_grid: Sequence[float],
    patience_grid: Sequence[int],
    device: torch.device,
    seed: int,
    predict_fn,
    expected_fn,
    build_prompt_fn,
    eval_profile,
    optimize: str = "balanced",
) -> Tuple[dict, dict]:
    best_score = -1.0
    best_params = {"cos_threshold": 0.95, "patience": 2}
    best_row = None
    trials = []

    for cos_thr in cos_grid:
        for patience in patience_grid:
            row = eval_fn(
                model,
                tokenizer,
                val_samples,
                cap=cap,
                min_n=min_n,
                cos_threshold=cos_thr,
                patience=patience,
                device=device,
                seed=seed,
                predict_fn=predict_fn,
                expected_fn=expected_fn,
                build_prompt_fn=build_prompt_fn,
                eval_profile=eval_profile,
            )
            timing = row.get("stop_timing_acc") or 0.0
            if optimize == "timing":
                score = timing * 5.0 + row["accuracy"]
            elif optimize == "balanced":
                score = row["accuracy"] * 5.0 + timing * 3.0
            else:
                score = row["accuracy"] * 10.0 + timing
            trials.append(
                {
                    "cos_threshold": cos_thr,
                    "patience": patience,
                    "accuracy": row["accuracy"],
                    "stop_timing_acc": timing,
                    "score": round(score, 4),
                }
            )
            if score > best_score + 1e-6:
                best_score = score
                best_params = {"cos_threshold": cos_thr, "patience": patience}
                best_row = row

    calibration = {
        **best_params,
        "optimize": optimize,
        "val_accuracy": best_row["accuracy"] if best_row else None,
        "val_stop_timing_acc": best_row.get("stop_timing_acc") if best_row else None,
        "trials": trials,
    }
    return best_params, calibration


class DeltaStopHead(nn.Module):
    """Predict adjustment {-1,0,+1} on blind BFS depth."""

    def __init__(self, hidden_dim: int = 768, max_steps: int = 8, dropout: float = 0.15):
        super().__init__()
        self.step_emb = nn.Embedding(max_steps + 1, 32)
        self.answer_emb = nn.Embedding(64, 16)
        layers: List[nn.Module] = [
            nn.Linear(hidden_dim + 32 + 16 + 1, 128),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(128, 3))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        hidden: torch.Tensor,
        step: torch.Tensor,
        answer_bucket: torch.Tensor,
        blind_d_norm: torch.Tensor,
    ) -> torch.Tensor:
        step_vec = self.step_emb(step)
        ans_vec = self.answer_emb(answer_bucket)
        x = torch.cat([hidden, step_vec, ans_vec, blind_d_norm.unsqueeze(-1)], dim=-1)
        return self.net(x)


def _delta_class(first_correct: Optional[int], d: int, cap: int) -> int:
    if first_correct is None:
        return 1
    raw = int(first_correct) - int(d)
    raw = max(-1, min(1, raw))
    return raw + 1


def build_delta_examples(
    model,
    tokenizer,
    samples: Sequence[dict],
    *,
    cap: int,
    device: torch.device,
    seed: int,
    predict_fn,
    expected_fn,
    build_prompt_fn,
    eval_profile,
    progress_cb=None,
) -> List[dict]:
    rows: List[dict] = []
    for idx, sample in enumerate(samples):
        d = blind_depth(sample)
        step_n = max(1, min(d, cap))
        first_correct, preds_by_step = first_correct_step(
            model,
            tokenizer,
            sample,
            cap=cap,
            device=device,
            seed=seed + idx * 31,
            predict_fn=predict_fn,
            expected_fn=expected_fn,
            eval_profile=eval_profile,
        )
        pred = preds_by_step.get(step_n, "")
        prompt = build_prompt_fn(
            sample,
            step_n,
            seed=seed + idx * 31 + step_n,
            choice_order=eval_profile.choice_order,
            shuffle_edges=eval_profile.prompt_mode != "fixed_edges",
        )
        input_ids = torch.tensor(
            [tokenizer.encode(prompt, add_special_tokens=False)],
            device=device,
        )
        hidden = extract_latent_hidden(model, input_ids, pass_idx=step_n - 1).cpu()
        rows.append(
            {
                "hidden": hidden,
                "step": step_n,
                "answer_bucket": encode_answer_bucket(pred),
                "blind_d": d,
                "delta_class": _delta_class(first_correct, d, cap),
                "sample_idx": idx,
            }
        )
        if progress_cb:
            progress_cb(idx + 1, len(samples))
    return rows


def train_delta_stop_head(
    train_examples: Sequence[dict],
    val_examples: Sequence[dict],
    *,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    early_stop_patience: int = 8,
) -> Tuple[DeltaStopHead, dict]:
    if not train_examples:
        raise ValueError("no delta training examples")

    def to_tensors(examples):
        h = torch.stack([ex["hidden"] for ex in examples]).to(device)
        s = torch.tensor([ex["step"] for ex in examples], dtype=torch.long, device=device)
        a = torch.tensor([ex["answer_bucket"] for ex in examples], dtype=torch.long, device=device)
        d = torch.tensor([ex["blind_d"] / 8.0 for ex in examples], dtype=torch.float32, device=device)
        y = torch.tensor([ex["delta_class"] for ex in examples], dtype=torch.long, device=device)
        return h, s, a, d, y

    train_tensors = to_tensors(train_examples)
    val_tensors = to_tensors(val_examples) if val_examples else None

    head = DeltaStopHead().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = math.inf
    best_state = None
    stale = 0
    stopped = epochs
    history = []

    for epoch in range(epochs):
        head.train()
        h, s, a, d, y = train_tensors
        opt.zero_grad()
        logits = head(h, s, a, d)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        train_loss = float(loss.item())

        val_loss = None
        if val_tensors is not None:
            head.eval()
            with torch.no_grad():
                vh, vs, va, vd, vy = val_tensors
                vlogits = head(vh, vs, va, vd)
                val_loss = float(F.cross_entropy(vlogits, vy).item())
            if val_loss + 1e-5 < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= early_stop_patience:
                stopped = epoch + 1
                break
        history.append({"epoch": epoch + 1, "train_loss": round(train_loss, 4), "val_loss": val_loss})

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        h, s, a, d, y = train_tensors
        preds = head(h, s, a, d).argmax(dim=-1)
        acc = float((preds == y).float().mean().item())

    metrics = {
        "train_rows": len(train_examples),
        "val_rows": len(val_examples),
        "train_acc": round(acc, 4),
        "epochs_requested": epochs,
        "epochs_ran": stopped,
        "best_val_loss": round(best_val, 4) if val_examples else None,
        "history_tail": history[-3:],
    }
    return head, metrics


@torch.no_grad()
def evaluate_delta_route_stop(
    head: DeltaStopHead,
    model,
    tokenizer,
    samples: Sequence[dict],
    *,
    cap: int,
    min_n: int,
    device: torch.device,
    seed: int,
    predict_fn,
    expected_fn,
    build_prompt_fn,
    eval_profile,
    progress_cb=None,
) -> dict:
    head.eval()
    correct = total = stop_sum = timing_hits = timing_total = 0
    stop_hist: Dict[str, int] = {}

    for idx, sample in enumerate(samples):
        expected = expected_fn(sample, eval_profile)
        d = blind_depth(sample)
        first_correct, preds_by_step = first_correct_step(
            model,
            tokenizer,
            sample,
            cap=cap,
            device=device,
            seed=seed + idx * 31,
            predict_fn=predict_fn,
            expected_fn=expected_fn,
            eval_profile=eval_profile,
        )
        step_n = max(1, min(d, cap))
        pred = preds_by_step.get(step_n, "")
        prompt = build_prompt_fn(
            sample,
            step_n,
            seed=seed + idx * 31 + step_n,
            choice_order=eval_profile.choice_order,
            shuffle_edges=eval_profile.prompt_mode != "fixed_edges",
        )
        input_ids = torch.tensor(
            [tokenizer.encode(prompt, add_special_tokens=False)],
            device=device,
        )
        hidden = extract_latent_hidden(model, input_ids, pass_idx=step_n - 1).to(device)
        logits = head(
            hidden.unsqueeze(0),
            torch.tensor([step_n], device=device),
            torch.tensor([encode_answer_bucket(pred)], device=device),
            torch.tensor([d / 8.0], device=device),
        )
        delta = int(logits.argmax(dim=-1).item()) - 1
        stop_n = max(min_n, min(cap, d + delta))
        final_pred = preds_by_step.get(stop_n, preds_by_step.get(cap, ""))

        total += 1
        if final_pred == expected:
            correct += 1
        stop_sum += stop_n
        stop_hist[str(stop_n)] = stop_hist.get(str(stop_n), 0) + 1
        if first_correct is not None:
            timing_total += 1
            if stop_n == first_correct:
                timing_hits += 1
        if progress_cb:
            progress_cb(idx + 1, len(samples))

    return _timing_row(
        strategy_id="delta_route_stop",
        strategy_label="delta_route · BFS深度 + Δ∈{-1,0,+1}",
        correct=correct,
        total=total,
        stop_sum=stop_sum,
        stop_hist=stop_hist,
        timing_hits=timing_hits,
        timing_total=timing_total,
        params={"min_n": min_n, "cap": cap, "head": "delta"},
    )


def run_track_first_correct_timing(
    model,
    tokenizer,
    train_sub,
    val_sub,
    test_set,
    *,
    cap,
    stop_min_n,
    device,
    profile,
    predict_fn,
    expected_fn,
    build_prompt_fn,
    train_epochs,
    status_cb=None,
    progress_cb=None,
):
    if status_cb:
        status_cb("labeling_train", 1, f"标注 first_correct train · 0/{len(train_sub)}")

    def label_cb(done, total):
        if status_cb:
            status_cb("labeling_train", 1, f"标注 first_correct train · {done}/{total}")

    train_examples = build_rich_stop_examples_for_samples(
        model,
        tokenizer,
        train_sub,
        cap=cap,
        device=device,
        seed=42,
        predict_fn=predict_fn,
        expected_fn=expected_fn,
        build_prompt_fn=build_prompt_fn,
        eval_profile=profile,
        progress_cb=label_cb,
        label_mode="first_correct",
    )
    val_examples = build_rich_stop_examples_for_samples(
        model,
        tokenizer,
        val_sub,
        cap=cap,
        device=device,
        seed=142,
        predict_fn=predict_fn,
        expected_fn=expected_fn,
        build_prompt_fn=build_prompt_fn,
        eval_profile=profile,
        label_mode="first_correct",
    ) if val_sub else []

    if status_cb:
        status_cb("training_stop_head", 2, "训练 first_correct RichStopHead")
    head, train_metrics = train_rich_stop_head(
        train_examples,
        val_examples,
        epochs=train_epochs,
        device=device,
    )
    if status_cb:
        status_cb("calibrating_threshold", 3, "val 优先 timing 阈值校准")
    calibrated_threshold, calibration = calibrate_rich_threshold(
        head,
        model,
        tokenizer,
        val_sub,
        cap=cap,
        min_n=stop_min_n,
        device=device,
        seed=77,
        predict_fn=predict_fn,
        expected_fn=expected_fn,
        build_prompt_fn=build_prompt_fn,
        eval_profile=profile,
        optimize="timing",
    )
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
        predict_fn=predict_fn,
        expected_fn=expected_fn,
        build_prompt_fn=build_prompt_fn,
        eval_profile=profile,
        progress_cb=progress_cb,
    )
    row["strategy_id"] = "first_correct_timing_stop"
    row["strategy_label"] = f"first_correct_timing · thr={calibrated_threshold:.2f} · optimize=timing"
    row["params"]["label_mode"] = "first_correct"
    row["params"]["optimize"] = "timing"
    return head, train_metrics, calibration, calibrated_threshold, row


@torch.no_grad()
def evaluate_route_gated_correctness_stop(
    head,
    model,
    tokenizer,
    samples: Sequence[dict],
    *,
    cap: int,
    min_n: int,
    threshold: float,
    device: torch.device,
    seed: int,
    predict_fn,
    expected_fn,
    build_prompt_fn,
    eval_profile,
    progress_cb=None,
) -> dict:
    """At least blind_depth steps; stop when is_correct head fires (v4-style)."""
    head.eval()
    correct = total = stop_sum = timing_hits = timing_total = 0
    stop_hist: Dict[str, int] = {}

    for idx, sample in enumerate(samples):
        expected = expected_fn(sample, eval_profile)
        d = blind_depth(sample)
        first_correct, preds_by_step = first_correct_step(
            model,
            tokenizer,
            sample,
            cap=cap,
            device=device,
            seed=seed + idx * 31,
            predict_fn=predict_fn,
            expected_fn=expected_fn,
            eval_profile=eval_profile,
        )
        floor_n = max(min_n, min(d, cap))
        stop_n = cap
        final_pred = preds_by_step.get(cap, "")
        prev_pred = ""
        streak = 0
        fired = False

        for n in range(1, cap + 1):
            final_pred = preds_by_step[n]
            answer_bucket, streak, changed = _rich_step_features(final_pred, prev_pred, streak)
            prev_pred = final_pred
            if n < floor_n:
                continue
            prompt = build_prompt_fn(
                sample,
                n,
                seed=seed + idx * 31 + n,
                choice_order=eval_profile.choice_order,
                shuffle_edges=eval_profile.prompt_mode != "fixed_edges",
            )
            input_ids = torch.tensor(
                [tokenizer.encode(prompt, add_special_tokens=False)],
                device=device,
            )
            hidden = extract_latent_hidden(model, input_ids, pass_idx=n - 1).to(device)
            step_t = torch.tensor([n], dtype=torch.long, device=device)
            ans_t = torch.tensor([answer_bucket], dtype=torch.long, device=device)
            streak_t = torch.tensor([streak], dtype=torch.long, device=device)
            changed_t = torch.tensor([changed], dtype=torch.float32, device=device)
            prob = torch.sigmoid(
                head(hidden.unsqueeze(0), step_t, ans_t, streak_t, changed_t)
            ).item()
            stop_n = n
            if prob >= threshold:
                fired = True
                break

        if not fired:
            stop_n = floor_n
            final_pred = preds_by_step.get(stop_n, final_pred)

        total += 1
        if final_pred == expected:
            correct += 1
        stop_sum += stop_n
        stop_hist[str(stop_n)] = stop_hist.get(str(stop_n), 0) + 1
        if first_correct is not None:
            timing_total += 1
            if stop_n == first_correct:
                timing_hits += 1
        if progress_cb:
            progress_cb(idx + 1, len(samples))

    return _timing_row(
        strategy_id="route_gated_correctness_stop",
        strategy_label=f"route_gated · n≥blind_depth ∧ head · thr={threshold:.2f}",
        correct=correct,
        total=total,
        stop_sum=stop_sum,
        stop_hist=stop_hist,
        timing_hits=timing_hits,
        timing_total=timing_total,
        params={
            "min_n": min_n,
            "threshold": threshold,
            "cap": cap,
            "uses_blind_depth_floor": True,
            "head": "rich_v4",
        },
    )


def calibrate_route_gated_threshold(
    head,
    model,
    tokenizer,
    val_samples: Sequence[dict],
    *,
    cap: int,
    min_n: int,
    device: torch.device,
    seed: int,
    predict_fn,
    expected_fn,
    build_prompt_fn,
    eval_profile,
    optimize: str = "balanced",
) -> Tuple[float, dict]:
    if not val_samples:
        return 0.55, {"reason": "empty_val", "threshold": 0.55}

    grid = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    best_threshold = 0.55
    best_score = -1.0
    best_row = None
    trials = []

    for thr in grid:
        row = evaluate_route_gated_correctness_stop(
            head,
            model,
            tokenizer,
            val_samples,
            cap=cap,
            min_n=min_n,
            threshold=thr,
            device=device,
            seed=seed,
            predict_fn=predict_fn,
            expected_fn=expected_fn,
            build_prompt_fn=build_prompt_fn,
            eval_profile=eval_profile,
        )
        timing = row.get("stop_timing_acc") or 0.0
        if optimize == "accuracy":
            score = row["accuracy"] * 10.0 + timing
        elif optimize == "balanced":
            score = row["accuracy"] * 5.0 + timing * 3.0
        else:
            score = timing * 2.0 + row["accuracy"]
        trials.append(
            {
                "threshold": thr,
                "accuracy": row["accuracy"],
                "stop_timing_acc": timing,
                "score": round(score, 4),
            }
        )
        if score > best_score + 1e-6:
            best_score = score
            best_threshold = thr
            best_row = row

    return best_threshold, {
        "threshold": best_threshold,
        "optimize": optimize,
        "val_accuracy": best_row["accuracy"] if best_row else None,
        "val_stop_timing_acc": best_row.get("stop_timing_acc") if best_row else None,
        "trials": trials,
    }


def _calibrate_route_gated_generic(
    eval_fn,
    head,
    model,
    tokenizer,
    val_samples: Sequence[dict],
    *,
    cap: int,
    min_n: int,
    device: torch.device,
    seed: int,
    predict_fn,
    expected_fn,
    build_prompt_fn,
    eval_profile,
    optimize: str = "balanced",
) -> Tuple[float, dict]:
    if not val_samples:
        return 0.55, {"reason": "empty_val", "threshold": 0.55, "optimize": optimize}

    grid = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    best_threshold = 0.55
    best_score = -1.0
    best_row = None
    trials = []

    for thr in grid:
        row = eval_fn(
            head,
            model,
            tokenizer,
            val_samples,
            cap=cap,
            min_n=min_n,
            threshold=thr,
            device=device,
            seed=seed,
            predict_fn=predict_fn,
            expected_fn=expected_fn,
            build_prompt_fn=build_prompt_fn,
            eval_profile=eval_profile,
        )
        timing = row.get("stop_timing_acc") or 0.0
        if optimize == "accuracy":
            score = row["accuracy"] * 10.0 + timing
        elif optimize == "balanced":
            score = row["accuracy"] * 5.0 + timing * 3.0
        else:
            score = timing * 2.0 + row["accuracy"]
        trials.append(
            {
                "threshold": thr,
                "accuracy": row["accuracy"],
                "stop_timing_acc": timing,
                "score": round(score, 4),
            }
        )
        if score > best_score + 1e-6:
            best_score = score
            best_threshold = thr
            best_row = row

    return best_threshold, {
        "threshold": best_threshold,
        "optimize": optimize,
        "val_accuracy": best_row["accuracy"] if best_row else None,
        "val_stop_timing_acc": best_row.get("stop_timing_acc") if best_row else None,
        "trials": trials,
    }


@torch.no_grad()
def evaluate_route_gated_autoroute_cap_stop(
    head,
    model,
    tokenizer,
    samples: Sequence[dict],
    *,
    cap: int,
    min_n: int,
    threshold: float,
    device: torch.device,
    seed: int,
    predict_fn,
    expected_fn,
    build_prompt_fn,
    eval_profile,
    progress_cb=None,
) -> dict:
    """n≥blind_depth; head may stop early; never exceed auto_route depth d."""
    head.eval()
    correct = total = stop_sum = timing_hits = timing_total = 0
    stop_hist: Dict[str, int] = {}

    for idx, sample in enumerate(samples):
        expected = expected_fn(sample, eval_profile)
        d = blind_depth(sample)
        auto_n = min(d, cap)
        floor_n = max(min_n, auto_n)
        first_correct, preds_by_step = first_correct_step(
            model,
            tokenizer,
            sample,
            cap=cap,
            device=device,
            seed=seed + idx * 31,
            predict_fn=predict_fn,
            expected_fn=expected_fn,
            eval_profile=eval_profile,
        )
        stop_n = auto_n
        final_pred = preds_by_step.get(auto_n, preds_by_step.get(cap, ""))
        prev_pred = ""
        streak = 0

        for n in range(floor_n, auto_n + 1):
            final_pred = preds_by_step[n]
            answer_bucket, streak, changed = _rich_step_features(final_pred, prev_pred, streak)
            prev_pred = final_pred
            prompt = build_prompt_fn(
                sample,
                n,
                seed=seed + idx * 31 + n,
                choice_order=eval_profile.choice_order,
                shuffle_edges=eval_profile.prompt_mode != "fixed_edges",
            )
            input_ids = torch.tensor(
                [tokenizer.encode(prompt, add_special_tokens=False)],
                device=device,
            )
            hidden = extract_latent_hidden(model, input_ids, pass_idx=n - 1).to(device)
            step_t = torch.tensor([n], dtype=torch.long, device=device)
            ans_t = torch.tensor([answer_bucket], dtype=torch.long, device=device)
            streak_t = torch.tensor([streak], dtype=torch.long, device=device)
            changed_t = torch.tensor([changed], dtype=torch.float32, device=device)
            prob = torch.sigmoid(
                head(hidden.unsqueeze(0), step_t, ans_t, streak_t, changed_t)
            ).item()
            stop_n = n
            if prob >= threshold:
                break

        total += 1
        if final_pred == expected:
            correct += 1
        stop_sum += stop_n
        stop_hist[str(stop_n)] = stop_hist.get(str(stop_n), 0) + 1
        if first_correct is not None:
            timing_total += 1
            if stop_n == first_correct:
                timing_hits += 1
        if progress_cb:
            progress_cb(idx + 1, len(samples))

    return _timing_row(
        strategy_id="route_gated_autoroute_cap_stop",
        strategy_label=f"route_gated_cap · n∈[d,d] head · thr={threshold:.2f}",
        correct=correct,
        total=total,
        stop_sum=stop_sum,
        stop_hist=stop_hist,
        timing_hits=timing_hits,
        timing_total=timing_total,
        params={
            "min_n": min_n,
            "threshold": threshold,
            "cap": cap,
            "uses_blind_depth_floor": True,
            "uses_autoroute_cap": True,
            "head": "rich_v4",
        },
    )


def calibrate_route_gated_autoroute_cap_threshold(
    head,
    model,
    tokenizer,
    val_samples: Sequence[dict],
    *,
    cap: int,
    min_n: int,
    device: torch.device,
    seed: int,
    predict_fn,
    expected_fn,
    build_prompt_fn,
    eval_profile,
    optimize: str = "balanced",
) -> Tuple[float, dict]:
    return _calibrate_route_gated_generic(
        evaluate_route_gated_autoroute_cap_stop,
        head,
        model,
        tokenizer,
        val_samples,
        cap=cap,
        min_n=min_n,
        device=device,
        seed=seed,
        predict_fn=predict_fn,
        expected_fn=expected_fn,
        build_prompt_fn=build_prompt_fn,
        eval_profile=eval_profile,
        optimize=optimize,
    )


@torch.no_grad()
def evaluate_first_correct_gated_stop(
    model,
    tokenizer,
    samples: Sequence[dict],
    *,
    cap: int,
    min_n: int,
    device: torch.device,
    seed: int,
    predict_fn,
    expected_fn,
    build_prompt_fn,
    eval_profile,
    progress_cb=None,
) -> dict:
    """Stop at first_correct if n≥blind_depth, else at blind_depth (structure floor)."""
    correct = total = stop_sum = timing_hits = timing_total = 0
    stop_hist: Dict[str, int] = {}

    for idx, sample in enumerate(samples):
        expected = expected_fn(sample, eval_profile)
        d = blind_depth(sample)
        floor_n = max(min_n, min(d, cap))
        first_correct, preds_by_step = first_correct_step(
            model,
            tokenizer,
            sample,
            cap=cap,
            device=device,
            seed=seed + idx * 31,
            predict_fn=predict_fn,
            expected_fn=expected_fn,
            eval_profile=eval_profile,
        )
        if first_correct is not None and first_correct >= floor_n:
            stop_n = first_correct
        else:
            stop_n = floor_n
        final_pred = preds_by_step.get(stop_n, preds_by_step.get(cap, ""))

        total += 1
        if final_pred == expected:
            correct += 1
        stop_sum += stop_n
        stop_hist[str(stop_n)] = stop_hist.get(str(stop_n), 0) + 1
        if first_correct is not None:
            timing_total += 1
            if stop_n == first_correct:
                timing_hits += 1
        if progress_cb:
            progress_cb(idx + 1, len(samples))

    return _timing_row(
        strategy_id="first_correct_gated_stop",
        strategy_label="first_correct_gated · stop@首次答对步 if n≥d else d",
        correct=correct,
        total=total,
        stop_sum=stop_sum,
        stop_hist=stop_hist,
        timing_hits=timing_hits,
        timing_total=timing_total,
        params={
            "min_n": min_n,
            "cap": cap,
            "uses_blind_depth_floor": True,
            "label_mode": "first_correct_gated",
        },
    )


def _first_correct_stop_n(
    *,
    first_correct: Optional[int],
    floor_n: int,
    auto_n: int,
    cap_n: int,
) -> int:
    if first_correct is not None and first_correct >= floor_n:
        return min(first_correct, cap_n)
    return auto_n


@torch.no_grad()
def evaluate_first_correct_capped_stop(
    model,
    tokenizer,
    samples: Sequence[dict],
    *,
    cap: int,
    min_n: int,
    device: torch.device,
    seed: int,
    predict_fn,
    expected_fn,
    build_prompt_fn,
    eval_profile,
    progress_cb=None,
) -> dict:
    """Exp24 capped at auto_route depth d (never stop after d even if fc later)."""
    correct = total = stop_sum = timing_hits = timing_total = 0
    stop_hist: Dict[str, int] = {}

    for idx, sample in enumerate(samples):
        expected = expected_fn(sample, eval_profile)
        d = blind_depth(sample)
        auto_n = min(d, cap)
        floor_n = max(min_n, auto_n)
        first_correct, preds_by_step = first_correct_step(
            model,
            tokenizer,
            sample,
            cap=cap,
            device=device,
            seed=seed + idx * 31,
            predict_fn=predict_fn,
            expected_fn=expected_fn,
            eval_profile=eval_profile,
        )
        stop_n = _first_correct_stop_n(
            first_correct=first_correct,
            floor_n=floor_n,
            auto_n=auto_n,
            cap_n=auto_n,
        )
        final_pred = preds_by_step.get(stop_n, preds_by_step.get(cap, ""))

        total += 1
        if final_pred == expected:
            correct += 1
        stop_sum += stop_n
        stop_hist[str(stop_n)] = stop_hist.get(str(stop_n), 0) + 1
        if first_correct is not None:
            timing_total += 1
            if stop_n == first_correct:
                timing_hits += 1
        if progress_cb:
            progress_cb(idx + 1, len(samples))

    return _timing_row(
        strategy_id="first_correct_capped_stop",
        strategy_label="first_correct_capped · min(fc,d) if fc≥d else d",
        correct=correct,
        total=total,
        stop_sum=stop_sum,
        stop_hist=stop_hist,
        timing_hits=timing_hits,
        timing_total=timing_total,
        params={
            "min_n": min_n,
            "cap": cap,
            "uses_blind_depth_floor": True,
            "uses_autoroute_cap": True,
            "label_mode": "first_correct_capped",
        },
    )


@torch.no_grad()
def evaluate_hop_plus_one_capped_stop(
    model,
    tokenizer,
    samples: Sequence[dict],
    *,
    cap: int,
    min_n: int,
    device: torch.device,
    seed: int,
    predict_fn,
    expected_fn,
    build_prompt_fn,
    eval_profile,
    progress_cb=None,
) -> dict:
    """4-hop may use d+1 cap; 3-hop capped at d."""
    correct = total = stop_sum = timing_hits = timing_total = 0
    stop_hist: Dict[str, int] = {}

    for idx, sample in enumerate(samples):
        expected = expected_fn(sample, eval_profile)
        d = blind_depth(sample)
        auto_n = min(d, cap)
        floor_n = max(min_n, auto_n)
        cap_n = min(auto_n + 1, cap) if d >= 4 else auto_n
        first_correct, preds_by_step = first_correct_step(
            model,
            tokenizer,
            sample,
            cap=cap,
            device=device,
            seed=seed + idx * 31,
            predict_fn=predict_fn,
            expected_fn=expected_fn,
            eval_profile=eval_profile,
        )
        stop_n = _first_correct_stop_n(
            first_correct=first_correct,
            floor_n=floor_n,
            auto_n=auto_n,
            cap_n=cap_n,
        )
        final_pred = preds_by_step.get(stop_n, preds_by_step.get(cap, ""))

        total += 1
        if final_pred == expected:
            correct += 1
        stop_sum += stop_n
        stop_hist[str(stop_n)] = stop_hist.get(str(stop_n), 0) + 1
        if first_correct is not None:
            timing_total += 1
            if stop_n == first_correct:
                timing_hits += 1
        if progress_cb:
            progress_cb(idx + 1, len(samples))

    return _timing_row(
        strategy_id="hop_plus_one_capped_stop",
        strategy_label="hop_plus_one · 3跳≤d · 4跳≤d+1 · min(fc,cap)",
        correct=correct,
        total=total,
        stop_sum=stop_sum,
        stop_hist=stop_hist,
        timing_hits=timing_hits,
        timing_total=timing_total,
        params={
            "min_n": min_n,
            "cap": cap,
            "uses_blind_depth_floor": True,
            "hop_plus_one_for_4": True,
            "label_mode": "hop_plus_one_capped",
        },
    )


def compute_hop_breakdown(
    row: dict,
    samples: Sequence[dict],
    *,
    stop_ns: List[int],
    first_corrects: List[Optional[int]],
) -> dict:
    """Per blind_depth bucket metrics for heuristic stops."""
    buckets: Dict[int, dict] = {}
    for sample, stop_n, fc in zip(samples, stop_ns, first_corrects):
        d = blind_depth(sample)
        b = buckets.setdefault(
            d,
            {"depth": d, "total": 0, "correct": 0, "timing_hits": 0, "timing_total": 0},
        )
        b["total"] += 1
        # caller passes whether this sample was correct separately if needed
        if fc is not None:
            b["timing_total"] += 1
            if stop_n == fc:
                b["timing_hits"] += 1
    out = []
    for d in sorted(buckets):
        b = buckets[d]
        t = b["timing_total"]
        out.append(
            {
                "blind_depth": d,
                "total": b["total"],
                "timing_acc": round(b["timing_hits"] / t, 4) if t else None,
                "timing_hits": b["timing_hits"],
                "timing_total": t,
            }
        )
    return {"by_blind_depth": out}


@torch.no_grad()
def evaluate_first_correct_gated_with_breakdown(
    model,
    tokenizer,
    samples: Sequence[dict],
    *,
    cap: int,
    min_n: int,
    device: torch.device,
    seed: int,
    predict_fn,
    expected_fn,
    build_prompt_fn,
    eval_profile,
    progress_cb=None,
) -> Tuple[dict, dict]:
    """Same as exp24 with per-hop accuracy/timing breakdown."""
    correct = total = stop_sum = timing_hits = timing_total = 0
    stop_hist: Dict[str, int] = {}
    hop_buckets: Dict[int, dict] = {}

    for idx, sample in enumerate(samples):
        expected = expected_fn(sample, eval_profile)
        d = blind_depth(sample)
        floor_n = max(min_n, min(d, cap))
        first_correct, preds_by_step = first_correct_step(
            model,
            tokenizer,
            sample,
            cap=cap,
            device=device,
            seed=seed + idx * 31,
            predict_fn=predict_fn,
            expected_fn=expected_fn,
            eval_profile=eval_profile,
        )
        if first_correct is not None and first_correct >= floor_n:
            stop_n = first_correct
        else:
            stop_n = floor_n
        final_pred = preds_by_step.get(stop_n, preds_by_step.get(cap, ""))
        is_correct = final_pred == expected

        total += 1
        if is_correct:
            correct += 1
        stop_sum += stop_n
        stop_hist[str(stop_n)] = stop_hist.get(str(stop_n), 0) + 1
        if first_correct is not None:
            timing_total += 1
            if stop_n == first_correct:
                timing_hits += 1

        hb = hop_buckets.setdefault(
            d,
            {
                "blind_depth": d,
                "total": 0,
                "correct": 0,
                "timing_hits": 0,
                "timing_total": 0,
                "mean_stop_sum": 0.0,
            },
        )
        hb["total"] += 1
        hb["correct"] += int(is_correct)
        hb["mean_stop_sum"] += stop_n
        if first_correct is not None:
            hb["timing_total"] += 1
            if stop_n == first_correct:
                hb["timing_hits"] += 1

        if progress_cb:
            progress_cb(idx + 1, len(samples))

    row = _timing_row(
        strategy_id="first_correct_gated_hop_report",
        strategy_label="first_correct_gated · 同实验24 + 分跳数报告",
        correct=correct,
        total=total,
        stop_sum=stop_sum,
        stop_hist=stop_hist,
        timing_hits=timing_hits,
        timing_total=timing_total,
        params={
            "min_n": min_n,
            "cap": cap,
            "uses_blind_depth_floor": True,
            "label_mode": "first_correct_gated_hop_report",
        },
    )

    breakdown = []
    for d in sorted(hop_buckets):
        hb = hop_buckets[d]
        t = hb["total"]
        tt = hb["timing_total"]
        breakdown.append(
            {
                "blind_depth": d,
                "total": t,
                "accuracy": round(hb["correct"] / t, 4) if t else None,
                "mean_stop_n": round(hb["mean_stop_sum"] / t, 2) if t else None,
                "timing_acc": round(hb["timing_hits"] / tt, 4) if tt else None,
                "timing_hits": hb["timing_hits"],
                "timing_total": tt,
            }
        )
    return row, {"by_blind_depth": breakdown}


@torch.no_grad()
def evaluate_soft_floor_first_correct_stop(
    model,
    tokenizer,
    samples: Sequence[dict],
    *,
    cap: int,
    min_n: int,
    device: torch.device,
    seed: int,
    predict_fn,
    expected_fn,
    build_prompt_fn,
    eval_profile,
    progress_cb=None,
) -> dict:
    """Stop at first_correct if n≥min_n (no blind_depth floor); else stop at d."""
    correct = total = stop_sum = timing_hits = timing_total = 0
    stop_hist: Dict[str, int] = {}

    for idx, sample in enumerate(samples):
        expected = expected_fn(sample, eval_profile)
        d = blind_depth(sample)
        floor_n = max(min_n, min(d, cap))
        first_correct, preds_by_step = first_correct_step(
            model,
            tokenizer,
            sample,
            cap=cap,
            device=device,
            seed=seed + idx * 31,
            predict_fn=predict_fn,
            expected_fn=expected_fn,
            eval_profile=eval_profile,
        )
        if first_correct is not None and first_correct >= min_n:
            stop_n = first_correct
        else:
            stop_n = floor_n
        final_pred = preds_by_step.get(stop_n, preds_by_step.get(cap, ""))

        total += 1
        if final_pred == expected:
            correct += 1
        stop_sum += stop_n
        stop_hist[str(stop_n)] = stop_hist.get(str(stop_n), 0) + 1
        if first_correct is not None:
            timing_total += 1
            if stop_n == first_correct:
                timing_hits += 1
        if progress_cb:
            progress_cb(idx + 1, len(samples))

    return _timing_row(
        strategy_id="soft_floor_first_correct_stop",
        strategy_label="soft_floor_fc · stop@fc if n≥min_n else d",
        correct=correct,
        total=total,
        stop_sum=stop_sum,
        stop_hist=stop_hist,
        timing_hits=timing_hits,
        timing_total=timing_total,
        params={
            "min_n": min_n,
            "cap": cap,
            "uses_blind_depth_floor": False,
            "label_mode": "soft_floor_first_correct",
        },
    )


@torch.no_grad()
def evaluate_hop_split_first_correct_stop(
    model,
    tokenizer,
    samples: Sequence[dict],
    *,
    cap: int,
    min_n: int,
    split_depth: int,
    device: torch.device,
    seed: int,
    predict_fn,
    expected_fn,
    build_prompt_fn,
    eval_profile,
    progress_cb=None,
) -> dict:
    """d<split: Exp24 strict (fc≥d); d≥split: soft floor (fc≥min_n)."""
    correct = total = stop_sum = timing_hits = timing_total = 0
    stop_hist: Dict[str, int] = {}

    for idx, sample in enumerate(samples):
        expected = expected_fn(sample, eval_profile)
        d = blind_depth(sample)
        floor_n = max(min_n, min(d, cap))
        first_correct, preds_by_step = first_correct_step(
            model,
            tokenizer,
            sample,
            cap=cap,
            device=device,
            seed=seed + idx * 31,
            predict_fn=predict_fn,
            expected_fn=expected_fn,
            eval_profile=eval_profile,
        )
        if d >= split_depth:
            if first_correct is not None and first_correct >= min_n:
                stop_n = first_correct
            else:
                stop_n = floor_n
        elif first_correct is not None and first_correct >= floor_n:
            stop_n = first_correct
        else:
            stop_n = floor_n
        final_pred = preds_by_step.get(stop_n, preds_by_step.get(cap, ""))

        total += 1
        if final_pred == expected:
            correct += 1
        stop_sum += stop_n
        stop_hist[str(stop_n)] = stop_hist.get(str(stop_n), 0) + 1
        if first_correct is not None:
            timing_total += 1
            if stop_n == first_correct:
                timing_hits += 1
        if progress_cb:
            progress_cb(idx + 1, len(samples))

    return _timing_row(
        strategy_id="hop_split_first_correct_stop",
        strategy_label=f"hop_split_fc · d<{split_depth} strict · d≥{split_depth} soft",
        correct=correct,
        total=total,
        stop_sum=stop_sum,
        stop_hist=stop_hist,
        timing_hits=timing_hits,
        timing_total=timing_total,
        params={
            "min_n": min_n,
            "cap": cap,
            "split_depth": split_depth,
            "uses_blind_depth_floor": True,
            "label_mode": "hop_split_first_correct",
        },
    )


@torch.no_grad()
def evaluate_hop_hybrid_fc_v4_stop(
    head,
    model,
    tokenizer,
    samples: Sequence[dict],
    *,
    cap: int,
    min_n: int,
    split_depth: int,
    threshold: float,
    device: torch.device,
    seed: int,
    predict_fn,
    expected_fn,
    build_prompt_fn,
    eval_profile,
    progress_cb=None,
) -> dict:
    """d<split: Exp24 first_correct_gated; d≥split: route_gated v4 head (Exp20)."""
    head.eval()
    correct = total = stop_sum = timing_hits = timing_total = 0
    stop_hist: Dict[str, int] = {}

    for idx, sample in enumerate(samples):
        expected = expected_fn(sample, eval_profile)
        d = blind_depth(sample)
        floor_n = max(min_n, min(d, cap))
        first_correct, preds_by_step = first_correct_step(
            model,
            tokenizer,
            sample,
            cap=cap,
            device=device,
            seed=seed + idx * 31,
            predict_fn=predict_fn,
            expected_fn=expected_fn,
            eval_profile=eval_profile,
        )

        if d < split_depth:
            if first_correct is not None and first_correct >= floor_n:
                stop_n = first_correct
            else:
                stop_n = floor_n
            final_pred = preds_by_step.get(stop_n, preds_by_step.get(cap, ""))
        else:
            stop_n = cap
            final_pred = preds_by_step.get(cap, "")
            prev_pred = ""
            streak = 0
            fired = False

            for n in range(1, cap + 1):
                final_pred = preds_by_step[n]
                answer_bucket, streak, changed = _rich_step_features(final_pred, prev_pred, streak)
                prev_pred = final_pred
                if n < floor_n:
                    continue
                prompt = build_prompt_fn(
                    sample,
                    n,
                    seed=seed + idx * 31 + n,
                    choice_order=eval_profile.choice_order,
                    shuffle_edges=eval_profile.prompt_mode != "fixed_edges",
                )
                input_ids = torch.tensor(
                    [tokenizer.encode(prompt, add_special_tokens=False)],
                    device=device,
                )
                hidden = extract_latent_hidden(model, input_ids, pass_idx=n - 1).to(device)
                step_t = torch.tensor([n], dtype=torch.long, device=device)
                ans_t = torch.tensor([answer_bucket], dtype=torch.long, device=device)
                streak_t = torch.tensor([streak], dtype=torch.long, device=device)
                changed_t = torch.tensor([changed], dtype=torch.float32, device=device)
                prob = torch.sigmoid(
                    head(hidden.unsqueeze(0), step_t, ans_t, streak_t, changed_t)
                ).item()
                stop_n = n
                if prob >= threshold:
                    fired = True
                    break

            if not fired:
                stop_n = floor_n
                final_pred = preds_by_step.get(stop_n, final_pred)

        total += 1
        if final_pred == expected:
            correct += 1
        stop_sum += stop_n
        stop_hist[str(stop_n)] = stop_hist.get(str(stop_n), 0) + 1
        if first_correct is not None:
            timing_total += 1
            if stop_n == first_correct:
                timing_hits += 1
        if progress_cb:
            progress_cb(idx + 1, len(samples))

    return _timing_row(
        strategy_id="hop_hybrid_fc_v4_stop",
        strategy_label=(
            f"hop_hybrid · d<{split_depth} fc_gated · d≥{split_depth} v4 head · thr={threshold:.2f}"
        ),
        correct=correct,
        total=total,
        stop_sum=stop_sum,
        stop_hist=stop_hist,
        timing_hits=timing_hits,
        timing_total=timing_total,
        params={
            "min_n": min_n,
            "threshold": threshold,
            "cap": cap,
            "split_depth": split_depth,
            "uses_blind_depth_floor": True,
            "head": "rich_v4",
            "label_mode": "hop_hybrid_fc_v4",
        },
    )


def _combo_score(row: dict, optimize: str) -> float:
    timing = row.get("stop_timing_acc") or 0.0
    if optimize == "timing":
        return timing * 5.0 + row["accuracy"]
    if optimize == "balanced":
        return row["accuracy"] * 5.0 + timing * 3.0
    return row["accuracy"] * 10.0 + timing


@torch.no_grad()
def evaluate_rich_or_stable_stop(
    head,
    model,
    tokenizer,
    samples: Sequence[dict],
    *,
    cap: int,
    min_n: int,
    threshold: float,
    patience: int,
    device: torch.device,
    seed: int,
    predict_fn,
    expected_fn,
    build_prompt_fn,
    eval_profile,
    progress_cb=None,
) -> dict:
    """Online stop: learned head OR answer-stable streak — no BFS floor."""
    head.eval()
    correct = total = stop_sum = timing_hits = timing_total = 0
    stop_hist: Dict[str, int] = {}

    for idx, sample in enumerate(samples):
        expected = expected_fn(sample, eval_profile)
        first_correct, preds_by_step = first_correct_step(
            model,
            tokenizer,
            sample,
            cap=cap,
            device=device,
            seed=seed + idx * 31,
            predict_fn=predict_fn,
            expected_fn=expected_fn,
            eval_profile=eval_profile,
        )
        stop_n = cap
        final_pred = preds_by_step.get(cap, "")
        prev_pred = ""
        streak = 0

        for n in range(1, cap + 1):
            final_pred = preds_by_step[n]
            answer_bucket, streak, changed = _rich_step_features(final_pred, prev_pred, streak)
            prev_pred = final_pred
            prompt = build_prompt_fn(
                sample,
                n,
                seed=seed + idx * 31 + n,
                choice_order=eval_profile.choice_order,
                shuffle_edges=eval_profile.prompt_mode != "fixed_edges",
            )
            input_ids = torch.tensor(
                [tokenizer.encode(prompt, add_special_tokens=False)],
                device=device,
            )
            hidden = extract_latent_hidden(model, input_ids, pass_idx=n - 1).to(device)
            step_t = torch.tensor([n], dtype=torch.long, device=device)
            ans_t = torch.tensor([answer_bucket], dtype=torch.long, device=device)
            streak_t = torch.tensor([streak], dtype=torch.long, device=device)
            changed_t = torch.tensor([changed], dtype=torch.float32, device=device)
            prob = torch.sigmoid(
                head(hidden.unsqueeze(0), step_t, ans_t, streak_t, changed_t)
            ).item()
            stop_n = n
            stable = streak >= patience
            if n >= min_n and (prob >= threshold or stable):
                break

        total += 1
        if final_pred == expected:
            correct += 1
        stop_sum += stop_n
        stop_hist[str(stop_n)] = stop_hist.get(str(stop_n), 0) + 1
        if first_correct is not None:
            timing_total += 1
            if stop_n == first_correct:
                timing_hits += 1
        if progress_cb:
            progress_cb(idx + 1, len(samples))

    return _timing_row(
        strategy_id="online_rich_or_stable_stop",
        strategy_label=(
            f"online · head∨stable · thr={threshold:.2f} · streak≥{patience} · no BFS"
        ),
        correct=correct,
        total=total,
        stop_sum=stop_sum,
        stop_hist=stop_hist,
        timing_hits=timing_hits,
        timing_total=timing_total,
        params={
            "min_n": min_n,
            "threshold": threshold,
            "patience": patience,
            "cap": cap,
            "head": "rich",
            "mode": "head_or_stable",
            "uses_blind_depth": False,
        },
    )


@torch.no_grad()
def evaluate_rich_or_conv_stop(
    head,
    model,
    tokenizer,
    samples: Sequence[dict],
    *,
    cap: int,
    min_n: int,
    threshold: float,
    cos_threshold: float,
    device: torch.device,
    seed: int,
    predict_fn,
    expected_fn,
    build_prompt_fn,
    eval_profile,
    progress_cb=None,
) -> dict:
    """Online stop: learned head OR hidden-state convergence — no BFS floor."""
    head.eval()
    correct = total = stop_sum = timing_hits = timing_total = 0
    stop_hist: Dict[str, int] = {}

    for idx, sample in enumerate(samples):
        expected = expected_fn(sample, eval_profile)
        first_correct, preds_by_step = first_correct_step(
            model,
            tokenizer,
            sample,
            cap=cap,
            device=device,
            seed=seed + idx * 31,
            predict_fn=predict_fn,
            expected_fn=expected_fn,
            eval_profile=eval_profile,
        )
        stop_n = cap
        final_pred = preds_by_step.get(cap, "")
        prev_pred = ""
        streak = 0
        prev_hidden = None

        for n in range(1, cap + 1):
            final_pred = preds_by_step[n]
            answer_bucket, streak, changed = _rich_step_features(final_pred, prev_pred, streak)
            prev_pred = final_pred
            prompt = build_prompt_fn(
                sample,
                n,
                seed=seed + idx * 31 + n,
                choice_order=eval_profile.choice_order,
                shuffle_edges=eval_profile.prompt_mode != "fixed_edges",
            )
            input_ids = torch.tensor(
                [tokenizer.encode(prompt, add_special_tokens=False)],
                device=device,
            )
            hidden = extract_latent_hidden(model, input_ids, pass_idx=n - 1).to(device)
            cos_sim = 0.0
            if prev_hidden is not None:
                cos_sim = float(F.cosine_similarity(hidden, prev_hidden, dim=0).item())
            prev_hidden = hidden.clone()
            step_t = torch.tensor([n], dtype=torch.long, device=device)
            ans_t = torch.tensor([answer_bucket], dtype=torch.long, device=device)
            streak_t = torch.tensor([streak], dtype=torch.long, device=device)
            changed_t = torch.tensor([changed], dtype=torch.float32, device=device)
            prob = torch.sigmoid(
                head(hidden.unsqueeze(0), step_t, ans_t, streak_t, changed_t)
            ).item()
            stop_n = n
            converged = prev_hidden is not None and cos_sim >= cos_threshold
            if n >= min_n and (prob >= threshold or converged):
                break

        total += 1
        if final_pred == expected:
            correct += 1
        stop_sum += stop_n
        stop_hist[str(stop_n)] = stop_hist.get(str(stop_n), 0) + 1
        if first_correct is not None:
            timing_total += 1
            if stop_n == first_correct:
                timing_hits += 1
        if progress_cb:
            progress_cb(idx + 1, len(samples))

    return _timing_row(
        strategy_id="online_rich_or_conv_stop",
        strategy_label=(
            f"online · head∨conv · thr={threshold:.2f} · cos≥{cos_threshold:.2f} · no BFS"
        ),
        correct=correct,
        total=total,
        stop_sum=stop_sum,
        stop_hist=stop_hist,
        timing_hits=timing_hits,
        timing_total=timing_total,
        params={
            "min_n": min_n,
            "threshold": threshold,
            "cos_threshold": cos_threshold,
            "cap": cap,
            "head": "rich",
            "mode": "head_or_conv",
            "uses_blind_depth": False,
        },
    )


@torch.no_grad()
def evaluate_rich_or_multi_stop(
    head,
    model,
    tokenizer,
    samples: Sequence[dict],
    *,
    cap: int,
    min_n: int,
    threshold: float,
    patience: int,
    cos_threshold: float,
    device: torch.device,
    seed: int,
    predict_fn,
    expected_fn,
    build_prompt_fn,
    eval_profile,
    progress_cb=None,
) -> dict:
    """Online stop: head OR stable OR conv — no BFS, no upfront budget."""
    head.eval()
    correct = total = stop_sum = timing_hits = timing_total = 0
    stop_hist: Dict[str, int] = {}

    for idx, sample in enumerate(samples):
        expected = expected_fn(sample, eval_profile)
        first_correct, preds_by_step = first_correct_step(
            model,
            tokenizer,
            sample,
            cap=cap,
            device=device,
            seed=seed + idx * 31,
            predict_fn=predict_fn,
            expected_fn=expected_fn,
            eval_profile=eval_profile,
        )
        stop_n = cap
        final_pred = preds_by_step.get(cap, "")
        prev_pred = ""
        streak = 0
        prev_hidden = None

        for n in range(1, cap + 1):
            final_pred = preds_by_step[n]
            answer_bucket, streak, changed = _rich_step_features(final_pred, prev_pred, streak)
            prev_pred = final_pred
            prompt = build_prompt_fn(
                sample,
                n,
                seed=seed + idx * 31 + n,
                choice_order=eval_profile.choice_order,
                shuffle_edges=eval_profile.prompt_mode != "fixed_edges",
            )
            input_ids = torch.tensor(
                [tokenizer.encode(prompt, add_special_tokens=False)],
                device=device,
            )
            hidden = extract_latent_hidden(model, input_ids, pass_idx=n - 1).to(device)
            cos_sim = 0.0
            if prev_hidden is not None:
                cos_sim = float(F.cosine_similarity(hidden, prev_hidden, dim=0).item())
            prev_hidden = hidden.clone()
            step_t = torch.tensor([n], dtype=torch.long, device=device)
            ans_t = torch.tensor([answer_bucket], dtype=torch.long, device=device)
            streak_t = torch.tensor([streak], dtype=torch.long, device=device)
            changed_t = torch.tensor([changed], dtype=torch.float32, device=device)
            prob = torch.sigmoid(
                head(hidden.unsqueeze(0), step_t, ans_t, streak_t, changed_t)
            ).item()
            stop_n = n
            stable = streak >= patience
            converged = n > 1 and cos_sim >= cos_threshold
            if n >= min_n and (prob >= threshold or stable or converged):
                break

        total += 1
        if final_pred == expected:
            correct += 1
        stop_sum += stop_n
        stop_hist[str(stop_n)] = stop_hist.get(str(stop_n), 0) + 1
        if first_correct is not None:
            timing_total += 1
            if stop_n == first_correct:
                timing_hits += 1
        if progress_cb:
            progress_cb(idx + 1, len(samples))

    return _timing_row(
        strategy_id="online_rich_or_multi_stop",
        strategy_label=(
            f"online · head∨stable∨conv · thr={threshold:.2f} · p≥{patience} · cos≥{cos_threshold:.2f}"
        ),
        correct=correct,
        total=total,
        stop_sum=stop_sum,
        stop_hist=stop_hist,
        timing_hits=timing_hits,
        timing_total=timing_total,
        params={
            "min_n": min_n,
            "threshold": threshold,
            "patience": patience,
            "cos_threshold": cos_threshold,
            "cap": cap,
            "head": "rich",
            "mode": "head_or_stable_or_conv",
            "uses_blind_depth": False,
        },
    )


def calibrate_rich_combo_stop(
    eval_fn,
    head,
    model,
    tokenizer,
    val_samples: Sequence[dict],
    *,
    cap: int,
    min_n: int,
    device: torch.device,
    seed: int,
    predict_fn,
    expected_fn,
    build_prompt_fn,
    eval_profile,
    param_grid: Dict[str, Sequence],
    optimize: str = "balanced",
) -> Tuple[dict, dict]:
    keys = list(param_grid.keys())
    if not keys:
        raise ValueError("param_grid must not be empty")

    best_score = -1.0
    best_params: dict = {}
    best_row = None
    trials: List[dict] = []

    def _recurse(depth: int, current: dict):
        nonlocal best_score, best_params, best_row
        if depth == len(keys):
            row = eval_fn(
                head,
                model,
                tokenizer,
                val_samples,
                cap=cap,
                min_n=min_n,
                device=device,
                seed=seed,
                predict_fn=predict_fn,
                expected_fn=expected_fn,
                build_prompt_fn=build_prompt_fn,
                eval_profile=eval_profile,
                **current,
            )
            score = _combo_score(row, optimize)
            trial = {**current, "accuracy": row["accuracy"], "stop_timing_acc": row.get("stop_timing_acc"), "score": round(score, 4)}
            trials.append(trial)
            if score > best_score + 1e-6:
                best_score = score
                best_params = dict(current)
                best_row = row
            return
        key = keys[depth]
        for value in param_grid[key]:
            current[key] = value
            _recurse(depth + 1, current)

    _recurse(0, {})
    calibration = {
        **best_params,
        "optimize": optimize,
        "val_accuracy": best_row["accuracy"] if best_row else None,
        "val_stop_timing_acc": best_row.get("stop_timing_acc") if best_row else None,
        "trials": trials,
    }
    return best_params, calibration


def run_track_online_rich_combo(
    model,
    tokenizer,
    train_sub,
    val_sub,
    test_set,
    *,
    track: int,
    eval_fn,
    param_grid: Dict[str, Sequence],
    cap,
    stop_min_n,
    device,
    profile,
    predict_fn,
    expected_fn,
    build_prompt_fn,
    train_epochs,
    optimize: str = "balanced",
    status_cb=None,
    progress_cb=None,
):
    """Train first_correct RichStopHead, calibrate online combo stop on val, eval test."""
    if status_cb:
        status_cb("labeling_train", 1, f"标注 first_correct train · 0/{len(train_sub)}")

    def label_cb(done, total):
        if status_cb:
            status_cb("labeling_train", 1, f"标注 first_correct train · {done}/{total}")

    train_examples = build_rich_stop_examples_for_samples(
        model,
        tokenizer,
        train_sub,
        cap=cap,
        device=device,
        seed=42,
        predict_fn=predict_fn,
        expected_fn=expected_fn,
        build_prompt_fn=build_prompt_fn,
        eval_profile=profile,
        progress_cb=label_cb,
        label_mode="first_correct",
    )
    val_examples = build_rich_stop_examples_for_samples(
        model,
        tokenizer,
        val_sub,
        cap=cap,
        device=device,
        seed=142,
        predict_fn=predict_fn,
        expected_fn=expected_fn,
        build_prompt_fn=build_prompt_fn,
        eval_profile=profile,
        label_mode="first_correct",
    ) if val_sub else []

    if status_cb:
        status_cb("training_stop_head", 2, "训练 online RichStopHead · first_correct")
    head, train_metrics = train_rich_stop_head(
        train_examples,
        val_examples,
        epochs=train_epochs,
        device=device,
    )
    if status_cb:
        status_cb("calibrating", 3, f"val 网格 · {eval_fn.__name__}")
    best_params, calibration = calibrate_rich_combo_stop(
        eval_fn,
        head,
        model,
        tokenizer,
        val_sub,
        cap=cap,
        min_n=stop_min_n,
        device=device,
        seed=77,
        predict_fn=predict_fn,
        expected_fn=expected_fn,
        build_prompt_fn=build_prompt_fn,
        eval_profile=profile,
        param_grid=param_grid,
        optimize=optimize,
    )
    row = eval_fn(
        head,
        model,
        tokenizer,
        test_set,
        cap=cap,
        min_n=stop_min_n,
        device=device,
        seed=99,
        predict_fn=predict_fn,
        expected_fn=expected_fn,
        build_prompt_fn=build_prompt_fn,
        eval_profile=profile,
        progress_cb=progress_cb,
        **best_params,
    )
    row["params"]["label_mode"] = "first_correct"
    row["params"]["optimize"] = optimize
    row["params"]["track"] = track
    return head, train_metrics, calibration, best_params, row
