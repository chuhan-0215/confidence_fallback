from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stop_head.models import LatentStopHead, RichStopHead, StopExample
from stop_head.latent import extract_latent_hidden
from stop_head.examples import first_correct_step
from stop_head.features import _rich_step_features
from stop_head.train import _rich_tensors, focal_bce_with_logits

def calibrate_rich_threshold(
    head: RichStopHead,
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
    thresholds: Optional[Sequence[float]] = None,
    optimize: str = "accuracy",
    min_accuracy: Optional[float] = None,
) -> Tuple[float, dict]:
    if not val_samples:
        return 0.5, {"reason": "empty_val", "threshold": 0.5}

    grid = list(thresholds or [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
    best_threshold = 0.5
    best_score = -1.0
    best_row = None
    trials = []

    for thr in grid:
        row = evaluate_rich_stop(
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
        if min_accuracy is not None and row["accuracy"] + 1e-6 < min_accuracy:
            score = -1.0
        elif optimize == "accuracy":
            score = row["accuracy"] * 10.0 + timing
        elif optimize == "balanced":
            score = row["accuracy"] * 5.0 + timing * 3.0
        elif optimize == "feasible":
            score = timing * 10.0 + row["accuracy"]
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
        "min_accuracy": min_accuracy,
        "val_accuracy": best_row["accuracy"] if best_row else None,
        "val_stop_timing_acc": best_row.get("stop_timing_acc") if best_row else None,
        "trials": trials,
    }


@torch.no_grad()
def evaluate_rich_stop(
    head: RichStopHead,
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
    head.eval()
    correct = 0
    total = 0
    stop_sum = 0
    stop_hist: Dict[str, int] = {}
    timing_hits = 0
    timing_total = 0

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
            step = torch.tensor([n], dtype=torch.long, device=device)
            ans = torch.tensor([answer_bucket], dtype=torch.long, device=device)
            st = torch.tensor([streak], dtype=torch.long, device=device)
            ch = torch.tensor([changed], dtype=torch.float32, device=device)
            prob = torch.sigmoid(head(hidden.unsqueeze(0), step, ans, st, ch)).item()
            stop_n = n
            if n >= min_n and prob >= threshold:
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

    acc = correct / total if total else 0.0
    return {
        "accuracy": round(acc, 4),
        "correct": correct,
        "total": total,
        "mean_stop_n": round(stop_sum / total, 2) if total else 0.0,
        "stop_n_histogram": stop_hist,
        "stop_timing_acc": round(timing_hits / timing_total, 4) if timing_total else None,
        "stop_timing_hits": timing_hits,
        "stop_timing_total": timing_total,
        "params": {"min_n": min_n, "threshold": threshold, "cap": cap, "head": "rich"},
    }


@torch.no_grad()
def evaluate_streak_gated_stop(
    head: RichStopHead,
    model,
    tokenizer,
    samples: Sequence[dict],
    *,
    cap: int,
    min_n: int,
    threshold: float,
    stable_min_n: int = 3,
    patience: int = 2,
    device: torch.device,
    seed: int,
    predict_fn,
    expected_fn,
    build_prompt_fn,
    eval_profile,
    progress_cb=None,
) -> dict:
    """Stop when answer is stable AND head agrees — avoids same-step AND deadlock."""
    head.eval()
    correct = 0
    total = 0
    stop_sum = 0
    stop_hist: Dict[str, int] = {}
    timing_hits = 0
    timing_total = 0

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
        recent: List[str] = []

        for n in range(1, cap + 1):
            final_pred = preds_by_step[n]
            answer_bucket, streak, changed = _rich_step_features(final_pred, prev_pred, streak)
            prev_pred = final_pred
            recent.append(final_pred)

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
            step = torch.tensor([n], dtype=torch.long, device=device)
            ans = torch.tensor([answer_bucket], dtype=torch.long, device=device)
            st = torch.tensor([streak], dtype=torch.long, device=device)
            ch = torch.tensor([changed], dtype=torch.float32, device=device)
            prob = torch.sigmoid(head(hidden.unsqueeze(0), step, ans, st, ch)).item()
            stop_n = n

            stable = (
                n >= stable_min_n
                and len(recent) >= patience
                and all(p == recent[-patience] and p for p in recent[-patience:])
            )
            if stable and n >= min_n and prob >= threshold:
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

    acc = correct / total if total else 0.0
    return {
        "accuracy": round(acc, 4),
        "correct": correct,
        "total": total,
        "mean_stop_n": round(stop_sum / total, 2) if total else 0.0,
        "stop_n_histogram": stop_hist,
        "stop_timing_acc": round(timing_hits / timing_total, 4) if timing_total else None,
        "stop_timing_hits": timing_hits,
        "stop_timing_total": timing_total,
        "params": {
            "min_n": min_n,
            "threshold": threshold,
            "cap": cap,
            "stable_min_n": stable_min_n,
            "patience": patience,
            "mode": "stable_gated_by_head",
        },
    }


def train_stop_head(
    examples: Sequence[StopExample],
    *,
    hidden_dim: int = 768,
    max_steps: int = 8,
    epochs: int = 25,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: torch.device,
) -> Tuple[LatentStopHead, dict]:
    if not examples:
        raise ValueError("no training examples")

    pos = sum(1 for ex in examples if ex.should_stop > 0.5)
    neg = len(examples) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    hiddens = torch.stack([ex.hidden for ex in examples]).to(device)
    steps = torch.tensor([ex.step for ex in examples], dtype=torch.long, device=device)
    labels = torch.tensor([ex.should_stop for ex in examples], dtype=torch.float32, device=device)

    loader = DataLoader(TensorDataset(hiddens, steps, labels), batch_size=batch_size, shuffle=True)
    head = LatentStopHead(hidden_dim=hidden_dim, max_steps=max_steps).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    history = []
    for epoch in range(epochs):
        head.train()
        total_loss = 0.0
        n_batches = 0
        for batch_h, batch_s, batch_y in loader:
            opt.zero_grad()
            logits = head(batch_h, batch_s)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        history.append({"epoch": epoch + 1, "loss": round(avg_loss, 4)})

    head.eval()
    with torch.no_grad():
        logits = head(hiddens, steps)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        acc = float((preds == labels).float().mean().item())
        pos_mask = labels > 0.5
        stop_recall = float((preds[pos_mask] == 1).float().mean().item()) if pos_mask.any() else 0.0

    metrics = {
        "train_rows": len(examples),
        "train_pos": pos,
        "train_neg": neg,
        "train_acc": round(acc, 4),
        "train_stop_recall": round(stop_recall, 4),
        "epochs": epochs,
        "history_tail": history[-3:],
    }
    return head, metrics


def split_train_val_samples(
    samples: Sequence[dict],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    n = len(samples)
    if n < 2:
        return list(samples), []
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng).tolist()
    val_n = max(1, min(n - 1, int(n * val_ratio)))
    val_ids = set(perm[:val_n])
    train_sub = [samples[i] for i in range(n) if i not in val_ids]
    val_sub = [samples[i] for i in val_ids]
    return train_sub, val_sub


def train_stop_head_v2(
    train_examples: Sequence[StopExample],
    val_examples: Sequence[StopExample],
    *,
    hidden_dim: int = 768,
    max_steps: int = 8,
    dropout: float = 0.2,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    early_stop_patience: int = 8,
    device: torch.device,
) -> Tuple[LatentStopHead, dict]:
    if not train_examples:
        raise ValueError("no training examples")

    pos = sum(1 for ex in train_examples if ex.should_stop > 0.5)
    neg = len(train_examples) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_h = torch.stack([ex.hidden for ex in train_examples]).to(device)
    train_s = torch.tensor([ex.step for ex in train_examples], dtype=torch.long, device=device)
    train_y = torch.tensor([ex.should_stop for ex in train_examples], dtype=torch.float32, device=device)
    train_loader = DataLoader(TensorDataset(train_h, train_s, train_y), batch_size=batch_size, shuffle=True)

    val_h = val_s = val_y = None
    if val_examples:
        val_h = torch.stack([ex.hidden for ex in val_examples]).to(device)
        val_s = torch.tensor([ex.step for ex in val_examples], dtype=torch.long, device=device)
        val_y = torch.tensor([ex.should_stop for ex in val_examples], dtype=torch.float32, device=device)

    head = LatentStopHead(hidden_dim=hidden_dim, max_steps=max_steps, dropout=dropout).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)

    history = []
    best_val_loss = math.inf
    best_state = None
    stale_epochs = 0
    stopped_epoch = epochs

    for epoch in range(epochs):
        head.train()
        total_loss = 0.0
        n_batches = 0
        for batch_h, batch_s, batch_y in train_loader:
            opt.zero_grad()
            logits = head(batch_h, batch_s)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1
        avg_train_loss = total_loss / max(n_batches, 1)

        val_loss = None
        if val_h is not None and val_y is not None and val_s is not None:
            head.eval()
            with torch.no_grad():
                val_logits = head(val_h, val_s)
                val_loss = float(loss_fn(val_logits, val_y).item())
            if val_loss + 1e-5 < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= early_stop_patience:
                stopped_epoch = epoch + 1
                break

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(avg_train_loss, 4),
                "val_loss": round(val_loss, 4) if val_loss is not None else None,
            }
        )

    if best_state is not None:
        head.load_state_dict(best_state)

    head.eval()
    with torch.no_grad():
        logits = head(train_h, train_s)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        acc = float((preds == train_y).float().mean().item())
        pos_mask = train_y > 0.5
        stop_recall = float((preds[pos_mask] == 1).float().mean().item()) if pos_mask.any() else 0.0

    metrics = {
        "train_rows": len(train_examples),
        "val_rows": len(val_examples),
        "train_pos": pos,
        "train_neg": neg,
        "train_acc": round(acc, 4),
        "train_stop_recall": round(stop_recall, 4),
        "epochs_requested": epochs,
        "epochs_ran": stopped_epoch,
        "best_val_loss": round(best_val_loss, 4) if val_examples else None,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "history_tail": history[-3:],
    }
    return head, metrics


def calibrate_stop_threshold(
    head: LatentStopHead,
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
    thresholds: Optional[Sequence[float]] = None,
) -> Tuple[float, dict]:
    if not val_samples:
        return 0.5, {"reason": "empty_val", "threshold": 0.5}

    grid = list(thresholds or [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
    best_threshold = 0.5
    best_score = -1.0
    best_row = None
    trials = []

    for thr in grid:
        row = evaluate_trained_stop(
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
        "val_accuracy": best_row["accuracy"] if best_row else None,
        "val_stop_timing_acc": best_row.get("stop_timing_acc") if best_row else None,
        "trials": trials,
    }


@torch.no_grad()
def evaluate_trained_stop(
    head: LatentStopHead,
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
    head.eval()
    correct = 0
    total = 0
    stop_sum = 0
    stop_hist: Dict[str, int] = {}
    timing_hits = 0
    timing_total = 0

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

        for n in range(1, cap + 1):
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
            step = torch.tensor([n], dtype=torch.long, device=device)
            prob = torch.sigmoid(head(hidden.unsqueeze(0), step)).item()
            final_pred = preds_by_step[n]
            stop_n = n
            if n >= min_n and prob >= threshold:
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

    acc = correct / total if total else 0.0
    return {
        "accuracy": round(acc, 4),
        "correct": correct,
        "total": total,
        "mean_stop_n": round(stop_sum / total, 2) if total else 0.0,
        "stop_n_histogram": stop_hist,
        "stop_timing_acc": round(timing_hits / timing_total, 4) if timing_total else None,
        "stop_timing_hits": timing_hits,
        "stop_timing_total": timing_total,
        "params": {"min_n": min_n, "threshold": threshold, "cap": cap},
    }


@torch.no_grad()
def evaluate_hybrid_stop(
    head: LatentStopHead,
    model,
    tokenizer,
    samples: Sequence[dict],
    *,
    cap: int,
    min_n: int,
    threshold: float,
    stable_min_n: int = 3,
    patience: int = 2,
    device: torch.device,
    seed: int,
    predict_fn,
    expected_fn,
    build_prompt_fn,
    eval_profile,
    progress_cb=None,
) -> dict:
    head.eval()
    correct = 0
    total = 0
    stop_sum = 0
    stop_hist: Dict[str, int] = {}
    timing_hits = 0
    timing_total = 0

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
        recent: List[str] = []

        for n in range(1, cap + 1):
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
            step = torch.tensor([n], dtype=torch.long, device=device)
            prob = torch.sigmoid(head(hidden.unsqueeze(0), step)).item()
            final_pred = preds_by_step[n]
            stop_n = n
            recent.append(final_pred)
            head_stop = n >= min_n and prob >= threshold
            stable = (
                n >= stable_min_n
                and len(recent) >= patience
                and all(p == recent[-patience] and p for p in recent[-patience:])
            )
            if head_stop and stable:
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

    acc = correct / total if total else 0.0
    return {
        "accuracy": round(acc, 4),
        "correct": correct,
        "total": total,
        "mean_stop_n": round(stop_sum / total, 2) if total else 0.0,
        "stop_n_histogram": stop_hist,
        "stop_timing_acc": round(timing_hits / timing_total, 4) if timing_total else None,
        "stop_timing_hits": timing_hits,
        "stop_timing_total": timing_total,
        "params": {
            "min_n": min_n,
            "threshold": threshold,
            "cap": cap,
            "stable_min_n": stable_min_n,
            "patience": patience,
            "mode": "head_and_stable",
        },
    }

