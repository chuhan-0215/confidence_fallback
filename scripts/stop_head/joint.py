from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stop_head.models import RichStopHead
from stop_head.latent import extract_latent_hidden, extract_latent_hidden_trainable
from stop_head.examples import first_correct_step, resolve_hybrid_stop_target
from stop_head.features import _rich_step_features
from stop_head.train import focal_bce_with_logits

def configure_coconut_joint_train(model, unfreeze_layers: int = 2) -> dict:
    """Unfreeze the last N GPT-2 blocks for stop-head joint training."""
    for param in model.parameters():
        param.requires_grad = False
    blocks = model.base_causallm.transformer.h
    start = max(0, len(blocks) - unfreeze_layers)
    trainable = []
    for block in blocks[start:]:
        for param in block.parameters():
            param.requires_grad = True
            trainable.append(param)
    return {
        "unfreeze_layers": unfreeze_layers,
        "total_blocks": len(blocks),
        "trainable_param_count": sum(p.numel() for p in trainable),
    }


def build_rich_stop_metadata_for_samples(
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
    label_mode: str = "is_correct",
) -> List[dict]:
    """Per-sample step metadata for joint training (no hidden cache)."""
    rows: List[dict] = []
    for idx, sample in enumerate(samples):
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
        expected = expected_fn(sample, eval_profile)
        if label_mode == "hybrid_stop":
            stop_target = resolve_hybrid_stop_target(
                sample,
                model=model,
                tokenizer=tokenizer,
                cap=cap,
                min_n=2,
                device=device,
                seed=seed + idx * 31,
                predict_fn=predict_fn,
                expected_fn=expected_fn,
                eval_profile=eval_profile,
            )
        else:
            stop_target = first_correct if first_correct is not None else cap
        prev_pred = ""
        streak = 0
        steps = []
        for n in range(1, cap + 1):
            pred = preds_by_step[n]
            if pred and pred == prev_pred:
                streak += 1
            else:
                streak = 1
            changed = 1.0 if prev_pred and pred != prev_pred else 0.0
            prev_pred = pred
            if label_mode == "is_correct":
                should_stop = 1.0 if pred == expected else 0.0
            elif label_mode == "stable_correct":
                should_stop = 1.0 if pred == expected and streak >= 2 else 0.0
            elif label_mode == "earliest_stop":
                fc_n = first_correct if first_correct is not None else cap
                first_ok = next((k for k in range(1, cap + 1) if preds_by_step[k] == expected), cap)
                stop_target = min(fc_n, first_ok)
                should_stop = 1.0 if n == stop_target else 0.0
            elif label_mode == "hybrid_stop":
                should_stop = 1.0 if n == stop_target else 0.0
            else:
                should_stop = 1.0 if n == stop_target else 0.0
            steps.append(
                {
                    "n": n,
                    "answer_bucket": encode_answer_bucket(pred),
                    "streak": streak,
                    "changed": changed,
                    "should_stop": should_stop,
                }
            )
        rows.append({"sample_idx": idx, "sample": sample, "steps": steps})
        if progress_cb:
            progress_cb(idx + 1, len(samples))
    return rows


def _rich_stop_loss_on_metadata(
    head: RichStopHead,
    model,
    tokenizer,
    metadata: Sequence[dict],
    *,
    cap: int,
    device: torch.device,
    seed: int,
    build_prompt_fn,
    eval_profile,
    pos_weight: torch.Tensor,
    focal_gamma: float,
    trainable_hidden: bool,
) -> Tuple[float, int]:
    total_loss = 0.0
    n_steps = 0
    head.eval() if not trainable_hidden else head.train()
    if trainable_hidden:
        model.train()
    else:
        model.eval()

    ctx = torch.enable_grad() if trainable_hidden else torch.no_grad()
    with ctx:
        for item in metadata:
            sample = item["sample"]
            sample_idx = item["sample_idx"]
            for step in item["steps"]:
                n = step["n"]
                prompt = build_prompt_fn(
                    sample,
                    n,
                    seed=seed + sample_idx * 31 + n,
                    choice_order=eval_profile.choice_order,
                    shuffle_edges=eval_profile.prompt_mode != "fixed_edges",
                )
                input_ids = torch.tensor(
                    [tokenizer.encode(prompt, add_special_tokens=False)],
                    device=device,
                )
                if trainable_hidden:
                    hidden = extract_latent_hidden_trainable(model, input_ids, pass_idx=n - 1)
                else:
                    hidden = extract_latent_hidden(model, input_ids, pass_idx=n - 1).to(device)
                step_t = torch.tensor([n], dtype=torch.long, device=device)
                ans_t = torch.tensor([step["answer_bucket"]], dtype=torch.long, device=device)
                streak_t = torch.tensor([step["streak"]], dtype=torch.long, device=device)
                changed_t = torch.tensor([step["changed"]], dtype=torch.float32, device=device)
                target = torch.tensor([step["should_stop"]], dtype=torch.float32, device=device)
                logits = head(
                    hidden.unsqueeze(0),
                    step_t,
                    ans_t,
                    streak_t,
                    changed_t,
                )
                loss = focal_bce_with_logits(
                    logits,
                    target,
                    pos_weight=pos_weight,
                    gamma=focal_gamma,
                )
                if trainable_hidden:
                    loss.backward()
                total_loss += float(loss.item())
                n_steps += 1
    return total_loss / max(n_steps, 1), n_steps


def train_joint_rich_stop_head(
    model,
    tokenizer,
    train_samples: Sequence[dict],
    val_samples: Sequence[dict],
    *,
    cap: int,
    device: torch.device,
    seed: int,
    predict_fn,
    expected_fn,
    build_prompt_fn,
    eval_profile,
    unfreeze_layers: int = 2,
    epochs: int = 3,
    head_lr: float = 1e-3,
    coconut_lr: float = 2e-5,
    weight_decay: float = 1e-4,
    focal_gamma: float = 2.0,
    early_stop_patience: int = 2,
    progress_cb=None,
    init_head_state: Optional[dict] = None,
    label_mode: str = "is_correct",
) -> Tuple[RichStopHead, dict]:
    ft_info = configure_coconut_joint_train(model, unfreeze_layers=unfreeze_layers)
    head = RichStopHead(hidden_dim=768, max_steps=cap, dropout=0.15).to(device)
    if init_head_state:
        head.load_state_dict(init_head_state)

    model.eval()
    train_meta = build_rich_stop_metadata_for_samples(
        model,
        tokenizer,
        train_samples,
        cap=cap,
        device=device,
        seed=seed,
        predict_fn=predict_fn,
        expected_fn=expected_fn,
        build_prompt_fn=build_prompt_fn,
        eval_profile=eval_profile,
        label_mode=label_mode,
    )
    val_meta = build_rich_stop_metadata_for_samples(
        model,
        tokenizer,
        val_samples,
        cap=cap,
        device=device,
        seed=seed + 999,
        predict_fn=predict_fn,
        expected_fn=expected_fn,
        build_prompt_fn=build_prompt_fn,
        eval_profile=eval_profile,
        label_mode=label_mode,
    ) if val_samples else []

    pos = sum(step["should_stop"] > 0.5 for item in train_meta for step in item["steps"])
    neg = sum(1 for item in train_meta for step in item["steps"]) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], device=device)

    coconut_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [
            {"params": head.parameters(), "lr": head_lr},
            {"params": coconut_params, "lr": coconut_lr},
        ],
        weight_decay=weight_decay,
    )

    history = []
    best_val_loss = math.inf
    best_head_state = None
    best_model_state = None
    stale_epochs = 0
    stopped_epoch = epochs

    for epoch in range(epochs):
        order = list(range(len(train_meta)))
        g = torch.Generator()
        g.manual_seed(seed + epoch * 17)
        order = torch.randperm(len(train_meta), generator=g).tolist()

        epoch_loss = 0.0
        n_samples = 0
        for ord_idx, meta_idx in enumerate(order):
            item = train_meta[meta_idx]
            opt.zero_grad(set_to_none=True)
            sample_loss = 0.0
            model.train()
            head.train()
            for step in item["steps"]:
                n = step["n"]
                sample_idx = item["sample_idx"]
                prompt = build_prompt_fn(
                    item["sample"],
                    n,
                    seed=seed + sample_idx * 31 + n,
                    choice_order=eval_profile.choice_order,
                    shuffle_edges=eval_profile.prompt_mode != "fixed_edges",
                )
                input_ids = torch.tensor(
                    [tokenizer.encode(prompt, add_special_tokens=False)],
                    device=device,
                )
                hidden = extract_latent_hidden_trainable(model, input_ids, pass_idx=n - 1)
                step_t = torch.tensor([n], dtype=torch.long, device=device)
                ans_t = torch.tensor([step["answer_bucket"]], dtype=torch.long, device=device)
                streak_t = torch.tensor([step["streak"]], dtype=torch.long, device=device)
                changed_t = torch.tensor([step["changed"]], dtype=torch.float32, device=device)
                target = torch.tensor([step["should_stop"]], dtype=torch.float32, device=device)
                logits = head(
                    hidden.unsqueeze(0),
                    step_t,
                    ans_t,
                    streak_t,
                    changed_t,
                )
                loss = focal_bce_with_logits(
                    logits,
                    target,
                    pos_weight=pos_weight,
                    gamma=focal_gamma,
                )
                loss.backward()
                sample_loss += float(loss.item())
            torch.nn.utils.clip_grad_norm_(
                list(head.parameters()) + coconut_params,
                max_norm=1.0,
            )
            opt.step()
            epoch_loss += sample_loss / max(len(item["steps"]), 1)
            n_samples += 1
            if progress_cb:
                progress_cb(epoch + 1, epochs, ord_idx + 1, len(order))

        avg_train_loss = epoch_loss / max(n_samples, 1)
        val_loss = None
        if val_meta:
            opt.zero_grad(set_to_none=True)
            val_loss, _ = _rich_stop_loss_on_metadata(
                head,
                model,
                tokenizer,
                val_meta,
                cap=cap,
                device=device,
                seed=seed + 777,
                build_prompt_fn=build_prompt_fn,
                eval_profile=eval_profile,
                pos_weight=pos_weight,
                focal_gamma=focal_gamma,
                trainable_hidden=False,
            )
            if val_loss + 1e-5 < best_val_loss:
                best_val_loss = val_loss
                best_head_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                best_model_state = {
                    name: param.detach().cpu().clone()
                    for name, param in model.named_parameters()
                    if param.requires_grad
                }
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

    if best_head_state is not None:
        head.load_state_dict(best_head_state)
    if best_model_state is not None:
        merged = model.state_dict()
        merged.update(best_model_state)
        model.load_state_dict(merged, strict=False)

    model.eval()
    head.eval()
    metrics = {
        **ft_info,
        "train_samples": len(train_meta),
        "val_samples": len(val_meta),
        "train_pos": pos,
        "train_neg": neg,
        "epochs_requested": epochs,
        "epochs_ran": stopped_epoch,
        "best_val_loss": round(best_val_loss, 4) if val_meta else None,
        "head_lr": head_lr,
        "coconut_lr": coconut_lr,
        "focal_gamma": focal_gamma,
        "history_tail": history[-3:],
    }
    return head, metrics
