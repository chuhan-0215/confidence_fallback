from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stop_head.models import RichStopHead, RichStopExample

def _rich_tensors(examples: Sequence[RichStopExample], device: torch.device):
    hiddens = torch.stack([ex.hidden for ex in examples]).to(device)
    steps = torch.tensor([ex.step for ex in examples], dtype=torch.long, device=device)
    answers = torch.tensor([ex.answer_bucket for ex in examples], dtype=torch.long, device=device)
    streaks = torch.tensor([ex.streak for ex in examples], dtype=torch.long, device=device)
    changed = torch.tensor([ex.changed for ex in examples], dtype=torch.float32, device=device)
    labels = torch.tensor([ex.should_stop for ex in examples], dtype=torch.float32, device=device)
    return hiddens, steps, answers, streaks, changed, labels


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pos_weight: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    probs = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, probs, 1.0 - probs)
    return (((1.0 - pt) ** gamma) * bce).mean()


def train_rich_stop_head(
    train_examples: Sequence[RichStopExample],
    val_examples: Sequence[RichStopExample],
    *,
    hidden_dim: int = 768,
    max_steps: int = 8,
    dropout: float = 0.15,
    epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    early_stop_patience: int = 10,
    focal_gamma: float = 2.0,
    device: torch.device,
    init_head_state: Optional[dict] = None,
) -> Tuple[RichStopHead, dict]:
    if not train_examples:
        raise ValueError("no training examples")

    pos = sum(1 for ex in train_examples if ex.should_stop > 0.5)
    neg = len(train_examples) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], device=device)

    train_h, train_s, train_a, train_st, train_c, train_y = _rich_tensors(train_examples, device)
    train_loader = DataLoader(
        TensorDataset(train_h, train_s, train_a, train_st, train_c, train_y),
        batch_size=batch_size,
        shuffle=True,
    )

    val_tensors = None
    if val_examples:
        val_tensors = _rich_tensors(val_examples, device)

    head = RichStopHead(hidden_dim=hidden_dim, max_steps=max_steps, dropout=dropout).to(device)
    if init_head_state:
        head.load_state_dict(init_head_state)
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
        for batch_h, batch_s, batch_a, batch_st, batch_c, batch_y in train_loader:
            opt.zero_grad()
            logits = head(batch_h, batch_s, batch_a, batch_st, batch_c)
            loss = focal_bce_with_logits(logits, batch_y, pos_weight=pos_weight, gamma=focal_gamma)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1
        avg_train_loss = total_loss / max(n_batches, 1)

        val_loss = None
        if val_tensors is not None:
            val_h, val_s, val_a, val_st, val_c, val_y = val_tensors
            head.eval()
            with torch.no_grad():
                val_logits = head(val_h, val_s, val_a, val_st, val_c)
                val_loss = float(
                    focal_bce_with_logits(val_logits, val_y, pos_weight=pos_weight, gamma=focal_gamma).item()
                )
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
        logits = head(train_h, train_s, train_a, train_st, train_c)
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
        "focal_gamma": focal_gamma,
        "history_tail": history[-3:],
    }
    return head, metrics
