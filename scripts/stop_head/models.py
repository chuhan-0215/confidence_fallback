"""Lightweight stop head trained on frozen Coconut latent hidden states."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class StopExample:
    hidden: torch.Tensor
    step: int
    should_stop: float
    sample_idx: int


@dataclass
class RichStopExample:
    hidden: torch.Tensor
    step: int
    answer_bucket: int
    streak: int
    changed: float
    should_stop: float
    sample_idx: int


def encode_answer_bucket(pred: str, num_buckets: int = 64) -> int:
    text = (pred or "").strip().lower()
    if not text:
        return 0
    try:
        return int(text) % num_buckets
    except ValueError:
        return hash(text) % num_buckets


class LatentStopHead(nn.Module):
    def __init__(self, hidden_dim: int = 768, max_steps: int = 8, dropout: float = 0.0):
        super().__init__()
        self.step_emb = nn.Embedding(max_steps + 1, 32)
        layers: List[nn.Module] = [
            nn.Linear(hidden_dim + 32, 128),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(128, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, hidden: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        step_vec = self.step_emb(step)
        x = torch.cat([hidden, step_vec], dim=-1)
        return self.net(x).squeeze(-1)


class RichStopHead(nn.Module):
    """Stop head with answer + stability features observable at inference."""

    def __init__(
        self,
        hidden_dim: int = 768,
        max_steps: int = 8,
        num_answer_buckets: int = 64,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.step_emb = nn.Embedding(max_steps + 1, 32)
        self.answer_emb = nn.Embedding(num_answer_buckets, 16)
        layers: List[nn.Module] = [
            nn.Linear(hidden_dim + 32 + 16 + 2, 128),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(128, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        hidden: torch.Tensor,
        step: torch.Tensor,
        answer_bucket: torch.Tensor,
        streak: torch.Tensor,
        changed: torch.Tensor,
    ) -> torch.Tensor:
        step_vec = self.step_emb(step)
        ans_vec = self.answer_emb(answer_bucket)
        streak_norm = (streak.float() / 8.0).unsqueeze(-1)
        changed_feat = changed.float().unsqueeze(-1)
        x = torch.cat([hidden, step_vec, ans_vec, streak_norm, changed_feat], dim=-1)
        return self.net(x).squeeze(-1)
