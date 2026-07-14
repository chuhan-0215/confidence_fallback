from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stop_head.models import encode_answer_bucket


def _rich_step_features(
    pred: str,
    prev_pred: str,
    streak: int,
) -> Tuple[int, int, float]:
    if pred and pred == prev_pred:
        streak += 1
    else:
        streak = 1
    changed = 1.0 if prev_pred and pred != prev_pred else 0.0
    return encode_answer_bucket(pred), streak, changed

