from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def split_dataset(dataset: Sequence[dict], train_ratio: float = 0.6, seed: int = 42) -> Tuple[List[dict], List[dict]]:
    n = len(dataset)
    order = list(range(n))
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng).tolist()
    train_n = max(1, int(n * train_ratio))
    train_idx = sorted(perm[:train_n])
    test_idx = sorted(perm[train_n:])
    train = [dataset[i] for i in train_idx]
    test = [dataset[i] for i in test_idx]
    return train, test
