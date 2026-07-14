"""Upfront boundary budget prediction · 通解：先定步数，再单次推理。

Tracks 31–34 test whether a *new question* can get a latent budget without
step-by-step probing at inference time.  Training may use oracle stop labels
(soft-floor teacher, same as Exp28); test deployment only runs **one**
``predict_at_n`` call per sample.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_utils import bfs_distances, graph_diameter, reasoning_hops, root_to_target_distance
from stop_head import first_correct_step


def blind_depth(sample: dict) -> int:
    from run_auto_submit_experiment import blind_choice_depth

    return blind_choice_depth(sample)


def soft_floor_optimal_stop(
    first_correct: Optional[int],
    *,
    d: int,
    min_n: int,
    cap: int,
) -> int:
    """Exp28 teacher label: fc≥min_n → stop@fc else floor d."""
    floor_n = max(min_n, min(d, cap))
    if first_correct is not None and first_correct >= min_n:
        return min(first_correct, cap)
    return floor_n


def extract_graph_features(sample: dict, *, cap: int = 8) -> torch.Tensor:
    """Observable graph/question features — no Coconut forward, no answer label."""
    n_nodes = len(sample.get("idx_to_symbol") or [])
    edges = sample.get("edges") or []
    dist = bfs_distances(edges, sample["root"], n_nodes)
    c1 = int(sample["target"])
    c2 = int(sample["neg_target"])
    d1 = float(dist.get(c1, 0))
    d2 = float(dist.get(c2, 0))
    d = max(d1, d2)
    diam = float(graph_diameter(sample))
    return torch.tensor(
        [
            d / cap,
            d1 / cap,
            d2 / cap,
            abs(d1 - d2) / cap,
            n_nodes / 64.0,
            len(edges) / 128.0,
            diam / cap,
            float(d >= 4),
        ],
        dtype=torch.float32,
    )


def extract_rich_graph_features(sample: dict, *, cap: int = 8) -> torch.Tensor:
    """Extended observable graph features (no forward, no labels)."""
    base = extract_graph_features(sample, cap=cap)
    n_nodes = len(sample.get("idx_to_symbol") or [])
    edges = sample.get("edges") or []
    dist = bfs_distances(edges, sample["root"], n_nodes)
    c1 = int(sample["target"])
    c2 = int(sample["neg_target"])
    d1 = float(dist.get(c1, 0))
    d2 = float(dist.get(c2, 0))
    hops = float(reasoning_hops(sample)) / cap
    rtt = float(root_to_target_distance(sample)) / cap
    reach = len(dist) / max(n_nodes, 1)
    avg_deg = (2 * len(edges)) / max(n_nodes, 1) / 8.0
    extra = torch.tensor(
        [
            hops,
            rtt,
            reach,
            avg_deg,
            min(d1, d2) / cap,
            max(d1, d2) / cap,
            float(d1 == d2),
            float(d1 < d2),
        ],
        dtype=torch.float32,
    )
    return torch.cat([base, extra])


@torch.no_grad()
def extract_coconut_prompt_hidden(
    model,
    tokenizer,
    sample: dict,
    *,
    seed: int,
    device: torch.device,
    eval_profile,
) -> torch.Tensor:
    """Coconut hidden at question prefix (n_latent=0, before answer generation)."""
    from eval_profile import parse_eval_profile
    from graph_utils import build_eval_prompt

    profile = eval_profile or parse_eval_profile(None)
    shuffle_edges = profile.prompt_mode != "fixed_edges"
    prompt = build_eval_prompt(
        sample,
        0,
        seed=seed,
        choice_order=profile.choice_order,
        shuffle_edges=shuffle_edges,
    )
    input_ids = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=False)],
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(
        0, input_ids.shape[1], dtype=torch.long, device=device
    ).reshape(1, -1)
    outputs = model.base_causallm(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_hidden_states=True,
    )
    return outputs.hidden_states[-1][0, -1, :].detach().cpu()


def build_soft_floor_budget_labels(
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
    eval_profile,
    progress_cb=None,
) -> List[dict]:
    """Offline teacher labels for training budget predictors."""
    rows: List[dict] = []
    for idx, sample in enumerate(samples):
        d = blind_depth(sample)
        first_correct, _ = first_correct_step(
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
        optimal_n = soft_floor_optimal_stop(
            first_correct, d=d, min_n=min_n, cap=cap
        )
        delta = max(-1, min(1, optimal_n - d))
        rows.append(
            {
                "features": extract_graph_features(sample, cap=cap),
                "rich_features": extract_rich_graph_features(sample, cap=cap),
                "optimal_n": optimal_n,
                "delta": delta,
                "blind_depth": d,
                "first_correct": first_correct,
            }
        )
        if progress_cb:
            progress_cb(idx + 1, len(samples))
    return rows


import torch
import torch.nn as nn
from typing import Dict, List, Optional, Sequence, Tuple


class GraphBudgetMLP(nn.Module):
    """Predict latent budget n ∈ {min_n..cap} from graph features only."""

    def __init__(self, in_dim: int = 8, hidden: int = 64, num_classes: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GraphDeltaMLP(nn.Module):
    """Predict Δ ∈ {-1,0,+1} added to blind_depth."""

    def __init__(self, in_dim: int = 8, hidden: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _rows_to_tensors(
    rows: Sequence[dict],
    *,
    min_n: int,
    label_key: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = torch.stack([r["features"] for r in rows])
    if label_key == "optimal_n":
        ys = torch.tensor([r["optimal_n"] - min_n for r in rows], dtype=torch.long)
    else:
        ys = torch.tensor([r["delta"] + 1 for r in rows], dtype=torch.long)
    return xs, ys


def train_graph_budget_mlp(
    train_rows: Sequence[dict],
    val_rows: Sequence[dict],
    *,
    min_n: int,
    cap: int,
    epochs: int = 80,
    lr: float = 1e-3,
) -> Tuple[GraphBudgetMLP, dict]:
    num_classes = cap - min_n + 1
    model = GraphBudgetMLP(num_classes=num_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    x_train, y_train = _rows_to_tensors(train_rows, min_n=min_n, label_key="optimal_n")
    x_val, y_val = _rows_to_tensors(val_rows, min_n=min_n, label_key="optimal_n")

    best_state = None
    best_val = math.inf
    history = []
    for epoch in range(epochs):
        model.train()
        logits = model(x_train)
        loss = F.cross_entropy(logits, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            v_logits = model(x_val)
            v_loss = float(F.cross_entropy(v_logits, y_val).item())
            v_acc = float((v_logits.argmax(-1) == y_val).float().mean().item())
        history.append({"epoch": epoch + 1, "train_loss": round(float(loss.item()), 4), "val_loss": round(v_loss, 4), "val_acc": round(v_acc, 4)})
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_acc = float((model(x_train).argmax(-1) == y_train).float().mean().item())
        val_acc = float((model(x_val).argmax(-1) == y_val).float().mean().item())
    metrics = {
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_acc": round(train_acc, 4),
        "val_acc": round(val_acc, 4),
        "best_val_loss": round(best_val, 4),
        "history_tail": history[-3:],
    }
    return model, metrics


def train_graph_delta_mlp(
    train_rows: Sequence[dict],
    val_rows: Sequence[dict],
    *,
    epochs: int = 80,
    lr: float = 1e-3,
) -> Tuple[GraphDeltaMLP, dict]:
    model = GraphDeltaMLP()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    x_train, y_train = _rows_to_tensors(train_rows, min_n=0, label_key="delta")
    x_val, y_val = _rows_to_tensors(val_rows, min_n=0, label_key="delta")

    best_state = None
    best_val = math.inf
    history = []
    for epoch in range(epochs):
        model.train()
        loss = F.cross_entropy(model(x_train), y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            v_logits = model(x_val)
            v_loss = float(F.cross_entropy(v_logits, y_val).item())
            v_acc = float((v_logits.argmax(-1) == y_val).float().mean().item())
        history.append({"epoch": epoch + 1, "val_loss": round(v_loss, 4), "val_acc": round(v_acc, 4)})
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_acc = float((model(x_train).argmax(-1) == y_train).float().mean().item())
        val_acc = float((model(x_val).argmax(-1) == y_val).float().mean().item())
    metrics = {
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_acc": round(train_acc, 4),
        "val_acc": round(val_acc, 4),
        "best_val_loss": round(best_val, 4),
        "history_tail": history[-3:],
    }
    return model, metrics


class GraphD4BinaryMLP(nn.Module):
    """For d>=4 only: predict budget 3 vs 4."""

    def __init__(self, in_dim: int = 8, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _asymmetry_bin(sample: dict, *, cap: int) -> int:
    feats = extract_graph_features(sample, cap=cap)
    return int(feats[3].item() > 0)


def train_d4_binary_mlp(
    train_rows: Sequence[dict],
    val_rows: Sequence[dict],
    *,
    epochs: int = 80,
    lr: float = 1e-3,
) -> Tuple[GraphD4BinaryMLP, dict]:
    train4 = [r for r in train_rows if r["blind_depth"] >= 4]
    val4 = [r for r in val_rows if r["blind_depth"] >= 4]
    if not train4 or not val4:
        raise ValueError("need d>=4 samples for d4 binary training")

    model = GraphD4BinaryMLP()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    x_train = torch.stack([r["features"] for r in train4])
    y_train = torch.tensor([0 if r["optimal_n"] <= 3 else 1 for r in train4], dtype=torch.long)
    x_val = torch.stack([r["features"] for r in val4])
    y_val = torch.tensor([0 if r["optimal_n"] <= 3 else 1 for r in val4], dtype=torch.long)

    best_state = None
    best_val = math.inf
    history = []
    for epoch in range(epochs):
        model.train()
        loss = F.cross_entropy(model(x_train), y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v_loss = float(F.cross_entropy(model(x_val), y_val).item())
            v_acc = float((model(x_val).argmax(-1) == y_val).float().mean().item())
        history.append({"epoch": epoch + 1, "val_loss": round(v_loss, 4), "val_acc": round(v_acc, 4)})
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_acc = float((model(x_train).argmax(-1) == y_train).float().mean().item())
        val_acc = float((model(x_val).argmax(-1) == y_val).float().mean().item())
    return model, {
        "train_rows_d4": len(train4),
        "val_rows_d4": len(val4),
        "train_acc": round(train_acc, 4),
        "val_acc": round(val_acc, 4),
        "best_val_loss": round(best_val, 4),
        "history_tail": history[-3:],
    }


class GraphD4MultiMLP(nn.Module):
    """For d>=4: predict budget in {2,3,4}."""

    def __init__(self, in_dim: int = 8, hidden: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


_D4_N_TO_CLS = {2: 0, 3: 1, 4: 2}
_D4_CLS_TO_N = {0: 2, 1: 3, 2: 4}


def train_d4_multiclass_mlp(
    train_rows: Sequence[dict],
    val_rows: Sequence[dict],
    *,
    epochs: int = 80,
    lr: float = 1e-3,
) -> Tuple[GraphD4MultiMLP, dict]:
    train4 = [r for r in train_rows if r["blind_depth"] >= 4 and r["optimal_n"] in _D4_N_TO_CLS]
    val4 = [r for r in val_rows if r["blind_depth"] >= 4 and r["optimal_n"] in _D4_N_TO_CLS]
    if not train4 or not val4:
        raise ValueError("need d>=4 samples with optimal_n in {2,3,4}")

    model = GraphD4MultiMLP()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    x_train = torch.stack([r["features"] for r in train4])
    y_train = torch.tensor([_D4_N_TO_CLS[int(r["optimal_n"])] for r in train4], dtype=torch.long)
    x_val = torch.stack([r["features"] for r in val4])
    y_val = torch.tensor([_D4_N_TO_CLS[int(r["optimal_n"])] for r in val4], dtype=torch.long)

    best_state = None
    best_val = math.inf
    history = []
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(x_train), y_train)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v_loss = float(F.cross_entropy(model(x_val), y_val).item())
            v_acc = float((model(x_val).argmax(-1) == y_val).float().mean().item())
        history.append({"epoch": epoch + 1, "val_loss": round(v_loss, 4), "val_acc": round(v_acc, 4)})
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_acc = float((model(x_train).argmax(-1) == y_train).float().mean().item())
        val_acc = float((model(x_val).argmax(-1) == y_val).float().mean().item())
    return model, {
        "train_rows_d4": len(train4),
        "val_rows_d4": len(val4),
        "train_acc": round(train_acc, 4),
        "val_acc": round(val_acc, 4),
        "best_val_loss": round(best_val, 4),
        "history_tail": history[-3:],
    }


def calibrate_d4_binary_threshold(
    head: GraphD4BinaryMLP,
    val_rows: Sequence[dict],
) -> Tuple[float, dict]:
    val4 = [r for r in val_rows if r["blind_depth"] >= 4]
    if not val4:
        return 0.5, {"threshold": 0.5, "val_acc": None}
    head.eval()
    x_val = torch.stack([r["features"] for r in val4])
    y_val = [0 if r["optimal_n"] <= 3 else 1 for r in val4]
    with torch.no_grad():
        prob3 = F.softmax(head(x_val), dim=-1)[:, 0]
    best_t = 0.5
    best_acc = -1.0
    trials = []
    for i in range(1, 20):
        t = i / 20.0
        hits = sum(int((p.item() >= t) == (y == 0)) for p, y in zip(prob3, y_val))
        acc = hits / len(val4)
        trials.append({"threshold": round(t, 2), "val_acc": round(acc, 4)})
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return best_t, {"threshold": round(best_t, 4), "val_acc": round(best_acc, 4), "trials": trials[-5:]}


import torch
import torch.nn as nn
from typing import Dict, List, Optional, Sequence, Tuple


def make_d4_threshold_budget_fn(
    head: GraphD4BinaryMLP,
    *,
    threshold: float,
    min_n: int,
    cap: int,
    device: torch.device,
) -> BudgetFn:
    head.eval()

    def _fn(sample: dict) -> int:
        d = blind_depth(sample)
        if d < 4:
            return max(min_n, min(cap, d))
        x = extract_graph_features(sample, cap=cap).unsqueeze(0).to(device)
        with torch.no_grad():
            p3 = F.softmax(head(x), dim=-1)[0, 0].item()
        n = 3 if p3 >= threshold else 4
        return max(min_n, min(cap, n))

    return _fn


def make_d4_multiclass_budget_fn(
    head: GraphD4MultiMLP,
    *,
    min_n: int,
    cap: int,
    device: torch.device,
) -> BudgetFn:
    head.eval()

    def _fn(sample: dict) -> int:
        d = blind_depth(sample)
        if d < 4:
            return max(min_n, min(cap, d))
        x = extract_graph_features(sample, cap=cap).unsqueeze(0).to(device)
        with torch.no_grad():
            cls = int(head(x).argmax(dim=-1).item())
        n = _D4_CLS_TO_N.get(cls, 4)
        return max(min_n, min(cap, n))

    return _fn


def make_d4_teacher_hybrid_budget_fn(
    teacher_map: Dict[int, int],
    *,
    min_n: int,
    cap: int,
) -> BudgetFn:
    """d<4: n=d; d>=4: perfect teacher budget (Exp42 upper bound on d4 fix)."""

    def _fn(sample: dict) -> int:
        d = blind_depth(sample)
        if d < 4:
            return max(min_n, min(cap, d))
        key = int(sample.get("_idx", -1))
        if key in teacher_map:
            return max(min_n, min(cap, teacher_map[key]))
        return max(min_n, min(cap, d))

    return _fn


def build_teacher_budget_rows(
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
    eval_profile,
    progress_cb=None,
) -> List[dict]:
    """Alias for build_prompt_budget_labels (Exp31–34 / Phase14+ teacher rows)."""
    return build_prompt_budget_labels(
        model,
        tokenizer,
        samples,
        cap=cap,
        min_n=min_n,
        device=device,
        seed=seed,
        predict_fn=predict_fn,
        expected_fn=expected_fn,
        eval_profile=eval_profile,
        progress_cb=progress_cb,
    )


def build_prompt_budget_labels(
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
    eval_profile,
    progress_cb=None,
) -> List[dict]:
    """Teacher labels + Coconut prefix hidden (n_latent=0) for model-self budget."""
    rows: List[dict] = []
    for idx, sample in enumerate(samples):
        d = blind_depth(sample)
        first_correct, _ = first_correct_step(
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
        optimal_n = soft_floor_optimal_stop(
            first_correct, d=d, min_n=min_n, cap=cap
        )
        hidden = extract_coconut_prompt_hidden(
            model,
            tokenizer,
            sample,
            seed=seed + idx * 31,
            device=device,
            eval_profile=eval_profile,
        )
        graph = extract_rich_graph_features(sample, cap=cap)
        rows.append(
            {
                "features": extract_graph_features(sample, cap=cap),
                "rich_features": graph,
                "prompt_hidden": hidden,
                "joint_features": torch.cat([graph, hidden]),
                "optimal_n": optimal_n,
                "blind_depth": d,
                "first_correct": first_correct,
            }
        )
        if progress_cb:
            progress_cb(idx + 1, len(samples))
    return rows


class GraphD3BinaryMLP(nn.Module):
    def __init__(self, in_dim: int = 16, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PromptD4BinaryMLP(nn.Module):
    """Rich graph + Coconut prefix hidden → d≥4 budget 3 vs 4."""

    def __init__(self, in_dim: int = 784, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_d3_binary_mlp(
    train_rows: Sequence[dict],
    val_rows: Sequence[dict],
    *,
    feature_key: str = "rich_features",
    epochs: int = 80,
    lr: float = 1e-3,
) -> Tuple[GraphD3BinaryMLP, dict]:
    train3 = [r for r in train_rows if r["blind_depth"] == 3]
    val3 = [r for r in val_rows if r["blind_depth"] == 3]
    if not train3 or not val3:
        raise ValueError("need d=3 samples for d3 binary training")

    in_dim = train3[0][feature_key].numel()
    model = GraphD3BinaryMLP(in_dim=in_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    x_train = torch.stack([r[feature_key] for r in train3])
    y_train = torch.tensor([0 if r["optimal_n"] <= 2 else 1 for r in train3], dtype=torch.long)
    x_val = torch.stack([r[feature_key] for r in val3])
    y_val = torch.tensor([0 if r["optimal_n"] <= 2 else 1 for r in val3], dtype=torch.long)

    best_state = None
    best_val = math.inf
    history = []
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(x_train), y_train)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v_loss = float(F.cross_entropy(model(x_val), y_val).item())
            v_acc = float((model(x_val).argmax(-1) == y_val).float().mean().item())
        history.append({"epoch": epoch + 1, "val_loss": round(v_loss, 4), "val_acc": round(v_acc, 4)})
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_acc = float((model(x_train).argmax(-1) == y_train).float().mean().item())
        val_acc = float((model(x_val).argmax(-1) == y_val).float().mean().item())
    return model, {
        "train_rows_d3": len(train3),
        "val_rows_d3": len(val3),
        "train_acc": round(train_acc, 4),
        "val_acc": round(val_acc, 4),
        "best_val_loss": round(best_val, 4),
        "history_tail": history[-3:],
    }


def train_d4_knn_bank(
    train_rows: Sequence[dict],
    *,
    feature_key: str = "rich_features",
) -> Tuple[dict, dict]:
    train4 = [r for r in train_rows if r["blind_depth"] >= 4]
    if not train4:
        raise ValueError("need d>=4 train rows for kNN bank")
    feats = torch.stack([r[feature_key] for r in train4])
    feats = F.normalize(feats, dim=1)
    labels = [3 if r["optimal_n"] <= 3 else 4 for r in train4]
    bank = {"features": feats, "labels": labels}
    meta = {
        "train_rows_d4": len(train4),
        "label_hist": {str(k): labels.count(k) for k in sorted(set(labels))},
    }
    return bank, meta


def train_d4_weighted_binary_mlp(
    train_rows: Sequence[dict],
    val_rows: Sequence[dict],
    *,
    feature_key: str = "rich_features",
    epochs: int = 80,
    lr: float = 1e-3,
) -> Tuple[GraphD4BinaryMLP, dict]:
    train4 = [r for r in train_rows if r["blind_depth"] >= 4]
    val4 = [r for r in val_rows if r["blind_depth"] >= 4]
    if not train4 or not val4:
        raise ValueError("need d>=4 samples for weighted d4 binary")

    in_dim = train4[0][feature_key].numel()
    model = GraphD4BinaryMLP(in_dim=in_dim, hidden=48)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    x_train = torch.stack([r[feature_key] for r in train4])
    y_train = torch.tensor([0 if r["optimal_n"] <= 3 else 1 for r in train4], dtype=torch.long)
    x_val = torch.stack([r[feature_key] for r in val4])
    y_val = torch.tensor([0 if r["optimal_n"] <= 3 else 1 for r in val4], dtype=torch.long)
    counts = torch.bincount(y_train, minlength=2).float().clamp(min=1.0)
    class_weight = (len(y_train) / (2.0 * counts)).to(x_train.dtype)

    best_state = None
    best_val = math.inf
    history = []
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(x_train), y_train, weight=class_weight)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v_loss = float(F.cross_entropy(model(x_val), y_val).item())
            v_acc = float((model(x_val).argmax(-1) == y_val).float().mean().item())
        history.append({"epoch": epoch + 1, "val_loss": round(v_loss, 4), "val_acc": round(v_acc, 4)})
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_acc = float((model(x_train).argmax(-1) == y_train).float().mean().item())
        val_acc = float((model(x_val).argmax(-1) == y_val).float().mean().item())
    return model, {
        "train_rows_d4": len(train4),
        "val_rows_d4": len(val4),
        "train_acc": round(train_acc, 4),
        "val_acc": round(val_acc, 4),
        "best_val_loss": round(best_val, 4),
        "class_weight": [round(float(w), 4) for w in class_weight],
        "history_tail": history[-3:],
    }


def train_prompt_d4_binary_mlp(
    train_rows: Sequence[dict],
    val_rows: Sequence[dict],
    *,
    epochs: int = 60,
    lr: float = 5e-4,
) -> Tuple[PromptD4BinaryMLP, dict]:
    train4 = [r for r in train_rows if r["blind_depth"] >= 4]
    val4 = [r for r in val_rows if r["blind_depth"] >= 4]
    if not train4 or not val4:
        raise ValueError("need d>=4 rows with prompt hidden")

    in_dim = train4[0]["joint_features"].numel()
    model = PromptD4BinaryMLP(in_dim=in_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    x_train = torch.stack([r["joint_features"] for r in train4])
    y_train = torch.tensor([0 if r["optimal_n"] <= 3 else 1 for r in train4], dtype=torch.long)
    x_val = torch.stack([r["joint_features"] for r in val4])
    y_val = torch.tensor([0 if r["optimal_n"] <= 3 else 1 for r in val4], dtype=torch.long)
    counts = torch.bincount(y_train, minlength=2).float().clamp(min=1.0)
    class_weight = (len(y_train) / (2.0 * counts)).to(x_train.dtype)

    best_state = None
    best_val = math.inf
    history = []
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(x_train), y_train, weight=class_weight)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v_loss = float(F.cross_entropy(model(x_val), y_val).item())
            v_acc = float((model(x_val).argmax(-1) == y_val).float().mean().item())
        history.append({"epoch": epoch + 1, "val_loss": round(v_loss, 4), "val_acc": round(v_acc, 4)})
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_acc = float((model(x_train).argmax(-1) == y_train).float().mean().item())
        val_acc = float((model(x_val).argmax(-1) == y_val).float().mean().item())
    return model, {
        "train_rows_d4": len(train4),
        "val_rows_d4": len(val4),
        "train_acc": round(train_acc, 4),
        "val_acc": round(val_acc, 4),
        "best_val_loss": round(best_val, 4),
        "history_tail": history[-3:],
    }


def make_d3_hybrid_budget_fn(
    d3_head: GraphD3BinaryMLP,
    *,
    min_n: int,
    cap: int,
    device: torch.device,
    feature_key: str = "rich_features",
) -> BudgetFn:
    d3_head.eval()

    def _fn(sample: dict) -> int:
        d = blind_depth(sample)
        if d < 3:
            return max(min_n, min(cap, d))
        if d == 3:
            x = extract_rich_graph_features(sample, cap=cap).unsqueeze(0).to(device)
            with torch.no_grad():
                cls = int(d3_head(x).argmax(dim=-1).item())
            n = 2 if cls == 0 else 3
            return max(min_n, min(cap, n))
        return max(min_n, min(cap, d))

    return _fn


def make_d4_knn_budget_fn(
    bank: dict,
    *,
    k: int,
    min_n: int,
    cap: int,
) -> BudgetFn:
    def _fn(sample: dict) -> int:
        d = blind_depth(sample)
        if d < 4:
            return max(min_n, min(cap, d))
        x = F.normalize(
            extract_rich_graph_features(sample, cap=cap).unsqueeze(0), dim=1
        )
        sims = bank["features"] @ x.T
        sims = sims.squeeze(1)
        topk = min(k, sims.numel())
        vals, idxs = sims.topk(topk)
        votes: Dict[int, float] = {}
        for i, w in zip(idxs.tolist(), vals.tolist()):
            label = bank["labels"][i]
            votes[label] = votes.get(label, 0.0) + max(w, 0.0)
        n = max(votes, key=votes.get)
        return max(min_n, min(cap, int(n)))

    return _fn


def make_d4_rich_binary_budget_fn(
    head: GraphD4BinaryMLP,
    *,
    min_n: int,
    cap: int,
    device: torch.device,
) -> BudgetFn:
    head.eval()

    def _fn(sample: dict) -> int:
        d = blind_depth(sample)
        if d < 4:
            return max(min_n, min(cap, d))
        x = extract_rich_graph_features(sample, cap=cap).unsqueeze(0).to(device)
        with torch.no_grad():
            cls = int(head(x).argmax(dim=-1).item())
        n = 3 if cls == 0 else 4
        return max(min_n, min(cap, n))

    return _fn


def make_prompt_d4_budget_fn(
    head: PromptD4BinaryMLP,
    model,
    tokenizer,
    *,
    min_n: int,
    cap: int,
    device: torch.device,
    eval_profile,
    seed_base: int = 99,
) -> BudgetFn:
    """Model-self budget: Coconut prefix encode + learned head (Exp47)."""
    head.eval()

    def _fn(sample: dict) -> int:
        d = blind_depth(sample)
        if d < 4:
            return max(min_n, min(cap, d))
        idx = int(sample.get("_idx", 0))
        hidden = extract_coconut_prompt_hidden(
            model,
            tokenizer,
            sample,
            seed=seed_base + idx * 31,
            device=device,
            eval_profile=eval_profile,
        ).to(device)
        graph = extract_rich_graph_features(sample, cap=cap).to(device)
        x = torch.cat([graph, hidden]).unsqueeze(0)
        with torch.no_grad():
            cls = int(head(x).argmax(dim=-1).item())
        n = 3 if cls == 0 else 4
        return max(min_n, min(cap, n))

    return _fn


def _prompt_d4_prob3(
    head: PromptD4BinaryMLP,
    model,
    tokenizer,
    sample: dict,
    *,
    cap: int,
    device: torch.device,
    eval_profile,
    seed_base: int,
) -> float:
    head.eval()
    idx = int(sample.get("_idx", 0))
    hidden = extract_coconut_prompt_hidden(
        model,
        tokenizer,
        sample,
        seed=seed_base + idx * 31,
        device=device,
        eval_profile=eval_profile,
    ).to(device)
    graph = extract_rich_graph_features(sample, cap=cap).to(device)
    x = torch.cat([graph, hidden]).unsqueeze(0)
    with torch.no_grad():
        return float(F.softmax(head(x), dim=-1)[0, 0].item())


def calibrate_prompt_d4_threshold(
    head: PromptD4BinaryMLP,
    val_rows: Sequence[dict],
) -> Tuple[float, dict]:
    val4 = [r for r in val_rows if r["blind_depth"] >= 4]
    if not val4:
        return 0.5, {"threshold": 0.5, "val_acc": None}
    head.eval()
    x_val = torch.stack([r["joint_features"] for r in val4])
    y_val = [0 if r["optimal_n"] <= 3 else 1 for r in val4]
    with torch.no_grad():
        prob3 = F.softmax(head(x_val), dim=-1)[:, 0]
    best_t = 0.5
    best_acc = -1.0
    trials = []
    for i in range(1, 20):
        t = i / 20.0
        hits = sum(int((p.item() >= t) == (y == 0)) for p, y in zip(prob3, y_val))
        acc = hits / len(val4)
        trials.append({"threshold": round(t, 2), "val_label_acc": round(acc, 4)})
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return best_t, {
        "threshold": round(best_t, 4),
        "val_label_acc": round(best_acc, 4),
        "trials": trials[-5:],
    }


def calibrate_prompt_d4_threshold_by_accuracy(
    head: PromptD4BinaryMLP,
    val_rows: Sequence[dict],
    val_samples: Sequence[dict],
    model,
    tokenizer,
    *,
    min_n: int,
    cap: int,
    device: torch.device,
    eval_profile,
    predict_fn,
    expected_fn,
    seed: int = 78,
    conservative: bool = False,
) -> Tuple[float, dict]:
    """Grid-search P(n=3) threshold maximizing val answer accuracy (d>=4)."""
    pairs = [
        (row, sample)
        for row, sample in zip(val_rows, val_samples)
        if row["blind_depth"] >= 4
    ]
    if not pairs:
        return 0.5, {"threshold": 0.5, "val_answer_acc": None}

    best_t = 0.5
    best_acc = -1.0
    trials = []
    for i in range(1, 20):
        t = i / 20.0
        correct = 0
        for row_idx, (row, sample) in enumerate(pairs):
            d = int(row["blind_depth"])
            p3 = _prompt_d4_prob3(
                head, model, tokenizer, sample,
                cap=cap, device=device, eval_profile=eval_profile, seed_base=seed + row_idx * 31,
            )
            if p3 >= t:
                n = max(min_n, min(cap, 3))
            elif conservative:
                n = max(min_n, min(cap, d))
            else:
                n = max(min_n, min(cap, 4))
            pred = predict_fn(
                model, tokenizer, sample, n, device,
                seed=seed + row_idx * 31 + n,
                eval_profile=eval_profile,
            )
            if pred == expected_fn(sample, eval_profile):
                correct += 1
        acc = correct / len(pairs)
        trials.append({"threshold": round(t, 2), "val_answer_acc": round(acc, 4)})
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return best_t, {
        "threshold": round(best_t, 4),
        "val_answer_acc": round(best_acc, 4),
        "conservative": conservative,
        "trials": trials[-5:],
    }


def make_prompt_d4_threshold_budget_fn(
    head: PromptD4BinaryMLP,
    model,
    tokenizer,
    *,
    threshold: float,
    min_n: int,
    cap: int,
    device: torch.device,
    eval_profile,
    seed_base: int = 99,
    conservative: bool = False,
) -> BudgetFn:
    head.eval()

    def _fn(sample: dict) -> int:
        d = blind_depth(sample)
        if d < 4:
            return max(min_n, min(cap, d))
        p3 = _prompt_d4_prob3(
            head, model, tokenizer, sample,
            cap=cap, device=device, eval_profile=eval_profile, seed_base=seed_base,
        )
        if p3 >= threshold:
            return max(min_n, min(cap, 3))
        if conservative:
            return max(min_n, min(cap, d))
        return max(min_n, min(cap, 4))

    return _fn


def make_prompt_knn_ensemble_budget_fn(
    head: PromptD4BinaryMLP,
    knn_bank: dict,
    model,
    tokenizer,
    *,
    min_n: int,
    cap: int,
    device: torch.device,
    eval_profile,
    seed_base: int = 99,
    k: int = 9,
    prompt_threshold: float = 0.5,
) -> BudgetFn:
    """n=3 only when Coconut prefix AND kNN both vote 3 (Exp51)."""
    head.eval()

    def _knn_n(sample: dict) -> int:
        x = F.normalize(
            extract_rich_graph_features(sample, cap=cap).unsqueeze(0), dim=1
        )
        sims = knn_bank["features"] @ x.T
        sims = sims.squeeze(1)
        topk = min(k, sims.numel())
        _, idxs = sims.topk(topk)
        votes: Dict[int, float] = {}
        for i in idxs.tolist():
            label = knn_bank["labels"][i]
            votes[label] = votes.get(label, 0.0) + 1.0
        return int(max(votes, key=votes.get))

    def _fn(sample: dict) -> int:
        d = blind_depth(sample)
        if d < 4:
            return max(min_n, min(cap, d))
        p3 = _prompt_d4_prob3(
            head, model, tokenizer, sample,
            cap=cap, device=device, eval_profile=eval_profile, seed_base=seed_base,
        )
        knn_vote = _knn_n(sample)
        use_three = p3 >= prompt_threshold and knn_vote <= 3
        n = 3 if use_three else 4
        return max(min_n, min(cap, n))

    return _fn


class PromptDeltaMLP(nn.Module):
    def __init__(self, in_dim: int = 784, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_prompt_delta_mlp(
    train_rows: Sequence[dict],
    val_rows: Sequence[dict],
    *,
    epochs: int = 60,
    lr: float = 5e-4,
) -> Tuple[PromptDeltaMLP, dict]:
    if not train_rows or not val_rows:
        raise ValueError("need train/val rows with prompt hidden")

    def _delta_cls(row: dict) -> int:
        delta = max(-1, min(1, int(row["optimal_n"] - row["blind_depth"])))
        return delta + 1

    in_dim = train_rows[0]["joint_features"].numel()
    model = PromptDeltaMLP(in_dim=in_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    x_train = torch.stack([r["joint_features"] for r in train_rows])
    y_train = torch.tensor([_delta_cls(r) for r in train_rows], dtype=torch.long)
    x_val = torch.stack([r["joint_features"] for r in val_rows])
    y_val = torch.tensor([_delta_cls(r) for r in val_rows], dtype=torch.long)

    best_state = None
    best_val = math.inf
    history = []
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(x_train), y_train)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v_loss = float(F.cross_entropy(model(x_val), y_val).item())
            v_acc = float((model(x_val).argmax(-1) == y_val).float().mean().item())
        history.append({"epoch": epoch + 1, "val_loss": round(v_loss, 4), "val_acc": round(v_acc, 4)})
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_acc = float((model(x_train).argmax(-1) == y_train).float().mean().item())
        val_acc = float((model(x_val).argmax(-1) == y_val).float().mean().item())
    return model, {
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_acc": round(train_acc, 4),
        "val_acc": round(val_acc, 4),
        "best_val_loss": round(best_val, 4),
        "history_tail": history[-3:],
    }


def make_prompt_delta_budget_fn(
    head: PromptDeltaMLP,
    model,
    tokenizer,
    *,
    min_n: int,
    cap: int,
    device: torch.device,
    eval_profile,
    seed_base: int = 99,
) -> BudgetFn:
    """Coconut prefix hidden → Δ, n=d+Δ (Exp48, model-self budget)."""
    head.eval()

    def _fn(sample: dict) -> int:
        d = blind_depth(sample)
        idx = int(sample.get("_idx", 0))
        hidden = extract_coconut_prompt_hidden(
            model,
            tokenizer,
            sample,
            seed=seed_base + idx * 31,
            device=device,
            eval_profile=eval_profile,
        ).to(device)
        graph = extract_rich_graph_features(sample, cap=cap).to(device)
        x = torch.cat([graph, hidden]).unsqueeze(0)
        with torch.no_grad():
            delta = int(head(x).argmax(dim=-1).item()) - 1
        return max(min_n, min(cap, d + delta))

    return _fn


from typing import Callable, Dict, List, Optional, Sequence

import torch


def train_asymmetry_lookup_table(
    train_rows: Sequence[dict],
    *,
    min_n: int,
    cap: int,
) -> Tuple[Dict[Tuple[int, int], int], dict]:
    from collections import Counter, defaultdict

    buckets: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
    for row in train_rows:
        d = int(row["blind_depth"])
        asym = int(row["features"][3].item() > 0)
        buckets[(d, asym)][int(row["optimal_n"])] += 1

    table: Dict[Tuple[int, int], int] = {}
    details = []
    for key in sorted(buckets):
        optimal_n, count = buckets[key].most_common(1)[0]
        n = max(min_n, min(cap, int(optimal_n)))
        table[key] = n
        details.append(
            {
                "blind_depth": key[0],
                "asymmetry": key[1],
                "optimal_n": n,
                "train_count": count,
                "train_total": sum(buckets[key].values()),
            }
        )
    return table, {"lookup_table": details}


def make_d4_binary_budget_fn(
    head: GraphD4BinaryMLP,
    *,
    min_n: int,
    cap: int,
    device: torch.device,
) -> BudgetFn:
    head.eval()

    def _fn(sample: dict) -> int:
        d = blind_depth(sample)
        if d < 4:
            return max(min_n, min(cap, d))
        x = extract_graph_features(sample, cap=cap).unsqueeze(0).to(device)
        with torch.no_grad():
            use_four = int(head(x).argmax(dim=-1).item()) == 1
        return max(min_n, min(cap, 4 if use_four else 3))

    return _fn


def make_asymmetry_lookup_budget_fn(
    table: Dict[Tuple[int, int], int],
    *,
    min_n: int,
    cap: int,
) -> BudgetFn:
    def _fn(sample: dict) -> int:
        d = blind_depth(sample)
        key = (d, _asymmetry_bin(sample, cap=cap))
        n = table.get(key, table.get((d, 0), d))
        return max(min_n, min(cap, int(n)))

    return _fn


def make_asymmetry_rule_budget_fn(*, min_n: int, cap: int) -> BudgetFn:
    """d=3→3; d>=4 and candidate depths differ→3; else d."""

    def _fn(sample: dict) -> int:
        d = blind_depth(sample)
        if d < 4:
            return max(min_n, min(cap, d))
        if _asymmetry_bin(sample, cap=cap):
            return max(min_n, min(cap, 3))
        return max(min_n, min(cap, d))

    return _fn


BudgetFn = Callable[[dict], int]


def make_d4_early_budget_fn(*, min_n: int, cap: int) -> BudgetFn:
    """4-hop: budget=3 (Exp26 insight); 3-hop: budget=d."""

    def _fn(sample: dict) -> int:
        d = blind_depth(sample)
        if d >= 4:
            return max(min_n, min(cap, 3))
        return max(min_n, min(cap, d))

    return _fn


def make_d_minus_one_budget_fn(*, min_n: int, cap: int) -> BudgetFn:
    """4-hop: budget=d-1; 3-hop: budget=d."""

    def _fn(sample: dict) -> int:
        d = blind_depth(sample)
        if d >= 4:
            return max(min_n, min(cap, d - 1))
        return max(min_n, min(cap, d))

    return _fn


def train_lookup_budget_table(
    train_rows: Sequence[dict],
    *,
    min_n: int,
    cap: int,
) -> Tuple[Dict[int, int], dict]:
    """Per blind_depth, pick mode optimal_n from train teacher labels."""
    from collections import Counter, defaultdict

    buckets: Dict[int, Counter] = defaultdict(Counter)
    for row in train_rows:
        buckets[int(row["blind_depth"])][int(row["optimal_n"])] += 1

    table: Dict[int, int] = {}
    details = []
    for d in sorted(buckets):
        optimal_n, count = buckets[d].most_common(1)[0]
        n = max(min_n, min(cap, int(optimal_n)))
        table[d] = n
        details.append({"blind_depth": d, "optimal_n": n, "train_count": count, "train_total": sum(buckets[d].values())})

    return table, {"lookup_table": details, "buckets": {str(k): dict(v) for k, v in buckets.items()}}


def make_lookup_budget_fn(
    table: Dict[int, int],
    *,
    min_n: int,
    cap: int,
) -> BudgetFn:
    def _fn(sample: dict) -> int:
        d = blind_depth(sample)
        n = table.get(d, d)
        return max(min_n, min(cap, int(n)))

    return _fn


def make_structure_budget_fn(*, min_n: int, cap: int) -> BudgetFn:
    def _fn(sample: dict) -> int:
        return max(min_n, min(cap, blind_depth(sample)))

    return _fn


def make_oracle_teacher_budget_fn(
    labels: Dict[int, int],
    *,
    min_n: int,
    cap: int,
    fallback: BudgetFn,
) -> BudgetFn:
    """Perfect teacher budget from precomputed sample index → optimal_n (Exp34)."""

    def _fn(sample: dict) -> int:
        key = int(sample.get("_idx", -1))
        if key in labels:
            return max(min_n, min(cap, labels[key]))
        return fallback(sample)

    return _fn


def make_mlp_budget_fn(
    head: GraphBudgetMLP,
    *,
    min_n: int,
    cap: int,
    device: torch.device,
) -> BudgetFn:
    head.to(device)
    head.eval()

    def _fn(sample: dict) -> int:
        x = extract_graph_features(sample, cap=cap).unsqueeze(0).to(device)
        with torch.no_grad():
            cls = int(head(x).argmax(dim=-1).item())
        return max(min_n, min(cap, min_n + cls))

    return _fn


def make_delta_mlp_budget_fn(
    head: GraphDeltaMLP,
    *,
    min_n: int,
    cap: int,
    device: torch.device,
) -> BudgetFn:
    head.to(device)
    head.eval()

    def _fn(sample: dict) -> int:
        d = blind_depth(sample)
        x = extract_graph_features(sample, cap=cap).unsqueeze(0).to(device)
        with torch.no_grad():
            delta = int(head(x).argmax(dim=-1).item()) - 1
        return max(min_n, min(cap, d + delta))

    return _fn


@torch.no_grad()
def evaluate_upfront_budget_stop(
    model,
    tokenizer,
    samples: Sequence[dict],
    budget_fn: BudgetFn,
    *,
    strategy_id: str,
    strategy_label: str,
    cap: int,
    min_n: int,
    device: torch.device,
    seed: int,
    predict_fn,
    expected_fn,
    eval_profile,
    progress_cb=None,
    extra_params: Optional[dict] = None,
) -> dict:
    """Single forward per sample; timing metric uses offline first_correct scan."""
    correct = total = stop_sum = timing_hits = timing_total = 0
    stop_hist: Dict[str, int] = {}
    budget_hist: Dict[str, int] = {}

    for idx, sample in enumerate(samples):
        expected = expected_fn(sample, eval_profile)
        n_pred = budget_fn(sample)
        pred = predict_fn(
            model,
            tokenizer,
            sample,
            n_pred,
            device,
            seed=seed + idx * 31 + n_pred,
            eval_profile=eval_profile,
        )
        first_correct, _ = first_correct_step(
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

        total += 1
        if pred == expected:
            correct += 1
        stop_sum += n_pred
        stop_hist[str(n_pred)] = stop_hist.get(str(n_pred), 0) + 1
        budget_hist[str(n_pred)] = budget_hist.get(str(n_pred), 0) + 1
        if first_correct is not None:
            timing_total += 1
            if n_pred == first_correct:
                timing_hits += 1
        if progress_cb:
            progress_cb(idx + 1, len(samples))

    acc = correct / total if total else 0.0
    params = {
        "min_n": min_n,
        "cap": cap,
        "single_forward": True,
        "inference_probes": 1,
        "label_mode": "upfront_budget",
    }
    if extra_params:
        params.update(extra_params)

    return {
        "strategy_id": strategy_id,
        "strategy_label": strategy_label,
        "accuracy": round(acc, 4),
        "correct": correct,
        "total": total,
        "mean_stop_n": round(stop_sum / total, 2) if total else None,
        "stop_n_histogram": stop_hist,
        "budget_histogram": budget_hist,
        "stop_timing_acc": round(timing_hits / timing_total, 4) if timing_total else None,
        "stop_timing_hits": timing_hits,
        "stop_timing_total": timing_total,
        "params": params,
        "eval_split": "test",
    }
