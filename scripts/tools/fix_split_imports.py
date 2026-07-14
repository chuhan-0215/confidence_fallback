#!/usr/bin/env python3
"""Add imports to split stop_head / boundary_budget submodules."""
from __future__ import annotations

from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]

TORCH_HEADER = '''from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

'''

STOP_HEAD_HEADERS = {
    "latent.py": TORCH_HEADER,
    "examples.py": TORCH_HEADER,
    "features.py": TORCH_HEADER,
    "train.py": TORCH_HEADER + "from stop_head.models import RichStopHead, RichStopExample\n\n",
    "eval.py": TORCH_HEADER + (
        "from stop_head.models import LatentStopHead, RichStopHead\n"
        "from stop_head.latent import extract_latent_hidden\n"
        "from stop_head.examples import first_correct_step\n"
        "from stop_head.features import _rich_step_features\n"
        "from stop_head.train import _rich_tensors, focal_bce_with_logits\n\n"
    ),
    "joint.py": TORCH_HEADER + (
        "from stop_head.models import RichStopHead\n"
        "from stop_head.latent import extract_latent_hidden, extract_latent_hidden_trainable\n"
        "from stop_head.examples import build_rich_stop_metadata_for_samples\n"
        "from stop_head.features import _rich_step_features\n"
        "from stop_head.train import focal_bce_with_logits\n\n"
    ),
    "splits.py": TORCH_HEADER,
}

BOUNDARY_HEADERS = {
    "graph_models.py": "from __future__ import annotations\n\nimport torch\nimport torch.nn as nn\nfrom typing import Dict, List, Optional, Sequence, Tuple\n\nfrom boundary_budget.core import extract_graph_features, extract_rich_graph_features\n\n",
    "prompt_models.py": "from __future__ import annotations\n\nimport torch\nimport torch.nn as nn\nfrom typing import Dict, List, Optional, Sequence, Tuple\n\nfrom boundary_budget.core import extract_coconut_prompt_hidden\nfrom boundary_budget.graph_models import train_d4_knn_bank\n\n",
    "factories.py": "from __future__ import annotations\n\nfrom typing import Callable, Dict, List, Optional, Sequence\n\nimport torch\n\nfrom boundary_budget.core import blind_depth\n\n",
    "eval.py": "from __future__ import annotations\n\nfrom typing import Callable, Dict, List, Optional, Sequence\n\nimport torch\n\nfrom boundary_budget.core import blind_depth\n\n",
}


def _strip_existing_header(text: str) -> str:
    if text.startswith("from __future__ import annotations"):
        parts = text.split("\n\n", 1)
        if len(parts) == 2 and parts[1]:
            return parts[1]
    return text


def apply_headers(pkg: Path, headers: dict[str, str]) -> None:
    for fname, header in headers.items():
        path = pkg / fname
        if not path.is_file():
            continue
        body = _strip_existing_header(path.read_text(encoding="utf-8"))
        path.write_text(header + body, encoding="utf-8")


def main() -> None:
    apply_headers(SCRIPTS / "stop_head", STOP_HEAD_HEADERS)
    apply_headers(SCRIPTS / "boundary_budget", BOUNDARY_HEADERS)
    print("Headers patched.")


if __name__ == "__main__":
    main()
