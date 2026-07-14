#!/usr/bin/env python3
"""Split monolithic modules into packages (one-time maintenance tool)."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"


def split_stop_head() -> None:
    src_path = SCRIPTS / "stop_head.py"
    if not src_path.is_file():
        return
    text = src_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    pkg = SCRIPTS / "stop_head"
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir()

    sections = [
        ("models.py", 0, 97),
        ("latent.py", 99, 207),
        ("examples.py", 208, 433),
        ("train.py", 435, 574),
        ("features.py", 576, 588),
        ("eval.py", 589, 1321),
        ("joint.py", 1322, 1695),
        ("splits.py", 1697, len(lines)),
    ]
    header = 'from __future__ import annotations\n\n'
    for fname, start, end in sections:
        chunk = "".join(lines[start:end])
        if fname != "models.py":
            chunk = header + chunk
        (pkg / fname).write_text(chunk, encoding="utf-8")

    init = '''"""Stop head package — re-exports for backward compatibility."""
from stop_head.models import *
from stop_head.latent import *
from stop_head.examples import *
from stop_head.features import *
from stop_head.train import *
from stop_head.eval import *
from stop_head.joint import *
from stop_head.splits import *

__all__ = [name for name in globals() if not name.startswith("_")]
'''
    (pkg / "__init__.py").write_text(init, encoding="utf-8")
    src_path.unlink()


def split_boundary_budget() -> None:
    src_path = SCRIPTS / "boundary_budget.py"
    if not src_path.is_file():
        return
    text = src_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    pkg = SCRIPTS / "boundary_budget"
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir()

    sections = [
        ("core.py", 0, 185),
        ("graph_models.py", 185, 506),
        ("prompt_models.py", 506, 1290),
        ("factories.py", 1290, 1502),
        ("eval.py", 1502, len(lines)),
    ]
    header = 'from __future__ import annotations\n\n'
    for fname, start, end in sections:
        chunk = "".join(lines[start:end])
        if fname != "core.py":
            chunk = header + chunk
        (pkg / fname).write_text(chunk, encoding="utf-8")

    init = '''"""Boundary budget package — re-exports for backward compatibility."""
from boundary_budget.core import *
from boundary_budget.graph_models import *
from boundary_budget.prompt_models import *
from boundary_budget.factories import *
from boundary_budget.eval import *

__all__ = [name for name in globals() if not name.startswith("_")]
'''
    (pkg / "__init__.py").write_text(init, encoding="utf-8")
    src_path.unlink()


def extract_dataset_specs() -> None:
    reg_path = SCRIPTS / "dataset_registry.py"
    text = reg_path.read_text(encoding="utf-8")
    m = re.search(r"^SLICE_SPECS:", text, re.M)
    n = re.search(r"^PATTERN_SLICE_IDS", text, re.M)
    if not m or not n:
        return
    specs_block = text[m.start():n.start()]
    ids_block = text[n.start(): text.index("_ALL_SPECS")]
    spec_path = SCRIPTS / "dataset_slice_specs.py"
    spec_path.write_text(
        '"""Dataset slice specification tables (data-only)."""\n'
        "from __future__ import annotations\n\n"
        "from typing import List\n\n"
        + specs_block
        + ids_block,
        encoding="utf-8",
    )
    new_reg = text[: m.start()] + (
        "from dataset_slice_specs import (  # noqa: E402\n"
        "    BOUNDARY_PUSH_DEEP_SLICE_IDS,\n"
        "    BOUNDARY_PUSH_DEEP_SLICE_SPECS,\n"
        "    BOUNDARY_PUSH_SLICE_IDS,\n"
        "    BOUNDARY_PUSH_SLICE_SPECS,\n"
        "    DEEP_SLICE_IDS,\n"
        "    DEEP_SLICE_SPECS,\n"
        "    PATTERN_SLICE_IDS,\n"
        "    PATTERN_SLICE_SPECS,\n"
        "    SLICE_SPECS,\n"
        "    VARIANT_SLICE_IDS,\n"
        "    VARIANT_SLICE_SPECS,\n"
        ")\n\n"
    ) + text[text.index("_ALL_SPECS") :]
    reg_path.write_text(new_reg, encoding="utf-8")


def patch_phase_commons() -> None:
    for path in SCRIPTS.glob("phase*/_phase*_common.py"):
        text = path.read_text(encoding="utf-8")
        m = re.search(r"def write_phase(\d+)_result\(", text)
        if not m:
            m2 = re.search(r"def write_phase\d+_result\(experiment_id", text)
            if not m2:
                continue
        phase_m = re.search(r"phase(\d+)", path.parent.name)
        if not phase_m:
            continue
        phase = int(phase_m.group(1))
        # Replace write function body with delegate
        fn_pattern = re.compile(
            r"def write_phase\d+_result\([^)]+\)[^:]*:\n(?:    .+\n)+?(?=\n\n|\ndef |\Z)",
            re.M,
        )
        param = "eid" if "eid: str" in text else "experiment_id"
        replacement = (
            f"def write_phase{phase}_result({param}: str, payload: dict) -> Path:\n"
            f"    from shared.phase_io import write_phase_result\n"
            f"    return write_phase_result({phase}, {param}, payload)\n"
        )
        if "from shared.phase_io import write_phase_result" in text:
            continue
        new_text = fn_pattern.sub(replacement, text, count=1)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")


def main() -> None:
    extract_dataset_specs()
    split_stop_head()
    split_boundary_budget()
    patch_phase_commons()
    print("Refactor split complete.")


if __name__ == "__main__":
    main()
