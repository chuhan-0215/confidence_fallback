#!/usr/bin/env python3
"""Y18 · hybrid 跨子集泛化（boundary_push）。"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

sys.argv = [sys.argv[0], "--group", "boundary_push", *sys.argv[1:]]
runpy.run_path(str(Path(__file__).resolve().parent / "run_y18_hybrid_slices_core.py"), run_name="__main__")
