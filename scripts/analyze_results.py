#!/usr/bin/env python3
"""Re-analyze existing experiment results (no model re-run)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from boundary_analysis import analyze_boundary, resolve_boundary  # noqa: E402
from evaluate_coconut import load_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(ROOT / "results" / "latest.json"),
    )
    parser.add_argument(
        "--data",
        default=str(ROOT / "data" / "prosqa_test_graph_4_coconut.json"),
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    inp = Path(args.input)
    data = json.loads(inp.read_text(encoding="utf-8"))
    if not data.get("ok"):
        raise SystemExit("Input results not ok: " + str(data.get("error")))

    count = data.get("dataset", {}).get("count")
    dataset = load_dataset(Path(args.data), max_samples=count)

    why = analyze_boundary(
        dataset,
        data.get("latent_sweep") or [],
        data.get("boundary") or {},
        data.get("by_reasoning_hops") or None,
    )
    peak_steps, peak_acc = resolve_boundary(data.get("latent_sweep") or [], data.get("boundary") or {})
    if data.get("boundary") is not None:
        data["boundary"]["recommended_latent_steps"] = peak_steps
        data["boundary"]["max_accuracy"] = peak_acc
        data["boundary"]["max_accuracy_at_steps"] = peak_steps
    data["why_analysis"] = why
    if data.get("theoretical") is not None:
        data["theoretical"]["graph_profile"] = why.get("dataset_graph_profile")

    out = Path(args.output) if args.output else inp
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(why, ensure_ascii=False, indent=2))
    print(f"Updated -> {out}")


if __name__ == "__main__":
    main()
