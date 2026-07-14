#!/usr/bin/env python3
"""将 partials 拼回 index.html（实验结论区由 build_findings_summary 单独注入）。"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAB_PAGE = ROOT / "lab.html"
LAB_PARTIAL = ROOT / "partials" / "lab-section.html"
LAB_START = "<!-- LAB_SECTION_START -->"
LAB_END = "<!-- LAB_SECTION_END -->"


def inject_lab_section(target: Path = LAB_PAGE) -> None:
    if not LAB_PARTIAL.is_file():
        raise SystemExit(f"Missing {LAB_PARTIAL}")
    lab = LAB_PARTIAL.read_text(encoding="utf-8").strip() + "\n"
    text = target.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(LAB_START) + r".*?" + re.escape(LAB_END), re.DOTALL)
    if not pattern.search(text):
        raise SystemExit(f"Lab markers not found in {target}")
    replacement = f"{LAB_START}\n{lab}{LAB_END}"
    target.write_text(pattern.sub(replacement, text, count=1), encoding="utf-8")


def main() -> None:
    inject_lab_section()
    print(f"Updated {LAB_PAGE} lab section from {LAB_PARTIAL}")


if __name__ == "__main__":
    main()
