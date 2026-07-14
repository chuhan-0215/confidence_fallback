#!/usr/bin/env python3
"""从 results/*.json 汇总实验发现，生成 JSON + 内嵌 HTML（刷新即见）。"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from glossary import write_glossary_html  # noqa: E402
from findings.render import render_appendix_html, render_findings_html  # noqa: E402

RESULTS = ROOT / "results"
INDEX = ROOT / "index.html"
APPENDIX = ROOT / "appendix.html"
OUT = RESULTS / "findings_summary.json"
MARKER_START = "<!-- FINDINGS_AUTO_START -->"
MARKER_END = "<!-- FINDINGS_AUTO_END -->"
APPENDIX_MARKER_START = "<!-- APPENDIX_AUTO_START -->"
APPENDIX_MARKER_END = "<!-- APPENDIX_AUTO_END -->"
FINDINGS_FRAGMENT = ROOT / "partials" / "findings-auto.html"
APPENDIX_FRAGMENT = ROOT / "partials" / "appendix-auto.html"


from findings.payload import build_payload  # noqa: E402
from findings.builders import validate_feedback_schedule_copy  # noqa: E402
from assemble_index import inject_lab_section  # noqa: E402
from findings.format import load_json  # noqa: E402

def inject_fragment(path: Path, start: str, end: str, fragment: str) -> None:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(text):
        raise SystemExit(f"Markers not found in {path}")
    replacement = f"{start}\n        {fragment}\n        {end}"
    path.write_text(pattern.sub(replacement, text, count=1), encoding="utf-8")


def main() -> None:
    payload = build_payload()
    fb_json = load_json("feedback_schedule_latest.json") or {}
    fb_errors = validate_feedback_schedule_copy(fb_json, payload.get("feedback_schedule"))
    if fb_errors:
        for err in fb_errors:
            print(f"VALIDATION ERROR: {err}", file=sys.stderr)
        raise SystemExit("feedback schedule copy validation failed")
    print("Validated experiment 9 copy against feedback_schedule_latest.json")
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")
    findings_fragment = render_findings_html(payload)
    inject_fragment(INDEX, MARKER_START, MARKER_END, findings_fragment)
    FINDINGS_FRAGMENT.parent.mkdir(parents=True, exist_ok=True)
    FINDINGS_FRAGMENT.write_text(findings_fragment, encoding="utf-8")
    print(f"Updated {INDEX} findings panel")
    print(f"Wrote {FINDINGS_FRAGMENT}")

    appendix_fragment = render_appendix_html(payload)
    inject_fragment(APPENDIX, APPENDIX_MARKER_START, APPENDIX_MARKER_END, appendix_fragment)
    APPENDIX_FRAGMENT.write_text(appendix_fragment, encoding="utf-8")
    print(f"Updated {APPENDIX} appendix panel")
    print(f"Wrote {APPENDIX_FRAGMENT}")

    gpath = write_glossary_html()
    print(f"Wrote {gpath}")
    inject_lab_section()
    print(f"Synced lab section in lab.html")


if __name__ == "__main__":
    main()
