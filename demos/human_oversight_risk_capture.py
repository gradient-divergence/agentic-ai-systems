"""Explicit chapter-37 sources; use the repository's output-capture function."""

import sys
from pathlib import Path

from demos import regions as _regions

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ROOT / "refund_agent/oversight/human_oversight_risk_queueing.py",
    ROOT / "refund_agent/oversight/human_oversight_risk_deadline.py",
    ROOT / "demos/human_oversight_risk_demo.py",
    ROOT / "tests/refund_agent/test_human_oversight_risk_lifecycle.py",
)


def capture_chapter(destination):
    extractor = vars(_regions)
    snippets = {s.snippet_id: s for source in SOURCES
                for s in extractor["extract_snippets_from_text"](source.read_text(), source)}
    outputs = extractor["capture_outputs"](snippets, sys.executable, timeout=900)
    out = Path(destination) / "book_outputs"
    out.mkdir(parents=True, exist_ok=True)
    for name, text in outputs.items():
        (out / f"_{name}.txt").write_text(text, encoding="utf-8")
    return outputs


if __name__ == "__main__":
    capture_chapter(ROOT)
