"""Capture explicit chapter sources with the repository function, never a scan."""

import sys
from pathlib import Path

from demos import regions as _regions

ROOT = Path(__file__).resolve().parents[1]


def capture_chapter(destination, *, integrated=False):
    extractor = vars(_regions)
    name = (
        "audits_explanations_integrated_capture.py" if integrated else "audits_explanations_demo.py"
    )
    path = ROOT / "demos" / name
    snippets = {
        s.snippet_id: s
        for s in extractor["extract_snippets_from_text"](
            path.read_text(), path
        )
    }
    outputs = extractor["capture_outputs"](
        snippets, sys.executable, timeout=1800
    )
    out = Path(destination) / "book_outputs"
    out.mkdir(parents=True, exist_ok=True)
    for name, captured in outputs.items():
        (out / f"_{name}.txt").write_text(captured, encoding="utf-8")
    return outputs
