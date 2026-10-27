"""Explicit chapter paths only; capture with the repository function."""

import sys
from pathlib import Path

from demos import regions as _regions

ROOT = Path(__file__).resolve().parents[1]


def capture_chapter(destination):
    extractor = vars(_regions)
    demo = ROOT / "demos/cloud_native_deployment_demo.py"
    snippets = {s.snippet_id: s for s in extractor["extract_snippets_from_text"](demo.read_text(), demo)}
    outputs = extractor["capture_outputs"](snippets, sys.executable, timeout=1800)
    out = Path(destination) / "book_outputs"
    out.mkdir(parents=True, exist_ok=True)
    for name, text in outputs.items():
        (out / f"_{name}.txt").write_text(text)
    return outputs
