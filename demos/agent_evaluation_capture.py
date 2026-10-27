"""Capture chapter 29's text and records from one fresh, final-code experiment."""

import os
import sys
from pathlib import Path

from demos import regions as _regions

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "fnd.agent-evaluation."


def capture_chapter(destination):
    destination = Path(destination).resolve()
    extractor = vars(_regions)
    snippets = {}
    for relative in (
        "demos/agent_evaluation_release_demo.py",
        "refund_agent/release/trials.py",
    ):
        path = ROOT / relative
        snippets.update(
            {
                s.snippet_id: s
                for s in extractor["extract_snippets_from_text"](path.read_text(), path)
                if s.snippet_id.startswith(PREFIX)
            }
        )
    prior = os.environ.get("AGENT_EVALUATION_CAPTURE_DIR")
    os.environ["AGENT_EVALUATION_CAPTURE_DIR"] = str(destination)
    try:
        outputs = extractor["capture_outputs"](snippets, sys.executable, timeout=600)
    finally:
        if prior is None:
            os.environ.pop("AGENT_EVALUATION_CAPTURE_DIR", None)
        else:
            os.environ["AGENT_EVALUATION_CAPTURE_DIR"] = prior
    out = destination / "book_outputs"
    out.mkdir(parents=True, exist_ok=True)
    for name, text in outputs.items():
        (out / f"_{name}.txt").write_text(text)
    return outputs
