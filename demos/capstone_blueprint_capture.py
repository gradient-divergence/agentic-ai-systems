"""Capture only chapter 39's explicit source, never enumerate demo data."""

import os
import sys
from pathlib import Path

from demos import regions as _regions

ROOT = Path(__file__).resolve().parents[1]


def capture_chapter(destination: Path, *, run_dir: Path) -> dict[str, str]:
    extractor = vars(_regions)
    path = ROOT / "demos/capstone_blueprint_demo.py"
    snippets = {
        s.snippet_id: s
        for s in extractor["extract_snippets_from_text"](
            path.read_text(), path
        )
        if s.snippet_id.startswith("fnd.capstone-blueprint.")
    }
    prior = os.environ.get("CAPSTONE_BLUEPRINT_RUN")
    os.environ["CAPSTONE_BLUEPRINT_RUN"] = str(run_dir.resolve())
    try:
        outputs = extractor["capture_outputs"](
            snippets, sys.executable, timeout=60
        )
    finally:
        if prior is None:
            os.environ.pop("CAPSTONE_BLUEPRINT_RUN", None)
        else:
            os.environ["CAPSTONE_BLUEPRINT_RUN"] = prior
    destination.mkdir(parents=True, exist_ok=True)
    for name, output in outputs.items():
        (destination / f"_{name}.txt").write_text(output)
    return outputs
