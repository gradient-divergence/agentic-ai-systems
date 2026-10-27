"""Capture only this chapter; sealed sources are never enumerated or executed."""

import subprocess
import sys
from pathlib import Path

import pytest

from demos import regions as _regions

ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR = vars(_regions)
capture_outputs = EXTRACTOR["capture_outputs"]
extract_snippets_from_text = EXTRACTOR["extract_snippets_from_text"]
SOURCE = ROOT / "demos/future_directions_demo.py"
PREFIX = "fnd.future-directions."
NAMES = (
    "requirement-01",
    "decision-01",
    "decision-02",
    "adopt-01",
    "watch-01",
    "power-01",
    "confirmation-01",
    "threshold-01",
    "keys-01",
)


@pytest.fixture(scope="module")
def capture_directory(tmp_path_factory):
    return tmp_path_factory.mktemp("future-directions-run")


@pytest.fixture(scope="module")
def captured(capture_directory):
    snippets = extract_snippets_from_text(SOURCE.read_text(), SOURCE)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("FUTURE_DIRECTIONS_RUN_DIRECTORY", str(capture_directory))
        return capture_outputs(
            {s.snippet_id: s for s in snippets},
            sys.executable,
        )


@pytest.mark.parametrize("name", NAMES)
def test_every_output_regenerates(name, captured):
    assert (
        captured[PREFIX + name].encode()
        == (ROOT / "book_outputs" / ("_" + PREFIX + name + ".txt")).read_bytes()
    )
    assert "companion/" not in captured[PREFIX + name]


def test_stored_run_emits_the_same_bytes(captured, capture_directory):
    text = subprocess.check_output(
        [sys.executable, str(SOURCE), "--read-run", "--run-dir", str(capture_directory)],
        text=True,
    )
    assert text == "".join(captured[PREFIX + name] for name in NAMES)


def test_printed_decision_runs_alone_and_fits_the_budget(captured, capture_directory):
    path = ROOT / "refund_agent/renewal/future_directions_decide.py"
    snippet = extract_snippets_from_text(path.read_text(), path)[0]
    lines = snippet.content.rstrip().splitlines()
    assert len(lines) <= 22
    assert all(len(line) <= 79 for line in lines)
    namespace = {}
    exec(compile(snippet.content, str(path), "exec"), namespace)
    from demos.future_directions_demo import load
    from refund_agent.renewal.future_directions_evidence import Evidence
    from refund_agent.renewal.future_directions_requirement import ReplacementRequirement

    result = load(capture_directory)
    req = ReplacementRequirement.model_validate(result["requirement"])
    ev = Evidence.model_validate(result["evidence"])
    assert namespace["decide"](req, ev, 1).verdict == "keep"



def test_region_reader_finds_the_chapter_output_contract():
    regions=extract_snippets_from_text(SOURCE.read_text(),SOURCE)
    assert {r.snippet_id for r in regions if r.output}=={PREFIX+name for name in NAMES}
