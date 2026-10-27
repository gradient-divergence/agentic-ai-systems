"""Appendix A includes each listing of demos/appendix_math_demo.py and its stdout
in a plain fence straight after the listing. These tests run the listings in
the order the appendix prints them, in one namespace as a reader would, and
check that each prints exactly what the book shows."""

import contextlib
import io
import re
from pathlib import Path

import numpy as np
import pytest

from demos import regions as _regions

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT.parent / "backmatter/appendix-math-foundations.qmd"
DEMO = ROOT / "demos/appendix_math_demo.py"
extract = vars(_regions)[
    "extract_snippets_from_text"
]
LISTING = re.compile(
    r"```python\n\{\{< include \.\./_includes/snippets/_(?P<id>fnd\.appendix-math-foundations"
    r"\.[\w-]+)\.py >\}\}\n```\n\n```\n"
    r"\{\{< include (?P<output>\.\./companion/book_outputs/_(?P=id)\.txt) >\}\}\n```",
    re.DOTALL,
)


def regions() -> dict[str, str]:
    return {s.snippet_id: s.content for s in extract(DEMO.read_text(encoding="utf-8"), DEMO)}


def printed_listings() -> list[tuple[str, str]]:
    return [(m["id"], (BOOK.parent / m["output"]).read_bytes().decode("utf-8"))
            for m in LISTING.finditer(BOOK.read_text(encoding="utf-8"))]


def run(code: str, namespace: dict[str, object]) -> str:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(code, str(DEMO), "exec"), namespace)
    return stdout.getvalue()


def test_every_demo_region_is_printed_with_its_output() -> None:
    assert set(regions()) == {snippet_id for snippet_id, _ in printed_listings()}


def test_printed_output_matches_a_run_in_print_order() -> None:
    code, namespace = regions(), {}
    listings = printed_listings()
    assert listings, "no listing with output found in the appendix"
    for snippet_id, expected in listings:
        assert run(code[snippet_id], namespace) == expected, snippet_id


@pytest.mark.parametrize("label, loss", [(0, 0.408), (2, 2.408)])
def test_cross_entropy_ignores_a_shift_of_all_scores(label: int, loss: float) -> None:
    namespace: dict[str, object] = {}
    for name in ("softmax-01", "crossentropy-01"):  # the order the book prints them
        run(regions()[f"fnd.appendix-math-foundations.{name}"], namespace)
    z = np.array([2.0, 1.0, 0.0])
    assert namespace["cross_entropy"](z, label) == pytest.approx(loss, abs=5e-4)
    assert namespace["cross_entropy"](z + 1000.0, label) == pytest.approx(loss, abs=5e-4)


@pytest.mark.parametrize("width", [3, 0, -2])
def test_position_functions_reject_an_odd_or_non_positive_width(width: int) -> None:
    namespace: dict[str, object] = {}
    run(regions()["fnd.appendix-math-foundations.shift-01"], namespace)
    for call in (lambda: namespace["positions"](8, width), lambda: namespace["shift"](width, 2)):
        with pytest.raises(ValueError, match="not positive and even"):
            call()
