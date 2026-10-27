"""Structural acceptance only; a person must judge whether each source supports its row."""

import re
from pathlib import Path

from refund_agent.learning_case import LearningCase, verdict

ROOT = Path(__file__).resolve().parents[3]


def check_submission(case, rendered):
    if not isinstance(case, LearningCase):
        case = LearningCase.model_validate(case)
    assert LearningCase.model_validate_json(case.model_dump_json()) == case
    assert case.verdict == verdict(case.rows)
    assert "threshold" in case.decision.lower()
    assert rendered == case.render()
    assert "autonomy unchanged: drafts only, a person executes" in rendered
    for row in case.rows:
        match = re.fullmatch(r"(companion/[^:;]+):(\d+)(?:;\s*(\d+))*", row.source)
        assert match, f"source must cite companion code or output lines: {row.source}"
        source = (ROOT / match[1]).resolve()
        allowed = [
            ROOT / "companion/refund_agent",
            ROOT / "companion/demos",
            ROOT / "companion/book_outputs",
        ]
        assert any(source.is_relative_to(directory) for directory in allowed)
        assert source.suffix in (".py", ".txt"), "cite code or a stored output"
        lines = source.read_text().splitlines()
        numbers = [int(n) for n in row.source.split(":", 1)[1].split(";")]
        assert all(1 <= n <= len(lines) and lines[n - 1].strip() for n in numbers), row.source
    return case
