"""The book layout preserves content, numeric spelling and Markdown columns."""

import json
import re
from collections import Counter

import pytest

from demos.output_wrap import print_wrapped


@pytest.mark.parametrize(
    "text",
    [
        "estimate -0.1149 [-0.1798, -0.0499]; " * 4,
        "  fingerprint: " + "a7a7cf68978cc0c8" * 4,
        "refund_cap_v4@2026-01-20#eligibility vs " * 4,
        "| Tier threshold, 0.080 | 880 | 119 | 12 (0.008 to 0.024) "
        "| 0.80 (0.61 to 1.01) | 0.27 (−0.01 to 0.59) |",
    ],
)
def test_reflow_preserves_tokens_and_numeric_forms(text, capsys):
    print_wrapped(text)
    output = capsys.readouterr().out
    assert max(map(len, output.splitlines())) <= 79
    if text.startswith("|"):
        rows = [line.strip("|").split("|") for line in output.splitlines()]
        expected = [cell.split() for cell in text.strip("|").split("|")]
        rebuilt = [" ".join(row[i] for row in rows).split() for i in range(len(expected))]
        assert rebuilt == expected
        assert all(line.count("|") == text.count("|") for line in output.splitlines())
    else:
        assert output.split() == text.split()
    assert Counter(re.findall(r"[+−-]?\d+(?:\.\d+)?", output)) == Counter(
        re.findall(r"[+−-]?\d+(?:\.\d+)?", text)
    )


def test_short_output_retains_exact_whitespace_and_end(capsys):
    print_wrapped("  x    y", "0.0000", sep=" : ", end="")
    assert capsys.readouterr().out == "  x    y : 0.0000"


def test_pretty_json_preserves_values_and_fingerprint(capsys):
    record = {"amount_cents": 4999, "tier": 1, "sha256": "a7a7cf68978cc0c8" * 4}
    print_wrapped(json.dumps(record, indent=2))
    output = capsys.readouterr().out
    assert json.loads(output) == record
    assert max(map(len, output.splitlines())) <= 79
