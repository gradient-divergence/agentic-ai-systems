"""Print book output within its 79-column measure without changing tokens."""

from itertools import zip_longest
from textwrap import wrap


def _wrapped(line, width):
    return wrap(line, width, break_long_words=False, break_on_hyphens=False) or [""]


def _table_lines(line):
    """Continue long cells on blank-leading rows of the same Markdown table."""
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    widths = [len(cell) for cell in cells]
    minimum = [max(map(len, cell.split()), default=0) for cell in cells]
    available = 79 - (3 * len(cells) + 1)
    if sum(minimum) > available:
        raise ValueError("Abbreviate the table headings to fit the book measure")
    while sum(widths) > available:
        column = max(
            (i for i in range(len(cells)) if widths[i] > minimum[i]),
            key=lambda i: widths[i],
        )
        widths[column] -= 1
    parts = [_wrapped(cell, max(1, width)) for cell, width in zip(cells, widths)]
    return ["| " + " | ".join(row) + " |" for row in zip_longest(*parts, fillvalue="")]


def print_wrapped(*values, sep=" ", end="\n"):
    """A presentation-only print: retain short lines, indent long continuations.

    Long words (notably dates, citations and fingerprints) stay intact. Callers
    pretty-print compact JSON before passing it here. Numeric spelling is never
    parsed, rounded or replaced.
    """
    text = sep.join(str(value) for value in values)
    lines = []
    for line in text.split("\n"):
        if len(line) <= 79:
            lines.append(line)
        elif line.startswith("|") and line.endswith("|"):
            lines.extend(_table_lines(line))
        else:
            indent = line[: len(line) - len(line.lstrip())]
            parts = _wrapped(line.strip(), 79 - len(indent) - 2)
            lines.extend(indent + ("  " if i else "") + part for i, part in enumerate(parts))
    print("\n".join(lines), end=end)
