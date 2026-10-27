"""Read marked examples and capture selected output regions from named sources."""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import tokenize
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent

SNIPPET_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
DEFAULT_OUTPUT_TIMEOUT = 300


@dataclass(frozen=True)
class Snippet:
    snippet_id: str
    source_path: Path
    start_line: int  # 1-based, first content line inside region
    end_line: int  # 1-based, last content line inside region
    content: str
    suffix: str
    output: bool = False
    callouts: dict[int, int] = field(default_factory=dict)  # printed line -> number


_STYLE_PATTERNS: list[tuple[str, re.Pattern[str], re.Pattern[str]]] = [
    (
        "hash",
        re.compile(r"^\s*#\s*region\s+book:(?P<id>[A-Za-z0-9_.-]+)(?P<output>\s+output)?\s*$"),
        re.compile(r"^\s*#\s*endregion(?:\s+book:(?P<id>[A-Za-z0-9_.-]+))?\s*$"),
    ),
    (
        "slash",
        re.compile(r"^\s*//\s*region\s+book:(?P<id>[A-Za-z0-9_.-]+)\s*$"),
        re.compile(r"^\s*//\s*endregion(?:\s+book:(?P<id>[A-Za-z0-9_.-]+))?\s*$"),
    ),
    (
        "html",
        re.compile(r"^\s*<!--\s*region\s+book:(?P<id>[A-Za-z0-9_.-]+)\s*-->\s*$"),
        re.compile(r"^\s*<!--\s*endregion(?:\s+book:(?P<id>[A-Za-z0-9_.-]+))?\s*-->\s*$"),
    ),
]


def _detect_style(line: str) -> tuple[str, re.Pattern[str], re.Pattern[str]] | None:
    for style, start_re, end_re in _STYLE_PATTERNS:
        if start_re.match(line):
            return style, start_re, end_re
    return None


def extract_snippets_from_text(text: str, source_path: Path) -> list[Snippet]:
    lines = text.splitlines(keepends=True)
    # Tokenize the source, not individual lines: marker-like text in a string
    # (including a multiline string) is code, never a callout comment.
    markers = {}
    if source_path.suffix == ".py" and re.search(r"#\s*\(\d+\)", text):
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.COMMENT:
                marker = re.search(r"#\s*\((\d+)\)\s*$", token.string)
                if marker:
                    markers[token.start[0]] = (token.start[1] + marker.start(), marker[1])

    snippets: list[Snippet] = []
    active: tuple[str, str, int, int, list[str], re.Pattern[str], bool] | None = None
    # active = (id, style, marker_line, content_start, content, end_regex, output)

    for idx0, line in enumerate(lines):
        line_no = idx0 + 1

        if active is None:
            style_match = _detect_style(line)
            if style_match is None:
                continue

            style, start_re, end_re = style_match
            match = start_re.match(line)
            if match is None:
                continue
            snippet_id = match.group("id")
            if not SNIPPET_ID_RE.match(snippet_id):
                raise ValueError(
                    f"{source_path}:{line_no}: invalid snippet id {snippet_id!r} "
                    f"(must match {SNIPPET_ID_RE.pattern})"
                )
            output = bool(match.groupdict().get("output"))
            if output and source_path.suffix != ".py":
                raise ValueError(
                    f"{source_path}:{line_no}: output capture requires a Python region"
                )
            active = (snippet_id, style, line_no, line_no + 1, [], end_re, output)
            continue

        snippet_id, style, start_marker_line, content_start_line, content_lines, end_re, output = (
            active
        )
        end_match = end_re.match(line)
        if end_match is not None:
            end_id = end_match.group("id")
            if end_id is not None and end_id != snippet_id:
                raise ValueError(
                    f"{source_path}:{line_no}: endregion id {end_id!r} "
                    f"does not match start id {snippet_id!r} (started at line {start_marker_line})"
                )
            callouts = {}
            for offset, original in enumerate(content_lines):
                marker = markers.get(content_start_line + offset)
                if marker is None:
                    continue
                column, number = marker
                if number != str(len(callouts) + 1):
                    raise ValueError(
                        f"{source_path}:{content_start_line + offset}: "
                        f"callouts in {snippet_id} must run 1, 2, 3 in line order"
                    )
                printed = original[:column].rstrip(" \t")
                if not printed.strip():
                    raise ValueError(
                        f"{source_path}:{content_start_line + offset}: "
                        "callout must follow a nonempty printed line"
                    )
                ending = original[len(original.rstrip("\r\n")) :]
                content_lines[offset] = printed + ending
                callouts[offset + 1] = int(number)
            content = dedent("".join(content_lines))
            # Keep original file suffix to aid humans, even though Quarto includes treat content as text.
            suffix = source_path.suffix or ".txt"
            # content_start_line is inside region; end_line is last content line
            end_line = line_no - 1
            snippets.append(
                Snippet(
                    snippet_id=snippet_id,
                    source_path=source_path,
                    start_line=content_start_line,
                    end_line=end_line,
                    content=content,
                    suffix=suffix,
                    output=output,
                    callouts=callouts,
                )
            )
            active = None
            continue

        # Still inside region
        content_lines.append(line)
        active = (
            snippet_id,
            style,
            start_marker_line,
            content_start_line,
            content_lines,
            end_re,
            output,
        )

    if active is not None:
        (
            snippet_id,
            _style,
            start_marker_line,
            _content_start_line,
            _content_lines,
            _end_re,
            _output,
        ) = active
        raise ValueError(
            f"{source_path}:{start_marker_line}: region book:{snippet_id} was not closed with an endregion marker"
        )

    return snippets


def capture_outputs(
    snippets: dict[str, Snippet],
    python: str,
    timeout: int = DEFAULT_OUTPUT_TIMEOUT,
) -> dict[str, str]:
    """Run each marked source through its last output region in a fresh process.

    Earlier regions and intervening setup run once in the same namespace; only
    the marked region's stdout is retained. Nothing after the last one runs.
    """
    outputs = {}
    sources = sorted({s.source_path for s in snippets.values() if s.output})
    runner = """import __future__, contextlib, io, json, sys
payload = json.load(sys.stdin)
lines = payload["text"].splitlines(keepends=True)
source = payload["source"]
flags = compile(payload["text"], source, "exec").co_flags & sum(
    getattr(__future__, name).compiler_flag for name in __future__.all_feature_names)
namespace = {"__name__": "__main__", "__file__": source}
outputs = {}
previous = 0
for region in payload["regions"]:
    start, end = region["start"], region["end"]
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile("\\n" * previous + "".join(lines[previous:start]), source,
                     "exec", flags=flags, dont_inherit=True), namespace)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile("\\n" * start + "".join(lines[start:end]), source,
                     "exec", flags=flags, dont_inherit=True), namespace)
    outputs[region["id"]] = stdout.getvalue()
    previous = end
print(json.dumps(outputs))
"""
    for source in sources:
        regions = sorted(
            (s for s in snippets.values() if s.source_path == source and s.output),
            key=lambda s: s.start_line,
        )
        last = regions[-1].end_line
        payload = {
            "source": str(source),
            "text": "".join(source.read_text(encoding="utf-8").splitlines(keepends=True)[:last]),
            "regions": [
                {"id": s.snippet_id, "start": s.start_line - 1, "end": s.end_line} for s in regions
            ],
        }
        with TemporaryDirectory(prefix="book-outputs-") as directory:
            try:
                result = subprocess.run(
                    [python, "-c", runner],
                    input=json.dumps(payload),
                    cwd=directory,
                    env={**os.environ, "PYTHONHASHSEED": "0"},
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise ValueError(
                    f"Output capture timed out after {timeout}s: {source}. "
                    "Increase --output-timeout or BOOK_OUTPUT_TIMEOUT_SECONDS."
                ) from exc
            except OSError as exc:
                raise ValueError(
                    f"Cannot run companion Python {python}: {exc}. Set BOOK_COMPANION_PYTHON "
                    "to the companion environment's Python."
                ) from exc
        if result.returncode:
            raise ValueError(
                f"Output capture failed with {python}: {source}\n{result.stderr}"
                "Set BOOK_COMPANION_PYTHON to a companion Python with the required dependencies."
            )
        outputs.update(json.loads(result.stdout))
    return outputs
