"""Printed Chapter 26 listings and their captured output bytes."""

import contextlib
import importlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from demos import regions as _regions

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demos/memory_rag_demo.py"
TOOLS = vars(_regions)
PREFIX = "fnd.memory-rag."
NAMES = (
    "funnel-01",
    "budget-01",
    "budget-02",
    "asof-01",
    "successor-01",
    "lifecycle-01",
    "support-01",
    "sweep-01",
    "keys-01",
    "journey-01",
)


def snippets():
    return {s.snippet_id: s for s in TOOLS["extract_snippets_from_text"](DEMO.read_text(), DEMO)}


def committed_output(relative):
    prefix = subprocess.check_output(
        ["git", "rev-parse", "--show-prefix"], cwd=ROOT, text=True
    ).strip()
    original = subprocess.check_output(["git", "show", f"HEAD:{prefix}{relative}"], cwd=ROOT)
    if relative == f"book_outputs/_{PREFIX}keys-01.txt":
        # The historical capture rounded this product one digit lower.
        original = original.replace(
            b"exercise 3 wrong product 0.771637\n",
            b"exercise 3 wrong product 0.771638\n",
            1,
        )
    return original


def test_printed_listings_run_in_chapter_order_without_hidden_setup():
    namespace = {"__name__": "__main__"}
    found = snippets()
    with contextlib.redirect_stdout(io.StringIO()) as printed:
        for name in ("funnel-01", "asof-01"):
            exec(compile(found[PREFIX + name].content, str(DEMO), "exec"), namespace)
    assert "beyond_notice" in printed.getvalue()
    assert namespace["before"][0].version == "2026-03-01"
    assert namespace["now"][0].version == "2026-05-01"


def test_printed_funnel_handles_an_all_miss_group(monkeypatch, capsys):
    from demos import memory_rag_demo as demo
    from refund_agent.memory.funnel import funnel

    flags = [{"wording": "miss", "ret": False, "rank": False, "kept": False}]
    monkeypatch.setattr(
        demo, "telecom_run", lambda seed: {"flags": flags, "unranked": funnel(flags, "wording")}
    )
    listing = snippets()[PREFIX + "funnel-01"]
    exec(compile(listing.content, str(DEMO), "exec"), {"__name__": "__main__"})
    printed = capsys.readouterr().out
    assert "rank|ret: undefined" in printed and "kept|rank: undefined" in printed
    assert "product undefined; end-to-end 0.000000" in printed
    assert "without reranking miss: counts [1, 0, 0, 0]; product undefined" in printed


def test_counting_excerpt_is_exact_source_and_preserves_funnel_behavior():
    source = ROOT / "refund_agent/memory/funnel.py"
    found = TOOLS["extract_snippets_from_text"](source.read_text(), source)
    assert [s.snippet_id for s in found] == [PREFIX + "funnel-02"]
    namespace = {
        "defaultdict": __import__("collections").defaultdict,
    }
    exec(compile(found[0].content, str(source), "exec"), namespace)
    flags = [{"wording": "miss", "ret": False, "rank": False, "kept": False}]
    from refund_agent.memory.funnel import funnel

    assert funnel(flags, "wording")[0]["counts"] == [1, 0, 0, 0]
    assert funnel(flags, "wording")[0]["rates"] == [0, None, None]
    assert funnel(flags, "wording")[0]["intervals"][1:] == [None, None]
    assert funnel(flags, "wording")[0]["product"] is None
    assert funnel(flags, "wording")[0]["end_to_end"] == 0
    from demos.ml_data_evaluation_demo import wilson_interval

    nested = [{"wording": "mixed", "ret": i < 3, "rank": i < 2, "kept": i < 1} for i in range(4)]
    row = funnel(nested, "wording")[0]
    assert row["counts"] == [4, 3, 2, 1]
    assert row["rates"] == [3 / 4, 2 / 3, 1 / 2]
    assert row["intervals"] == [
        list(wilson_interval(end, start)) for start, end in ((4, 3), (3, 2), (2, 1))
    ]
    assert row["product"] == row["end_to_end"] == 1 / 4
    namespace["funnel"](flags, "wording")
    flags[0]["kept"] = True
    with pytest.raises(ValueError, match="nested"):
        namespace["funnel"](flags, "wording")


def test_retrieval_journey_uses_the_cached_seeded_orders(monkeypatch):
    from demos import memory_rag_demo as demo

    assert PREFIX + "journey-01" in snippets(), "the new journey must be captured"
    experiment = demo.telecom_run(7)
    monkeypatch.setattr(demo, "telecom_library", lambda *args: pytest.fail("new randomness"))
    journey = demo.retrieval_journey(experiment)
    assert journey["question"]["id"] == "q000-paraphrase"
    assert journey["question"]["needed"] == "s12-p3-v2"
    assert journey["needed_words"] == 45
    assert [row["id"] for row in journey["orders"]["first stage"]["passages"]] == [
        "s12-p0-v2",
        "s12-p3-v2",
        "s12-p1-v3",
        "s12-p2-v2",
        "s00-p3-v1",
    ]
    assert [row["id"] for row in journey["orders"]["reranked"]["passages"]] == [
        "s12-p0-v2",
        "s12-p1-v3",
        "s12-p2-v2",
        "s12-p3-v2",
        "s00-p3-v1",
    ]
    for name, kept, used in (("first stage", True, 45), ("reranked", False, 140)):
        row = journey["orders"][name]
        assert row["rank"] is True and row["kept"] is kept
        assert row["words_before_needed"] == used
    assert [row["words"] for row in journey["orders"]["reranked"]["passages"]] == [
        45,
        95,
        95,
        45,
        45,
    ]
    assert [row["kept"] for row in journey["orders"]["reranked"]["passages"]] == [
        True,
        True,
        False,
        False,
        False,
    ]


def test_two_passage_example_preserves_committed_keys(capsys):
    from demos import memory_rag_demo as demo

    demo.print_keys()
    lines = capsys.readouterr().out.splitlines(keepends=True)
    path = ROOT / "book_outputs" / f"_{PREFIX}keys-01.txt"
    original = committed_output(path.relative_to(ROOT).as_posix())
    assert "".join(lines).encode() == path.read_bytes() == original
    assert lines[-1] == (
        "constructed two-passage case: recall 10/20 = 0.500000; complete evidence 0/10 = 0.000000; "
        "p1 = p2 = 0.500000; bounds [0.000000, 0.500000]; independent 0.250000\n"
    )


def test_exercise_three_wrong_product_rounds_the_exact_count_ratios(capsys):
    from demos import memory_rag_demo as demo

    demo.print_keys()
    # 0.95 * 0.855 * 0.95 is exactly 0.7716375, rounded to six places.
    assert capsys.readouterr().out.splitlines()[7] == "exercise 3 wrong product 0.771638"


def test_request_b_figure_listing_runs_alone_and_preserves_every_required_fact(capsys):
    found = snippets()
    assert PREFIX + "budget-02" in found
    listing = found[PREFIX + "budget-02"]
    exec(compile(listing.content, str(DEMO), "exec"), {"__name__": "__main__"})
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 12
    assert lines[:9] == [
        "shipped: 2026-04-02T00:00:00+00:00",
        "lost: 2026-04-05T00:00:00+00:00",
        "resent: 2026-04-10T00:00:00+00:00",
        "delivered: 2026-04-16T00:00:00+00:00 (latest delivery)",
        "return_scan: 2026-05-02T00:00:00+00:00",
        "return_scan: 2026-05-03T00:00:00+00:00",
        "return_scan: 2026-05-04T00:00:00+00:00",
        "return_scan: 2026-05-05T00:00:00+00:00",
        "return_scan: 2026-05-06T00:00:00+00:00",
    ]
    fields = [field for line in lines[9:] for field in line.removeprefix("required facts: ").split("; ")]
    assert len(fields) == 15
    assert dict(field.split(": ", 1) for field in fields) == {
        "request_id": "constructed-b",
        "at": "2026-05-12T00:00:00+00:00",
        "known_at": "2026-05-12T00:00:00+00:00",
        "snapshot_at": "2026-05-11T23:00:00+00:00",
        "purchased_at": "2026-04-01T00:00:00+00:00",
        "delivered_at": "2026-04-16T00:00:00+00:00",
        "delivery_age_days": "26.0",
        "future_evidence": "False",
        "amount_cents": "2499",
        "reason": "changed_mind",
        "balance_cents": "4999",
        "policy_version": "2026-05-01",
        "return_days": "30",
        "cap_cents": "90000",
        "named_approver": "morgan",
    }


@pytest.fixture(scope="module")
def captured():
    first = TOOLS["capture_outputs"](snippets(), sys.executable, timeout=300)
    assert first == TOOLS["capture_outputs"](snippets(), sys.executable, timeout=300)
    return first


@pytest.mark.parametrize("name", NAMES)
def test_every_stored_output_regenerates_byte_for_byte(name, captured):
    path = ROOT / "book_outputs" / f"_{PREFIX}{name}.txt"
    assert captured[PREFIX + name].encode() == path.read_bytes()
    if name in {"lifecycle-01", "support-01"}:
        assert max(map(len, path.read_text().splitlines())) <= 79
        assert path.read_bytes().split() == committed_output(
            path.relative_to(ROOT).as_posix()
        ).split()
    elif name != "budget-02":  # New figure data has no baseline at this stage.
        assert path.read_bytes() == committed_output(path.relative_to(ROOT).as_posix())


def test_import_does_not_print_or_generate_data():
    code = """import contextlib, io
from refund_agent.data import generate
def forbidden(*args, **kwargs):
    raise AssertionError('import generated data')
generate.generate = forbidden
with contextlib.redirect_stdout(io.StringIO()) as printed:
    import demos.memory_rag_demo
assert printed.getvalue() == ''
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_figure_and_support_data_regenerate():
    demo = importlib.import_module("demos.memory_rag_demo")
    report = demo.build_experiment()
    for name, value in (("figures", demo.figure_data(report)), ("support", report["support"])):
        path = ROOT / "demos/data" / f"memory_rag_{name}.json"
        assert path.read_text() == json.dumps(value, indent=2, sort_keys=True) + "\n"


def test_chapter_run_never_consults_golden_or_validation(monkeypatch):
    from refund_agent import split
    from refund_agent.memory import assemble

    def forbidden(*args, **kwargs):
        raise AssertionError("Chapter 26 consulted frozen rows")

    monkeypatch.setattr(split.GoldenSet, "consult", forbidden)
    monkeypatch.setattr(split, "freeze", forbidden)
    original = assemble.required_facts
    seen = []

    def training_only(state):
        assert state.request.at < split.CUTS[0]
        seen.append(state.request.request_id)
        return original(state)

    monkeypatch.setattr(assemble, "required_facts", training_only)
    demo = importlib.import_module("demos.memory_rag_demo")
    demo.build_experiment.cache_clear()
    report = demo.build_experiment()
    assert report["training_count"] == 398 and seen
    assert report["accesses"] == []


def test_stored_outputs_are_byte_identical_to_committed_ones():
    # The regeneration checks below exercise the loop and index, rather than pinning source.
    paths = subprocess.check_output(
        ["git", "ls-files", "book_outputs"], cwd=ROOT, text=True
    ).splitlines()
    assert paths
    for relative in paths:
        # Authorized diagnostic split changes this suite identity; Chapter 29 replays it.
        if relative == "book_outputs/_fnd.agent-evaluation.suite-01.txt":
            continue
        committed = committed_output(relative)
        assert (ROOT / relative).read_bytes() == committed, relative


def test_retirement_changes_no_spine_behavior_or_unrelated_source_map_row():
    path = ROOT / "spine/agent/memory.py"
    from spine.agent.memory import MemoryRecord, MemoryStore, lookup_memory

    store = MemoryStore()
    records = [
        MemoryRecord(key="Refund", value="eligible"),
        MemoryRecord(key="delivery", value="refund requested"),
    ]
    for record in records:
        store.add(record)
    assert lookup_memory(store, "REFUND") == records
    assert lookup_memory(store, "refund", limit=1) == records[:1]
    assert lookup_memory(store, "missing") == []
    for name in ("core-01", "example-01"):
        assert PREFIX + name not in path.read_text()
        assert not (ROOT / "book_outputs" / f"_{PREFIX}{name}.txt").exists()
    source_map = ROOT / "docs/source-map/snippet_plan.csv"
    original = subprocess.check_output(
        ["git", "show", f"HEAD:{source_map.relative_to(ROOT)}"], cwd=ROOT
    )

    def untouched(data):
        return [
            line for line in data.splitlines(keepends=True) if not line.startswith(b"SNIP-041,")
        ]

    assert untouched(source_map.read_bytes()) == untouched(original)


def test_earlier_solve_record_json_regenerates_byte_for_byte():
    from demos import constrained_optimization_demo as demo

    records = {
        policy: {day["day"]: day["record"].to_dict() for day in result["days"]}
        for policy, result in demo.window_results().items()
    }
    records["robust_monday"] = demo.robust_plan().to_dict()
    records["deadline_key"] = demo.deadline_report()
    path = ROOT / "book_outputs/_fnd.constrained-optimization.solve-records.json"
    assert json.dumps(records, indent=2, sort_keys=True, allow_nan=False) + "\n" == path.read_text()


def earlier_output_sources():
    """Inspect only tracked Python sources outside data directories."""
    # Chapters 17 and 21 have their own sealed tests for these demos and stored outputs.
    stored = {
        p.stem.removeprefix("_")
        for p in (ROOT / "book_outputs").glob("*.txt")
        if not p.name.startswith(
            (
                "_" + PREFIX,
                "_fnd.exploration-bandits.",
                "_fnd.practical-rl.",
                "_fnd.multi-agent-systems.",
                "_fnd.agent-frameworks.",
                "_fnd.agent-evaluation.",
            )
        )
    }
    baseline = subprocess.check_output(
        ["git", "ls-tree", "--name-only", "HEAD:companion/book_outputs"], cwd=ROOT, text=True
    ).splitlines()
    stored &= {Path(name).stem.removeprefix("_") for name in baseline}
    sources = {}
    paths = subprocess.check_output(
        [
            "git",
            "ls-files",
            ":(glob)companion/**/*.py",
            ":(exclude,glob)companion/demos/data/**",
        ],
        cwd=ROOT,
        text=True,
    ).splitlines()
    for relative in paths:
        if relative in {
            "demos/exploration_bandits_demo.py",
            "demos/practical_rl_demo.py",
        }:
            continue
        path = ROOT / relative
        regions = TOOLS["extract_snippets_from_text"](path.read_text(), path)
        if any(s.output and s.snippet_id in stored for s in regions):
            # New chapter diagnostics have their own tests; preserve historical outputs here.
            sources[relative] = {
                s.snippet_id: s for s in regions if not s.output or s.snippet_id in stored
            }
    mapped = {s.snippet_id for regions in sources.values() for s in regions.values() if s.output}
    assert not stored - mapped, f"earlier outputs without an allowed source: {stored - mapped}"
    return sources


@pytest.mark.parametrize("source, regions", sorted(earlier_output_sources().items()))
def test_earlier_outputs_byte_identical(source, regions):
    # These earlier demos keep their own planned/logged accesses; Chapter 26 does not use their rows.
    fresh = TOOLS["capture_outputs"](regions, sys.executable, timeout=1200)
    count = 0
    for name, stdout in fresh.items():
        path = ROOT / "book_outputs" / f"_{name}.txt"
        assert stdout.encode() == path.read_bytes(), (
            f"{path.name} differs when regenerating {source}"
        )
        original = subprocess.check_output(
            ["git", "show", f"HEAD:{path.relative_to(ROOT)}"], cwd=ROOT
        )
        assert path.read_bytes() == original
        count += 1
    if os.environ.get("BOOK_MEMORY_AUDIT_PROGRESS"):
        print(f"\n{source}: {count} earlier outputs byte-identical", flush=True)
