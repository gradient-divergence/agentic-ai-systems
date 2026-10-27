"""Printed chapter-31 regions, silent imports and final figure-data coverage."""

import subprocess
import sys
from pathlib import Path

import pytest

from demos.regions import (
    extract_snippets_from_text,
)
from refund_agent.batch_stream.experiment import figure_run
from refund_agent.batch_stream.generate import opening
from refund_agent.batch_stream.view import shipment_as_of

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def capture_directory(tmp_path_factory):
    return tmp_path_factory.mktemp("chapter31-capture")


@pytest.fixture(scope="module")
def captured(capture_directory):
    import tempfile

    from demos.batch_stream_capture import capture_chapter

    # Capture replaces generated snapshots; stale files are not runtime stores.
    for relative in (
        "demos/data/batch-stream-records.json",
        "exercises/batch-stream/scans-03.json",
        "exercises/batch-stream/payment-05.json",
        "book_outputs/_fnd.batch-stream.keys-01.txt",
    ):
        path = capture_directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale generated snapshot\n")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("TMPDIR", str(capture_directory))
        patch.setattr(tempfile, "tempdir", str(capture_directory))
        return capture_chapter(capture_directory)


def test_import_has_no_output_or_files(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", "import demos.batch_stream_demo"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert not result.stdout and not result.stderr
    assert not tuple(tmp_path.iterdir())


def test_stored_outputs_regenerate_through_ordinary_capture(
    captured, capture_directory
):
    assert len(captured) == 8
    for name, output in captured.items():
        assert (
            output.encode()
            == (ROOT / "book_outputs" / f"_{name}.txt").read_bytes()
        )
        if not name.endswith(("keys-01", "figure-01")):
            assert all(len(line) <= 79 for line in output.splitlines())
    assert {p.name for p in capture_directory.iterdir()} == {
        "book_outputs", "demos", "exercises"
    }


def test_printed_regions_execute_their_source_prefix(capsys):
    data, log, request = opening()
    for module, region in (("view", "view-01"), ("replay", "replay-01")):
        path = ROOT / f"refund_agent/batch_stream/{module}.py"
        text = path.read_text()
        snippets = extract_snippets_from_text(text, path)
        shown = next(
            s for s in snippets if s.snippet_id == f"fnd.batch-stream.{region}"
        )
        assert max(map(len, shown.content.splitlines())) <= 79
        prefix = "".join(text.splitlines(keepends=True)[: shown.end_line])
        namespace = {
            "__name__": "batch_stream_listing_probe",
            "__file__": str(path),
        }
        exec(compile(prefix, str(path), "exec"), namespace)
        if module == "view":
            build_view = namespace["shipment_as_of"]
            assert callable(build_view)
            assert build_view(
                log, request.order_reference, request.at
            ) == shipment_as_of(log, request.order_reference, request.at)
        else:
            rebuild = namespace["replay"]
            assert callable(rebuild)
            rebuilt = rebuild(log, through=request.at)
            assert rebuilt.offset == len(log.arrived_by(request.at)) - 1
            assert [
                e.status
                for e in rebuilt.shipment(
                    request.order_reference, request.at
                ).events
            ] == ["shipped"]
    assert not capsys.readouterr().out


def test_figure_data_includes_finality_and_each_copy():
    result = figure_run()
    assert len(result["points"]) == 408
    assert sum(p["copy"] for p in result["points"]) == 8
    assert result["counts"][-1]["pane"] == "final"
    assert result["counts"][-1]["n_final"] == 400


def test_stored_records_reproduce_in_temporary_directory(
    captured, capture_directory
):
    for relative in (
        "demos/data/batch-stream-records.json",
        "exercises/batch-stream/scans-03.json",
        "exercises/batch-stream/payment-05.json",
    ):
        assert (capture_directory / relative).read_bytes() == (
            ROOT / relative
        ).read_bytes()
