"""Chapter 30's explicit-path captures, standalone listing and preservation proof."""

import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from demos import regions as _regions

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demos/data_systems_demo.py"
API = vars(_regions)
PREFIX = "fnd.data-systems."
INPUT_SPINE = "0060bbab"
FINAL_CORE = "2d3776c5"
NAMES = (
    "contract-01",
    "budget-01",
    "run-01",
    "releases-01",
    "generator-01",
    "lineage-01",
    "keys-01",
    "inputs-access-01",
    "release-01",
    "release-record-01",
)


def snippets():
    return {s.snippet_id: s for s in API["extract_snippets_from_text"](DEMO.read_text(), DEMO)}


def test_data_systems_listing_is_self_contained_and_within_79():
    found = snippets()
    region = found[PREFIX + "contract-01"]
    assert all(len(line) <= 79 for line in region.content.splitlines())
    with contextlib.redirect_stdout(io.StringIO()) as output:
        ns = {"__name__": "__main__"}
        exec(compile(region.content, str(DEMO), "exec"), ns)
    assert "contract_vocabulary:shipment.status" in output.getvalue()
    assert ns["shipment_contract"].vocabularies["status"] == ("shipped", "delivered")
    assert all(len(line) <= 79 for s in found.values() for line in s.content.splitlines())


@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    directory = tmp_path_factory.mktemp("chapter30-capture")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("TMPDIR", str(directory))
        patch.setattr(tempfile, "tempdir", str(directory))
        # The hardened core recomputes the transitive closure on every hook.
        outputs = API["capture_outputs"](
            snippets(), sys.executable,
            timeout=int(os.environ.get("BOOK_OUTPUT_TIMEOUT_SECONDS", "1800")),
        )
    for name, text in outputs.items():
        (directory / f"_{name}.txt").write_text(text)
    return outputs


@pytest.mark.parametrize("name", NAMES)
def test_data_systems_outputs_reproduce_exactly(name, captured):
    path = ROOT / "book_outputs" / f"_{PREFIX}{name}.txt"
    assert path.read_bytes() == captured[PREFIX + name].encode()
    assert "companion/" not in captured[PREFIX + name]
    # The JSON evidence records are machine artifacts, not verbatim listings.
    if name not in ("inputs-access-01", "release-record-01"):
        assert all(len(line) <= 79 for line in captured[PREFIX + name].splitlines())


def test_data_systems_release_output_is_a_real_pass(captured):
    record = json.loads(captured[PREFIX + "release-record-01"])
    assert record["decision"] == "pass" and not record["failed_cases"]
    assert record["floor"]["pass"]
    assert record["must_pay_cases"] == 5
    assert record["must_pay_cents_per_trial"] == [62598] * 5
    assert (
        record["candidate"]["tools_composition"]["components"][0]["settings"]["registry_version"]
        == "data-systems.registry@1"
    )
    assert record["signature"] is None


def test_data_systems_original_outputs_and_run_are_byte_identical():
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", FINAL_CORE, "--", "book_outputs"],
        cwd=ROOT, text=True,
    ).splitlines()
    paths = [p for p in paths if not Path(p).name.startswith("_" + PREFIX)]
    assert paths
    for name in paths + ["refund_agent/run.py"]:
        before = subprocess.check_output(["git", "show", f"{FINAL_CORE}:{name}"], cwd=ROOT)
        if name == "book_outputs/_fnd.agent-evaluation.suite-01.txt":
            from refund_agent.release.suite import build_suite

            old_fingerprint = before.split(b"; fingerprint\n", 1)[1].splitlines()[0]
            before = before.replace(old_fingerprint, build_suite().fingerprint.encode(), 1)
        assert (
            hashlib.sha256((ROOT / name).read_bytes()).digest() == hashlib.sha256(before).digest()
        ), name


def test_data_systems_other_operations_regions_unchanged():
    path = ROOT / "demos/engineering_ops_demo.py"
    before = subprocess.check_output(
        ["git", "show", f"{INPUT_SPINE}:{path.relative_to(ROOT)}"], cwd=ROOT, text=True
    )
    old = API["extract_snippets_from_text"](before, path)
    after = {
        s.snippet_id: s.content for s in API["extract_snippets_from_text"](path.read_text(), path)
    }
    retired = {PREFIX + name for name in ("core-01", "example-01", "duckdb-polars-01")}
    retired |= {"fnd.mlops-mle." + name for name in ("core-01", "example-01", "sota-01")}
    retired |= {
        "fnd.fairness-bias.core-01",
        "fnd.fairness-bias.example-01",
        "fnd.explainability.core-01",
        "fnd.explainability.example-01",
    }
    assert set(after) == {s.snippet_id for s in old} - retired
    for region in old:
        if region.snippet_id not in retired:
            assert after[region.snippet_id] == region.content


def test_data_systems_demo_import_is_silent():
    result = subprocess.run(
        [sys.executable, "-c", "import demos.data_systems_demo"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    assert not result.stdout and not result.stderr
