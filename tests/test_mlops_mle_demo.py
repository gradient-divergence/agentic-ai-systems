"""Fresh live records plus independent renderer/printed-listing acceptance."""

import contextlib
import io
import json
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from demos.mlops_mle_demo_helpers import collision_bound
from demos.regions import capture_outputs, extract_snippets_from_text
from refund_agent.canary.cloud_native_deployment_window import ShadowWindowResult
from refund_agent.promotion.mlops_mle_bundle import VersionBundle, from_candidate
from refund_agent.promotion.mlops_mle_record import PromotionRecord
from refund_agent.release.candidate import build_candidate
from refund_agent.release.suite import build_suite

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "book_outputs"
NAMES = (
    "bundle-01",
    "requirements-01",
    "week-01",
    "records-01",
    "rollback-01",
    "collision-01",
    "keys-01",
)


def saved(name):
    return (OUTPUTS / ("_fnd.mlops-mle." + name + ".txt")).read_text()


@pytest.fixture(scope="module")
def evidence():
    return json.loads((ROOT / "demos/data/mlops_mle_data_facts.json").read_text())


def test_printed_region_runs_verbatim_and_fits_width():
    source = ROOT / "demos/mlops_mle_demo.py"
    snippets = extract_snippets_from_text(source.read_text(), source)
    listing = next(s for s in snippets if s.snippet_id == "fnd.mlops-mle.bundle-01")
    assert len(listing.content.splitlines()) <= 32
    assert all(len(line) <= 79 for line in listing.content.splitlines())
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(listing.content, {"__name__": "mlops_mle_listing"})
    assert out.getvalue() == saved("bundle-01")


def test_every_output_has_printable_paths_and_width():
    for name in NAMES:
        text = saved(name)
        assert text.endswith("\n") and "companion/" not in text
        assert max(map(len, text.splitlines())) <= 79


def test_collision_and_component_arithmetic_is_exact():
    assert collision_bound(1000000, 16) == Fraction(499999500000, 2**64)
    assert collision_bound(10000, 7) == Fraction(49995000, 16**7)
    assert collision_bound(50000, 12) > Fraction(1, 1000000)
    assert collision_bound(50000, 13) < Fraction(1, 1000000)
    assert "versions 17 bundles 288" in saved("collision-01")
    with pytest.raises(ValueError):
        collision_bound(True, 12)


def test_live_window_is_the_current_frozen_bundle(evidence):
    recorded = evidence["candidate0"]
    suite = build_suite(components=recorded["tools_composition"]["components"])
    candidate = build_candidate(suite)
    bundle = from_candidate(candidate)
    window = ShadowWindowResult.model_validate_json(json.dumps(evidence["window"]))
    assert candidate.record() == recorded
    assert window.candidate_fingerprint == candidate.fingerprint
    assert window.bundle_fingerprint == bundle.fingerprint
    assert window.required_requests == 299 and window.measures_pass
    assert window.wrongful_denials <= window.baseline_wrongful_denials


def test_fortnight_timing_and_all_lineage_stamps(evidence):
    data = json.loads((ROOT / "demos/data/mlops_mle_figures.json").read_text())[
        "requests"
    ]
    assert not any(row["policy_not_in_bundle"] for row in data["early"])
    for row in data["late"]:
        assert row["policy_not_in_bundle"] == (8 <= row["day"] <= 10)
    for row in data["rolled"]:
        assert row["policy_not_in_bundle"] == (8 <= row["day"] <= 10 or row["day"] >= 12)
    assert all(row["lineage_stamped"] for rows in data.values() for row in rows)
    controls = [row for row in data["late"] if row["policy_not_in_bundle"]]
    assert len(controls) == 12
    assert sum(row["control_matches_label"] for row in controls) == 8
    exceptions = {
        row["request_id"]: (row["control_reason"], row["label_reason"])
        for row in controls
        if not row["control_matches_label"]
    }
    assert exceptions == {
        "live-r-7-0036": ("stale_order", "eligible"),
        "live-r-7-0039": ("amount_over_policy", "amount_over_policy"),
        "live-r-7-0041": ("amount_over_policy", "amount_over_policy"),
        "live-r-7-0046": ("outside_return_window", "outside_return_window"),
    }
    assert all(
        row["control_outcome"] == "refer" for row in controls if not row["control_matches_label"]
    )
    from datetime import datetime

    from refund_agent.data.live import generate_live

    requests = generate_live(
        7, datetime.fromisoformat(evidence["announcement"]), 14, 4
    ).data.requests
    named_review = {"live-r-7-0039", "live-r-7-0041", "live-r-7-0046"}
    assert all(
        request.amount_cents > 50000 for request in requests if request.request_id in named_review
    )
    assert all(
        row["label_outcome"] == "deny" for row in controls if row["request_id"] in named_review
    )


def test_fortnight_control_uses_terminal_decisions_for_both_review_tiers(evidence):
    from datetime import datetime

    from demos.mlops_mle_demo_helpers import run_fortnight
    from refund_agent import contracts as c
    from refund_agent.data.live import LiveStream, generate_live

    suite = build_suite(components=evidence["candidate0"]["tools_composition"]["components"])
    candidate = build_candidate(suite)
    b0 = from_candidate(candidate)
    b1 = VersionBundle.from_record(evidence["bundle1"])
    stream = generate_live(7, datetime.fromisoformat(evidence["announcement"]), 14, 4)
    wanted = {"live-r-7-0037", "live-r-7-0040"}
    data = stream.data.model_copy(
        update={
            "requests": [
                request for request in stream.data.requests if request.request_id in wanted
            ]
        }
    )
    measured = run_fortnight(
        LiveStream(data, stream.manifest),
        c.Policy.model_validate_json(json.dumps(evidence["policy"])),
        b0,
        b1,
        candidate,
        candidate,
        promote_day=11,
    )
    assert {row["request_id"] for row in measured} == wanted
    assert all(row["policy_not_in_bundle"] and row["control_matches_label"] for row in measured)


def test_hypothetical_score_is_held_on_its_four_missing_items(evidence):
    record = PromotionRecord.from_record(evidence["records"][3])
    assert record.status == "held"
    assert set(record.unmet) == {
        "release",
        "calibration",
        "signed_cost_matrix",
        "threshold_experiment",
    }
    assert evidence["risk_evidence"]["rebuild"]["passed"]
    assert (
        evidence["risk_evidence"]["rebuild"]["receipt"]
        == evidence["risk_evidence"]["rebuild"]["rebuilt_receipt"]
    )


def test_canary_rehearsal_restores_the_promotion_incumbent(evidence):
    promotion = PromotionRecord.from_record(evidence["records"][2])
    assert promotion.rollback_target == evidence["bundle0"]["fingerprint"]
    assert (
        evidence["rehearsal"]["rollback_target_fingerprint"]
        == evidence["bundle0"]["refs"]["candidate"]
    )


def test_rebuild_never_constructs_or_consults_a_frozen_set(monkeypatch, evidence):
    from demos import supervised_learning_demo
    from demos.mlops_mle_demo_helpers import rebuilt_risk

    def denied(*args, **kwargs):
        raise AssertionError("frozen labels are not rebuild evidence")

    monkeypatch.setattr(supervised_learning_demo, "frozen", denied)
    bundle = VersionBundle.from_record(evidence["bundle0"])
    _, records = rebuilt_risk(bundle)
    assert records["rebuild"]["passed"]


def test_starter_fails_for_missing_row_and_reference_passes(tmp_path):
    source = ROOT / "exercises/mlops-mle"
    names = (
        "requirements_06.py",
        "reference_06.py",
        "mlops_mle_check_06.py",
        "test_acceptance.py",
        "candidate_06.json",
        "b1_06.json",
        "candidate-05.json",
        "incumbent-05.json",
        "record-05.json",
        "mlops_mle_evidence_06.json",
    )
    for name in names:
        shutil.copyfile(source / name, tmp_path / name)
    command = [
        sys.executable,
        "-m",
        "pytest",
        str(tmp_path / "test_acceptance.py"),
        "-p",
        "no:cacheprovider",
    ]
    starter = subprocess.run(command, capture_output=True, text=True, check=False)
    assert starter.returncode == 1
    assert "test_runtime_change_is_held" in starter.stdout
    assert "NotImplementedError: match every change" in starter.stdout
    shutil.copyfile(tmp_path / "reference_06.py", tmp_path / "requirements_06.py")
    reference = subprocess.run(command, capture_output=True, text=True, check=False)
    assert reference.returncode == 0, reference.stdout + reference.stderr


def test_outputs_regenerate_through_capture(tmp_path):
    source = ROOT / "demos/mlops_mle_demo.py"
    # The freshly replayed Chapter 32 window is this chapter's input fixture.
    # Execute the real Chapter 33 workflow, including all 168 request rows.
    measured = ROOT / "demos/data/mlops_mle_window.json"
    capture_source = tmp_path / "mlops_mle_demo.py"
    capture_source.write_text(
        f"""import json as _json
from pathlib import Path as _Path
from demos import mlops_mle_demo_helpers as _helpers
from refund_agent.canary.cloud_native_deployment_window import ShadowWindowResult as _Window
_window = _Window.model_validate_json(
    _json.dumps(_json.loads(_Path({str(measured)!r}).read_text())["window"])
)
def _measured_window(candidate, start, *, bundle_fingerprint):
    if (candidate.fingerprint, bundle_fingerprint, start) != (
        _window.candidate_fingerprint, _window.bundle_fingerprint, _window.start
    ):
        raise ValueError("measured window belongs to another candidate, bundle or start")
    return _window
_helpers.run_window = _measured_window
"""
        + source.read_text()
    )
    fixture_paths = [
        ROOT / "demos/data" / name
        for name in ("mlops_mle_window.json", "mlops_mle_figures.json", "mlops_mle_data_facts.json")
    ]
    fixture_paths += [
        ROOT / "exercises/mlops-mle" / name
        for name in (
            "incumbent-05.json",
            "candidate-05.json",
            "candidate_06.json",
            "b1_06.json",
            "record-05.json",
            "mlops_mle_evidence_06.json",
        )
    ]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in fixture_paths}
    snippets = {
        s.snippet_id: s
        for s in extract_snippets_from_text(capture_source.read_text(), capture_source)
    }
    captured = capture_outputs(snippets, sys.executable, timeout=2300)
    assert {key.split("mlops-mle.")[1]: value for key, value in captured.items()} == {
        name: saved(name) for name in NAMES
    }
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in fixture_paths} == before


@pytest.mark.parametrize("write_data", [False, True])
def test_demo_writes_fixtures_only_with_flag(tmp_path, monkeypatch, write_data):
    import runpy

    from demos import mlops_mle_demo_helpers as helpers

    marker = tmp_path / "fixture.json"
    # Isolate CLI routing from the expensive experiment and printers.
    monkeypatch.setattr(helpers, "experiment", lambda: {})
    for name in ("bundle", "requirements", "week", "records", "rollback", "collision", "keys"):
        monkeypatch.setattr(helpers, "print_" + name, lambda *args: None)
    monkeypatch.setattr(helpers, "write_data", lambda result: marker.write_text("written"))
    monkeypatch.setattr(sys, "argv", ["mlops_mle_demo", *(["--write-data"] if write_data else [])])
    runpy.run_module("demos.mlops_mle_demo", run_name="__main__")
    assert marker.exists() == write_data
