"""Behavior checks on constructed partner records, not substitute operating runs."""

import json

import pytest
from capstone_blueprint_fixtures import (
    AT,
    OWNERS,
    evidence,
    packet_fixture,
)

from refund_agent.launch.capstone_blueprint_gate import check
from refund_agent.launch.capstone_blueprint_packet import (
    CohortRule,
    assemble,
    write_record,
)
from refund_agent.launch.capstone_blueprint_refresh import refresh
from refund_agent.launch.capstone_blueprint_scan import LaunchLogs, stop_event


def judge(req, packet, registry, *, owners=OWNERS, logs=None):
    return check(
        packet, req, registry, logs or LaunchLogs(), owners, req.gate_day
    )


def test_cli_holds_without_partner_context_and_refuses_admission(
    tmp_path, capsys
):
    from refund_agent.launch.__main__ import main

    req, _, _ = packet_fixture(tmp_path)
    path = tmp_path / "requirement.json"
    path.write_text(req.model_dump_json())
    args = ["--run", str(tmp_path), "--requirement", str(path)]
    assert main(["gate", *args]) == 1
    recorded = json.loads(
        (tmp_path / "capstone-blueprint-gate-cli.json").read_text()
    )
    assert recorded["verdict"] == "hold"
    assert main(["admit", *args]) == 1
    assert main(["sign", *args]) == 1
    assert main(["refresh", *args]) == 1
    assert not (tmp_path / "capstone-blueprint-admission.json").exists()
    assert not (tmp_path / "capstone-blueprint-decision.json").exists()
    assert "current gate holds" in capsys.readouterr().out


def test_clean_and_failed_comparison_are_different(tmp_path):
    req, packet, registry = packet_fixture(tmp_path)
    assert judge(req, packet, registry).verdict == "launch"
    bad = evidence(
        req.rows[0].records[0],
        record_id="comparison-new",
        day=62,
        verdict="not better",
    )
    write_record(tmp_path, bad)
    verdict = judge(req, assemble(tmp_path, registry, req), registry)
    assert verdict.verdict == "hold"
    assert {(f.row, f.check) for f in verdict.failures} == {("1", "complete")}


@pytest.mark.parametrize(
    "kind,bad",
    [("release", "fail"), ("segment_audit", "breach"), ("rehearsal", "fail")],
)
def test_current_signed_failure_cannot_count(tmp_path, kind, bad):
    req, packet, registry = packet_fixture(tmp_path, {kind: {"verdict": bad}})
    verdict = judge(req, packet, registry)
    assert any(
        f.kind == kind and f.check == "complete" for f in verdict.failures
    )


def test_tampered_file_and_wrong_claims_hold(tmp_path):
    req, packet, registry = packet_fixture(tmp_path)
    target = tmp_path / packet.records[0].path
    raw = json.loads(target.read_text())
    raw["payload"]["verdict"] = "better than recorded"
    target.write_text(json.dumps(raw))
    assert "file differs" in judge(req, packet, registry).failures[0].reason
    rebuilt = assemble(tmp_path, registry, req)
    assert (
        "producer's claims" in judge(req, rebuilt, registry).failures[0].reason
    )


def test_missing_and_latest_record_are_checked(tmp_path):
    req, packet, registry = packet_fixture(tmp_path)
    missing = packet.model_copy(
        update={
            "records": tuple(
                r for r in packet.records if r.claims.kind != "release"
            )
        }
    )
    assert any(
        f.kind == "release" and f.reason == "missing record"
        for f in judge(req, missing, registry).failures
    )
    need = next(
        n for row in req.rows for n in row.records if n.kind == "release"
    )
    write_record(
        tmp_path,
        evidence(need, record_id="new-failed-release", day=63, verdict="fail"),
    )
    verdict = judge(req, assemble(tmp_path, registry, req), registry)
    assert any(
        f.kind == "release" and "says fail" in f.reason
        for f in verdict.failures
    )


def test_rehearsal_names_launched_bundle(tmp_path):
    from capstone_blueprint_fixtures import rehearsal_packet

    req, packet, registry = rehearsal_packet(tmp_path)
    verdict = judge(req, packet, registry)
    assert {(f.row, f.check) for f in verdict.failures} == {
        ("7", "consistent")
    }
    calls = []

    def producer(**kwargs):
        calls.append(kwargs)
        return evidence(
            req.rows[6].records[0], record_id="new-B6-rehearsal", day=62
        )

    result = refresh(
        verdict,
        req,
        tmp_path,
        packet=packet,
        registry=registry,
        producers={"rehearsal": producer},
    )
    assert result.retained == ("constructed-rehearsal",)
    assert calls[0]["bundle"] == "B6" and calls[0]["rollback_target"] == "B5"
    refreshed = assemble(tmp_path, registry, req)
    assert len(refreshed.records) == len(packet.records) + 1
    assert judge(req, refreshed, registry).verdict == "launch"


@pytest.mark.parametrize("name", ["RehearsalFixtureRegistry", "rehearsal_packet"])
@pytest.mark.parametrize("surface", ["module", "package"])
def test_constructed_rehearsal_fixture_is_not_shipped(surface, name):
    import refund_agent.launch as launch
    from refund_agent.launch import capstone_blueprint_exercises

    shipped = capstone_blueprint_exercises if surface == "module" else launch
    assert not hasattr(shipped, name)
    assert name not in getattr(shipped, "__all__", ())


def test_wrong_rehearsal_target_also_holds(tmp_path):
    req, packet, registry = packet_fixture(
        tmp_path, {"rehearsal": {"rollback_target": "B0"}}
    )
    assert any(
        "target differs" in f.reason
        for f in judge(req, packet, registry).failures
    )


def test_covered_record_uses_real_diff_requirement(tmp_path):
    req, packet, registry = packet_fixture(
        tmp_path, {"injection_suite": {"bundle": "B5"}}
    )
    assert judge(req, packet, registry).verdict == "launch"
    registry.required.add("injection_suite")
    assert any(
        f.kind == "injection_suite"
        and "requires this record again" in f.reason
        for f in judge(req, packet, registry).failures
    )


def test_unknown_producer_or_coverage_cannot_pass(tmp_path):
    req, packet, _ = packet_fixture(tmp_path)
    verdict = judge(req, packet, None)
    assert verdict.verdict == "hold"
    assert any("parser unavailable" in f.reason for f in verdict.failures)


@pytest.mark.parametrize("bundle", [None, "unlinked"])
def test_unknown_coverage_is_isolated_from_producer_parsing(tmp_path, bundle):
    req, packet, registry = packet_fixture(
        tmp_path, {"injection_suite": {"bundle": bundle}}
    )
    verdict = judge(req, packet, registry)
    assert {(f.row, f.kind, f.check) for f in verdict.failures} == {
        ("3", "injection_suite", "consistent")
    }
    assert all(row.checks["complete"] for row in verdict.rows)


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"release": {"signed_by": "alex"}}, "signer"),
        ({"release": {"signed_day": 2}}, "precede run"),
        (
            {
                "release": {"policy_versions": ("P1",)},
                "rehearsal": {"policy_versions": ("P2",)},
            },
            "policy-version",
        ),
    ],
)
def test_role_clock_and_policy_invariants(tmp_path, changes, reason):
    req, packet, registry = packet_fixture(tmp_path, changes)
    assert any(
        reason in f.reason for f in judge(req, packet, registry).failures
    )


def test_cohort_identity_is_one_hash_with_one_canary(tmp_path):
    hash_rule = "SHA-256(canary_id:requester_id), first 8 bytes, modulo 10000"
    req, packet, registry = packet_fixture(
        tmp_path,
        {
            "release": {
                "cohort": CohortRule(canary_id="A", hash_rule=hash_rule)
            },
            "rehearsal": {
                "cohort": CohortRule(canary_id="B", hash_rule=hash_rule)
            },
        },
    )
    assert any(
        "cohort" in f.reason for f in judge(req, packet, registry).failures
    )


def test_every_layer_has_an_owner(tmp_path):
    req, packet, registry = packet_fixture(tmp_path)
    assert any(
        "layer" in f.reason
        for f in judge(
            req,
            packet,
            registry,
            owners={"roles": OWNERS["roles"], "layers": {}},
        ).failures
    )


def test_mechanism_and_post_signature_events_are_required(tmp_path):
    req, packet, registry = packet_fixture(tmp_path)
    registry.absent_mechanisms.add(req.rows[0].reversals[0].mechanism)
    assert any(
        "mechanism absent" in f.reason
        for f in judge(req, packet, registry).failures
    )
    registry.absent_mechanisms.clear()
    end = stop_event(record_id="stop", trigger="time_box", day=40, at=AT)
    assert (
        judge(req, packet, registry, logs=LaunchLogs(events=(end,))).verdict
        == "launch"
    )
    defect = stop_event(record_id="defect", trigger="defect", day=40, at=AT)
    assert (
        judge(req, packet, registry, logs=LaunchLogs(events=(defect,))).verdict
        == "hold"
    )
    registry.false.add(packet.records[0].claims.record_id)
    assert any(
        "marked counted record false" in f.reason
        for f in judge(req, packet, registry).failures
    )


def test_failed_verdict_is_never_refreshed(tmp_path):
    req, packet, registry = packet_fixture(
        tmp_path,
        {
            "comparison": {"verdict": "not better"},
            "rehearsal": {"bundle": "B5", "verdict": "fail"},
        },
    )
    result = refresh(
        judge(req, packet, registry),
        req,
        tmp_path,
        packet=packet,
        registry=registry,
        producers={},
    )
    assert result.refreshed == () and result.unavailable == ()


def test_unmarked_kind_is_not_refreshed_when_only_consistency_fails(tmp_path):
    req, packet, registry = packet_fixture(
        tmp_path, {"scope": {"bundle": "B5"}}
    )
    verdict = judge(req, packet, registry)
    assert {(f.row, f.kind, f.check) for f in verdict.failures} == {
        ("2", "scope", "consistent")
    }

    def producer(**kwargs):
        need = next(
            n for row in req.rows for n in row.records if n.kind == "scope"
        )
        return evidence(need, record_id="scope-rerun", day=62)

    before = packet.fingerprint
    result = refresh(
        verdict,
        req,
        tmp_path,
        packet=packet,
        registry=registry,
        producers={"scope": producer},
    )
    assert result.refreshed == () and result.unavailable == ()
    assert assemble(tmp_path, registry, req).fingerprint == before


@pytest.mark.parametrize("command", ["admit", "sign"])
@pytest.mark.parametrize(
    "fault",
    [None, "file", "packet", "false", "coverage", "reversal", "requirement"],
)
def test_cli_rechecks_current_evidence_before_using_saved_verdict(
    tmp_path, monkeypatch, capsys, command, fault
):
    from types import SimpleNamespace

    from refund_agent.launch import __main__ as cli
    from refund_agent.launch.capstone_blueprint_decision import (
        StandingControl,
        admit,
    )

    req, packet, registry = packet_fixture(
        tmp_path, {"injection_suite": {"bundle": "B5"}}
    )
    context = {
        "registry": registry,
        "owners": OWNERS,
        "logs": LaunchLogs(),
        "producers": {},
        "open_items": (),
        "controls": (
            StandingControl(name="audit", owner="morgan", period="monthly"),
        ),
    }
    monkeypatch.setattr(
        cli,
        "import_module",
        lambda name: SimpleNamespace(
            context=lambda **kwargs: context,
        ),
    )
    req_path = tmp_path / "requirement.json"
    req_path.write_text(req.model_dump_json())
    args = [
        "--run",
        str(tmp_path),
        "--requirement",
        str(req_path),
        "--context",
        "refund_agent.constructed:context",
    ]
    assert cli.main(["gate", *args]) == 0
    saved = judge(req, packet, registry)
    if command == "sign":
        admission = admit(saved, "morgan", 64, registry)
        (tmp_path / "capstone-blueprint-admission.json").write_text(
            admission.model_dump_json()
        )
    if fault == "file":
        target = tmp_path / packet.records[0].path
        raw = json.loads(target.read_text())
        raw["payload"]["verdict"] = "not better"
        target.write_text(json.dumps(raw))
    elif fault == "packet":
        write_record(
            tmp_path,
            evidence(
                req.rows[0].records[0], record_id="new-comparison", day=63
            ),
        )
    elif fault == "false":
        registry.false.add(packet.records[0].claims.record_id)
    elif fault == "coverage":
        registry.required.add("injection_suite")
    elif fault == "reversal":
        context["logs"] = LaunchLogs(
            events=(
                stop_event(
                    record_id="new-stop", trigger="defect", day=65, at=AT
                ),
            )
        )
    elif fault == "requirement":
        changed = req.model_copy(
            update={
                "cost_matrix": req.cost_matrix.model_copy(
                    update={"review_cents": 401}
                ),
            }
        )
        req_path.write_text(changed.model_dump_json())
    before = tuple(registry.entries)
    capsys.readouterr()
    if fault is None:
        assert cli.main([command, *args, "--day", "66"]) == 0
        target = "admission" if command == "admit" else "decision"
        assert (tmp_path / f"capstone-blueprint-{target}.json").exists()
        return
    assert cli.main([command, *args, "--day", "66"]) == 1
    assert "refused" in capsys.readouterr().out
    assert tuple(registry.entries) == before
    target = "admission" if command == "admit" else "decision"
    assert not (tmp_path / f"capstone-blueprint-{target}.json").exists()


def test_no_same_day_latest_tie_or_path_escape(tmp_path):
    req, packet, registry = packet_fixture(tmp_path)
    write_record(
        tmp_path, evidence(req.rows[0].records[0], record_id="same-day-other")
    )
    assert (
        "ambiguous latest"
        in judge(req, assemble(tmp_path, registry, req), registry)
        .failures[0]
        .reason
    )
    escaped = packet.records[0].model_copy(update={"path": "../outside.json"})
    altered = packet.model_copy(
        update={"records": (escaped,) + packet.records[1:]}
    )
    assert "leaves the run" in judge(req, altered, registry).failures[0].reason
