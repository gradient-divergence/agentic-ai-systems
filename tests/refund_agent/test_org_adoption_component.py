"""Selected authority joins through chapter 29; unselected identity stays fixed."""

from pathlib import Path

import pytest

from demos.org_adoption_fixtures import constructed_plan
from refund_agent import contracts as c
from refund_agent.release.candidate import (
    build_candidate,
    build_tools,
    run_request,
)
from refund_agent.release.suite import build_suite
from refund_agent.rollout.org_adoption_authority import StageAuthority
from refund_agent.rollout.org_adoption_desk import TEXT_BALANCE, balance_data


def test_registration_leaves_unselected_candidate_unchanged(tmp_path):
    before = build_candidate(build_suite(trials=1))
    from refund_agent.release.extensions import RuntimeStore
    from refund_agent.rollout.org_adoption_component import selector

    SELECTOR = selector(RuntimeStore(str(tmp_path / "org-adoption.jsonl"), "morgan", "append"))

    after = build_candidate(build_suite(trials=1))
    assert before.record() == after.record()
    assert SELECTOR["name"].startswith("org-adoption.authority.")


@pytest.mark.parametrize("runtime", ["own", "second"])
def test_selected_candidate_covers_owned_modules_and_requires_bound_authority(
    tmp_path,
    runtime,
):
    from refund_agent.release.extensions import RuntimeStore
    from refund_agent.rollout.org_adoption_component import selector

    SELECTOR = selector(RuntimeStore(str(tmp_path / "org-adoption.jsonl"), "morgan", "append"))

    suite = build_suite(trials=1, components=[SELECTOR])
    candidate = build_candidate(suite, runtime=runtime)
    record = candidate.tools_composition["components"][0]
    assert record["version"] == "org-adoption.authority@1"
    assert any("org_adoption_authority" in key for key in record["modules"])
    data = balance_data(TEXT_BALANCE)
    with pytest.raises(ValueError, match="org_adoption_authority_required"):
        build_tools(data, candidate, "execute")
    wrong_authority = StageAuthority(
        constructed_plan("another-candidate"),
        tmp_path / "org-adoption.jsonl",
    )
    with pytest.raises(ValueError, match="another candidate"):
        build_tools(
            data, candidate, "execute", org_adoption_authority=wrong_authority
        )
    authority = StageAuthority(
        constructed_plan(candidate.fingerprint),
        tmp_path / "org-adoption.jsonl",
    )
    tools = build_tools(
        data, candidate, "execute", org_adoption_authority=authority
    )
    trace = run_request(candidate, tools, data.requests[0])
    assert trace.decision.outcome == c.Outcome.REFER
    assert trace.decision.reason == "org_adoption_no_signed_step"
    assert not tools.payments and len(tools.shadow_traces) == 1
    assert tools.shadow_traces[0].decision.outcome == c.Outcome.APPROVE
    assert not tools.shadow_recorder.payments


def test_selected_source_edit_changes_identity_but_unselected_source_does_not(
    monkeypatch,
    tmp_path,
):
    from refund_agent.release.extensions import RuntimeStore
    from refund_agent.rollout.org_adoption_component import selector

    SELECTOR = selector(RuntimeStore(str(tmp_path / "org-adoption.jsonl"), "morgan", "append"))

    selected = build_candidate(build_suite(trials=1, components=[SELECTOR]))
    unselected = build_candidate(build_suite(trials=1))
    original = Path.read_bytes
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda path: original(path)
        + (b"\n" if path.name == "org_adoption_authority.py" else b""),
    )
    changed = build_candidate(build_suite(trials=1, components=[SELECTOR]))
    assert changed.fingerprint != selected.fingerprint
    assert (
        build_candidate(build_suite(trials=1)).fingerprint
        == unselected.fingerprint
    )
