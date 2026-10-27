"""Exercise 6 acceptance; PRACTICAL_RL_SOLUTION selects the reader's submission.

The default starter deliberately fails. All consultations use a temporary copy.
The record test first lists and copies canonical sealed evidence and requires
FND_ALLOW_SEALED=1, set only for a separately authorized sealed-execution run.
"""

import importlib.util
import json
import os
import shutil
from pathlib import Path

import pytest

from demos import practical_rl_demo as demo
from refund_agent.offers.evaluation import OfferPackage, SealedRecords
from refund_agent.offers.notices import OFFERS
from refund_agent.offpolicy.policies import Policy
from refund_agent.offpolicy.record import EvaluationRecord
from refund_agent.offpolicy.rule import Rule, decide

PATH = Path(os.environ.get("PRACTICAL_RL_SOLUTION", Path(__file__).with_name("starter.py")))
SPEC = importlib.util.spec_from_file_location("practical_rl_exercise_solution", PATH)
assert SPEC is not None and SPEC.loader is not None, f"cannot load solution {PATH}"
SOLUTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOLUTION)


@pytest.mark.parametrize(
    "reason,chosen",
    [("changed_mind", "standard"), ("damaged", "standard"), ("not_delivered", "referenced")],
)
def test_candidate_c_implements_the_specified_fixed_policy(reason, chosen):
    frozen_at = "2026-02-01T00:00:00+00:00"
    policy = SOLUTION.candidate_c(frozen_at)
    assert policy.version == "candidate-c-1"
    assert policy.frozen_at == frozen_at
    assert {cell for cell, _ in policy.tables} == {"changed_mind", "damaged", "not_delivered"}
    assert policy.probabilities(reason) == {action: float(action == chosen) for action in OFFERS}


@pytest.mark.sealed
def test_candidate_c_record_and_single_journaled_read(tmp_path, monkeypatch):
    canonical = list(demo.PACKAGE_DIRECTORY.glob("*/*.access.jsonl")) + [
        demo.DATA / "practical_rl_evaluation.access.jsonl"
    ]
    canonical_before = {path: path.read_bytes() for path in canonical}
    directory = shutil.copytree(demo.PACKAGE_DIRECTORY, tmp_path / "package")
    # Reader receipts belong to the source package, never to this fresh test run.
    for copied_journal in directory.glob("*/*.access.jsonl"):
        copied_journal.write_bytes(b"")
    monkeypatch.setattr(demo, "PACKAGE_DIRECTORY", directory)
    monkeypatch.chdir(tmp_path)
    journal = directory / "evaluation/outcomes.access.jsonl"
    truth_journal = directory / "evaluation/truth.access.jsonl"
    frozen_at = OfferPackage.load(directory).candidate.frozen_at.isoformat()
    specification = json.loads(Path(__file__).with_name("candidate_c.json").read_text())
    expected = Policy.deterministic(
        specification["version"], specification["choices"], OFFERS, frozen_at=frozen_at
    )
    audit = tmp_path / "practical_rl_evaluation.access.jsonl"
    original_open = Path.open
    original_consult = SealedRecords.consult
    consultations = []

    def guarded_open(path, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+"):
            assert path.resolve().is_relative_to(tmp_path), "test write outside temporary directory"
        return original_open(path, mode, *args, **kwargs)

    def guarded_consult(store, reader, purpose):
        assert store.manifest["name"] == "outcomes", "exercise consulted simulation truth"
        assert audit.is_file(), "candidate read outcomes before journaling the rule"
        entries = [json.loads(line) for line in audit.read_text().splitlines()]
        assert entries[0]["kind"] == "rule"
        assert entries[0]["fingerprint"] == Rule().fingerprint
        assert entries[0]["candidates"]["C"] == expected.fingerprint
        assert entries[-1]["kind"] == "outcomes"
        consultations.append((reader, purpose))
        return original_consult(store, reader, purpose)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(SealedRecords, "consult", guarded_consult)
    assert SOLUTION.candidate_c(frozen_at).fingerprint == expected.fingerprint
    record = SOLUTION.evaluate_candidate_c(tmp_path)
    assert len(consultations) == 1, "one actual outcome consultation per evaluation"
    after = journal.read_text().splitlines()
    assert len(after) == 1
    assert json.loads(after[-1])["reader"] == "exercise-6"
    assert truth_journal.read_bytes() == b""
    written = EvaluationRecord.read(tmp_path / "candidate-c-record.json")
    assert written.data == record.data
    assert (tmp_path / "candidate-c-output.txt").read_text() == record.render() + "\n"
    assert record.data["access"][0]["kind"] == "rule"
    assert record.data["access"][0]["fingerprint"] == Rule().fingerprint
    assert record.data["access"][1]["kind"] == "outcomes"
    assert record.data["outcome_reads"] == 1
    assert len(record.data["policies"]["C"]["support"]) == 9
    card = record.data["policies"]["C"]
    assert card["policy"] == json.loads(json.dumps(expected.record()))
    assert card["judged"]
    assert all(card[k] is not None for k in ("ips", "sn", "dm", "dr", "n_eff", "interval"))
    assert (card["decision"], card["reasons"]) == decide(card, Rule())
    assert record.data["autonomy"] == "drafts only; a person executes"
    assert canonical_before == {path: path.read_bytes() for path in canonical}
    assert [json.loads(line)["kind"] for line in audit.read_text().splitlines()] == [
        "rule",
        "outcomes",
    ]
    receipts = {path: path.read_bytes() for path in tmp_path.rglob("*.access.jsonl")}
    repeated = SOLUTION.evaluate_candidate_c(tmp_path)
    assert len(consultations) == 2, "identical rerun consults once and reuses its receipt"
    assert repeated.data["outcome_reads"] == 1
    assert repeated.data == record.data
    assert receipts == {path: path.read_bytes() for path in tmp_path.rglob("*.access.jsonl")}
    assert canonical_before == {path: path.read_bytes() for path in canonical}
