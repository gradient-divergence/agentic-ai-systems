"""Chapter-order listings, deterministic capture and side-effect boundaries.

Tests using isolated_package or captured first copy canonical sealed evidence.
They require FND_ALLOW_SEALED=1, set only for a separately authorized
sealed-execution run; temporary copies do not authorize reading the source.
"""

import contextlib
import io
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from demos import regions as _regions

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demos/practical_rl_demo.py"
TOOLS = vars(_regions)
PREFIX = "fnd.practical-rl."
NAMES = (
    "bank-01",
    "box-01",
    "edge-01",
    "dr-01",
    "coverage-01",
    "horizon-01",
    "refund-01",
    "keys-01",
)


def snippets():
    return {s.snippet_id: s for s in TOOLS["extract_snippets_from_text"](DEMO.read_text(), DEMO)}


def copy_package(directory):
    package = shutil.copytree(
        ROOT / "demos/data/exploration_bandits_package", directory / "package"
    )
    for journal in package.glob("*/*.access.jsonl"):
        journal.write_bytes(b"")
    return package


@pytest.fixture(autouse=True)
def temporary_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def isolated_package(tmp_path, monkeypatch):
    from demos import practical_rl_demo as demo

    package = copy_package(tmp_path)
    monkeypatch.setattr(demo, "PACKAGE_DIRECTORY", package)
    monkeypatch.setenv("PRACTICAL_RL_PACKAGE_DIRECTORY", str(package))
    return package


@pytest.mark.sealed
def test_printed_listings_run_in_chapter_order_without_hidden_setup(isolated_package):
    namespace = {"__name__": "__main__"}
    found = snippets()
    for name in ("bank-01", "dr-01", "refund-01"):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(compile(found[PREFIX + name].content, str(DEMO), "exec"), namespace)
        assert (
            stdout.getvalue().encode()
            == (ROOT / "book_outputs" / f"_{PREFIX}{name}.txt").read_bytes()
        )
    assert namespace["record"].data["autonomy"] == "drafts only; a person executes"


@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    directory = tmp_path_factory.mktemp("practical-rl-capture")
    package = copy_package(directory)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("PRACTICAL_RL_PACKAGE_DIRECTORY", str(package))
        first = TOOLS["capture_outputs"](snippets(), sys.executable)
        journals = {p: p.read_bytes() for p in directory.rglob("*.access.jsonl")}
        assert first == TOOLS["capture_outputs"](snippets(), sys.executable)
        assert journals == {p: p.read_bytes() for p in directory.rglob("*.access.jsonl")}
    return first


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.sealed
def test_stored_output_regenerates_byte_for_byte(name, captured):
    assert (
        captured[PREFIX + name].encode()
        == (ROOT / "book_outputs" / f"_{PREFIX}{name}.txt").read_bytes()
    )


def test_import_does_not_print_read_or_write(tmp_path):
    code = """from refund_agent.offers.evaluation import OfferPackage
from pathlib import Path
def forbidden(*args, **kwargs):
    raise AssertionError('import touched data')
Path.read_text = Path.write_text = forbidden
import demos.practical_rl_demo
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert list(tmp_path.iterdir()) == []


def test_output_regions_are_in_outline_order():
    assert list(snippets()) == [PREFIX + name for name in NAMES]
    assert all(s.output for s in snippets().values())


def test_five_record_illustration_reports_only_its_computed_floor():
    from demos import practical_rl_demo as demo

    namespace = dict(demo.__dict__, __name__="__main__")
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(snippets()[PREFIX + "edge-01"].content, str(DEMO), "exec"), namespace)
    assert stdout.getvalue().splitlines()[-1] == (
        "decision hold; candidate effective size below 400"
    )


@pytest.mark.sealed
def test_fit_and_candidate_preparation_read_no_frozen_outcomes_or_golden(
    monkeypatch, isolated_package
):
    from demos import practical_rl_demo as demo
    from refund_agent.data import generate, history
    from refund_agent.offers.evaluation import SealedRecords
    from refund_agent.split import GoldenSet

    def forbidden(*args, **kwargs):
        raise AssertionError("preparation read forbidden data")

    monkeypatch.setattr(SealedRecords, "consult", forbidden)
    monkeypatch.setattr(GoldenSet, "consult", forbidden)
    monkeypatch.setattr(generate, "generate", forbidden)
    monkeypatch.setattr(history, "generate_history", forbidden)
    package, candidates, fixed, model, window = demo.prepare_offer_evaluation()
    assert len(package.learning.rows) == 9837
    assert all(row.decided_at < package.candidate.frozen_at for row in package.learning.rows)
    assert all(
        cell == "not_delivered" or candidates["B"].probabilities(cell)["standard"] == 1
        for cell, _ in candidates["B"].tables
    )
    assert model.source_fingerprint and fixed.source_fingerprint
    assert window.reads == 0


@pytest.mark.sealed
def test_offer_evaluation_logs_each_sealed_read_and_truth_only_after_record(
    tmp_path, monkeypatch, isolated_package
):
    import json

    from demos import practical_rl_demo as demo
    from refund_agent.offers.evaluation import SealedRecords
    from refund_agent.split import GoldenSet

    original_consult = SealedRecords.consult
    original_open = Path.open
    active = []
    calls = []
    record_path = tmp_path / "record.json"
    package, candidates, fixed, model, window = demo.prepare_offer_evaluation()
    journals_before = {
        kind: (demo.PACKAGE_DIRECTORY / f"evaluation/{kind}.access.jsonl").read_text().splitlines()
        for kind in ("outcomes", "truth")
    }

    def guarded_consult(self, reader, purpose):
        kind = self.manifest["name"]
        if kind == "outcomes":
            assert window.log[0]["kind"] == "rule"
            assert window.log[0]["fingerprint"] == demo.Rule().fingerprint
            assert not calls
        else:
            assert calls == ["outcomes"]
            assert (
                json.loads(record_path.read_text())["rule_fingerprint"] == demo.Rule().fingerprint
            )
        calls.append(kind)
        active.append(kind)
        try:
            return original_consult(self, reader, purpose)
        finally:
            active.pop()

    def guarded_open(path, *args, **kwargs):
        if path.name.endswith(".sealed.jsonl"):
            kind = path.name.split(".")[0]
            assert active == [kind], "sealed data read outside a named consultation"
            journal = path.with_name(f"{kind}.access.jsonl")
            entries = [json.loads(line) for line in original_open(journal).read().splitlines()]
            assert entries[-1]["purpose"] == purpose_by_kind[kind]
        return original_open(path, *args, **kwargs)

    purpose_by_kind = {}

    def named_consult(self, reader, purpose):
        purpose_by_kind[self.manifest["name"]] = purpose
        return guarded_consult(self, reader, purpose)

    monkeypatch.setattr(SealedRecords, "consult", named_consult)
    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(
        GoldenSet,
        "consult",
        lambda *a, **k: pytest.fail("Chapter 21 consulted Chapter 4's golden set"),
    )
    record = demo.evaluate(
        candidates,
        fixed,
        window,
        demo.Rule(),
        model=model,
        reader="chapter-21-guard-test",
        learning_records=len(package.learning.rows),
    )
    record.write(record_path)
    truth = demo.truth_check(package, candidates, fixed, record, record_path)
    assert calls == ["outcomes", "truth"]
    assert truth["after_record"]
    assert record.data["outcome_reads"] == 1
    assert record.data["policies"]["A"]["decision"] == "hold"
    assert record.data["policies"]["B"]["decision"] == "hold"
    for kind, prior in journals_before.items():
        after = (
            (demo.PACKAGE_DIRECTORY / f"evaluation/{kind}.access.jsonl").read_text().splitlines()
        )
        assert after[:-1] == prior
        assert len(after) == len(prior) + 1


@pytest.mark.sealed
def test_truth_consultation_is_refused_without_written_record(
    tmp_path, monkeypatch, isolated_package
):
    from demos import practical_rl_demo as demo

    package, candidates, fixed, model, window = demo.prepare_offer_evaluation()
    record = demo.evaluate(candidates, fixed, window, demo.Rule(), model=model)

    def forbidden(*args, **kwargs):
        raise AssertionError("truth consulted before record")

    monkeypatch.setattr(package.evaluation.truth, "consult", forbidden)
    with pytest.raises(FileNotFoundError):
        demo.truth_check(package, candidates, fixed, record, tmp_path / "absent.json")


def test_coverage_is_seeded_independently_fitted_and_matches_derived_file():
    import json

    from demos import practical_rl_demo as demo

    data = demo.coverage_experiment()
    assert data == demo.coverage_experiment()
    assert json.loads((demo.DATA / "practical_rl_coverage.json").read_text()) == data
    assert data["true_value"] == pytest.approx(0.735)
    assert data["logs"] == 500 and data["alerts"] == 2000
    assert [row["supported"] for row in data["rows"]] == [True] * 6 + [False]
    assert all(abs(row["estimates"]["dr"]["mean"] - 0.735) < 0.015 for row in data["rows"][:-1])
    zero = data["rows"][-1]
    assert all(
        abs(stats["mean"] - data["true_value"]) > 0.01 for stats in zero["estimates"].values()
    )
    assert zero["n_eff_median"] > 400  # A large floor does not repair zero support.


def test_horizon_reproduces_zero_matches_at_ten_steps():
    from demos import practical_rl_demo as demo

    a, b = demo.horizon_experiment(), demo.horizon_experiment()
    assert a == b
    assert [(r["matched"], r["n_eff"]) for r in a["rows"]] == [(523, 523), (73, 73), (0, 0)]
    assert a["rows"][-1]["expected"] == 1000 / 1024


def test_exercise_report_a_ignores_chances_on_supported_cells():
    from demos import practical_rl_demo as demo
    from refund_agent.offpolicy.policies import Policy

    fixture = demo.exercise_reports()
    rows = [demo.Record(**row) for row in fixture["records"]]
    spec = fixture["candidate"]
    policy = Policy(spec["version"], tuple(spec["tables"]))
    subset = [r for r in rows if r.context in fixture["marking"]["supported_cells"]]
    assert demo.supported(demo.support_table(subset, policy))
    assert len(subset) == 800
    matches = [r for r in subset if policy.probabilities(r.context)[r.action]]
    assert {r.logged[r.action] for r in matches} == {0.2, 0.8}
    rates = {
        cell: sum(r.reward for r in matches if r.context == cell)
        / sum(r.context == cell for r in matches)
        for cell in {r.context for r in subset}
    }
    assert rates == {"damaged": 1.0, "changed_mind": 0.75}
    naive = sum(r.reward for r in matches) / len(matches)
    assert fixture["report_a"]["value"] == naive == 0.8
    assert fixture["report_a"]["comparator"] == 0.75
    assert fixture["report_a"]["cells"] == ["changed_mind", "damaged"]
    weights = demo.policy_ratios(policy, subset)
    rewards = [r.reward for r in subset]
    assert demo.ips(weights, rewards) == 0.875
    assert demo.self_normalized_ips(weights, rewards) == 0.875
    assert naive != demo.ips(weights, rewards)
    assert fixture["marking"]["recomputed"]["ips"] == 0.875
    assert fixture["marking"]["recomputed"]["sn"] == 0.875


def test_exercise_report_b_claims_a_judgment_with_structural_zero_support():
    import json

    from demos import practical_rl_demo as demo
    from refund_agent.offpolicy.policies import Policy
    from refund_agent.offpolicy.reward_model import CellModel

    fixture = demo.exercise_reports()
    assert (demo.DATA / "practical_rl_reports_05.json").read_bytes() == (
        json.dumps(fixture, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    rows = [demo.Record(**row) for row in fixture["records"]]
    spec = fixture["candidate"]
    policy = Policy(spec["version"], tuple(spec["tables"]))
    model = CellModel(tuple(fixture["model"]))
    assert fixture["report_b"]["judged"]
    assert fixture["report_b"]["value"] == pytest.approx(0.86)
    assert fixture["report_b"]["value"] == pytest.approx(
        demo.per_record_terms(policy, model, rows).mean()
    )
    assert fixture["report_b"]["interval"][1] - fixture["report_b"]["interval"][0] < 0.05
    assert not demo.supported(demo.support_table(rows, policy))
    table = demo.support_table(rows, policy)
    assert {(r["cell"], r["offer"]) for r in table if r["acts"] and not r["positive_share"]} == {
        ("not_delivered", "dated")
    }
    assert model.prediction("not_delivered", "dated", supported=False)["assumption"]
    supported_rows = [r for r in rows if r.context in fixture["marking"]["supported_cells"]]
    recomputed = fixture["marking"]["recomputed"]
    assert recomputed["dr"] == pytest.approx(
        demo.per_record_terms(policy, model, supported_rows).mean()
    )
    assert recomputed["dr"] == pytest.approx(0.875)
    assert recomputed["improvement"] == pytest.approx(0.25)
    assert fixture["marking"]["decision"] == "reject both reports"


def test_plot_exports_and_label_sizes(tmp_path):
    pytest.importorskip("matplotlib", reason="plot rendering requires installed Matplotlib")
    plot = runpy.run_path(str(ROOT.parent / "figures/plots/fig_practical_rl.py"))
    plot["main"](tmp_path)
    for name in ("coverage", "support"):
        for suffix in (".pdf", ".png", "-gray.png"):
            assert (tmp_path / f"fig-practical-rl-{name}{suffix}").stat().st_size > 0
    assert plot["LABEL"] == 7.2
