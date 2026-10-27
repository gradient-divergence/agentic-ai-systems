"""Delegation boundaries, the standalone listing and the selection evidence."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demos/delegated_model_demo.py"
OUTPUTS = ROOT / "book_outputs"
ENV = {
    **os.environ,
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}


def demo():
    return importlib.import_module("demos.delegated_model_demo")


def snippets(source=None):
    from demos.regions import extract_snippets_from_text

    sources = [source] if source else [DEMO, DEMO.with_name("delegated_search_lab.py")]
    return {
        s.snippet_id: s for path in sources
        for s in extract_snippets_from_text(path.read_text(), path)
    }


def test_printed_listing_runs_on_its_own(tmp_path):
    script = tmp_path / "printed.py"
    script.write_text(snippets()["fnd.automl-and-selection.inspect-01"].content)
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=ENV,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert (
        result.stdout.encode()
        == (OUTPUTS / "_fnd.automl-and-selection.inspect-01.txt").read_bytes()
    )
    assert list(tmp_path.iterdir()) == [script]


@pytest.fixture(scope="module")
def built():
    return demo().build()


def test_budget_fingerprints_and_planted_fault(built):
    d = demo()
    assert len(built.trials) == built.commission.budget == 20
    assert built.commission.intact() and built.receipt == d.seal(built)
    assert built.checks["search winner"].passed
    assert built.checks["clean agent"].passed
    bad = built.checks["agent delivery"]
    assert bad.results == (False, False, True, True)
    assert bad.honest_loss > bad.reproduced_loss


@pytest.fixture(scope="module")
def captured():
    from demos.regions import capture_outputs

    return capture_outputs(snippets(), sys.executable, timeout=120)


def test_all_sixteen_outputs_regenerate(captured):
    assert len(captured) == 16
    for name, value in captured.items():
        assert value.encode() == (OUTPUTS / f"_{name}.txt").read_bytes(), name


def test_compact_selection_record_retains_every_decision_and_complete_settings(captured, built):
    from refund_agent.selection import fingerprint

    table = captured["fnd.automl-and-selection.record-01"]
    assert all(len(line) <= 79 for line in table.splitlines())
    table = " ".join(table.replace("|", " ").split())
    assert "max_iter=" not in table and "Record fingerprint" not in table
    complete = json.loads(captured["fnd.automl-and-selection.record-full-01"])
    assert complete["commission"] == json.loads(json.dumps(built.commission.payload())) | {
        "sha256": built.commission.sha256
    }
    assert complete["sha256"] == fingerprint(
        {key: value for key, value in complete.items() if key != "sha256"}
    )
    assert complete["receipt"] == built.receipt
    assert complete["selected"] == built.commission.baseline
    for delivery in complete["deliveries"]:
        assert delivery["settings"] and delivery["trials"]
        assert f"Delivery: {delivery['name']}" in table
        assert delivery["builder"] in table
        assert f"raw loss {delivery['reported_loss']:.4f}" in table
    for name, decision in complete["decisions"].items():
        assert f"{name}: {decision}" in table
    for field in ("selected", "spent", "grants", "signer"):
        assert complete[field] in table
    assert all(dependency in table for dependency in complete["dependencies"])
    assert len(complete["log"]) == 3
    assert "7 allowed inputs; raw validation log loss" in table
    assert "budget 20, seed 7" in table


def test_seed_reproduces_every_output(captured):
    from demos.regions import capture_outputs

    assert capture_outputs(snippets(), sys.executable, timeout=120) == captured


def test_earlier_outputs_regenerate_byte_for_byte():
    from demos.regions import capture_outputs

    regions = {}
    for name in ("ml_data_evaluation", "supervised_learning", "semi_self_supervised"):
        regions.update(snippets(ROOT / f"demos/{name}_demo.py"))
    values = capture_outputs(regions, sys.executable, timeout=300)
    assert len(values) == 45  # chapters 4 to 6, including the clustering-choice diagnostics
    for name, value in values.items():
        assert value.encode() == (OUTPUTS / f"_{name}.txt").read_bytes(), name


def test_import_has_no_effects(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", "import demos.delegated_model_demo"],
        cwd=tmp_path,
        env=ENV,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "" and list(tmp_path.iterdir()) == []


def test_builder_boundary_and_commission_precedes_trials(monkeypatch, built):
    d = demo()
    original = d.fit_delivery
    fitted = []

    def guarded(name, builder, settings, window, commission, trials=None):
        assert commission.intact()
        assert set(window) == {"training", "validation"}
        assert all(r.at < d.GO_LIVE for rows in window.values() for r in rows)
        fitted.append(name)
        return original(name, builder, settings, window, commission, trials)

    monkeypatch.setattr(d, "fit_delivery", guarded)
    other = d.build(built.window)
    assert fitted and other.events[0]["event"] == "commission"
    assert other.events[-1]["event"] == "sealed"
    assert [e["entry"] for e in other.events if e["event"] == "trial"] == list(range(1, 21))
    assert other.trials == built.trials
    assert other.receipt == built.receipt


def test_one_new_consult_after_every_fit_and_check(monkeypatch, built, tmp_path):
    d = demo()
    _, golden = d.prepare()

    def forbidden(*args, **kwargs):
        pytest.fail("fitting or inspection after sealing/consultation")

    monkeypatch.setattr(d, "fit_delivery", forbidden)
    monkeypatch.setattr(d, "inspect_delivery", forbidden)
    record, _ = d.evaluate(built, golden, tmp_path)
    assert [e["entry"] for e in golden.log] == [1, 2, 3]
    assert [e["reader"] for e in golden.log] == [
        "stage 1 report",
        "stage 1 report",
        "selection record",
    ]
    assert golden.intact() and record.intact()
    assert record.selected == d.BASELINE
    assert record.decisions == {
        "logistic regression": "not adopted",
        "search winner": "not adopted",
        "agent delivery": "sent back",
    }
    assert built.events[-1]["event"] == "consult"
    # A later run sees the real journal, and spends exactly one more read.
    _, resumed = d.prepare()
    again, _ = d.evaluate(built, resumed, tmp_path)
    assert [e["entry"] for e in again.log] == [1, 2, 3, 4]
    assert record.sha256 != again.sha256


def test_changed_seal_stops_before_consult(built, tmp_path):
    import copy

    d = demo()
    changed = copy.deepcopy(built)
    changed.candidates["agent delivery"].reported_loss += 0.01
    _, golden = d.prepare()
    with pytest.raises(ValueError, match="changed after sealing"):
        d.evaluate(changed, golden, tmp_path)
    assert golden.log == [] and list(tmp_path.iterdir()) == []


def test_changed_fitted_model_stops_before_consult(built, tmp_path):
    import copy

    d = demo()
    changed = copy.deepcopy(built)
    # What is sealed, and what August will score, is the inspector's rebuild.
    changed.rebuilds["logistic regression"].model[-1].intercept_ += 1
    _, golden = d.prepare()
    with pytest.raises(ValueError, match="changed after sealing"):
        d.evaluate(changed, golden, tmp_path)
    assert golden.log == []


def switched(delivery, honest_for):
    """The same delivery, answering 0.5 for every row where `honest_for(row)` is false."""
    d = demo()

    class Switched(d.Delivery):
        def predict(self, rows, mapped=True):
            p = super().predict(rows, mapped)
            return np.where([honest_for(r) for r in rows], p, 0.5)

    return Switched(**{k: getattr(delivery, k) for k in d.Delivery.__dataclass_fields__})


def test_a_predictor_that_changes_unseen_rows_cannot_reach_august(built, tmp_path):
    """The fix checker's first attack: agree on the rows you were given, change the rest."""
    import copy

    d = demo()
    window, commission = built.window, built.commission
    honest = built.candidates["logistic regression"]
    held = {r.request_id for rows in window.values() for r in rows}
    probe = d.probes(window)
    assert held.isdisjoint(r.request_id for r in probe)
    assert min(r.at for r in probe) >= d.GO_LIVE  # dated where no builder's rows are

    # A predictor that answers honestly only for the rows it held fails the third check.
    memorizer = switched(honest, lambda r: r.request_id in held)
    check = d.inspect_delivery(memorizer, window, commission)
    assert check.results == (True, True, False, True)
    assert check.findings == (
        "not reproduced: the delivered model predicts differently from its rebuild",
    )

    # No comparison can catch one that changes exactly the frozen requests. It passes
    # every check, and it changes nothing, because August scores the inspector's rebuild.
    _, golden = d.prepare()
    frozen = {r.request_id for r in golden.requests}
    oracle = switched(honest, lambda r: r.request_id not in frozen)
    rows = list(golden.requests)
    assert (oracle.predict(rows) != honest.predict(rows)).sum() == len(rows) == 999
    attacked = copy.copy(built)
    attacked.candidates = {**built.candidates, "logistic regression": oracle}
    attacked.checks = {
        **built.checks,
        "logistic regression": d.inspect_delivery(
            oracle, window, commission, built.rebuilds["logistic regression"]
        ),
    }
    attacked.events = list(built.events)
    attacked.receipt = d.seal(attacked)
    assert attacked.checks["logistic regression"].passed
    assert attacked.receipt == built.receipt
    plain = copy.copy(built)
    plain.events = list(built.events)
    expected, _ = d.evaluate(plain, d.prepare()[1], tmp_path / "plain")
    record, differences = d.evaluate(attacked, golden, tmp_path / "attacked")
    assert record.evidence == expected.evidence and record.decisions == expected.decisions
    assert differences["logistic regression"].mean() == pytest.approx(
        expected.evidence[1]["difference"][0]
    )

    # The same holds for a delivered model changed after the seal: it is never asked.
    late = copy.deepcopy(built)
    late.candidates["logistic regression"].model[-1].intercept_ += 4
    again, _ = d.evaluate(late, d.prepare()[1], tmp_path / "late")
    assert again.evidence == expected.evidence


@pytest.mark.parametrize(
    "eligible,mean,bounds,expected",
    [
        (False, -0.1, (-0.2, -0.05), "sent back"),
        (True, -0.1, (-0.2, -0.05), "accepted"),
        (True, -0.1, (-0.2, 0), "not adopted"),
        (True, 0, (0, 0), "not adopted"),
        (True, 0.1, (0.01, 0.2), "not adopted"),
    ],
)
def test_decision_rule(eligible, mean, bounds, expected):
    from refund_agent.selection import decide

    assert decide(eligible, mean, bounds) == expected


def test_exercise_fixture_has_only_the_trial_budget_fault(built):
    d = demo()
    report = json.loads((d.DATA / "delegated_model_delivery_05.json").read_text())
    delivery = d.fit_delivery(
        report["name"], report["builder"], report["settings"], built.window, built.commission
    )
    for key, value in report.items():
        setattr(delivery, key, value)
    check = d.inspect_delivery(delivery, built.window, built.commission)
    assert check.results == (True, True, True, False)
    # One fault and no other: sixty distinct, in-space, reproducible trials under a
    # budget of twenty, whose first twenty are the chapter's own search.
    assert check.findings == ("trial log: 60 entries under a budget of 20",)
    assert len(delivery.trials) == 60
    assert len({json.dumps(t["settings"], sort_keys=True) for t in delivery.trials}) == 60
    assert delivery.trials[:20] == built.trials
    lowest = min(delivery.trials, key=lambda t: t["loss"])
    assert lowest["settings"] == delivery.settings and lowest["entry"] > 20
    assert lowest["loss"] < min(t["loss"] for t in built.trials)


def test_model_altered_before_inspection_fails_the_reproduction_check(built):
    """The reviewer's first reproduction: the delivered model, not only a rebuild, is read."""
    import copy

    from demos.supervised_learning_demo import arrays, log_loss

    d = demo()
    validation = built.window["validation"]
    changed = copy.deepcopy(built.candidates["logistic regression"])
    changed.model[-1].intercept_ += 4  # before any inspection or seal
    actual = log_loss(arrays(validation)[1], changed.predict(validation, mapped=False))
    assert round(changed.reported_loss, 4) == 0.1685 and round(actual, 4) == 0.8576
    check = d.inspect_delivery(changed, built.window, built.commission)
    assert check.results == (True, True, False, True)
    assert check.findings == (
        "not reproduced: the delivered model predicts differently from its rebuild",
    )
    # A changed map is part of the delivered pipeline, and fails the same check.
    remapped = copy.deepcopy(built.candidates["search winner"])
    remapped.mapping = lambda p: p
    assert d.inspect_delivery(remapped, built.window, built.commission).results == (
        True,
        True,
        False,
        True,
    )
    # A report that claims a better loss than its own settings give does not reproduce.
    flattering = copy.deepcopy(built.candidates["logistic regression"])
    flattering.reported_loss -= 0.01
    flattering.trials[0]["loss"] -= 0.01
    faults = d.inspect_delivery(flattering, built.window, built.commission)
    assert faults.results == (True, True, False, False)


def test_trial_log_is_held_to_the_commission(built):
    """The reviewer's second reproduction: settings and losses of the log are checked."""
    import copy

    d = demo()
    window, commission = built.window, built.commission

    def fourth(delivery):
        check = d.inspect_delivery(delivery, window, commission)
        assert check.results[:3] == (True, True, True)
        assert not check.results[3]
        return check.findings[0]

    # A searched logistic model with C=2.0, which the commissioned space does not hold.
    outside = d.fit_delivery(
        "search winner", "seeded search", {"family": "logistic", "C": 2.0}, window, commission
    )
    unseeded = "not the seed's configurations in their order"
    assert fourth(outside) == f"trial log: settings outside the commissioned space; {unseeded}"
    # A logged loss changed to 99.
    relogged = copy.deepcopy(built.candidates["search winner"])
    relogged.trials[3]["loss"] = 99.0
    assert fourth(relogged) == "trial log: a logged loss does not reproduce"
    # A log that hides a lower loss than the delivery's.
    hidden = copy.deepcopy(built.candidates["search winner"])
    hidden.trials[3]["loss"] = 0.01
    assert fourth(hidden) == (
        "trial log: a logged loss does not reproduce; the delivery is not the log's lowest loss"
    )
    # One configuration logged twice, inside the budget.
    repeated = copy.deepcopy(built.candidates["search winner"])
    repeated.trials[1] = {**repeated.trials[0], "entry": 2}
    assert fourth(repeated) == f"trial log: a configuration logged more than once; {unseeded}"
    # The fix checker's second attack: the trials reordered, then renumbered.
    reordered = copy.deepcopy(built.candidates["search winner"])
    reordered.trials = [{**t, "entry": n} for n, t in enumerate(reversed(reordered.trials), 1)]
    assert [t["entry"] for t in reordered.trials] == list(range(1, 21))
    assert fourth(reordered) == f"trial log: {unseeded}"
    # A search that shows its five best trials and hides the other fifteen.
    picked = copy.deepcopy(built.candidates["search winner"])
    best_five = sorted(picked.trials, key=lambda t: t["loss"])[:5]
    picked.trials = [{**t, "entry": n} for n, t in enumerate(best_five, 1)]
    assert fourth(picked) == f"trial log: {unseeded}"
    # The seed's own configurations, in their order, are what the search logged.
    assert [t["settings"] for t in built.trials] == d.configurations(20, commission.seed)
    # A search may not return a pipeline of its own; the commission allows the agent one.
    own = d.fit_delivery(
        "search winner",
        "seeded search",
        {"family": "group", "clean": True, "C": 1.0},
        window,
        commission,
    )
    assert fourth(own) == f"trial log: settings outside the commissioned space; {unseeded}"
    assert built.checks["clean agent"].results[3]
    # An undeclared setting beside the declared ones is outside the space too.
    assert not d.allowed({"family": "logistic", "C": 1.0, "tol": 1.0}, d.AGENT, commission.space)
    assert not d.allowed({"family": "logistic", "C": 2.0}, d.AGENT, commission.space)
    assert d.allowed(built.baseline.settings, "person", commission.space)
    # Every trial of the chapter's search and the July practice run stays inside it.
    assert all(d.allowed(t["settings"], "seeded search", commission.space) for t in built.trials)


def test_frozen_log_loss_is_the_commissions_measure_on_august(built, tmp_path):
    """Both loss rows are of the scores as trained; chapter 5 printed the baseline's."""
    import re

    d = demo()
    _, golden = d.prepare()
    record, _ = d.evaluate(built, golden, tmp_path)
    printed = (OUTPUTS / "_fnd.supervised-learning.numbers-01.txt").read_text()
    earlier = re.search(r"^boosted trees: August log loss (\d\.\d{4}),", printed, re.M)
    assert earlier is not None
    assert f"{record.evidence[0]['frozen_loss']:.4f}" == earlier[1]
    rows = [row for row, _ in zip(golden.requests, golden.labels)]
    y = np.array([label.found_false_30d for label in golden.labels])
    for row in record.evidence:
        delivery = built.baseline if row["name"] == d.BASELINE else built.candidates[row["name"]]
        p = delivery.predict(rows, mapped=False)
        assert row["frozen_loss"] == pytest.approx(
            float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        )


def test_clean_statistic_omits_each_rows_own_label_and_all_validation_labels(built):
    from demos.delegated_model_pipeline import GroupRisk

    train, validation = built.window["training"], built.window["validation"]
    original = GroupRisk(clean=True).fit(train, validation)
    row = train[0]
    flipped = [
        r.model_copy(update={"found_false_30d": 1 - r.found_false_30d})
        if r.request_id == row.request_id
        else r
        for r in train
    ]
    changed = GroupRisk(clean=True).fit(flipped, validation)
    np.testing.assert_array_equal(
        original.features([row], leave_out=True), changed.features([flipped[0]], leave_out=True)
    )
    poisoned_validation = [
        r.model_copy(update={"found_false_30d": 1 - r.found_false_30d}) for r in validation
    ]
    other = GroupRisk(clean=True).fit(train, poisoned_validation)
    assert original.provenance() == other.provenance()
    np.testing.assert_array_equal(original.predict(validation), other.predict(validation))


def test_frozen_rows_are_rejected_by_builder(built):
    d = demo()
    _, golden = d.prepare()
    with pytest.raises(ValueError, match="development rows only"):
        d.build({"training": built.window["training"], "validation": list(golden.requests)})


def test_box_and_exercise_arithmetic(captured):
    import math

    d = demo()
    data = json.loads(captured["fnd.automl-and-selection.optimism-01"])
    assert data["family_chance"] == pytest.approx(0.2262190625)
    assert data["union_bound"] == 0.25 and data["bonferroni"] == 0.01
    for row, expected in zip(data["normal_maxima"], (1.163, 1.868, 2.508)):
        assert abs(row["mean_max"] - expected) < 0.015
        assert row["mean_max"] < row["bound"]
        assert row["bound"] == pytest.approx(math.sqrt(2 * math.log(row["candidates"])))
    assert data["allowance"] == pytest.approx(data["standard_error"] * math.sqrt(2 * math.log(20)))
    assert d.luck_allowance(0.004, 50) == pytest.approx(0.01118859849)
    keys = captured["fnd.automl-and-selection.keys-01"]
    assert "[-0.0392, -0.0008], accepted" in keys and "[-0.0420, 0.0020], not adopted" in keys
    assert "0.0225" in keys and "2.45 at 20 candidates, 2.86 at 60" in keys
    assert d.luck_allowance(1, 60) == pytest.approx(math.sqrt(2 * math.log(60)))
    # A lead over the bound is no pass: the tail is the union bound's question.
    from statistics import NormalDist

    tail = 1 - NormalDist().cdf(0.012 / 0.004)
    assert 0.012 > d.luck_allowance(0.004, 50) > 0.004 * 2.249
    assert f"chance {1 - (1 - tail) ** 50:.4f} if independent, at most {50 * tail:.4f}" in keys
    assert 50 * tail > 0.05  # over the bound, and still not rare
    sd = 0.62 / math.sqrt(4000)
    tail = 1 - NormalDist().cdf(0.02 / sd)
    assert f"chance {1 - (1 - tail) ** 14:.4f} if independent, at most {14 * tail:.4f}" in keys
    assert "leads over the baseline's 0.1683: 0.0017 and 0.0019" in keys


def test_what_one_month_can_settle(built, captured):
    """power-01: a half-width, the months a small saving would clear the rule, three challengers."""
    import math
    import re
    from statistics import NormalDist

    lines = captured["fnd.automl-and-selection.power-01"].splitlines()
    spread = float(re.search(r"sd \$(\d+\.\d+)", lines[0])[1])
    z, error = NormalDist().inv_cdf(0.975), spread / math.sqrt(999)
    assert f"95% half-width at 999 requests: ${z * error:.4f}" == lines[1]
    for line, saving in zip(lines[2:4], (0.02, 0.01)):
        needed = math.ceil((z * spread / saving) ** 2)
        chance = NormalDist().cdf(saving / error - z)
        assert f"${saving:.2f}" in line and f"{needed:,} requests" in line
        assert line.endswith(f"chance {chance:.3f}")
        assert needed > 999 and chance < 0.5  # one month is short of both
    assert "at most 0.075 that one is" in lines[4] and lines[4].endswith("at most 0.025")
    assert lines[4].startswith("If the bootstrap upper bounds attain nominal one-sided coverage:")
    assert "do not establish those coverage guarantees" in lines[5]
    assert len(built.commission.challengers) * built.commission.alpha / 2 == pytest.approx(0.075)


def test_bootstrap_pairing_and_best_five_win_counts(built):
    d = demo()
    best, wins, error, lead, own = d.search_summary(built)
    assert len(best) == 5 and wins.sum() == 200
    assert built.searched[best[0]].name == "search winner"
    assert error > 0 and lead == pytest.approx(0.00165426156926754)
    assert 0 < own <= error and 0.4 < lead / own < 0.6  # about half its own standard error
    assert d.interval(np.zeros(30)) == (0, 0, 0)


def test_fixtures_and_figure_data_are_the_results(built, captured):
    d = demo()
    for name, fixture in (("agent delivery", "agent_delivery"), ("clean agent", "agent_clean")):
        assert (
            json.loads((d.DATA / f"delegated_model_{fixture}.json").read_text())
            == built.candidates[name].report()
        )
    data = json.loads(captured["fnd.automl-and-selection.truth-01"])
    assert len(data["models"]) == 23  # 22 plotted models plus off-scale agent
    winner = next(r for r in data["models"] if r["name"] == "search winner")
    assert winner["measured_lead"] > winner["true_lead"]
    assert winner["true_saving"] < 0  # the constructed expectation does not establish a gain
    assert data["winner_true_rank"] >= 1


def test_unprinted_lab_acquisition_and_equal_budgets(built):
    from demos.delegated_search_lab import (
        compare,
        expected_improvement,
        feature_construction,
        stacking,
    )

    # At the same predicted mean, uncertainty offers improvement; zero variance at
    # the current best offers none. This tests acquisition, rather than mean sorting.
    gain = expected_improvement([0.1, 0.1], [0.0, 0.05], 0.1)
    assert gain[0] == 0 and gain[1] > 0
    results = compare(built.window, budget=4)
    assert set(results) == {"grid", "random", "model-based"}
    assert all(len(rows) == 4 for rows in results.values())
    assert all(
        len({json.dumps(t["settings"], sort_keys=True) for t in rows}) == 4
        for rows in results.values()
    )
    stack = stacking(built.window)
    assert stack["meta_rows"] > 0 and stack["validation_loss"] > 0
    assert all(f["latest_label"] <= f["first_held"] for f in stack["folds"])
    assert feature_construction(built.window)["own_label_left_out"]


def test_finite_space_rejects_impossible_budget():
    with pytest.raises(ValueError, match="finite"):
        demo().configurations(1000)


def test_july_exercise_reference_solution_reproduces_with_one_read(tmp_path):
    runs = []
    for name in ("first", "second"):
        result = subprocess.run(
            [sys.executable, str(DEMO), "--exercise-july"],
            cwd=tmp_path,
            env={**ENV, "BOOK_GOLDEN_DIRECTORY": str(tmp_path / name)},
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        runs.append(result.stdout)
        data = json.loads(result.stdout)
        record = data["record"]
        assert data["events"][0]["event"] == "commission"
        assert len([e for e in data["events"] if e["event"] == "trial"]) == 40
        assert data["events"][-2]["event"] == "sealed" and data["events"][-1]["event"] == "consult"
        assert len(record["log"]) == 1 and record["log"][0]["entry"] == 1
        assert record["frozen"]["name"] == "history-july" and record["frozen"]["requests"] == 985
        assert record["commission"]["budget"] == 40
        from refund_agent.selection import decide, fingerprint

        assert record["sha256"] == fingerprint({k: v for k, v in record.items() if k != "sha256"})
        for row in record["evidence"][1:]:
            assert record["decisions"][row["name"]] == decide(
                all(record["checks"][row["name"]]["results"]),
                row["difference"][0],
                row["difference"][1:],
            )
    assert runs[0] == runs[1]
