"""Numerical contracts, isolated printed listings and captured chapter 16 outputs."""

import contextlib
import io
import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from demos import regions as _regions

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demos/rl_foundations_demo.py"
CASE = ROOT / "refund_agent/learning_case.py"
TOOLS = vars(_regions)
PREFIX = "fnd.rl-foundations."
NAMES = (
    "model-01",
    "values-01",
    "vi-01",
    "stopping-01",
    "td-01",
    "learn-01",
    "state-01",
    "reward-01",
    "keys-01",
    "case-01",
)


def snippets():
    return {
        s.snippet_id: s
        for source in (DEMO, CASE)
        for s in TOOLS["extract_snippets_from_text"](source.read_text(), source)
    }


def demo():
    from demos import rl_foundations_demo

    return rl_foundations_demo


def test_printed_listings_run_in_chapter_order_without_hidden_setup():
    namespace = {"__name__": "__main__"}
    found = snippets()
    for name in ("vi-01", "td-01", "case-01"):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(compile(found[PREFIX + name].content, name, "exec"), namespace)
        stored = ROOT / "book_outputs" / f"_{PREFIX}{name}.txt"
        assert stdout.getvalue().encode() == stored.read_bytes()


def test_stopping_listing_runs_alone_and_prints_computed_costs():
    region = snippets().get(PREFIX + "stopping-01")
    assert region is not None, "the rising-scan counterexample needs a captured region"
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(region.content, "stopping-01", "exec"), {"__name__": "__main__"})
    assert stdout.getvalue() == (
        "rising scans (0, 0.9, 0), amount 10, wait 1: "
        "one-stage rule pays day 0, cost 10.00; optimal waits, cost 3.00\n"
    )


@pytest.fixture(scope="module")
def captured():
    return TOOLS["capture_outputs"](snippets(), sys.executable)


@pytest.mark.parametrize("name", NAMES)
def test_every_stored_output_regenerates_byte_for_byte(name, captured):
    path = ROOT / "book_outputs" / f"_{PREFIX}{name}.txt"
    assert captured[PREFIX + name].encode() == path.read_bytes()


def test_capture_is_deterministic_in_fresh_processes(captured):
    assert captured == TOOLS["capture_outputs"](snippets(), sys.executable)


def test_every_named_region_has_a_nonempty_stored_output():
    found = snippets()
    assert set(found) == {PREFIX + name for name in NAMES}
    assert all(region.output for region in found.values())
    for name in NAMES:
        assert (ROOT / "book_outputs" / f"_{PREFIX}{name}.txt").read_text().strip()


def test_import_has_no_stdout_or_file_effects(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import demos.rl_foundations_demo; import refund_agent.learning_case",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert list(tmp_path.iterdir()) == []


def test_belief_and_scan_probability_follow_bayes_rule():
    m = demo().LateParcel()
    for s in range(m.last_day + 1):
        survival = 0.1 + 0.9 * 0.4**s
        assert m.lost_belief(s) == pytest.approx(0.1 / survival)
        assert m.scan_probability(s) == pytest.approx(0.54 * 0.4**s / survival)


def test_parcel_backward_induction_and_undiscounted_sweeps():
    m = demo()
    parcel = m.LateParcel()
    mdp = m.parcel_mdp(parcel)
    values, policy, history = m.value_iteration(mdp)
    assert policy == {s: "wait" if s < 5 else "pay" for s in range(11)}
    assert round(values[0], 2) == -11.41
    backward = {"paid": 0.0, "scanned": 0.0}
    for s in reversed(range(11)):
        backward[s] = max(m.bellman_backup(mdp, backward, s).values())
    assert values == pytest.approx(backward)
    assert history[-1]["sweep"] == 11
    # Backup policies use the previous sweep's values, so the boundary travels back.
    assert [min(s for s, a in row["policy"].items() if a == "pay") for row in history[1:7]] == [
        10,
        9,
        8,
        7,
        6,
        5,
    ]
    for s in range(11):
        assert values[s] == pytest.approx(max(m.bellman_backup(mdp, values, s).values()))


@pytest.mark.parametrize("k", range(11))
def test_policy_evaluation_matches_independent_recursion(k):
    m = demo()
    parcel = m.LateParcel()
    policy = {s: "pay" if s >= k else "wait" for s in range(11)}
    values = m.policy_evaluation(m.parcel_mdp(parcel), policy)
    expected = -parcel.amount
    for s in reversed(range(k)):
        expected = -parcel.wait_cost + (1 - parcel.scan_probability(s)) * expected
    assert values[0] == pytest.approx(expected)


@pytest.mark.parametrize("amount", (24.99, 49.99))
@pytest.mark.parametrize("price", (0, 1, 3, 10, 27))
def test_one_step_rule_matches_planning_for_falling_scan_probabilities(amount, price):
    m = demo()
    model = m.LateParcel(amount=amount, wait_cost=price)
    _, policy, _ = m.value_iteration(m.parcel_mdp(model))
    assert policy == m.wait_rule(model)


def test_one_step_rule_fails_when_scan_probability_rises():
    m = demo()

    class RisingScans(m.LateParcel):
        def scan_probability(self, s):
            return (0, 0.9, 0)[s]

    model = RisingScans(last_day=2, amount=10, wait_cost=1)
    mdp = m.parcel_mdp(model)
    values, policy, _ = m.value_iteration(mdp)
    assert m.wait_rule(model)[0] == "pay"
    assert policy[0] == "wait"
    assert values[0] == pytest.approx(-3)
    assert m.policy_evaluation(mdp, m.wait_rule(model))[0] == -10


@pytest.mark.parametrize("discount", (0.5, 0.8, 0.9))
def test_discounted_contraction_on_one_state(discount):
    m = demo()
    optimum = max(3, 1 / (1 - discount))
    _, _, history = m.value_iteration(m.one_state_example(discount))
    all_errors = np.array([abs(optimum - row["values"]["open"]) for row in history])
    assert np.all(all_errors[1:] <= discount * all_errors[:-1] + 2e-14)
    errors = [abs(optimum - row["values"]["open"]) for row in history[1:]]
    if discount == 0.5:
        assert errors == [0.0]
    else:
        np.testing.assert_allclose(errors[1:], np.array(errors[:-1]) * discount, atol=2e-14)


def test_terminal_aware_single_updates():
    m = demo()
    assert m.td0_update(2, 1, 4, 0.9, 0.1, False) == pytest.approx((2.26, 4.6, 2.6))
    assert m.td0_update(2, 1, 400, 0.9, 0.1, True) == pytest.approx((1.9, 1, -1))
    assert m.q_learning_update(-46, -3, [-49.99, -51.20], 1, 0.1, False)[0] == pytest.approx(
        -46.699
    )
    assert m.q_learning_update(-46, -3, [999], 1, 0.1, True)[0] == pytest.approx(-41.7)
    assert m.td0_update(-20, -3, -12, 1, 0.5, False)[0] == -17.5
    # Exercise 5 supplies next value -15, yielding -19; a scan needs no next value.
    assert m.td0_update(-20, -3, -15, 1, 0.5, False)[0] == -19
    assert m.td0_update(-20, -3, -15, 1, 0.5, True)[0] == -11.5


def test_discount_zero_inverse_count_step_is_the_running_mean():
    m = demo()
    value = 0.0
    rewards = [1, -2, 4, 0]
    for n, reward in enumerate(rewards, 1):
        value, _, _ = m.q_learning_update(value, reward, [1e6], 0, 1 / n, False)
        assert value == pytest.approx(sum(rewards[:n]) / n)


def test_q_learning_run_is_seeded_and_counts_pairs():
    m = demo()
    first = m.q_learning_run(m.LateParcel(), claims=100, seed=7)
    second = m.q_learning_run(m.LateParcel(), claims=100, seed=7)
    assert first == second
    assert sum(first["counts"][0]) == 100
    assert first != m.q_learning_run(m.LateParcel(), claims=100, seed=8)


def test_q_learning_bootstraps_only_permitted_actions_at_last_day():
    m = demo()
    run = m.q_learning_run(m.LateParcel(lost_share=1, last_day=1), claims=100, seed=1)
    assert run["counts"][1][1] == 0  # Waiting is not an action at the compulsory boundary.
    assert run["q"][0][1] < -45  # The unused terminal-day wait entry cannot supply a zero target.


def test_exploration_simulation_agrees_with_exact_expected_cost():
    m = demo()
    model = m.LateParcel()
    _, rule, _ = m.value_iteration(m.parcel_mdp(model))
    rng = np.random.default_rng(71)

    def behavior(s):
        if s == model.last_day:
            return "pay"
        return ("pay", "wait")[int(rng.integers(2))] if rng.random() < 0.1 else rule[s]

    mean_cost = np.mean([m.simulate_claim(model, behavior, rng)["cost"] for _ in range(50000)])
    planned_cost = -m.policy_evaluation(m.parcel_mdp(model), rule)[0]
    exact = m.exploration_cost(model)
    assert round(exact["extra_cost"], 2) == 2.94
    assert abs(mean_cost - planned_cost - exact["extra_cost"]) < 0.30


def test_large_followup_log_recovers_true_rule():
    m = demo()
    model = m.LateParcel()
    learned = m.plan_from_logs(model, claims=200000, seed=17)
    assert learned["policy"] == m.wait_rule(model)
    assert len(learned["at_risk"]) == 10
    assert learned["paid_day"] == 0


def test_shaping_keeps_rule_but_flat_bonus_changes_it():
    result = demo().shaping_check()
    assert result["potential_pay_days"] == [5, 5, 5]
    assert result["flat_bonus_pay_day"] == 6
    assert result["shaped_cost"] == pytest.approx(result["raw_cost"] - 4.999)


@pytest.mark.parametrize(
    "potential_index, value_shift",
    [(0, 10.867391304347827), (1, 1.0), (2, -0.8414709848078965)],
    ids=["lost-belief", "linear", "sine"],
)
def test_shaping_output_names_potentials_in_pay_day_order(
    captured, monkeypatch, potential_index, value_shift
):
    m = demo()
    shaped_models = []
    original_shaped_mdp = m.shaped_mdp

    def record_shaped_mdp(mdp, potential):
        shaped = original_shaped_mdp(mdp, potential)
        shaped_models.append(shaped)
        return shaped

    monkeypatch.setattr(m, "shaped_mdp", record_shaped_mdp)
    m.shaping_check()
    raw_values, _, _ = m.value_iteration(m.parcel_mdp(m.LateParcel()))
    shaped_values, _, _ = m.value_iteration(shaped_models[potential_index])
    # V'(1) - V(1) = -Phi(1); lost belief at day 1 is 0.1 / 0.46.
    assert shaped_values[1] - raw_values[1] == pytest.approx(value_shift)
    assert captured[PREFIX + "reward-01"].splitlines()[2] == (
        "three potentials' pay days (-49.99 * lost belief, -1 * s, sin(s)): 5, 5, 5"
    )


def test_carrier_aware_policy_is_no_worse_than_pooled():
    result = demo().two_carriers()
    assert result["carrier_pay_days"] == [5, 10]
    assert result["pooled_pay_day"] == 10
    assert result["pooled_cost"] >= result["aware_cost"]
    assert round(result["pooled_cost"] - result["aware_cost"], 2) == 0.54


def test_student_support_scores_and_incident_reward_error():
    m = demo()
    assert m.student_support_rewards() == {"spam_help": (0.20, -4.20), "targeted": (0.16, -1.14)}
    assert m.value_iteration(m.incident_mdp("planted"))[1]["unchecked"] == "close"
    assert m.value_iteration(m.incident_mdp("corrected"))[1]["unchecked"] == "wait"
    fixture = json.loads(
        (ROOT / "demos/data/rl_foundations_incident_05.json").read_text()
    )
    assert fixture == m.incident_report()
    assert np.all(np.diff([row["reward"] for row in fixture["reward_curve"]]) > 0)


def test_figure_data_matches_runs_without_refitting():
    m = demo()
    vi = json.loads((ROOT / "demos/data/rl_foundations_vi.json").read_text())
    learning = json.loads((ROOT / "demos/data/rl_foundations_learning.json").read_text())
    assert vi == m.vi_figure_data()
    assert learning == m.learning_experiment()


def test_figure_and_exercise_data_regenerate_byte_for_byte(tmp_path):
    demo().write_data(tmp_path)
    for name in ("vi", "learning", "incident_05"):
        filename = f"rl_foundations_{name}.json"
        stored = ROOT / "demos/data" / filename
        assert (tmp_path / filename).read_bytes() == stored.read_bytes()


def test_no_golden_or_history_read_and_no_consultation_log_writes(monkeypatch, tmp_path):
    from refund_agent import split
    from refund_agent.data import generate, history
    from refund_agent.split import GoldenSet

    def forbidden(*args, **kwargs):
        raise AssertionError("chapter 16 read project data or a frozen set")

    monkeypatch.setattr(GoldenSet, "consult", forbidden)
    monkeypatch.setattr(GoldenSet, "_journal", forbidden)
    monkeypatch.setattr(GoldenSet, "persist", forbidden)
    monkeypatch.setattr(split, "freeze", forbidden)
    monkeypatch.setattr(generate, "generate", forbidden)
    monkeypatch.setattr(history, "generate_history", forbidden)
    monkeypatch.setattr(history, "load_history", forbidden)
    demo().learning_experiment.cache_clear()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(DEMO)])
    with contextlib.redirect_stdout(io.StringIO()):
        for source in (DEMO, CASE):
            runpy.run_path(str(source), run_name="__main__")
    assert list(tmp_path.iterdir()) == []
