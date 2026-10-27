"""Constructed bank examples and Chapter 17's frozen offer-policy judgment.

The three printed regions run in order without hidden setup. Importing reads,
prints and writes nothing. Each evaluation reads outcomes once, reusing the
receipt for an identical access;
simulation truth is consulted only after its evaluation record is on disk.
NumPy and the standard library suffice. --write-data writes derived data only.
"""

import os
from pathlib import Path

from refund_agent.offers.evaluation import OfferPackage
from refund_agent.offers.notices import OFFERS
from refund_agent.offpolicy.estimators import (
    Record,
    improvement_terms,
    per_record_terms,
    policy_ratios,
    ratios,
    self_normalized_interval,
)
from refund_agent.offpolicy.frozen import FrozenLog, from_offers
from refund_agent.offpolicy.policies import Policy, offer_candidates
from refund_agent.offpolicy.record import EvaluationRecord, evaluate
from refund_agent.offpolicy.reward_model import CellModel, fit_cells
from refund_agent.offpolicy.rule import Rule
from refund_agent.offpolicy.support import support_table, supported

DATA = Path(__file__).resolve().parent / "data"
PACKAGE_DIRECTORY = Path(
    os.environ.get("PRACTICAL_RL_PACKAGE_DIRECTORY", DATA / "exploration_bandits_package")
)


def bank_records(scale=1):
    return [
        Record("bank", a, int(i < answers * scale), {"text": 0.8, "call": 0.2})
        for a, count, answers in (("text", 80, 48), ("call", 20, 16))
        for i in range(count * scale)
    ]


def bank_policies():
    return Policy("calls-60-percent", (("bank", (("text", 0.4), ("call", 0.6))),)), Policy(
        "bank-logger", (("bank", (("text", 0.8), ("call", 0.2))),)
    )


def bank_model(text=0.55, call=0.85):
    return CellModel((("bank", "text", text, 1), ("bank", "call", call, 1)))


def coverage_experiment(seed=7, logs=500, alerts=2000):
    import numpy as np

    if alerts % 2:
        raise ValueError("equal-sized amount bands required")
    rng = np.random.default_rng(seed)
    band = np.repeat([0, 1], alerts // 2)
    rates = np.array([[0.55, 0.85], [0.65, 0.80]])
    true_value = float(np.mean(0.4 * rates[:, 0] + 0.6 * rates[:, 1]))
    rows = []
    for q in (0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0):
        call_chance = np.where(band == 0, q, 0.5)
        calls = rng.random((logs, alerts)) < call_chance
        rewards = rng.random((logs, alerts)) < rates[band, calls.astype(int)]
        # A separate training log of this design; the two bands are pooled.
        training_calls = rng.random((logs, alerts)) < call_chance
        training_rewards = rng.random((logs, alerts)) < rates[band, training_calls.astype(int)]
        model = np.column_stack(
            [
                ((training_rewards & (training_calls == a)).sum(axis=1) + 1)
                / ((training_calls == a).sum(axis=1) + 2)
                for a in (0, 1)
            ]
        )
        logged = np.where(calls, call_chance, 1 - call_chance)
        weights = np.where(calls, 0.6, 0.4) / logged
        ips_values = (weights * rewards).mean(axis=1)
        sn_values = (weights * rewards).sum(axis=1) / weights.sum(axis=1)
        dm_values = model @ np.array([0.4, 0.6])
        predictions = np.take_along_axis(model, calls.astype(int), axis=1)
        dr_values = dm_values + (weights * (rewards - predictions)).mean(axis=1)
        neff = weights.sum(axis=1) ** 2 / (weights**2).sum(axis=1)
        statistics = {}
        for name, values in (
            ("ips", ips_values),
            ("sn", sn_values),
            ("dm", dm_values),
            ("dr", dr_values),
        ):
            statistics[name] = dict(
                mean=float(values.mean()),
                median=float(np.median(values)),
                low=float(np.quantile(values, 0.025)),
                high=float(np.quantile(values, 0.975)),
                samples=values.tolist(),
            )
        rows.append(
            dict(q=q, supported=q > 0, n_eff_median=float(np.median(neff)), estimates=statistics)
        )
    return dict(
        seed=seed,
        logs=logs,
        alerts=alerts,
        true_value=true_value,
        answer_rates=rates.tolist(),
        training="independent same-design log; pooled bands; add-one smoothing",
        rows=rows,
    )


def print_coverage(data):
    print(
        f"simulated bank: seed {data['seed']}; logs {data['logs']}; alerts {data['alerts']}; true value {data['true_value']:.6f}"
    )
    print("q support IPS_mean SN_mean DM_mean DR_mean median_n_eff")
    for row in data["rows"]:
        print(
            f"{row['q']:.3f} {row['supported']} "
            + " ".join(f"{row['estimates'][key]['mean']:.6f}" for key in ("ips", "sn", "dm", "dr"))
            + f" {row['n_eff_median']:.3f}"
        )
    print("central 95% ranges: q estimator low high")
    for row in data["rows"]:
        for name, stats in row["estimates"].items():
            print(f"{row['q']:.3f} {name} {stats['low']:.6f} {stats['high']:.6f}")
    print("q=0 raw numbers are not policy judgments; missing small-call support")


def horizon_experiment(seed=7):
    import numpy as np

    matches = np.random.default_rng(seed).random((1000, 10)) < 0.5
    rows = []
    for horizon in (1, 4, 10):
        match = matches[:, :horizon].all(axis=1)
        ratio = 2**horizon
        rows.append(
            dict(
                steps=horizon,
                expected=1000 / ratio,
                ratio=ratio,
                matched=int(match.sum()),
                n_eff=effective_sample_size(match.astype(float) * ratio),
            )
        )
    return dict(
        seed=seed,
        trajectories=1000,
        reward_timing="action A_t followed by reward R_(t+1)",
        rows=rows,
    )


def prepare_offer_evaluation():
    package = OfferPackage.load(PACKAGE_DIRECTORY)
    candidates, fixed = offer_candidates(package)
    model = fit_cells(from_offers(package.learning.rows))
    window = FrozenLog.from_offer_window(
        package.evaluation,
        manifest=dict(
            dates=["2026-02-01", "2026-06-01"],
            bounds="start inclusive; end exclusive; decision date",
        ),
        journal_path=PACKAGE_DIRECTORY.parent / "practical_rl_evaluation.access.jsonl",
    )
    return package, candidates, fixed, model, window


def truth_check(package, candidates, fixed, record, record_path):
    # Readback proves that a complete record was written before consulting truth.
    written = EvaluationRecord.read(record_path)
    if (
        written.data["rule_fingerprint"] != record.data["rule_fingerprint"]
        or written.data != record.data
    ):
        raise ValueError("the evaluation record must be written before the truth read")
    truth = package.evaluation.truth.consult(
        "chapter-21", f"post-record simulation check; rule {record.data['rule_fingerprint']}"
    )
    result: dict = dict(reads=1, after_record=True, policies={})
    for name, policy in {**candidates, "fixed": fixed}.items():
        value = sum(
            sum(
                p * row["answers"][a]["reward"]
                for a, p in policy.probabilities(row["reason"]).items()
            )
            for row in truth
        ) / len(truth)
        result["policies"][name] = dict(value=value, improvement=0.0)
    baseline = result["policies"]["fixed"]["value"]
    for row in result["policies"].values():
        row["improvement"] = row["value"] - baseline
    return result


def print_truth(result):
    print(f"simulation truth: reads {result['reads']}; after record {result['after_record']}")
    for name, row in result["policies"].items():
        print(f"truth {name}: value {row['value']:.6f}; improvement {row['improvement']:.6f}")
    print("truth does not change the rule, candidate or decision")


def support_figure_data(package, candidates, fixed, record):
    zero = {
        name: support_table(from_offers(package.cap_zero.log.rows), policy)
        for name, policy in {**candidates, "fixed": fixed}.items()
    }
    return dict(
        offers=list(OFFERS),
        cells=sorted({r.reason for r in package.evaluation.log.rows}),
        chances=list(package.logger.probabilities()),
        cap_zero_chances=[1, 0, 0],
        cap=package.logger.cap,
        support={name: card["support"] for name, card in record.data["policies"].items()},
        cap_zero_support=zero,
        cap_zero_judged={name: supported(table) for name, table in zero.items()},
        n_eff={name: card["n_eff"] for name, card in record.data["policies"].items()},
    )


def exercise_reports():
    from dataclasses import replace

    # An explicitly constructed exercise window, not a second refund log.
    rows = [
        Record("damaged", a, r, {"standard": 0.8, "dated": 0.2, "referenced": 0.0})
        for a, r in (
            ("standard", 1),
            ("standard", 1),
            ("standard", 0),
            ("standard", 1),
            ("standard", 1),
            ("standard", 1),
            ("standard", 0),
            ("standard", 1),
            ("dated", 1),
            ("dated", 1),
        )
    ]
    rows += [
        Record("changed_mind", a, r, {"standard": 0.2, "dated": 0.8, "referenced": 0.0})
        for a, r in (
            ("standard", 1),
            ("standard", 0),
            ("dated", 1),
            ("dated", 1),
            ("dated", 1),
            ("dated", 1),
            ("dated", 1),
            ("dated", 1),
            ("dated", 0),
            ("dated", 0),
        )
    ]
    rows += [
        Record("not_delivered", "standard", r, {"standard": 1.0, "dated": 0.0, "referenced": 0.0})
        for r in (1, 1, 0, 1, 0)
    ]
    # Fixed counts in 1,000 distinct decisions make B's wrong reassurance narrow.
    rows = [
        replace(
            row,
            decision_id=f"exercise-{i:04d}",
            request_id=f"request-{i:04d}",
            customer_id=f"customer-{i:04d}",
        )
        for i, row in enumerate(rows * 40)
    ]
    policy = Policy.deterministic(
        "exercise-new-notice",
        {"changed_mind": "dated", "damaged": "dated", "not_delivered": "dated"},
        OFFERS,
    )
    baseline = Policy.deterministic(
        "exercise-fixed",
        {"changed_mind": "standard", "damaged": "standard", "not_delivered": "standard"},
        OFFERS,
    )
    model = CellModel(
        tuple(
            (cell, a, mean, 10)
            for cell, mean in (("changed_mind", 0.75), ("damaged", 1.0), ("not_delivered", 0.8))
            for a in OFFERS
        )
    )
    cells = ["changed_mind", "damaged"]
    subset = [r for r in rows if r.context in cells]
    matching = [
        r.reward
        for r in subset
        if r.reward is not None and policy.probabilities(r.context)[r.action]
    ]
    estimates = dict(
        ips=ips(policy_ratios(policy, subset), [r.reward for r in subset]),
        sn=self_normalized_ips(policy_ratios(policy, subset), [r.reward for r in subset]),
        dr=float(per_record_terms(policy, model, subset).mean()),
        improvement=float(improvement_terms(policy, baseline, model, subset).mean()),
        n_eff=effective_sample_size(policy_ratios(policy, subset)),
    )
    return dict(
        name="constructed-frozen-offer-exercise-05",
        records=[r.record() for r in rows],
        candidate=policy.record(),
        baseline=baseline.record(),
        model=model.cells,
        report_a=dict(
            method="unweighted matching mean",
            cells=cells,
            value=sum(matching) / len(matching),
            comparator=sum(r.reward for r in subset if r.reward is not None) / len(subset),
        ),
        report_b=dict(
            method="doubly robust",
            value=float(per_record_terms(policy, model, rows).mean()),
            interval=list(normal_interval(per_record_terms(policy, model, rows))),
            judged=True,
        ),
        marking=dict(
            report_a="ignores applied chances",
            report_b="reports a model guess in an unsupported cell as judged",
            supported_cells=cells,
            recomputed=estimates,
            decision="reject both reports",
            remedies=[
                "restrict the candidate to supported cells",
                "collect the missing action under a signed cap",
            ],
        ),
    )


def evaluate_candidate_c(output_dir, policy_factory):
    """Evaluate the reader's policy, constructed before any frozen outcomes read."""
    package, _, fixed, model, _ = prepare_offer_evaluation()
    policy = policy_factory(package.candidate.frozen_at.isoformat())
    window = FrozenLog.from_offer_window(
        package.evaluation,
        manifest=dict(dates=["2026-02-01", "2026-06-01"]),
        journal_path=PACKAGE_DIRECTORY.parent / "practical_rl_evaluation.access.jsonl",
    )
    record = evaluate(
        {"C": policy},
        fixed,
        window,
        Rule(),
        model=model,
        reader="exercise-6",
        learning_records=len(package.learning.rows),
    )
    record.write(Path(output_dir) / "candidate-c-record.json")
    (Path(output_dir) / "candidate-c-output.txt").write_text(record.render() + "\n")
    return record


# region book:fnd.practical-rl.bank-01 output
import numpy as np

from refund_agent.offpolicy.estimators import normal_interval


def ips(weights, rewards):
    return float(np.mean(np.asarray(weights) * rewards))


def self_normalized_ips(weights, rewards):
    weights = np.asarray(weights, dtype=float)
    return float(weights @ rewards / weights.sum()) if weights.sum() else None


def effective_sample_size(weights):
    weights = np.asarray(weights, dtype=float)
    return (
        float(weights.sum() ** 2 / (weights @ weights))
        if np.any(weights)
        else 0.0
    )


if __name__ == "__main__":
    reward = np.array([1] * 48 + [0] * 32 + [1] * 16 + [0] * 4)
    rho = np.array([0.5] * 80 + [3.0] * 20)
    improvement = (rho - 1) * reward
    print(
        "channel alerts answers logged_chance target_chance ratio answer_rate"
    )
    print(
        f"text 80 {reward[:80].sum()} 0.8 0.4 "
        f"{rho[0]:.1f} {reward[:80].mean():.3f}"
    )
    print(
        f"call 20 {reward[80:].sum()} 0.2 0.6 "
        f"{rho[-1]:.1f} {reward[80:].mean():.3f}"
    )
    print(
        f"logged mean {reward.mean():.3f}; IPS {ips(rho, reward):.3f}; "
        f"SN {self_normalized_ips(rho, reward):.3f}; "
        f"n_eff {effective_sample_size(rho):.3f}"
    )
    print(
        "IPS interval ["
        + ", ".join(f"{v:.3f}" for v in normal_interval(rho * reward))
        + "]"
    )
    print(
        f"improvement {improvement.mean():.3f}; interval ["
        + ", ".join(f"{v:.3f}" for v in normal_interval(improvement))
        + "]"
    )
# endregion book:fnd.practical-rl.bank-01


# region book:fnd.practical-rl.box-01 output
if __name__ == "__main__":
    weights, answers = [2, 1, 0.5], [1, 0, 1]
    print("ratios 2, 1, 0.5; rewards 1, 0, 1")
    print(
        f"IPS {ips(weights, answers):.3f}; SN {self_normalized_ips(weights, answers):.3f}; n_eff {effective_sample_size(weights):.3f}"
    )
# endregion book:fnd.practical-rl.box-01


# region book:fnd.practical-rl.edge-01 output
if __name__ == "__main__":
    weights = np.array([200.0] * 5 + [0.0] * 995)
    answers = np.array([1.0] * 5 + [0.0] * 995)
    print("records 1000; matching 5; ratio 200; all five answered; fixed comparator 0.6")
    print(
        f"IPS {ips(weights, answers):.3f}; interval ["
        + ", ".join(f"{v:.3f}" for v in normal_interval(weights * answers))
        + "]"
    )
    sn_bounds = self_normalized_interval(weights, answers)
    print(
        f"SN {self_normalized_ips(weights, answers):.3f}; interval [{sn_bounds[0]:.3f}, {sn_bounds[1]:.3f}]; improvement {self_normalized_ips(weights, answers) - 0.6:.3f}"
    )
    print(
        f"n_eff {effective_sample_size(weights):.3f}; one-sided 97.5% exact lower bound {0.025 ** (1 / 5):.3f}"
    )
    edge_rule = Rule()
    if effective_sample_size(weights) < edge_rule.n_eff_floor:
        print(f"decision hold; candidate effective size below {edge_rule.n_eff_floor}")
# endregion book:fnd.practical-rl.edge-01


# region book:fnd.practical-rl.dr-01 output
def doubly_robust(weights, rewards, target_prediction, logged_prediction):
    return float(
        np.mean(
            target_prediction
            + np.asarray(weights) * (np.asarray(rewards) - logged_prediction)
        )
    )


if __name__ == "__main__":
    model_text, model_call = 0.55, 0.85
    direct = 0.4 * model_text + 0.6 * model_call
    logged_prediction = np.array([model_text] * 80 + [model_call] * 20)
    correction = float(np.mean(rho * (reward - logged_prediction)))
    print(f"reward model text {model_text:.2f}, call {model_call:.2f}")
    print(
        f"DM {direct:.3f}; correction {correction:.3f}; "
        f"DR {doubly_robust(rho, reward, direct, logged_prediction):.3f}"
    )
# endregion book:fnd.practical-rl.dr-01


# region book:fnd.practical-rl.coverage-01 output
if __name__ == "__main__":
    coverage = coverage_experiment()
    print_coverage(coverage)
# endregion book:fnd.practical-rl.coverage-01


# region book:fnd.practical-rl.horizon-01 output
if __name__ == "__main__":
    horizons = horizon_experiment()
    print(
        f"student support: seed {horizons['seed']}; trajectories {horizons['trajectories']}; {horizons['reward_timing']}"
    )
    print("steps expected_matches ratio observed_matches n_eff")
    for row in horizons["rows"]:
        print(
            f"{row['steps']} {row['expected']:.6f} {row['ratio']} {row['matched']} {row['n_eff']:.3f}"
        )
# endregion book:fnd.practical-rl.horizon-01


# region book:fnd.practical-rl.refund-01 output
from pathlib import Path

from demos import practical_rl_demo as offer_demo
from refund_agent.offpolicy.record import evaluate as evaluate_offline
from refund_agent.offpolicy.rule import Rule

if __name__ == "__main__":
    package, candidates, fixed, model, window = (
        offer_demo.prepare_offer_evaluation()
    )
    rule = Rule()
    record = evaluate_offline(
        candidates,
        fixed,
        window,
        rule,
        model=model,
        learning_records=len(package.learning.rows),
    )
    record_path = record.write(Path("practical-rl-evaluation.json"))
    print(record.render())
    truth_result = offer_demo.truth_check(
        package, candidates, fixed, record, record_path
    )
    offer_demo.print_truth(truth_result)
# endregion book:fnd.practical-rl.refund-01


# region book:fnd.practical-rl.keys-01 output
if __name__ == "__main__":
    for chance in (0.3, 0.05):
        weights = ratios([0.9, 0.1, 0.8, 0.5], [chance, 0.6, 0.4, 0.5])
        print(
            f"exercise 3: first logged chance {chance:.2f}; ratios "
            + ", ".join(f"{v:.6f}" for v in weights)
        )
        print(
            f"IPS {ips(weights, [1, 0, 1, 1]):.3f}; SN {self_normalized_ips(weights, [1, 0, 1, 1]):.3f}; n_eff {effective_sample_size(weights):.3f}"
        )
    target, bank_logger = bank_policies()
    rows = bank_records()
    direct = 0.4 * 0.5 + 0.6 * 0.9
    correction = float(
        np.mean(
            policy_ratios(target, rows)
            * (
                np.array([r.reward for r in rows])
                - np.array([0.5 if r.action == "text" else 0.9 for r in rows])
            )
        )
    )
    print(
        f"exercise 4a: DM {direct:.3f}; correction {correction:.3f}; DR {direct + correction:.3f}"
    )
    thousand = evaluate(
        {"bank": target},
        bank_logger,
        FrozenLog.from_records(bank_records(10)),
        Rule(estimator="ips", estimand="bank candidate minus logging policy"),
    )
    card = thousand.data["policies"]["bank"]
    print(
        f"exercise 4b: alerts 1000; n_eff {card['n_eff']:.3f}; improvement {card['improvement']:.3f}; interval ["
        + ", ".join(f"{v:.3f}" for v in card["interval"])
        + f"]; exact p {card['exact']['p_value']:.6f}; draws {card['exact'].get('draws', 0)}; decision {card['decision']}"
    )
    for steps in (4, 10):
        print(
            f"exercise 4c: weeks {steps}; expected matching {1000 / 2**steps:.6f}; ratio {2**steps}"
        )
    sparse_target = Policy.deterministic("candidate", {"bank": "call"}, ("text", "call"))
    sparse_fixed = Policy.deterministic("fixed", {"bank": "text"}, ("text", "call"))
    sparse = [
        Record("bank", "call" if i < 500 else "text", int(i < 5), {"text": 0.5, "call": 0.5})
        for i in range(1000)
    ]
    sparse_record = evaluate(
        {"sparse": sparse_target}, sparse_fixed, FrozenLog.from_records(sparse), Rule()
    )
    sparse_card = sparse_record.data["policies"]["sparse"]
    print(
        f"sparse check: records 1000; answers 5 vs 0; effective sizes {sparse_card['n_eff']:.0f}, {sparse_card['baseline_n_eff']:.0f}; improvement {sparse_card['improvement']:.6f}; interval ["
        + ", ".join(f"{v:.6f}" for v in sparse_card["interval"])
        + f"]; exact p {sparse_card['exact']['p_value']:.6f}; decision {sparse_card['decision']}"
    )
    zero_tables = {
        name: support_table(from_offers(package.cap_zero.log.rows), policy)
        for name, policy in {**candidates, "fixed": fixed}.items()
    }
    print(
        "cap-zero judged: "
        + ", ".join(f"{name} {supported(table)}" for name, table in zero_tables.items())
    )
# endregion book:fnd.practical-rl.keys-01


def write_data(coverage, package, candidates, fixed, record, truth_result, directory=DATA):
    import json

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "practical_rl_coverage.json": coverage,
        "practical_rl_support.json": support_figure_data(package, candidates, fixed, record),
        "practical_rl_reports_05.json": exercise_reports(),
        "practical_rl_evaluation_record.json": record.data,
        "practical_rl_truth_check.json": truth_result,
    }
    for name, payload in artifacts.items():
        (directory / name).write_text(
            json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-data", action="store_true")
    args = parser.parse_args()
    if args.write_data:
        write_data(coverage, package, candidates, fixed, record, truth_result)
