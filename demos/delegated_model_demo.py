"""A bounded delegate, four inspections and one logged frozen-window decision.

Importing fits nothing. The script emits ten chapter outputs. Only inspect-01 is
a printed listing; it imports its setup explicitly and reads no frozen cases.
Raw validation log loss ranks configurations; each score gets chapter 5's map
fitted on validation before the final cost comparison at threshold 0.080.

The inspections rebuild a delivery from its settings and seed and hold the
delivered model to that rebuild: its predictions on every training and validation
row and on probe rows the delegate never held, as trained and after its map, and
every entry of its trial log (inside the commissioned space and budget, distinct,
each loss reproduced, the delivery the log's lowest, and for a seeded search the
seed's own configurations in their order). No comparison of predictions can speak
for every input, so the object a delegate hands over never meets a frozen row: the
inspector's rebuild is what is sealed and what August scores.

August is frozen before fitting. On first use its durable journal replays the
actual stored chapter 6 record, including chapter 5's read. An existing journal
is resumed without replay. BOOK_GOLDEN_DIRECTORY selects the journal directory;
capture runs in a fresh directory, while normal repeated runs append real reads.
--write-fixtures writes the constructed reports. --exercise-july runs the
unprinted 40-configuration reference solution on July (practice evidence).
"""

# ruff: noqa: E402
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == "__main__":
    for pool in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[pool] = "1"

import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import timedelta
from functools import lru_cache
from statistics import NormalDist
from typing import Callable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from demos.delegated_model_pipeline import GroupRisk
from demos.supervised_learning_demo import (
    GO_LIVE,
    arrays,
    boosted,
    calibration_error,
    frozen,
    gate,
    generate_history,
    log_loss,
    planted_risk,
    platt,
    realized_cost,
    reliability,
    risk,
    windows,
)
from refund_agent.data.history import FEATURE_COLUMNS
from refund_agent.selection import Commission, SelectionRecord, decide, fingerprint

OUTPUTS = Path(__file__).resolve().parents[1] / "book_outputs"
DATA = Path(__file__).resolve().parent / "data"
SPACE = {
    "logistic": {"C": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "boosted": {
        "max_iter": [50, 100, 200],
        "learning_rate": [0.025, 0.05, 0.1],
        "max_leaf_nodes": [3, 7, 15],
        "min_samples_leaf": [50, 100, 200],
        "l2_regularization": [0.0, 1.0, 10.0],
    },
}
BASELINE = "boosted trees with map"
# The commission's `delegate_changes` lets one builder write a pipeline of its own.
AGENT = "constructed agent"
# A seeded search has no say in what it tries: the seed fixes its configurations.
SEARCH = "seeded search"
# A seeded rebuild agrees with what it rebuilds to rounding; a changed model does not.
TOLERANCE = 1e-9


def prepare(month="august"):
    """Freeze before fitting; return builders only training and validation rows."""
    paid = [r for r in generate_history(seed=7) if r.paid]
    window = windows(paid)
    rows = window.pop("test" if month == "august" else "open")
    golden = frozen(rows, f"history-{month}")
    return {name: window[name] for name in ("training", "validation")}, golden


def commission_for(window, budget=20, seed=7):
    spans = {}
    for name, rows in window.items():
        spans[name] = {
            "first": rows[0].at.date().isoformat(),
            "last": rows[-1].at.date().isoformat(),
            "requests": len(rows),
            "row_sha256": fingerprint([r.model_dump(mode="json") for r in rows]),
        }
    return Commission("outcome-1", FEATURE_COLUMNS, spans, SPACE, budget=budget, seed=seed)


def make_model(settings, seed=7):
    params = {k: v for k, v in settings.items() if k != "family"}
    if settings["family"] == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, **params))
    return HistGradientBoostingClassifier(early_stopping=False, random_state=seed, **params)


def configurations(budget, seed=7):
    """Draw distinct configurations from the declared conditional space."""
    available = sum(
        math.prod(len(values) for values in params.values()) for params in SPACE.values()
    )
    if not 1 <= budget <= available:
        raise ValueError("budget must fit the finite declared space")
    rng, rows = np.random.default_rng(seed), []
    while len(rows) < budget:
        family = str(rng.choice(list(SPACE)))
        setting = {"family": family}
        setting.update({key: rng.choice(values).item() for key, values in SPACE[family].items()})
        if setting not in rows:
            rows.append(setting)
    return rows


@dataclass
class Delivery:
    name: str
    builder: str
    settings: dict
    seed: int
    commission_sha256: str
    reported_loss: float
    trials: list[dict]
    statistics: dict
    model: object
    mapping: Callable[[np.ndarray], np.ndarray]

    def predict(self, rows, mapped=True):
        p = (
            self.model.predict(rows)
            if isinstance(self.model, GroupRisk)
            else risk(self.model, arrays(rows)[0])
        )
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return self.mapping(p) if mapped else p

    def report(self):
        return {key: value for key, value in vars(self).items() if key not in ("model", "mapping")}


def fit_delivery(name, builder, settings, window, commission, trials=None):
    """Only development rows enter this function; the calibration fit is declared."""
    training, validation = window["training"], window["validation"]
    y = arrays(validation)[1]
    if settings["family"] == "group":
        model = GroupRisk(clean=settings["clean"], C=settings["C"], seed=commission.seed)
        model.fit(training, validation)
        p = model.predict(validation)
        statistics = model.provenance()
    else:
        model = make_model(settings, commission.seed).fit(*arrays(training)[:2])
        p = risk(model, arrays(validation)[0])
        statistics = {
            "inputs": list(FEATURE_COLUMNS),
            "fit_rows": len(training),
            "row_sha256": fingerprint(tuple(r.request_id for r in training)),
            "own_label_left_out": True,
        }
    mapping, slope, intercept = platt(np.clip(p, 1e-12, 1 - 1e-12), y)
    statistics["calibration"] = {
        "window": "validation",
        "slope": slope,
        "intercept": intercept,
        "row_sha256": commission.windows["validation"]["row_sha256"],
    }
    loss = log_loss(y, np.clip(p, 1e-12, 1 - 1e-12))
    own_trial = {
        "entry": 1,
        "settings": settings,
        "seed": commission.seed,
        "commission_sha256": commission.sha256,
        "loss": loss,
    }
    return Delivery(
        name,
        builder,
        settings,
        commission.seed,
        commission.sha256,
        loss,
        trials if trials is not None else [own_trial],
        statistics,
        model,
        mapping,
    )


def random_search(window, commission, events, budget=None):
    """Fit and log each drawn configuration; the lowest validation loss is the delivery."""
    trials, deliveries = [], []
    budget = commission.budget if budget is None else budget
    for entry, settings in enumerate(configurations(budget, commission.seed), 1):
        events.append({"event": "trial", "entry": entry, "commission": commission.sha256})
        delivery = fit_delivery(f"trial {entry:02}", SEARCH, settings, window, commission)
        trials.append({**delivery.trials[0], "entry": entry})
        deliveries.append(delivery)
    winner = min(deliveries, key=lambda d: d.reported_loss)
    winner.name, winner.trials = "search winner", trials
    return deliveries, trials, winner


@dataclass(frozen=True)
class Inspection:
    results: tuple[bool, bool, bool, bool]
    reproduced_loss: float
    honest_loss: float
    findings: tuple[str, ...]

    @property
    def passed(self):
        return all(self.results)


def close(a, b):
    """True when two numbers or two arrays agree to rounding; nan and non-numbers never do."""
    try:
        return bool(np.allclose(a, b, rtol=0, atol=TOLERANCE))
    except (TypeError, ValueError):
        return False


def allowed(settings, builder, space):
    """True if the commission lets this builder try these settings.

    A family the commission declares is held to its declared values, whoever built it.
    Any other family is a pipeline of the delegate's own, which the commission allows
    the agent and nobody else; the allowlist and fitted-statistic checks govern it.
    """
    declared = space.get(settings.get("family"))
    if declared is None:
        return builder == AGENT
    return set(settings) == {"family", *declared} and all(
        settings[key] in values for key, values in declared.items()
    )


def rebuild(delivery, window, commission):
    """The inspector's own fit of what a delivery says it is, from its settings and seed."""
    return fit_delivery(delivery.name, delivery.builder, delivery.settings, window, commission)


def probes(window):
    """Requests no delegate held: each validation request renamed, a cent dearer, two months on."""
    return [
        r.model_copy(
            update={
                "request_id": f"probe-{r.request_id}",
                "amount_cents": r.amount_cents + 1,
                "at": r.at + timedelta(days=61),
                "label_final_at": r.label_final_at + timedelta(days=61),
            }
        )
        for r in window["validation"]
    ]


def reproduction_faults(delivery, rebuilt, window, commission):
    """Every way the delivery departs from its seeded rebuild; empty when it reproduces.

    The delivered model is asked for every row its delegate held and for probe rows it
    never held. A model that answers for its rebuild there can still differ elsewhere,
    which is why `build` seals the rebuild and `evaluate` scores it.
    """
    faults = []
    if delivery.seed != commission.seed or delivery.commission_sha256 != commission.sha256:
        faults.append("cites another seed or commission")
    if not close(delivery.reported_loss, rebuilt.reported_loss):
        faults.append("a rebuild from the settings and seed gives another loss")
    rows = window["training"] + window["validation"] + probes(window)
    if not all(
        close(delivery.predict(rows, mapped), rebuilt.predict(rows, mapped))
        for mapped in (False, True)
    ):
        faults.append("the delivered model predicts differently from its rebuild")
    return faults


def trial_faults(delivery, rebuilt, window, commission):
    """Every way the trial log departs from the commission; empty when it is in order."""
    trials, faults, losses = delivery.trials, [], {}
    if not 0 < len(trials) <= commission.budget:
        faults.append(f"{len(trials)} entries under a budget of {commission.budget}")
    if any(
        t["entry"] != n
        or t["commission_sha256"] != commission.sha256
        or t["seed"] != commission.seed
        for n, t in enumerate(trials, 1)
    ):
        faults.append("entries misnumbered, or under another commission or seed")
    inside = [allowed(t["settings"], delivery.builder, commission.space) for t in trials]
    if not all(inside):
        faults.append("settings outside the commissioned space")
    keys = [json.dumps(t["settings"], sort_keys=True) for t in trials]
    if len(set(keys)) < len(keys):
        faults.append("a configuration logged more than once")
    if delivery.builder == SEARCH:
        # Reordered, renumbered or hand-picked entries are not what the seed draws.
        try:
            drawn = configurations(len(trials), commission.seed)
        except ValueError:
            drawn = None
        if [t["settings"] for t in trials] != drawn:
            faults.append("not the seed's configurations in their order")

    def refit(settings, key):
        # One fit for each distinct configuration; the delivery's own was rebuilt already.
        if settings == delivery.settings:
            return rebuilt.reported_loss
        if key not in losses:
            try:
                trial = fit_delivery("trial", delivery.builder, settings, window, commission)
                losses[key] = trial.reported_loss
            except (KeyError, TypeError, ValueError):
                losses[key] = math.nan  # an entry that cannot be rebuilt reproduces nothing
        return losses[key]

    if not all(
        close(t["loss"], refit(t["settings"], key))
        for ok, t, key in zip(inside, trials, keys)
        if ok
    ):
        faults.append("a logged loss does not reproduce")
    best = min(trials, key=lambda t: t["loss"], default=None)
    if (
        best is None
        or best["settings"] != delivery.settings
        or not close(best["loss"], delivery.reported_loss)
    ):
        faults.append("the delivery is not the log's lowest loss")
    return faults


def inspect_delivery(delivery, window, commission, rebuilt=None):
    """Run the four checks on rebuilds from the rows the delegate held; no frozen row.

    The third check holds the delivered model itself to its rebuild, so a model changed
    after its report was written fails it. The fourth holds every logged trial to the
    commission. Neither can speak for every input, and neither has to: what goes on to
    the seal and the frozen read is the rebuild.
    """
    if rebuilt is None:
        rebuilt = rebuild(delivery, window, commission)
    inputs_ok = set(delivery.statistics["inputs"]) <= set(commission.inputs)
    own_left_out = delivery.statistics["own_label_left_out"]
    training_sha = fingerprint(tuple(r.request_id for r in window["training"]))
    statistics_ok = (
        delivery.statistics["row_sha256"] == training_sha
        and own_left_out
        and delivery.statistics == rebuilt.statistics
    )
    reproduction = reproduction_faults(delivery, rebuilt, window, commission)
    trial_log = trial_faults(delivery, rebuilt, window, commission)
    honest_loss = rebuilt.reported_loss
    if delivery.settings["family"] == "group":
        honest = fit_delivery(
            "honest refit",
            delivery.builder,
            {**delivery.settings, "clean": True},
            window,
            commission,
        )
        honest_loss = honest.reported_loss
    results = inputs_ok, statistics_ok, not reproduction, not trial_log
    messages = (
        "label-built input outside the decision-time allowlist",
        "statistic includes validation rows or each training row's own outcome",
        "not reproduced: " + "; ".join(reproduction),
        "trial log: " + "; ".join(trial_log),
    )
    return Inspection(
        results,
        rebuilt.reported_loss,
        honest_loss,
        tuple(msg for ok, msg in zip(results, messages) if not ok),
    )


@dataclass
class Built:
    window: dict
    commission: Commission
    baseline: Delivery
    candidates: dict
    searched: list[Delivery]
    trials: list[dict]
    checks: dict
    events: list[dict]
    rebuilds: dict  # the inspector's own fit of each delivery, by name
    receipt: str = ""


def seal(built):
    """Fingerprint what was delivered, what was found, and the rebuilds that will be scored."""
    return fingerprint(
        {
            "commission": built.commission.sha256,
            "deliveries": [d.report() for d in (built.baseline, *built.candidates.values())],
            "predictions": [
                built.rebuilds[d.name].predict(built.window["validation"]).tolist()
                for d in (built.baseline, *built.candidates.values())
            ],
            "checks": {k: asdict(v) for k, v in built.checks.items()},
        }
    )


def build(window=None, budget=20, seed=7, write_fixtures=False):
    if window is None:
        window, _ = prepare()
    commission = commission_for(window, budget, seed)
    events = [{"event": "commission", "sha256": commission.sha256}]
    if not all(r.at < GO_LIVE for rows in window.values() for r in rows):
        raise ValueError("builders receive development rows only")
    baseline = fit_delivery(
        BASELINE,
        "person",
        {"family": "boosted", **{k: boosted().get_params()[k] for k in SPACE["boosted"]}},
        window,
        commission,
    )
    logistic = fit_delivery(
        "logistic regression", "person", {"family": "logistic", "C": 1.0}, window, commission
    )
    searched, trials, winner = random_search(window, commission, events)
    agent = fit_delivery(
        "agent delivery",
        "constructed agent",
        {"family": "group", "clean": False, "C": 1.0},
        window,
        commission,
    )
    clean = fit_delivery(
        "clean agent",
        "constructed agent",
        {"family": "group", "clean": True, "C": 1.0},
        window,
        commission,
    )
    if budget == 20:
        for delivery, name in ((agent, "agent_delivery"), (clean, "agent_clean")):
            path = DATA / f"delegated_model_{name}.json"
            if write_fixtures:
                path.write_text(json.dumps(delivery.report(), indent=2, sort_keys=True) + "\n")
            report = json.loads(path.read_text())
            # Run the actual pipeline named in the fixture, then inspect its claims.
            delivery = fit_delivery(
                report["name"], report["builder"], report["settings"], window, commission
            )
            for key, value in report.items():
                setattr(delivery, key, value)
            if name == "agent_delivery":
                agent = delivery
            else:
                clean = delivery
        if write_fixtures:
            # The exercise has one fault: this commission's search, run on to sixty
            # distinct configurations, each logged faithfully, under a budget of twenty.
            *_, sixty = random_search(window, commission, [], budget=60)
            sixty.name = "delivery 05"
            (DATA / "delegated_model_delivery_05.json").write_text(
                json.dumps(sixty.report(), indent=2, sort_keys=True) + "\n"
            )
    candidates = {d.name: d for d in (logistic, winner, agent, clean)}
    delivered = (baseline, *candidates.values())
    # From here on a delivered object is only compared with the inspector's rebuild.
    rebuilds = {d.name: rebuild(d, window, commission) for d in delivered}
    checks = {d.name: inspect_delivery(d, window, commission, rebuilds[d.name]) for d in delivered}
    events.append(
        {"event": "inspections", "sha256": fingerprint({k: asdict(v) for k, v in checks.items()})}
    )
    built = Built(
        window, commission, baseline, candidates, searched, trials, checks, events, rebuilds
    )
    built.receipt = seal(built)
    events.append({"event": "sealed", "sha256": built.receipt})
    return built


def binary_losses(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def search_summary(built, draws=200):
    rows, y = built.window["validation"], arrays(built.window["validation"])[1]
    losses = np.array([binary_losses(y, d.predict(rows, mapped=False)) for d in built.searched])
    baseline = binary_losses(y, built.baseline.predict(rows, mapped=False))
    index = np.random.default_rng(built.commission.seed).integers(0, len(y), (draws, len(y)))
    leads = baseline - losses
    resampled = leads[:, index].mean(axis=2)
    errors = resampled.std(axis=1, ddof=1)
    best = np.argsort(losses.mean(axis=1), kind="stable")[:5]
    wins = np.bincount(np.argmax(resampled[best], axis=0), minlength=5)
    # The largest error is the box's s; the winner's own says how far its lead is from zero.
    return best, wins, float(errors.max()), float(leads.mean(axis=1).max()), float(errors[best[0]])


def luck_allowance(s, candidates):
    if s < 0 or candidates < 1:
        raise ValueError("nonnegative standard error and positive count required")
    return s * math.sqrt(2 * math.log(candidates))


def selection_optimism(counts=(5, 20, 100), draws=50000, seed=7):
    """Simulate equal, independent standard-normal leads; the bound needs no independence."""
    rng = np.random.default_rng(seed)
    return [
        {
            "candidates": n,
            "mean_max": float(rng.standard_normal((draws, n)).max(axis=1).mean()),
            "bound": luck_allowance(1, n),
        }
        for n in counts
    ]


@lru_cache(maxsize=None)
def fresh_truth(seed=71):
    """One generated June pool per seed and process, shared by teaching regions."""
    pool = windows([r for r in generate_history(seed=seed, rows=72000) if r.paid])["validation"]
    mu, amount = np.array([planted_risk(r) for r in pool]), arrays(pool)[2]
    return pool, mu, amount


def truth(built, seed=71):
    """Fresh June requests; integrate outcomes using planted risk, never fit on them."""
    pool, mu, amount = fresh_truth(seed)
    validation = built.window["validation"]
    y = arrays(validation)[1]
    base_loss = binary_losses(mu, built.baseline.predict(pool, mapped=False)).mean()
    base_val = built.baseline.reported_loss
    base_cost = np.where(gate(built.baseline.predict(pool)), mu * amount, 4.0)
    rows = []
    for d in (
        *built.searched,
        built.baseline,
        built.candidates["logistic regression"],
        built.candidates["agent delivery"],
    ):
        p = d.predict(pool, mapped=False)
        cost = np.where(gate(d.predict(pool)), mu * amount, 4.0)
        rows.append(
            {
                "name": d.name,
                "builder": d.builder,
                "settings": d.settings,
                "measured_lead": float(base_val - log_loss(y, d.predict(validation, mapped=False))),
                "true_lead": float(base_loss - binary_losses(mu, p).mean()),
                "true_cost": float(cost.mean()),
                "true_saving": float((base_cost - cost).mean()),
            }
        )
    searched = sorted(rows[: len(built.searched)], key=lambda row: row["true_lead"], reverse=True)
    return {
        "seed": seed,
        "requests": len(pool),
        "window": "fresh June, generator expectation conditional on requests",
        "models": rows,
        "winner_true_rank": next(
            i for i, r in enumerate(searched, 1) if r["name"] == "search winner"
        ),
    }


def resume_history(golden, directory):
    text = (OUTPUTS / "_fnd.semi-self-supervised.holdout-01.txt").read_text()
    block = text.split("name: history-september")[0]
    sha = re.search(r"^sha256: ([a-f0-9]+)$", block, re.M)
    if sha is None or not golden.manifest["sha256"].startswith(sha[1]):
        raise ValueError("chapter 6's stored August fingerprint differs")
    prior = [
        {"entry": int(n), "set": name, "reader": reader, "purpose": purpose}
        for n, name, reader, purpose in re.findall(
            r"^  (\d+): ([^,]+), ([^,]+), (.+)$", block, re.M
        )
    ]
    if [e["entry"] for e in prior] != [1, 2]:
        raise ValueError("chapter 6 must hold both earlier August consultations")
    golden.persist(directory, prior_log=prior)


def interval(values, alpha=0.05, draws=2000, seed=7):
    values = np.asarray(values)
    index = np.random.default_rng(seed).integers(0, len(values), (draws, len(values)))
    low, high = np.quantile(values[index].mean(axis=1), [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(low), float(high)


def evaluate(built, golden, directory, replay=True):
    """Verify the seal before the sole consultation; no fits or checks follow it.

    Every frozen row is shown to the inspector's rebuilds alone. A delivered object,
    whatever it would answer on rows nobody could compare, is never asked.
    """
    if not built.commission.intact() or built.receipt != seal(built):
        raise ValueError("commission, deliveries or inspections changed after sealing")
    if replay:
        resume_history(golden, directory)
    else:
        golden.persist(directory)
    purpose = f"fixed commission {built.commission.sha256[:16]}; deliveries and checks {built.receipt[:16]}; one paired cost comparison"
    read = golden.consult("selection record", purpose)
    built.events.append(
        {"event": "consult", "entry": golden.log[-1]["entry"], "receipt": built.receipt}
    )
    rows = [r for r, _ in read]
    y = np.array([d.found_false_30d for _, d in read])
    amount = np.array([r.amount_cents for r in rows]) / 100
    baseline_cost = realized_cost(y, amount, gate(built.rebuilds[BASELINE].predict(rows)))
    evidence, differences = [], {}
    for name in (BASELINE, *built.commission.challengers):
        judged = built.rebuilds[name]
        p = judged.predict(rows)
        pay = gate(p, threshold=built.commission.threshold)
        cost = realized_cost(y, amount, pay)
        delta = cost - baseline_cost
        differences[name] = delta
        evidence.append(
            {
                "name": name,
                "validation_loss": judged.reported_loss,
                # The commission's measure on both windows: the score as trained, no map.
                "frozen_loss": log_loss(y, judged.predict(rows, mapped=False)),
                "cost": interval(cost),
                "difference": interval(delta, built.commission.alpha),
                "split_difference": interval(
                    delta, built.commission.alpha / len(built.commission.challengers)
                ),
                "calibration_error": calibration_error(reliability(y, p)),
                "referred": int((~pay).sum()),
                "wrong_paid": int(y[pay].sum()),
            }
        )
    decisions = {
        r["name"]: decide(
            built.checks[r["name"]].passed, r["difference"][0], tuple(r["difference"][1:])
        )
        for r in evidence[1:]
    }
    accepted = [r for r in evidence[1:] if decisions[r["name"]] == "accepted"]
    selected = (
        min(accepted, key=lambda r: (r["cost"][0], r["name"]))["name"] if accepted else BASELINE
    )
    record = SelectionRecord(
        asdict(built.commission),
        built.receipt,
        tuple(
            d.report()
            for d in (built.baseline, *(built.candidates[n] for n in built.commission.challengers))
        ),
        {k: asdict(v) for k, v in built.checks.items()},
        dict(golden.manifest),
        tuple(golden.log),
        tuple(evidence),
        decisions,
        selected,
        f"{golden.manifest['name']} spent for these candidates",
    )
    return record, differences


CHECKS = ("inputs", "statistics", "reproduced", "trial log")


def print_inspections(built):
    from demos.output_wrap import print_wrapped

    for name in (BASELINE, "logistic regression", "search winner", "agent delivery", "clean agent"):
        check = built.checks[name]
        delivery = built.baseline if name == BASELINE else built.candidates[name]
        print_wrapped(name + ":")
        print_wrapped("  " + "; ".join(
            f"{head}: {'pass' if ok else 'FAIL'}" for head, ok in zip(CHECKS, check.results)
        ))
        print_wrapped(
            f"  reported {delivery.reported_loss:.4f}; honest refit {check.honest_loss:.4f}"
        )


def table_number(value):
    # Four decimals for a printed table, where a negative number takes a true minus sign.
    return f"{value:.4f}".replace("-", "\N{MINUS SIGN}")


def frozen_table(evidence):
    # One column per candidate, so that a measure reads across a row and no cell wraps.
    def row(label, cell, challengers_only=False):
        # The baseline has no difference from itself, so its cells in those rows stay empty.
        cells = ["" if challengers_only and r["name"] == BASELINE else cell(r) for r in evidence]
        return f"| {label} | " + " | ".join(cells) + " |"

    def between(triple):
        return f"{table_number(triple[1])} to {table_number(triple[2])}"

    return [
        row("Measure", lambda r: r["name"].capitalize()),
        "|---" * (len(evidence) + 1) + "|",
        row("Validation log loss", lambda r: table_number(r["validation_loss"])),
        row("August log loss", lambda r: table_number(r["frozen_loss"])),
        row("Cost per request", lambda r: table_number(r["cost"][0])),
        row("Cost, 95% interval", lambda r: between(r["cost"])),
        row("Difference from baseline", lambda r: table_number(r["difference"][0]), True),
        row("Difference, 95% interval", lambda r: between(r["difference"]), True),
        row("Difference, split interval", lambda r: between(r["split_difference"]), True),
        row("Calibration error", lambda r: table_number(r["calibration_error"])),
        row("Referred", lambda r: str(r["referred"])),
        row("Wrong payments paid", lambda r: str(r["wrong_paid"])),
    ]


if __name__ == "__main__":
    window, august = prepare("july" if "--exercise-july" in sys.argv else "august")
    built = build(
        window,
        budget=40 if "--exercise-july" in sys.argv else 20,
        write_fixtures="--write-fixtures" in sys.argv,
    )
    if "--exercise-july" in sys.argv:
        record, _ = evaluate(
            built,
            august,
            Path(os.environ.get("BOOK_GOLDEN_DIRECTORY", "golden-sets")),
            replay=False,
        )
        print(
            json.dumps({"events": built.events, "record": asdict(record)}, indent=2, sort_keys=True)
        )
        raise SystemExit(0)


# region book:fnd.automl-and-selection.commission-01 output
if __name__ == "__main__":
    print(json.dumps(asdict(built.commission), indent=2, sort_keys=True))
# endregion book:fnd.automl-and-selection.commission-01


# region book:fnd.automl-and-selection.search-01 output
if __name__ == "__main__":
    best, wins, s, lead, own = search_summary(built)
    print(
        f"{len(built.trials)} distinct configurations, seed {built.commission.seed}; 200 validation resamples"
    )
    print(f"baseline raw validation loss: {built.baseline.reported_loss:.4f}")
    print("best five                       loss    lead  resample wins (among these five)")
    for index, count in zip(best, wins):
        d = built.searched[index]
        print(
            f"{d.name:16} {d.settings['family']:8} {d.reported_loss:.4f}  {built.baseline.reported_loss - d.reported_loss:.4f}  {count}/200"
        )
    print(f"winner lead {lead:.6f}, standard error {own:.6f}; largest standard error {s:.6f}")
# endregion book:fnd.automl-and-selection.search-01


# region book:fnd.automl-and-selection.optimism-01 output
if __name__ == "__main__":
    optimism = {
        "family_chance": 1 - 0.95**5,
        "union_bound": 5 * 0.05,
        "bonferroni": 0.05 / 5,
        "simulation_draws": 50000,
        "seed": 7,
        "normal_maxima": selection_optimism(),
        "standard_error": s,
        "allowance": luck_allowance(s, 20),
        "winner_lead": lead,
        "curve": selection_optimism(
            sorted(set([1, 5, 20, 100, 1000] + [int(n) for n in np.geomspace(2, 1000, 25)])),
            draws=10000,
        ),
    }
    print(json.dumps(optimism, indent=2, sort_keys=True))
# endregion book:fnd.automl-and-selection.optimism-01


# region book:fnd.automl-and-selection.truth-01 output
if __name__ == "__main__":
    true_results = truth(built)
    print(json.dumps(true_results, indent=2, sort_keys=True))
# endregion book:fnd.automl-and-selection.truth-01


# region book:fnd.automl-and-selection.inspect-01 output
from demos import delegated_model_demo as delegated

if __name__ == "__main__":
    # No earlier setup is needed; builders receive development rows only.
    inspected = delegated.build()
    for delivery in (inspected.baseline, *inspected.candidates.values()):
        inspected.checks[delivery.name] = delegated.inspect_delivery(
            delivery, inspected.window, inspected.commission
        )
    delegated.print_inspections(inspected)
# endregion book:fnd.automl-and-selection.inspect-01


# region book:fnd.automl-and-selection.frozen-01 output
from demos.output_wrap import print_wrapped

if __name__ == "__main__":
    record, differences = evaluate(
        built, august, Path(os.environ.get("BOOK_GOLDEN_DIRECTORY", "golden-sets"))
    )
    for key, value in august.manifest.items():
        print_wrapped(f"{key}: {value[:16] if key == 'sha256' else value}")
    print_wrapped("access log, every entry:")
    for entry in august.log:
        print_wrapped(
            f"  {entry['entry']}: {entry['set']}, {entry['reader']}, {entry['purpose']}"
        )
    print_wrapped(f"fingerprint intact: {august.intact()}; consultations: {len(august.log)}")
# endregion book:fnd.automl-and-selection.frozen-01


# region book:fnd.automl-and-selection.frozen-02 output
from demos.output_wrap import print_wrapped

if __name__ == "__main__":
    table = frozen_table(record.evidence)
    print_wrapped("Models: " + "; ".join(
        row["name"].capitalize() for row in record.evidence
    ))
    print_wrapped()
    print_wrapped("| Measure | Baseline | Logistic | Search | Agent |")
    print_wrapped("\n".join(table[1:]))
# endregion book:fnd.automl-and-selection.frozen-02


# region book:fnd.automl-and-selection.power-01 output
if __name__ == "__main__":
    # What one frozen month can and cannot settle, by the normal approximation.
    spread = float(differences["search winner"].std(ddof=1))
    requests = len(differences["search winner"])
    error, z = spread / math.sqrt(requests), NormalDist().inv_cdf(1 - built.commission.alpha / 2)
    print(f"August, search winner less baseline: paired cost difference, sd ${spread:.6f}")
    print(f"95% half-width at {requests} requests: ${z * error:.4f}")
    for saving in (0.02, 0.01):
        needed = math.ceil((z * spread / saving) ** 2)
        chance = NormalDist().cdf(saving / error - z)
        print(
            f"a true saving of ${saving:.2f} at that spread: half-width below it from {needed:,} requests; {requests} requests clear the rule with chance {chance:.3f}"
        )
    named, side = len(built.commission.challengers), built.commission.alpha / 2
    print(
        f"If the bootstrap upper bounds attain nominal one-sided coverage: {named} challengers no better than the baseline, each cleared with chance at most {side:.3f}: at most {named * side:.3f} that one is; with the split intervals, at most {side:.3f}"
    )
    print("These percentile-bootstrap intervals do not establish those coverage guarantees.")
    winner_truth = next(r for r in true_results["models"] if r["name"] == "search winner")
    print(
        f"generator, fresh draw of {true_results['requests']:,} requests: the search winner costs ${-winner_truth['true_saving']:.6f} more per request than the baseline"
    )
    print("A half-width is not a test's power; each chance is of the rule's one-sided acceptance.")
# endregion book:fnd.automl-and-selection.power-01


# region book:fnd.automl-and-selection.record-01 output
from demos.output_wrap import print_wrapped

if __name__ == "__main__":
    print_wrapped("| Selection record field | Value |\n|---|---|")
    c = built.commission
    print_wrapped(
        f"| Commission | {c.label_policy}; {len(c.inputs)} allowed inputs; {c.measure} |"
    )
    print_wrapped(
        f"| Windows | training {c.windows['training']['requests']}, {c.windows['training']['first']} to {c.windows['training']['last']}; validation {c.windows['validation']['requests']}, {c.windows['validation']['first']} to {c.windows['validation']['last']} |"
    )
    print_wrapped(f"| Delegation | {c.delegate_changes}; budget {c.budget}, seed {c.seed} |")
    print_wrapped(f"| Rule fixed in the commission | {c.acceptance} |")
    for d in record.deliveries:
        trials = f"{len(d['trials'])} trial" + ("" if len(d["trials"]) == 1 else "s")
        print_wrapped(
            f"| Delivery: {d['name']} | {d['builder']}; {trials}; raw loss {d['reported_loss']:.4f} |"
        )
    failed = "; ".join(
        f"{name} fails {' and '.join(label for label, ok in zip(CHECKS, check['results']) if not ok)}"
        for name, check in record.checks.items()
        if not all(check["results"])
    )
    print_wrapped(
        f"| Checks and fitted deliveries sealed | {failed or 'no check fails'}; every other check passes |"
    )
    print_wrapped(
        f"| Frozen evidence | {record.frozen['name']}, {record.frozen['requests']} requests; threshold {c.threshold:.3f}; review ${c.review_cost:.2f}; 95% and split intervals in the frozen table |"
    )
    print_wrapped(
        f"| Consultation history | {len(record.log)} entries; readers: {', '.join(e['reader'] for e in record.log)} |"
    )
    print_wrapped(
        f"| Decisions | {'; '.join(f'{name}: {result}' for name, result in record.decisions.items())} |"
    )
    print_wrapped(f"| Selected baseline | {record.selected} |")
    print_wrapped(f"| Spent and granted | {record.spent}; {record.grants} |")
    print_wrapped(f"| Depends on | {', '.join(record.dependencies)} |")
    print_wrapped(f"| Signer | {record.signer} |")
# endregion book:fnd.automl-and-selection.record-01


# region book:fnd.automl-and-selection.record-full-01 output
if __name__ == "__main__":
    print(json.dumps(asdict(record), indent=2, sort_keys=True))
# endregion book:fnd.automl-and-selection.record-full-01


# region book:fnd.automl-and-selection.keys-01 output
if __name__ == "__main__":

    def by_luck(lead, error, candidates):
        """The chance that at least one of several equal candidates shows such a lead."""
        tail = 1 - NormalDist().cdf(lead / error)
        return 1 - (1 - tail) ** candidates, candidates * tail

    sd = 0.62 / math.sqrt(4000)
    z95, zsplit = NormalDist().inv_cdf(0.975), NormalDist().inv_cdf(0.9875)
    print(f"Exercise 2: independent {1 - 0.95**5:.4f}; bound {5 * 0.05:.4f}")
    print(
        f"Exercise 3: bound {luck_allowance(0.004, 50):.4f}; mean of 50 independent equals {0.004 * 2.249:.4f}; the lead of 0.006 is under both"
    )
    free, linked = by_luck(0.012, 0.004, 50)
    print(
        f"Exercise 3: a lead of 0.012 is over both, at {0.012 / 0.004:.1f} standard errors; by luck in one of 50 equals with chance {free:.4f} if independent, at most {linked:.4f}"
    )
    plain = (-0.02 - z95 * sd, -0.02 + z95 * sd)
    split = (-0.02 - zsplit * sd, -0.02 + zsplit * sd)
    print(
        f"Exercise 4: 95% [{plain[0]:.4f}, {plain[1]:.4f}], {decide(True, -0.02, plain)}; split [{split[0]:.4f}, {split[1]:.4f}], {decide(True, -0.02, split)}; bound for 14 candidates {luck_allowance(sd, 14):.4f}"
    )
    free, linked = by_luck(0.02, sd, 14)
    print(
        f"Exercise 4: a lead of 0.020 is {0.02 / sd:.2f} standard errors; by luck in one of 14 equals with chance {free:.4f} if independent, at most {linked:.4f}"
    )
    logged = json.loads((DATA / "delegated_model_delivery_05.json").read_text())["trials"]
    distinct = len({json.dumps(t["settings"], sort_keys=True) for t in logged})
    budget, base = built.commission.budget, built.baseline.reported_loss
    within, every = min(t["loss"] for t in logged[:budget]), min(t["loss"] for t in logged)
    print(
        f"Exercise 5: {len(logged)} logged trials, {distinct} distinct, under a budget of {budget}; lowest loss of the first {budget}: {within:.4f}, of all {len(logged)}: {every:.4f}; leads over the baseline's {base:.4f}: {base - within:.4f} and {base - every:.4f}"
    )
    print(
        f"Exercise 5: bound on the expected best lead, in standard errors: {luck_allowance(1, budget):.2f} at {budget} candidates, {luck_allowance(1, len(logged)):.2f} at {len(logged)}"
    )
# endregion book:fnd.automl-and-selection.keys-01
