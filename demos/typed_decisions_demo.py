"""Chapter 20: declared answers, proper scores and one sealed retest of fresh months.

Importing generates and fits nothing. The printed regions typed-01, scores-01 and
conformal-01 run in chapter order without intervening setup. Seed 7 supplies only
training and validation; seed 20 supplies a second July, August and September.
Both held-out months are frozen before thresholds, then consulted once per run.
BOOK_GOLDEN_DIRECTORY selects the durable journal; capture uses a fresh directory.
--write-fixtures writes the exercise report, figure data and offline risk record.
--exercise-july prints only the scripted stand-in's July diagnostics, without
consulting or freezing either held-out month. No product is measured here.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import numpy as np
from scipy.stats import beta

from demos.ml_data_evaluation_demo import wilson_interval
from demos.supervised_learning_demo import (
    GO_LIVE,
    QUEUES,
    ROUTER,
    arrays,
    boosted,
    calibration_error,
    expected_costs,
    frozen,
    generate_history,
    planted_risk,
    platt,
    reliability,
    risk,
    windows,
)
from refund_agent.model.history_baseline import feature_matrix
from refund_agent.selection import fingerprint
from refund_agent.typed import (
    PAY_EDGE,
    RISK_QUESTION,
    THIRD_EDGE,
    RiskSignalRecord,
    ScriptedVendorModel,
)

DATA = Path(__file__).with_name("data")
MODEL_VERSION = "chapter5-boosted-platt-seed7-v1"
HISTORY_SEED, HISTORY_ROWS, ALPHA = 20, 36000, .10
JULY = datetime(2026, 7, 1, tzinfo=timezone.utc)
SCRIPTED_RELIABILITY_LIMIT = .0005  # illustrative acceptance rule, set before July is read


def expected_score(mu, p, rule="brier"):
    """Expected penalty of one report; impossible log reports remain unbounded."""
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (mu, p)):
        raise ValueError("mu and p must be probabilities")
    if rule == "brier":
        return (p - mu) ** 2 + mu * (1 - mu)
    if rule == "absolute":
        return mu + p * (1 - 2 * mu)
    if rule == "log":
        if (p == 0 and mu > 0) or (p == 1 and mu < 1):
            return math.inf
        return -(mu * math.log(p) if mu else 0) - ((1 - mu) * math.log1p(-p) if mu < 1 else 0)
    raise ValueError("unknown scoring rule")


def smoothed_minimizer(mu, epsilon=.05):
    return epsilon + (1 - 2 * epsilon) * mu


def decomposition(y, p, bins=5):
    """Murphy's parts and the within-bin residual of a continuous forecast."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or not len(y):
        raise ValueError("one prediction per nonempty outcome vector required")
    if bins is None:
        groups = [np.flatnonzero(p == value) for value in np.unique(p)]
    else:
        if bins < 1:
            raise ValueError("positive bin count required")
        groups = [g for g in np.array_split(np.argsort(p, kind="stable"), bins) if len(g)]
    rate = float(y.mean())
    table = [dict(requests=len(g), prediction=float(p[g].mean()), observed=float(y[g].mean()))
             for g in groups]
    rel = sum(r["requests"] * (r["prediction"] - r["observed"]) ** 2 for r in table) / len(y)
    res = sum(r["requests"] * (r["observed"] - rate) ** 2 for r in table) / len(y)
    unc, score = rate * (1 - rate), float(np.mean((p - y) ** 2))
    return dict(brier=score, reliability=rel, resolution=res, uncertainty=unc,
                residual=score - (rel - res + unc), bins=table)


def conformal_threshold(scores, alpha=.1):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or not 0 < alpha < 1:
        raise ValueError("finite calibration scores and alpha strictly between zero and one required")
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    return rank, math.inf if rank > len(scores) else float(np.sort(scores)[rank - 1])


def nonconformity_scores(probabilities):
    """Sum the other answers' probabilities, sharing exact operations at ties."""
    p = np.asarray(probabilities, dtype=float)
    return np.stack([p[..., :c].sum(axis=-1) + p[..., c + 1:].sum(axis=-1)
                     for c in range(p.shape[-1])], axis=-1)


def prediction_set(probabilities, threshold):
    """Indices of answers whose nonconformity scores pass their thresholds."""
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or (p < 0).any() or abs(p.sum() - 1) > 1e-9:
        raise ValueError("a probability vector over the declared answers is required")
    return tuple(int(c) for c in np.flatnonzero(nonconformity_scores(p) <= threshold))


def coverage_interval(covered, n):
    if not 0 <= covered <= n or n < 1:
        raise ValueError("coverage needs a positive count and covered between zero and n")
    if n >= 50:
        return wilson_interval(covered, n)
    return (0.0 if covered == 0 else float(beta.ppf(.025, covered, n - covered + 1)),
            1.0 if covered == n else float(beta.ppf(.975, covered + 1, n - covered)))


def coverage_verdict(covered, n):
    low, high = coverage_interval(covered, n)
    if low >= .87:
        return "coverage shown"
    return "undercoverage shown" if high < .90 else "inconclusive"


def coverage_summary(p, y, threshold):
    p, y = np.asarray(p), np.asarray(y, dtype=int)
    sets = [prediction_set([1 - value, value], threshold) for value in p]
    covered = np.array([int(label) in answers for label, answers in zip(y, sets)])
    counts = {name: sets.count(answers) for name, answers in
              (("empty", ()), ("sound", (0,)), ("wrong", (1,)), ("both", (0, 1)))}
    result = dict(requests=len(y), covered=int(covered.sum()), rate=float(covered.mean()),
                  interval=list(coverage_interval(int(covered.sum()), len(y))),
                  verdict=coverage_verdict(int(covered.sum()), len(y)), sets=counts)
    for label, name in ((0, "sound"), (1, "wrong")):
        selected = y == label
        n, k = int(selected.sum()), int(covered[selected].sum())
        result[name] = dict(requests=n, covered=k, rate=k / n if n else None,
                            interval=list(coverage_interval(k, n)) if n else None)
    return result


def band_statistics(p, y, amount, mu=None):
    p, y, amount = (np.asarray(a, dtype=float) for a in (p, y, amount))
    if not len(p):
        return dict(requests=0)
    row: dict = dict(requests=len(p), prediction=float(p.mean()), observed=float(y.mean()),
               weighted_prediction=float(np.average(p, weights=amount)),
               weighted_observed=float(np.average(y, weights=amount)),
               predicted_loss=float(np.mean(amount * p)), realized_loss=float(np.mean(amount * y)))
    if np.isin(y, [0, 1]).all():
        row["interval"] = list(wilson_interval(int(y.sum()), len(y)))
    if mu is not None:
        row["planted_risk"] = float(np.mean(mu))
    return row


def band_report(p, y, amount, mu):
    assigned = np.array([band(value) for value in p])
    return {name: band_statistics(p[assigned == name], y[assigned == name],
                                 amount[assigned == name], mu[assigned == name])
            for name in ("pay", "review", "third")}


def router_report():
    # Two ten-ticket groups share one forecast: 7/0/3 and 9/1/0 true queues.
    # Top-label calibration holds over twenty, not separately in each group.
    groups = ([0] * 7 + [2] * 3, [0] * 9 + [1])
    labels = np.concatenate(groups)
    counts = [np.bincount(group, minlength=3).tolist() for group in groups]
    vector, corrected = [.8, .15, .05], (np.array(counts[0]) / len(groups[0])).tolist()
    costs, corrected_costs = expected_costs(vector, ROUTER), expected_costs(corrected, ROUTER)
    return dict(requests=len(labels), top_probability=max(vector),
                top_correct=int(np.sum(labels == np.argmax(vector))),
                escalation_prediction=vector[2], escalation_count=int(np.sum(labels == 2)),
                vector=vector, corrected_vector=corrected, groups=counts,
                costs=costs, corrected_costs=corrected_costs,
                stated_route=QUEUES[int(np.argmin(costs))],
                corrected_route=QUEUES[int(np.argmin(corrected_costs))])


@dataclass
class Scores:
    proposed: object
    weighted: object
    mapping: Callable
    base_rate: float
    validation: dict
    sha256: str

    def predict(self, rows):
        X = np.nan_to_num(feature_matrix(rows))
        return {"proposed": self.mapping(risk(self.proposed, X)),
                "class-weighted": risk(self.weighted, X),
                "base rate": np.full(len(rows), self.base_rate)}


def rebuild_scores():
    """Exactly chapter 5's two fits and its proposed map, without a frozen read."""
    window = windows([r for r in generate_history(seed=7) if r.paid])
    training, validation = window["training"], window["validation"]
    del window
    Xt, yt, _ = arrays(training)
    Xv, yv, _ = arrays(validation)
    proposed, weighted = boosted().fit(Xt, yt), boosted(class_weight="balanced").fit(Xt, yt)
    pv, pw = risk(proposed, Xv), risk(weighted, Xv)
    mapping, slope, intercept = platt(pv, yv)
    report = dict(requests=len(yv), wrong=int(yv.sum()), boosted_mean=float(pv.mean()),
                  boosted_error=calibration_error(reliability(yv, pv)),
                  weighted_mean=float(pw.mean()), weighted_error=calibration_error(reliability(yv, pw)),
                  proposed_mean=float(mapping(pv).mean()),
                  proposed_error=calibration_error(reliability(yv, mapping(pv))),
                  slope=slope, intercept=intercept)
    receipt = fingerprint(dict(version=MODEL_VERSION, settings=proposed.get_params(),
                               training=[r.request_id for r in training],
                               validation=[r.request_id for r in validation],
                               predictions=pv.tolist(), mapping=[slope, intercept]))
    return Scores(proposed, weighted, mapping, float(yt.mean()), report, receipt)


def redrawn_windows():
    months = {name: [] for name in ("july", "august", "september")}
    for r in generate_history(seed=HISTORY_SEED, rows=HISTORY_ROWS):
        if r.paid and r.amount_cents < 5000 and r.at >= JULY:
            months[{7: "july", 8: "august", 9: "september"}[r.at.month]].append(r)
    return months


def july_rows():
    """Exercise data: July only, before any held-out outcome can be read."""
    return [r for r in generate_history(seed=HISTORY_SEED, rows=HISTORY_ROWS)
            if r.paid and r.amount_cents < 5000 and JULY <= r.at < GO_LIVE]


def scripted_diagnostics(rows=None):
    if rows is None:
        rows = july_rows()
    model, y = ScriptedVendorModel(), arrays(rows)[1]
    p, negated = np.array(model.predict(rows)), np.array(model.predict(rows, negated=True))
    return dict(seed=HISTORY_SEED, requests=len(rows), wrong=int(y.sum()),
                model_version=model.version, decomposition=decomposition(y, p),
                reliability_limit=SCRIPTED_RELIABILITY_LIMIT,
                negation_violations=int(np.sum(np.abs(p + negated - 1) > 1e-9)),
                largest_negation_gap=float(np.max(np.abs(p + negated - 1))))


def referral_summary(p, marginal, conditional):
    """Band referrals and conformal alternatives, each request counted only once."""
    assigned = np.array([band(value) for value in p])
    band_refers = assigned != "pay"
    result = {}
    # All paid tier 1 requests are in-domain, so the amount refusal count is zero.
    result["bands"] = dict(review=int(np.sum(assigned == "review")),
                           third=int(np.sum(assigned == "third")), cannot_tell=0)
    for name, refers in (("bands", band_refers),
                         ("marginal", band_refers | np.array([
                             prediction_set([1 - value, value], marginal) != (0,) for value in p])),
                         ("class-conditional", band_refers | np.array([
                             prediction_set([1 - value, value], conditional) != (0,) for value in p]))):
        k, n = int(refers.sum()), len(p)
        result.setdefault(name, {}).update(requests=n, referred=k, share=k / n,
                                          interval=list(wilson_interval(k, n)))
    return result


def summarize_month(rows, scores, thresholds):
    X, y, amount = arrays(rows)
    del X
    forecasts = scores.predict(rows)
    mu = np.array([planted_risk(r) for r in rows])
    report: dict = dict(requests=len(y), wrong=int(y.sum()), rate=float(y.mean()),
                  rate_interval=list(wilson_interval(int(y.sum()), len(y))), scores={}, bands={})
    for name, p in forecasts.items():
        log_probability = np.empty_like(p, dtype=float)
        with np.errstate(divide="ignore"):
            np.log(p, out=log_probability, where=y == 1)
            np.log1p(-p, out=log_probability, where=y == 0)
        report["scores"][name] = dict(brier=float(np.mean((p - y) ** 2)),
            log=float(-np.mean(log_probability)),
            maximum=float(p.max()), truth_error=float(np.mean((p - mu) ** 2)),
            intrinsic=float(np.mean(mu * (1 - mu))))
        if name != "base rate":
            report["bands"][name] = band_report(p, y, amount, mu)
    p = forecasts["proposed"]
    report["decomposition"] = decomposition(y, p)
    report["marginal"] = coverage_summary(p, y, thresholds["marginal"])
    conditional = np.array([thresholds["sound"], thresholds["wrong"]])
    report["class-conditional"] = coverage_summary(p, y, conditional)
    report["referrals"] = referral_summary(p, thresholds["marginal"], conditional)
    report["scripted"] = coverage_summary(ScriptedVendorModel().predict(rows), y, thresholds["scripted"])
    figure = dict(sound=p[y == 0].tolist(), wrong=p[y == 1].tolist())
    return report, figure


def build_experiment(directory=None):
    scores = rebuild_scores()
    months = redrawn_windows()
    held = {name: frozen(months.pop(name), "typed-decisions-seed20-" + name)
            for name in ("august", "september")}
    directory = Path(directory if directory is not None else os.environ.get("BOOK_GOLDEN_DIRECTORY", "golden-sets"))
    for golden in held.values():
        golden.persist(directory)
    july = months.pop("july")
    p, y = scores.predict(july)["proposed"], arrays(july)[1]
    zeta = nonconformity_scores(np.column_stack((1 - p, p)))[np.arange(len(y)), y.astype(int)]
    thresholds, ranks = {}, {}
    for name, selected in (("marginal", np.ones(len(y), dtype=bool)), ("sound", y == 0), ("wrong", y == 1)):
        ranks[name], thresholds[name] = conformal_threshold(zeta[selected], ALPHA)
    scripted_p = np.array(ScriptedVendorModel().predict(july))
    scripted_zeta = nonconformity_scores(np.column_stack((1 - scripted_p, scripted_p)))[np.arange(len(y)), y.astype(int)]
    ranks["scripted"], thresholds["scripted"] = conformal_threshold(scripted_zeta, ALPHA)
    calibration = dict(window="redrawn July 2026, policy 2026-07-01", requests=len(y),
                       wrong=int(y.sum()), alpha=ALPHA, thresholds=thresholds, ranks=ranks,
                       model_sha256=scores.sha256,
                       july_sha256=fingerprint([r.model_dump(mode="json") for r in july]),
                       verdict_rule="lower >= 0.87: coverage shown; else upper < 0.90: undercoverage shown; else inconclusive; exact bounds below 50")
    threshold_sha = fingerprint(calibration)
    diagnostics = scripted_diagnostics(july)
    del july, months, p, y, zeta, scripted_p, scripted_zeta
    reports, figure_months, journals = {}, {}, {}
    for name, golden in held.items():
        read = golden.consult("typed decision record", f"thresholds fixed {threshold_sha}; one proper-score, band, conformal and figure record")
        reports[name], figure_months[name] = summarize_month([r for r, _ in read], scores, thresholds)
        journals[name] = dict(manifest=golden.manifest, access=[dict(entry) for entry in golden.log])
        del read
    record = RiskSignalRecord(
        question=RISK_QUESTION, model_version=MODEL_VERSION, model_sha256=scores.sha256,
        class_weighted_version="chapter5-balanced-boosted-seed7-v1",
        cost_matrix_version="illustrative-review4-wrongdenial20-v1", signer_status="illustrative, unsigned",
        band_edges=(PAY_EDGE, THIRD_EDGE), action_map=ACTION_MAP, seed=HISTORY_SEED,
        history_rows=HISTORY_ROWS, population="paid tier 1 requests, by decision date",
        calibration=calibration, threshold_sha256=threshold_sha, retest=reports, journals=journals,
    ).model_dump(mode="json")
    grid = np.linspace(0, 1, 101).tolist()
    figures = dict(seed=HISTORY_SEED, population=record["population"], month="redrawn August 2026",
                   model_sha256=scores.sha256, threshold_sha256=threshold_sha,
                   scores=figure_months["august"],
                   boundaries=dict(tier=PAY_EDGE, marginal=thresholds["marginal"],
                                   class_conditional=1 - thresholds["wrong"],
                                   class_sound=thresholds["sound"]),
                   curves=dict(mu=.7, reports=grid,
                               brier=[expected_score(.7, v) for v in grid],
                               absolute=[expected_score(.7, v, "absolute") for v in grid]))
    return dict(months=reports, record=record, figures=figures,
                validation=scores.validation, scripted_july=diagnostics)


def acceptance_fixture(experiment):
    return dict(model_version=ScriptedVendorModel.version, seed=HISTORY_SEED,
                population="paid tier 1 requests", window="redrawn August 2026",
                calibration_window="redrawn July 2026", alpha=ALPHA,
                threshold=experiment["record"]["calibration"]["thresholds"]["scripted"],
                coverage=experiment["months"]["august"]["scripted"],
                agent_conclusion="The marginal coverage is about 0.90, so the sets catch nine in ten wrong payments.")


def print_retest(experiment):
    print("second draw: seed 20, 36000 requests; paid tier 1, by decision date")
    months = list(experiment["months"].values())
    print("\n| Measure | Redrawn August | Redrawn September |")
    print("|---|---|---|")

    def row(label, values, precision=6):
        cells = [str(v) if precision is None else f"{v:.{precision}f}" for v in values]
        print(f"| {label} | {' | '.join(cells)} |")

    for key, label, precision in (("requests", "Paid tier 1 requests", None),
                                  ("wrong", "Wrong payments", None), ("rate", "Wrong payment rate", 6)):
        row(label, [m[key] for m in months], precision)
    for name in ("proposed", "class-weighted", "base rate"):
        for key, label in (("brier", "Brier"), ("log", "Log loss, nats")):
            row(f"{label}, {name}", [m["scores"][name][key] for m in months])
    for key, label in (("reliability", "Reliability (REL)"), ("resolution", "Resolution (RES)"),
                       ("uncertainty", "Uncertainty (UNC)"), ("residual", "Within-bin residual")):
        row(label + ", proposed", [m["decomposition"][key] for m in months])
    for k in range(5):
        values = [m["decomposition"]["bins"][k] for m in months]
        cells = [f"{v['requests']} / {v['prediction']:.6f} / {v['observed']:.6f}" for v in values]
        print(f"| Equal-count bin {k + 1}: count / forecast / rate | {' | '.join(cells)} |")
    for name in ("proposed", "class-weighted", "base rate"):
        row("Generator squared probability error, " + name, [m["scores"][name]["truth_error"] for m in months])
        row("Generator intrinsic uncertainty, " + name, [m["scores"][name]["intrinsic"] for m in months])
        row("Generator expected Brier, " + name,
            [m["scores"][name]["truth_error"] + m["scores"][name]["intrinsic"] for m in months])
    print()
    print("ten-forecast inputs: five at 0.10 with 1 event; five at 0.70 with 3 events; constant forecast 0.40")
    example = decomposition([1, 0, 0, 0, 0, 1, 1, 1, 0, 0], [.1] * 5 + [.7] * 5, bins=None)
    print(f"ten-forecast example: {example['brier']:.2f} = {example['reliability']:.2f} - {example['resolution']:.2f} + {example['uncertainty']:.2f}; base-rate Brier 0.24")


def print_bands(experiment):
    print("bands: pay below 0.080; review through 0.80, including ties; third above 0.80")
    print("action map: pay only if every gate condition holds; review, third and cannot tell refer")
    print("\n| Window, score, band | Count | Mean forecast | Wrong rate (95% Wilson) | Planted risk |")
    print("|---|---|---|---|---|")
    for month, r in experiment["months"].items():
        for score, bands in r["bands"].items():
            for name, b in bands.items():
                label = f"{month}, {score}, {name}"
                if not b["requests"]:
                    print(f"| {label} | 0 | undefined | undefined | undefined |")
                    continue
                low, high = b["interval"]
                print(f"| {label} | {b['requests']} | {b['prediction']:.4f} | {b['observed']:.4f} ({low:.4f}, {high:.4f}) | {b['planted_risk']:.4f} |")
    print("\n| Window, score, band | Loss if paid: predicted / realized ($) | Amount-weighted forecast / rate |")
    print("|---|---|---|")
    for month, r in experiment["months"].items():
        for score, bands in r["bands"].items():
            for name, b in bands.items():
                if b["requests"]:
                    print(f"| {month}, {score}, {name} | {b['predicted_loss']:.6f} / {b['realized_loss']:.6f} | {b['weighted_prediction']:.6f} / {b['weighted_observed']:.6f} |")
    print()
    for month, r in experiment["months"].items():
        for name in r["bands"]:
            print(f"{month}, {name}: maximum {r['scores'][name]['maximum']:.6f}; third edge 0.80")
        f = r["referrals"]["bands"]
        low, high = f["interval"]
        print(f"{month}, complete band referrals: review {f['review']}, third {f['third']}, cannot tell {f['cannot_tell']}; {f['referred']}/{f['requests']} = {f['share']:.6f} ({low:.6f}, {high:.6f})")
        for name in ("marginal", "class-conditional"):
            f = r["referrals"][name]
            print(f"  with {name} sets requiring a sound singleton: {f['referred']}/{f['requests']} = {f['share']:.6f} ({f['interval'][0]:.6f}, {f['interval'][1]:.6f}); union, counted once")
    print("two-request example: mean prediction 0.10, mean risk 0.10; amount-weighted prediction 0.10, risk 0.13; predicted loss $2.50, expected $3.25")
    print("offline shares for a later queue projection; at stage 3 every draft reaches a person")


def print_conformal(experiment):
    from demos.output_wrap import print_wrapped

    c = experiment["record"]["calibration"]
    print_wrapped(f"redrawn July: {c['requests']} paid tier 1, {c['wrong']} wrong (rate {c['wrong'] / c['requests']:.6f}); alpha {c['alpha']:.2f}")
    print_wrapped(f"marginal: rank {c['ranks']['marginal']}, threshold {c['thresholds']['marginal']:.6f}")
    print_wrapped(f"class-conditional: sound rank {c['ranks']['sound']}, threshold {c['thresholds']['sound']:.6f}; wrong rank {c['ranks']['wrong']}, threshold {c['thresholds']['wrong']:.6f}")
    for month, r in experiment["months"].items():
        for name in ("marginal", "class-conditional"):
            s, n = r[name], r["requests"]
            print_wrapped(f"{month}, {name} sets:")
            for answer, count in s["sets"].items():
                print_wrapped(f"  {answer}: {count}/{n} = {count / n:.6f}")
            print_wrapped(f"  coverage: {s['covered']}/{n} = {s['rate']:.6f} ({s['interval'][0]:.6f}, {s['interval'][1]:.6f}); {s['verdict']}")
            for answer in ("sound", "wrong"):
                a = s[answer]
                if not a["requests"]:
                    print_wrapped(f"  {answer} coverage: undefined, no observations")
                    continue
                print_wrapped(f"  {answer} coverage: {a['covered']}/{a['requests']} = {a['rate']:.6f} ({a['interval'][0]:.6f}, {a['interval'][1]:.6f}); diagnostic only")
    print_wrapped("95% intervals: exact Clopper-Pearson below 50, Wilson otherwise")
    print_wrapped("verdict fixed before reads: lower >= 0.87 shows coverage; else upper < 0.90 shows undercoverage; otherwise inconclusive")
    print_wrapped(f"model {MODEL_VERSION}; costs illustrative, unsigned; seed {HISTORY_SEED}")
    print_wrapped(f"threshold record sha256: {experiment['record']['threshold_sha256']}")
    for name, j in experiment["record"]["journals"].items():
        print_wrapped(f"{name} manifest sha256: {j['manifest']['sha256']}")
        for e in j["access"]:
            print_wrapped(f"  access {e['entry']}: {e['reader']}; {e['purpose']}")


def print_keys(experiment):
    print(f"exercise 2: Brier {expected_score(.8, .8):.2f}, {expected_score(.8, .6):.2f}; absolute {expected_score(.8, .8, 'absolute'):.2f}, {expected_score(.8, 1, 'absolute'):.2f}")
    print("exercise 2b: report 0.85, third band; costs favor deny, action map refers")
    y, p = [0] * 8 + [1] * 2 + [0] * 6 + [1] * 3 + [0], [.05] * 8 + [.2] * 8 + [.5] * 4
    d = decomposition(y, p, bins=None)
    print(f"exercise 3: Brier {d['brier']:.4f}; REL {d['reliability']:.4f}, RES {d['resolution']:.4f}, UNC {d['uncertainty']:.4f}; base-rate {d['uncertainty']:.4f}, recalibrated {d['uncertainty'] - d['resolution']:.4f}")
    scores = [.02, .05, .08, .10, .12, .15, .18, .20, .22, .25, .28, .30, .35, .40, .45, .50, .55, .62, .80]
    rank, threshold = conformal_threshold(scores, .1)
    print(f"exercise 4: rank {rank}, threshold {threshold:.2f}; sets {prediction_set([.6, .3, .1], threshold)}, {prediction_set([.35, .33, .32], threshold)}; index 0 is level 1")
    s = acceptance_fixture(experiment)["coverage"]
    wrong = s["wrong"]
    low, high = wilson_interval(wrong["covered"], wrong["requests"])
    print(f"exercise 5: marginal {s['covered']}/{s['requests']} = {s['rate']:.6f}; wrong {wrong['covered']}/{wrong['requests']} = {wrong['rate']:.6f}, Wilson ({low:.6f}, {high:.6f}); marginal coverage is not class coverage")
    d = experiment["scripted_july"]
    parts = d["decomposition"]
    print(f"exercise 6: seed {d['seed']}, July {d['requests']} paid, {d['wrong']} wrong; Brier {parts['brier']:.6f}, REL {parts['reliability']:.6f}, RES {parts['resolution']:.6f}, UNC {parts['uncertainty']:.6f}, residual {parts['residual']:+.6f}")
    print(f"  refuse: reliability {parts['reliability']:.6f} exceeds illustrative limit {d['reliability_limit']:.6f}; negation {d['negation_violations']}/{d['requests']} violations, largest gap {d['largest_negation_gap']:.6f}")
    print("  no August or September outcome read for exercise 6; no answer maps to denial")
    print("exercise 2 inputs: true probability 0.80; Brier reports 0.80 and 0.60; absolute reports 0.80 and 1.00")
    print("exercise 3 inputs (count, forecast, wrong): (8, 0.05, 0), (8, 0.20, 2), (4, 0.50, 3)")
    print(f"exercise 4 inputs: {scores}; alpha 0.10; vectors (0.60, 0.30, 0.10), (0.35, 0.33, 0.32)")


if __name__ == "__main__" and "--exercise-july" in sys.argv:
    print(json.dumps(scripted_diagnostics(), sort_keys=True, indent=2))
    raise SystemExit(0)


# region book:fnd.typed-decisions.typed-01 output
from pydantic import ValidationError

from refund_agent.typed import (
    ACTION_MAP,
    TypedAnswer,
    TypedQuestion,
    band,
    risk_answer,
)

if __name__ == "__main__":
    question = TypedQuestion(
        name="Is paying this tier 1 refund a wrong payment?",
        kind="true_false",
        answers=("sound", "wrong"),
        cannot_tell=True,
    )
    print(question.name)
    print("declared answers: sound, wrong; declared refusal: cannot tell")
    for p, cents in ((0.03, 4000), (0.12, 4000), (0.03, 7500)):
        answer = risk_answer(p, cents, "chapter5-boosted-platt-seed7-v1")
        name = (
            "cannot_tell"
            if answer.cannot_tell
            else band(answer.probabilities["wrong"])
        )
        report = (
            "cannot tell"
            if answer.cannot_tell
            else f"wrong probability {p:.2f}"
        )
        print(
            f"${cents / 100:.2f}: {report}, band {name}, action "
            f"{ACTION_MAP[name]}"
        )
    try:
        TypedAnswer(
            question=question,
            probabilities={"sound": 0.9, "unknown": 0.1},
            model_version="test",
        )
    except ValidationError as error:
        print(
            "refused:", error.errors()[0]["msg"].removeprefix("Value error, ")
        )
# endregion book:fnd.typed-decisions.typed-01


# region book:fnd.typed-decisions.router-01 output
from demos.output_wrap import print_wrapped

if __name__ == "__main__":
    r = router_report()
    print_wrapped(f"{r['requests']} constructed tickets: top probability {r['top_probability']:.2f}, correct {r['top_correct']}/{r['requests']} = {r['top_correct'] / r['requests']:.2f}")
    print_wrapped(f"escalation: mean prediction {r['escalation_prediction']:.2f}, observed {r['escalation_count']}/{r['requests']} = {r['escalation_count'] / r['requests']:.2f}")
    print_wrapped(f"two ten-ticket groups, counts (standard, priority, escalation): {r['groups']}; correct the first group's vector")
    for prefix in ("", "corrected_"):
        costs = ', '.join(f"{cost:.2f}" for cost in r[prefix + 'costs'])
        print_wrapped(f"vector {r[prefix + 'vector']}; costs (hours) [{costs}]; route {r['corrected_route' if prefix else 'stated_route']}")
# endregion book:fnd.typed-decisions.router-01


# region book:fnd.typed-decisions.scores-01 output
from demos import typed_decisions_demo as td
from demos.output_wrap import print_wrapped

if __name__ == "__main__":
    print_wrapped("event probability 0.70; expected penalties (log in nats)")
    print_wrapped("report   Brier     log      absolute")
    for p in (0.5, 0.7, 0.9, 1.0):
        log = td.expected_score(0.7, p, "log")
        log_text = f"{log:.3f}" if log < float("inf") else "unbounded"
        print_wrapped(
            f"{p:.2f}     {td.expected_score(0.7, p):.2f}      {log_text:9} "
            f"{td.expected_score(0.7, p, 'absolute'):.2f}"
        )
    smoothed = td.smoothed_minimizer(0.04, 0.05)
    print_wrapped(
        f"label smoothing, targets 0.05 and 0.95: true risk 0.04 reports "
        f"{smoothed:.3f}, above 0.080"
    )
    print_wrapped(
        "hard-answer correctness reward: always yes gives 0.70; stated "
        "confidence is unscored"
    )
# endregion book:fnd.typed-decisions.scores-01


if __name__ == "__main__":
    experiment = build_experiment()

# region book:fnd.typed-decisions.retest-01 output
if __name__ == "__main__":
    print_retest(experiment)
# endregion book:fnd.typed-decisions.retest-01


# region book:fnd.typed-decisions.bands-01 output
if __name__ == "__main__":
    print_bands(experiment)
# endregion book:fnd.typed-decisions.bands-01


# region book:fnd.typed-decisions.conformal-01 output
from demos import typed_decisions_demo as td
from demos.output_wrap import print_wrapped

if __name__ == "__main__":
    triage_scores = [0.05, 0.10, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60]
    rank, threshold = td.conformal_threshold(triage_scores, 0.20)
    print_wrapped(
        f"triage: {len(triage_scores)} scores, alpha 0.20, rank {rank}, "
        f"threshold {threshold:.2f}"
    )
    print_wrapped(
        "triage set:",
        tuple(
            f"level {c + 1}"
            for c in td.prediction_set([0.7, 0.2, 0.1], threshold)
        ),
    )
    # Earlier output-only regions share this result; the three printed listings
    # alone reach it here, with no unprinted setup.
    if "experiment" not in globals():
        experiment = td.build_experiment()
    td.print_conformal(experiment)
# endregion book:fnd.typed-decisions.conformal-01


# region book:fnd.typed-decisions.figures-01 output
if __name__ == "__main__":
    print(json.dumps(experiment["figures"], sort_keys=True, indent=2, allow_nan=False))
# endregion book:fnd.typed-decisions.figures-01


# region book:fnd.typed-decisions.keys-01 output
if __name__ == "__main__":
    print_keys(experiment)
# endregion book:fnd.typed-decisions.keys-01


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-fixtures", action="store_true")
    parser.add_argument("--exercise-july", action="store_true")
    args = parser.parse_args()
    if args.write_fixtures:
        for name, value in (("typed_decisions_acceptance_05.json", acceptance_fixture(experiment)),
                            ("typed_decisions_figures.json", experiment["figures"]),
                            ("typed_decisions_record.json", experiment["record"])):
            (DATA / name).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
