"""Deterministic outputs for chapter 11, representations.

The three printed listings run in order without the setup between regions. Imports
do no work. The comparison fits on clause text, chooses on July, then reads chapter
4's frozen requests in one logged consultation. Each request produces two questions,
whose relevant version comes from its label; outcomes never become query features.

All version comparisons use the same retrospective five-version corpus and simulate
a fresh source read one hour before each question. This isolates date selection from
read age. The printed listings and the versions figure ask the worked request's
question at the time that request arrived, so the chapter has one clock.

The PairEncoder is trained only on pre-July pairs. Its comparison is omitted
because Chapter 6 trains no encoder. The retrieval contract is unchanged.
"""

from datetime import datetime, timezone
from pathlib import Path

from demos.ml_data_evaluation_demo import wilson_interval
from refund_agent.labels import LABEL_POLICY
from refund_agent.retrieval.encoders import (
    LinearTextEncoder,
    PairEncoder,
)
from refund_agent.retrieval.evaluate import mean_reciprocal_rank, recall_at_k, superseded_returned
from refund_agent.retrieval.index import rank
from refund_agent.retrieval.questions import policy_questions, question_requests
from refund_agent.split import CUTS, freeze, time_split

# Mid-August: the worked request is the eligible golden request nearest this time.
# No question is asked at it; every search runs at its own request's time.
MID_AUGUST = datetime(2026, 8, 15, tzinfo=timezone.utc)
EXERCISE_AT = datetime(2026, 10, 15, tzinfo=timezone.utc)
K = 3


def rankings(policies, encoder, questions, mode, k=K):
    found = []
    for q in questions:
        index = PolicyIndex.build(policies, q.as_of - timedelta(hours=1), encoder)
        if mode == "dated":
            hits = index.retrieve(q.text, q.as_of, k).hits
        elif mode in ("similarity", "newest"):
            vector = index.encoder.transform([q.text])[0]
            hits = (
                rank(index.clauses, vector, k, newest=mode == "newest")
                if index.encoder.has_terms(q.text)
                else ()
            )
        else:
            raise ValueError("unknown search")
        found.append(hits)
    return found


def score(policies, encoder, questions, mode):
    ranked = rankings(policies, encoder, questions, mode)
    labels = [q.relevant for q in questions]
    n = len(questions)
    hits = {
        k: sum(
            any(h.clause.citation == q.relevant for h in row[:k])
            for row, q in zip(ranked, questions)
        )
        for k in (1, K)
    }
    # A hit below first place is a tie when the clause listed above it has its score.
    tied = 0
    for row, q in zip(ranked, questions):
        place = next((i for i, h in enumerate(row[:K]) if h.clause.citation == q.relevant), 0)
        tied += place > 0 and round(row[place - 1].score, 12) == round(row[place].score, 12)
    return {
        "n": n,
        "hits": hits,
        "tied_below_first": tied,
        "recall": {k: recall_at_k(ranked, labels, k) for k in hits},
        "interval": {k: wilson_interval(hits[k], n) for k in hits},
        "mrr": mean_reciprocal_rank(ranked, labels),
        "superseded_top1": superseded_returned(
            [row[:1] for row in ranked], [q.as_of for q in questions]
        ),
    }


def build_experiment(seed=7):
    data = generate(seed)
    windows = time_split(data.requests, CUTS, timedelta(days=LABEL_POLICY.window_days))
    labels = {d.request_id: d for d in data.correct}
    golden = freeze(data, windows.pop("golden"), LABEL_POLICY)
    development = windows["training"] + windows["validation"]
    labels = {r.request_id: labels[r.request_id] for r in development}
    data = data.model_copy(update={"requests": development, "correct": list(labels.values())})

    clauses = policy_clauses(data.policies, MID_AUGUST)
    texts = [c.text for c in clauses]
    counts = CountEncoder().fit(texts)
    x = counts.transform(texts)
    encoders = {"counts": counts}
    errors = {"counts": None}
    for width in (1, 2):
        ae = LinearAutoencoder(width).fit(x)
        encoders[f"linear-{width}"] = LinearTextEncoder(counts, ae)
        errors[f"linear-{width}"] = ae.error(x)

    training = question_requests(data, windows["training"])
    training_questions = [
        q
        for wording in ("policy", "customer")
        for q in policy_questions(training, list(labels.values()), wording)
    ]
    training_clauses = {c.citation: c.text for c in clauses if c.effective_from < CUTS[0]}
    pairs = [(q.text, training_clauses[q.relevant]) for q in training_questions]
    pair_encoder = PairEncoder(2).fit(pairs, seed=seed)

    validation_rows = question_requests(data, windows["validation"])
    validation = {
        wording: policy_questions(validation_rows, list(labels.values()), wording)
        for wording in ("policy", "customer")
    }
    metrics = []
    for wording, questions in validation.items():
        for name, encoder in encoders.items():
            metrics.append(
                {
                    "window": "July",
                    "wording": wording,
                    "encoder": name,
                    "reconstruction": errors[name],
                    "similarity": score(data.policies, encoder, questions, "similarity"),
                    "dated": score(data.policies, encoder, questions, "dated"),
                }
            )
    primary = [row for row in metrics if row["wording"] == "policy"]
    chosen = max(primary, key=lambda row: row["dated"]["recall"][1])["encoder"]
    selection = {
        "encoder": chosen,
        "k": K,
        "chosen_on": "validation (July)",
        "tie_rule": "counts, linear-1, linear-2",
    }

    read = golden.consult("representations-demo", "report the fixed encoders and date filters")
    held_rows = question_requests(data, [r for r, _ in read])
    held_labels = [label for _, label in read]
    held = {
        wording: policy_questions(held_rows, held_labels, wording)
        for wording in ("policy", "customer")
    }
    for wording, questions in held.items():
        for name, encoder in encoders.items():
            metrics.append(
                {
                    "window": "golden",
                    "wording": wording,
                    "encoder": name,
                    "reconstruction": errors[name],
                    "similarity": score(data.policies, encoder, questions, "similarity"),
                    "dated": score(data.policies, encoder, questions, "dated"),
                }
            )

    eligible_ids = {
        r.request_id
        for r, label in read
        if label.reason == "outside_return_window" and r.at.month == 8
    }
    candidates = [
        row
        for row in held_rows
        if row.request.request_id in eligible_ids
        and row.days_since_delivery is not None
        and 30 < row.days_since_delivery <= 45
    ]
    example = min(candidates, key=lambda row: abs((row.request.at - MID_AUGUST).total_seconds()))
    worked = next(
        q
        for q in held["policy"]
        if q.request_id == example.request.request_id and q.section == "eligibility"
    )
    from refund_agent.data.generate import _label

    request = example.request
    order = next(o for o in data.orders if o.order_id == request.order_reference)
    shipment = next(s for s in data.shipments if s.order_id == order.order_id)
    current_label = next(label for r, label in read if r.request_id == request.request_id)
    latest_label = _label(request, order, shipment, data.policies[-1])
    versions = {
        mode: score(data.policies, counts, held["policy"], mode)
        for mode in ("similarity", "newest", "dated")
    }
    worked_hits = {mode: rankings(data.policies, counts, [worked], mode, 1)[0] for mode in versions}
    all_requests = development + [r for r, _ in read]
    injections = []
    for r in all_requests:
        if r.failure not in ("cached_policy", "stale_retrieval"):
            continue
        snapshot = next(s for s in data.retrievals if s.request_id == r.request_id)
        known = [p for p in data.policies if p.version <= snapshot.version]
        index = PolicyIndex.build(known, snapshot.observed_at, counts)
        result = index.retrieve("damaged days delivery", r.at, 1)
        # This is the same current-version comparison as rules.evidence_problem.
        current = data.policy_at(r.at)
        injections.append(
            {
                "failure": r.failure,
                "request": r.request_id,
                "age": (r.at - snapshot.observed_at).total_seconds(),
                "refusal": result.refusal,
                "citation": result.hits[0].clause.citation if result.hits else None,
                "at_use": "non_current_policy"
                if snapshot.version != current.version
                else "current",
            }
        )
    return {
        "policies": data.policies,
        "clauses": clauses,
        "encoders": encoders,
        "metrics": metrics,
        "selection": selection,
        "manifest": golden.manifest,
        "access_log": list(golden.log),
        "golden_questions": held,
        "worked": worked,
        "worked_hits": worked_hits,
        "versions": versions,
        "worked_decisions": {
            "amount_cents": request.amount_cents,
            "tier": int(current_label.tier),
            "current": current_label.reason,
            "latest": latest_label.reason,
        },
        "injections": sorted(injections, key=lambda row: row["failure"]),
        "pair_training": {
            "pairs": len(pairs),
            "first_loss": pair_encoder.losses[0],
            "last_loss": pair_encoder.losses[-1],
            "policy_versions": sorted(
                {q.relevant.split("@")[1].split("#")[0] for q in training_questions}
            ),
        },
    }


def interval_text(card, k):
    low, high = card["interval"][k]
    return f"{card['hits'][k]}/{card['n']} = {card['recall'][k]:.3f} [{low:.3f}, {high:.3f}]"


def print_recall(report):
    print(f"chosen on July: {report['selection']['encoder']}; k={K}; ties prefer counts")
    print(
        f"golden: {report['manifest']['requests']} requests, {len(report['golden_questions']['policy'])} questions per wording; one consultation"
    )
    print("each comparison simulates a fresh policy read 3600 seconds before use")
    print("reconstruction: mean squared row error on the ten centered clause vectors")
    print(
        "window | wording | representation | reconstruction | no dates R@1 | no dates R@3 | dated R@1 | dated R@3 | no dates MRR"
    )
    for row in report["metrics"]:
        error = "--" if row["reconstruction"] is None else f"{row['reconstruction']:.6f}"
        raw, dated = row["similarity"], row["dated"]
        cells = [
            row["window"],
            row["wording"],
            row["encoder"],
            error,
            interval_text(raw, 1),
            interval_text(raw, K),
            interval_text(dated, 1),
            interval_text(dated, K),
            f"{raw['mrr']:.3f}",
        ]
        print(" | ".join(cells))
    for row in report["metrics"]:
        raw = row["similarity"]
        if row["window"] == "golden" and raw["hits"][K]:
            print(
                f"golden | {row['wording']} | {row['encoder']} | no dates: {raw['hits'][K]} top-{K} hits, "
                f"{raw['hits'][1]} first; {raw['tied_below_first']} tie with the clause listed above"
            )
    for hits in (242, 0, 18):
        n = 20 if hits == 18 else 242
        low, high = wilson_interval(hits, n)
        print(f"arithmetic check: {hits}/{n} = {hits / n:.3f} [{low:.3f}, {high:.3f}]")
    print("pair-trained comparison omitted: Chapter 6 trains no encoder")
    print("Wilson intervals treat questions as Bernoulli trials; two questions share each request")
    print("access log:", json.dumps(report["access_log"], sort_keys=True))


def print_recall_table(report):
    """The golden rows of print_recall as a table of hits, for the chapter to include."""
    from demos.output_wrap import print_wrapped

    names = {
        "counts": "Counts",
        "linear-1": "Linear code, width 1",
        "linear-2": "Linear code, width 2",
    }
    print_wrapped("Raw: no dates; final columns: dated. MSE: reconstruction error.\n")
    print_wrapped(
        "| Wording | Encoding | MSE | First, raw "
        f"| Top {K}, raw | First, 95% CI | Top {K} |"
    )
    print_wrapped("|---|---|---|---|---|---|---|")
    for row in report["metrics"]:
        if row["window"] != "golden":
            continue
        error = "none" if row["reconstruction"] is None else f"{row['reconstruction']:.3f}"
        raw, dated = row["similarity"], row["dated"]
        low, high = dated["interval"][1]
        print_wrapped(
            f"| {row['wording'].capitalize()} | {names[row['encoder']]} | {error} "
            f"| {raw['hits'][1]} | {raw['hits'][K]} "
            f"| {dated['hits'][1]} ({low:.3f} to {high:.3f}) | {dated['hits'][K]} |"
        )


def print_versions(report):
    from demos.output_wrap import print_wrapped

    q = report["worked"]
    print_wrapped(f"worked request: {q.request_id}; at {q.as_of.isoformat()}; {q.text}")
    decision = report["worked_decisions"]
    print_wrapped(
        f"amount {decision['amount_cents']} cents, tier {decision['tier']}; "
        f"July: {decision['current']}; September: {decision['latest']}"
    )
    for mode, hits in report["worked_hits"].items():
        hit = hits[0]
        record = hit.clause
        print_wrapped(
            f"{mode}: {record.citation}; cosine {hit.score:.3f}; "
            f"[{record.effective_from.isoformat()}, {record.effective_to.isoformat() if record.effective_to else 'open'})"
        )
    for mode, card in report["versions"].items():
        print_wrapped(
            f"golden {mode}: R@1 {interval_text(card, 1)}; "
            f"out-of-interval clauses at rank 1: {card['superseded_top1']}"
        )
    for row in report["injections"]:
        print_wrapped(
            f"{row['failure']}: age {row['age']:.0f} seconds; index {row['refusal'] or row['citation']}; at use {row['at_use']}"
        )


def exercise_policies():
    policies = generate(7).policies
    latest = policies[-1]
    sixth = latest.model_copy(
        update={
            "version": "2026-10-01",
            "effective_at": datetime(2026, 10, 1, tzinfo=timezone.utc),
            "return_days": 60,
            "text": latest.text.replace("2026-09-01", "2026-10-01").replace(
                "within 45 days", "within 60 days"
            ),
        }
    )
    return policies + [sixth]


def exercise_fixture(report):
    policies = exercise_policies()
    counts = CountEncoder().fit([c.text for c in policy_clauses(policies, EXERCISE_AT)])
    index = PolicyIndex.build(policies, EXERCISE_AT, counts)
    records = [c.record() for c in index.clauses]
    for record in records:
        if record["version"] == "2026-09-01":
            record["effective_to"] = None
    # Deliberately not an attested build manifest: the exercise diagnoses this report.
    return {
        "records": records,
        "golden_recall_at_1": report["versions"]["dated"]["recall"][1],
        "golden_last": report["manifest"]["last"],
        "golden_questions": len(report["golden_questions"]["policy"]),
        "golden_sha256": report["manifest"]["sha256"],
    }


def exercise_acceptance(report):
    """Fixed sixth-version regression checks, including a preset rarity weighting."""
    from refund_agent.retrieval.questions import Question

    policies = exercise_policies()
    change = policies[-1].effective_at
    questions = [
        Question(
            f"exercise-6-{i}",
            "eligibility",
            "damaged 40 days after delivery",
            change + timedelta(seconds=i - 10),
            f"refund_cap_v4@{'2026-09-01' if i < 10 else '2026-10-01'}#eligibility",
        )
        for i in range(20)
    ]
    checks = []
    for rarity in (False, True):
        counts = CountEncoder(rarity=rarity).fit(
            [c.text for c in policy_clauses(policies, EXERCISE_AT)]
        )
        found = rankings(policies, counts, questions, "dated", 1)
        hits = sum(
            any(h.clause.citation == q.relevant for h in row) for row, q in zip(found, questions)
        )
        historical = score(policies, counts, report["golden_questions"]["policy"], "dated")
        stale = PolicyIndex.build(policies, change - timedelta(days=31), counts).retrieve(
            "delivery", change, 1
        )
        checks.append(
            {
                "rarity": rarity,
                "questions": len(questions),
                "hits": hits,
                "out_of_interval": superseded_returned(found, [q.as_of for q in questions]),
                "records_complete": all(
                    h.clause.citation
                    and h.clause.effective_from
                    and h.clause.observed_at
                    and hasattr(h.clause, "effective_to")
                    for row in found
                    for h in row
                ),
                "stale_refusal": stale.refusal,
                "golden_hits": historical["hits"][1],
                "manifest": PolicyIndex.build(policies, EXERCISE_AT, counts).manifest,
            }
        )
    return checks


def figure_data(report):
    counts = report["encoders"]["counts"]
    x = counts.transform([c.text for c in report["clauses"]])
    errors = [
        {"width": d, "error": LinearAutoencoder(d).fit(x).error(x)}
        for d in range(1, min(x.shape) + 1)
    ]
    question = report["worked"]
    searches = {
        mode: rankings(report["policies"], counts, [question], mode, 1)[0][0].clause.citation
        for mode in ("similarity", "newest", "dated")
    }
    return {
        "reconstruction": {
            "point": [3, 4],
            "code": [3],
            "reconstructed": [3, 0],
            "squared_error": 16,
            "widths": [2, 1, 2],
        },
        "width_error": errors,
        "versions": [
            {
                "version": p.version,
                "return_days": p.return_days,
                "from": p.effective_at.isoformat(),
                "to": report["policies"][i + 1].effective_at.isoformat()
                if i + 1 < len(report["policies"])
                else None,
            }
            for i, p in enumerate(report["policies"])
        ],
        "question": {
            "text": question.text,
            "at": question.as_of.isoformat(),
            "source_request": question.request_id,
            "note": "the worked request's own wording and time of arrival",
        },
        "searches": searches,
    }


def write_data(report, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("autoencoders_nlp_figures.json", figure_data(report)),
        ("autoencoders_nlp_index_report_05.json", exercise_fixture(report)),
    ):
        (directory / name).write_text(
            json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )


# region book:fnd.autoencoders-nlp.counts-01 output
from datetime import datetime, timezone

import numpy as np

from demos.output_wrap import print_wrapped
from refund_agent.data.generate import generate
from refund_agent.retrieval.clauses import policy_clauses
from refund_agent.retrieval.encoders import CountEncoder, cosine

if __name__ == "__main__":
    policies = generate(7).policies
    # The time of use: when the worked request, r-7-0117, arrived.
    at = datetime(2026, 8, 18, 18, 16, 54, tzinfo=timezone.utc)
    clauses = policy_clauses(policies, at)
    counts = CountEncoder().fit([c.text for c in clauses])
    x = counts.transform([c.text for c in clauses])
    question = counts.transform(["damaged 33 days after delivery"])[0]
    print_wrapped(
        f"clauses: {len(clauses)}; vocabulary: {len(counts.vocabulary)}"
    )
    print_wrapped("vocabulary:", ", ".join(counts.vocabulary))
    for i, j in ((0, 2), (4, 6), (0, 1)):
        print_wrapped(
            f"{clauses[i].citation} vs {clauses[j].citation}: "
            f"{cosine(x[i], x[j]):.3f}"
        )
    for i in range(0, len(clauses), 2):
        print_wrapped(
            f"question vs {clauses[i].citation}: {cosine(question, x[i]):.3f}"
        )
# endregion book:fnd.autoencoders-nlp.counts-01


# region book:fnd.autoencoders-nlp.autoencoder-01 output
from demos.output_wrap import print_wrapped
from refund_agent.retrieval.encoders import LinearAutoencoder


def show_rows(model, rows, names, wide=False):
    def values(vector):
        return [round(float(a), 3) if abs(a) > 0.0005 else 0.0 for a in vector]

    h = model.transform(rows)
    reconstructed = model.reconstruct(h)
    for name, code, rebuilt, row in zip(names, h, reconstructed, rows):
        # The reconstruction has one number per vocabulary word; omit it.
        middle = "" if wide else f", reconstruction {values(rebuilt)}"
        error = np.sum((row - rebuilt) ** 2)
        print_wrapped(
            f"{name}: code {values(code)}{middle}, squared error {error:.6f}"
        )


if __name__ == "__main__":
    points = np.array([[2.0, 0.0], [-2.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    toy = LinearAutoencoder(1).fit(points)
    show_rows(toy, np.array([[3.0, 4.0]]), ["point (3, 4)"])
    print_wrapped("four-point variances:", toy.eigenvalues.round(3).tolist())
    centered = points - toy.mean
    covariance = centered.T @ centered / len(points)
    _, eigenvectors = np.linalg.eigh(covariance)
    directions = ", ".join(
        "(" + ", ".join(f"{v:g}" for v in row) + ")"
        for row in eigenvectors[:, ::-1].T
    )
    print_wrapped(
        f"four-point covariance: {covariance.tolist()}; trace "
        f"{np.trace(covariance):g}; eigenvectors {directions}"
    )
    show_rows(toy, points, ["point " + str(i + 1) for i in range(4)])
    print_wrapped(
        f"four-point MSE, keep first axis: {toy.error(points):.3f}; keep "
        f"second: {np.mean(points[:, 0] ** 2):.3f}"
    )
    model = LinearAutoencoder(2).fit(x)
    print_wrapped(
        f"first clause direction, variance share: "
        f"{model.eigenvalues[0] / sum(model.eigenvalues):.6f}"
    )
    print_wrapped(
        f"clause MSE, width 1: {LinearAutoencoder(1).fit(x).error(x):.6f}; "
        f"width 2: {model.error(x):.6f}"
    )
    show_rows(model, x, [c.citation for c in clauses], wide=True)
    trained = LinearAutoencoder(2).fit(x, method="gradient_descent", seed=7)
    print_wrapped(
        f"clause MSE, closed form: {model.error(x):.9f}; untied gradient "
        f"descent: {trained.error(x):.9f}"
    )
# endregion book:fnd.autoencoders-nlp.autoencoder-01


# region book:fnd.autoencoders-nlp.index-01 output
import json
from datetime import timedelta

from demos.output_wrap import print_wrapped
from refund_agent.retrieval.index import PolicyIndex

if __name__ == "__main__":
    index = PolicyIndex.build(policies, at - timedelta(hours=1), counts)
    result = index.retrieve("damaged 33 days after delivery", at, k=1)
    for hit in result.hits:
        record = hit.clause
        end = (
            record.effective_to.isoformat() if record.effective_to else "open"
        )
        print_wrapped(f"{record.citation}; cosine {hit.score:.3f}")
        print_wrapped(f"effective_from: {record.effective_from.isoformat()}")
        print_wrapped(f"effective_to: {end}")
        print_wrapped(f"observed_at: {record.observed_at.isoformat()}")
    print_wrapped("manifest:", json.dumps(index.manifest, sort_keys=True))
    old = PolicyIndex.build(policies, at - timedelta(seconds=2592001), counts)
    stale = old.retrieve("delivery", at, k=1)
    print_wrapped(f"read age 2592001 seconds: {stale.refusal}")
# endregion book:fnd.autoencoders-nlp.index-01


if __name__ == "__main__":
    report = build_experiment(7)

# region book:fnd.autoencoders-nlp.recall-01 output
if __name__ == "__main__":
    print_recall(report)
# endregion book:fnd.autoencoders-nlp.recall-01

# region book:fnd.autoencoders-nlp.recall-02 output
if __name__ == "__main__":
    print_recall_table(report)
# endregion book:fnd.autoencoders-nlp.recall-02

# region book:fnd.autoencoders-nlp.versions-01 output
if __name__ == "__main__":
    print_versions(report)
# endregion book:fnd.autoencoders-nlp.versions-01


# region book:fnd.autoencoders-nlp.keys-01 output
def print_keys():
    point = np.array([3.0, 4.0])
    for direction in (np.array([0.6, 0.8]), np.array([0.0, 1.0])):
        h = point @ direction
        rebuilt = h * direction
        print(
            f"exercise 3: direction {direction.tolist()}; code {h:g}; "
            f"reconstruction {rebuilt.tolist()}; squared error {np.sum((point - rebuilt) ** 2):g}"
        )
    print(f"exercise 4: cosine {cosine([1, 1, 0, 2], [1, 0, 1, 2]):.3f}")
    print(
        f"exercise 4: recall@1 {22 / 40:.2f}; recall@3 {34 / 40:.2f}; Wilson [{wilson_interval(34, 40)[0]:.3f}, {wilson_interval(34, 40)[1]:.3f}]"
    )


if __name__ == "__main__":
    print_keys()
    for check in exercise_acceptance(report):
        print(
            f"exercise 6: rarity {check['rarity']}; {check['hits']}/{check['questions']} boundary hits; "
            f"out-of-interval {check['out_of_interval']}; complete records {check['records_complete']}; "
            f"31-day read {check['stale_refusal']}; golden hits {check['golden_hits']}"
        )
        print("exercise 6 manifest:", json.dumps(check["manifest"], sort_keys=True))
# endregion book:fnd.autoencoders-nlp.keys-01


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-data", action="store_true", help="regenerate the two chapter fixtures"
    )
    args = parser.parse_args()
    if args.write_data:
        write_data(report, Path(__file__).resolve().parent / "data")
