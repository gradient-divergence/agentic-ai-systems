"""Chapter 14's seeded adaptation demonstration, beside the unchanged agent loop.

The three printed listings run in chapter order without intervening setup. The
scripted adapter looks up the complete terms of the latest version cited by training
drafts; it trains no language model. The separate low-rank toy trains on training
arrays only. Imports do no work. --write-data regenerates figure inputs, records
and the planted report.
"""

from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from demos.data_science_strategy_demo import zero_failure_bound
from demos.ml_data_evaluation_demo import wilson_interval
from refund_agent import contracts as c
from refund_agent.adaptation.arms import Arm, dated_index, fit_adapter_terms, retrieval_arm
from refund_agent.adaptation.experiment import (
    BRIEF,
    AdaptationCommission,
    agreement,
    decide,
    drafts_needed,
    forbidden,
    selection_record,
    words_per_draft,
)
from refund_agent.adaptation.lora import fit
from refund_agent.adaptation.prompt import DRAFT_SHARE, WINDOW, draft_fields
from refund_agent.adaptation.suite import RegressionSuite, constructed_requests
from refund_agent.labels import LABEL_POLICY
from refund_agent.retrieval.clauses import policy_clauses
from refund_agent.retrieval.encoders import CountEncoder
from refund_agent.selection import fingerprint
from refund_agent.split import CUTS, freeze, time_split
from refund_agent.split import fingerprint as data_fingerprint


def objective(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    if (
        probabilities.ndim != 1
        or not len(probabilities)
        or np.any((probabilities <= 0) | (probabilities > 1))
        or not np.isfinite(probabilities).all()
    ):
        raise ValueError("draft token probabilities must be in (0, 1]")
    return float(-np.log(probabilities).mean())


def print_objective():
    probabilities = np.array([0.9, 0.8, 0.5, 0.25])
    print("draft token probabilities:", ", ".join(f"{p:g}" for p in probabilities))
    print(f"conditional draft probability: {np.prod(probabilities):.6f}")
    print("token losses (nats):", ", ".join(f"{v:.3f}" for v in -np.log(probabilities)))
    print(f"mean draft-only loss: {objective(probabilities):.3f} nats")


def lowrank_data(seed=7):
    """The existing toy's arrays, drawn once in their original order."""
    rng = np.random.default_rng(seed)
    width, n_train, n_test, noise = 16, 768, 2048, 0.1
    w0 = rng.normal(0, 0.1, (width, width))
    down, _ = np.linalg.qr(rng.normal(size=(width, 2)))
    up, _ = np.linalg.qr(rng.normal(size=(width, 2)))
    change = down @ np.diag([2.5, 1.5]) @ up.T
    x_train, x_test = rng.normal(size=(n_train, width)), rng.normal(size=(n_test, width))
    y_train = x_train @ (w0 + change) + rng.normal(0, noise, (n_train, width))
    y_test = x_test @ (w0 + change) + rng.normal(0, noise, (n_test, width))
    return w0, change, x_train, y_train, x_test, y_test, noise


def lowrank_experiment(seed=7):
    w0, change, x_train, y_train, x_test, y_test, noise = lowrank_data(seed)
    width, n_train, n_test = w0.shape[0], len(x_train), len(x_test)
    rows = []
    for rank in (1, 2, 4, 8, None):
        learned = fit(x_train, y_train, w0, rank, nu=8, seed=seed, steps=4000, step_size=0.1)
        rows.append(
            {
                "rank": rank,
                "method": "full" if rank is None else f"rank {rank}",
                "trainable": width**2 if rank is None else trainable_numbers(width, width, rank),
                "train_mse": float(np.mean((x_train @ learned - y_train) ** 2)),
                "test_mse": float(np.mean((x_test @ learned - y_test) ** 2)),
            }
        )
    full_error = rows[-1]["test_mse"]
    return {
        "seed": seed,
        "width": width,
        "needed_rank": int(np.linalg.matrix_rank(change)),
        "n_train": n_train,
        "n_test": n_test,
        "noise_sd": noise,
        "noise_floor": noise**2,
        "steps": 4000,
        "step_size": 0.1,
        "nu": 8,
        "results": rows,
        "rank_1_misses": rows[0]["test_mse"] >= 2 * noise**2,
        "within_full_10_percent": all(
            abs(row["test_mse"] / full_error - 1) <= 0.1 for row in rows[1:-1]
        ),
    }


def quantize_weights(weights, bits):
    """Symmetric nearest-level rounding; pack the actual 4- or 8-bit codes.

    One float64 step is separate metadata. The highest unsigned code is unused.
    """
    weights = np.asarray(weights, dtype=float)
    if bits not in (4, 8) or not weights.size or not np.isfinite(weights).all():
        raise ValueError("nonempty finite weights and 4 or 8 bits required")
    limit = 2 ** (bits - 1) - 1
    step = float(np.max(np.abs(weights)) / limit)
    codes = np.zeros_like(weights) if step == 0 else np.rint(weights / step)
    unsigned = (codes.ravel() + limit).astype(np.uint8)
    binary = (unsigned[:, None] >> np.arange(bits - 1, -1, -1)) & 1
    return codes * step, step, np.packbits(binary).tobytes()


def quantization_experiment(seed=7):
    w0, _, x_train, y_train, x_test, y_test, noise = lowrank_data(seed)
    weights = fit(
        x_train, y_train, w0, 2, nu=8, seed=seed, steps=4000, step_size=0.1
    ).astype(np.float32)
    rows = []
    for bits in (8, 4):
        rounded, step, payload = quantize_weights(weights, bits)
        rows.append({
            "bits": bits,
            "payload_bytes": len(payload),
            "metadata_bytes": np.asarray(step, dtype=np.float64).nbytes,
            "step": step,
            "half_step": step / 2,
            "max_error": float(np.max(np.abs(rounded - weights))),
            "test_mse": float(np.mean((x_test @ rounded - y_test) ** 2)),
        })
    return {
        "seed": seed,
        "width": w0.shape[0],
        "n_train": len(x_train),
        "n_test": len(x_test),
        "noise_floor": noise**2,
        "reference_bytes": weights.nbytes,
        "reference_mse": float(np.mean((x_test @ weights - y_test) ** 2)),
        "results": rows,
    }


def print_quantization(report):
    print(
        f"seed {report['seed']}; trained rank-2 merged {report['width']}-by-{report['width']} toy; "
        f"training {report['n_train']}; held-out {report['n_test']}"
    )
    print(
        f"float32 reference: {report['reference_bytes']} bytes; "
        f"held-out MSE {report['reference_mse']:.6f}; noise floor {report['noise_floor']:.4f}"
    )
    print("symmetric grid; one float64 step stored separately; no clipping or retraining")
    print("bits | packed weight bytes | step bytes | max error | half step | within bound | held-out MSE | MSE / floor")
    for row in report["results"]:
        print(
            f"{row['bits']} | {row['payload_bytes']} | {row['metadata_bytes']} | "
            f"{row['max_error']:.6f} | {row['half_step']:.6f} | "
            f"{row['max_error'] <= row['half_step'] + 1e-12} | "
            f"{row['test_mse']:.6f} | {row['test_mse'] / report['noise_floor']:.3f}"
        )


def soft_targets(scores, temperature):
    scores = np.asarray(scores, dtype=float)
    if (
        scores.ndim != 1 or not scores.size or not np.isfinite(scores).all()
        or not np.isfinite(temperature) or temperature <= 0
    ):
        raise ValueError("finite scores and positive finite temperature required")
    values = np.exp((scores - scores.max()) / temperature)
    return values / values.sum()


def cost_per_accepted_draft(mean_words, acceptance, word_price, review_cost):
    """One final draft per request, including a human redo after a rejection."""
    if (
        not np.isfinite([mean_words, acceptance, word_price, review_cost]).all()
        or min(mean_words, word_price, review_cost) < 0 or not 0 <= acceptance <= 1
    ):
        raise ValueError("nonnegative finite costs and words, acceptance in [0, 1] required")
    return word_price * mean_words + review_cost * (1 - acceptance)


def adaptation_costs(report, *, word_price, review_cost):
    """Label agreement stands in for acceptance; prices are constructed inputs."""
    rows = []
    for name in ("prompt", "retrieval", "adapter"):
        accepted = sum(report["agreements"][name])
        cases = len(report["agreements"][name])
        words = words_per_draft(report["runs"][name])["context"]
        rework = review_cost * (1 - accepted / cases)
        rows.append({
            "arm": name, "accepted": accepted, "cases": cases, "mean_words": words,
            "word_cost": word_price * words, "rework_cost": rework,
            "total_cost": cost_per_accepted_draft(words, accepted / cases, word_price, review_cost),
        })
    prompt, _, adapter = rows
    saved_words = prompt["mean_words"] - adapter["mean_words"]
    return {
        "arms": rows,
        "saved_words": saved_words,
        "break_even_word_price": (adapter["rework_cost"] - prompt["rework_cost"]) / saved_words,
    }


def print_costs(report):
    print("teacher scores: 2, 1, 0; soft targets = softmax(scores / temperature)")
    for temperature in (1, 2):
        targets = soft_targets([2, 1, 0], temperature)
        print(f"temperature {temperature}: " + ", ".join(f"{p:.6f}" for p in targets))
    fixed, provider, self_hosted = 120.0, 0.02, 0.005
    volume = fixed / (provider - self_hosted)
    print(
        f"constructed monthly hosting: fixed ${fixed:.2f}; provider ${provider:.3f}/draft; "
        f"self-hosted variable ${self_hosted:.3f}/draft"
    )
    print(
        f"break-even volume: {volume:.0f} drafts/month; provider ${provider * volume:.2f}; "
        f"self-hosted ${fixed + self_hosted * volume:.2f}"
    )
    word_price, review_cost = 0.001, 4.0
    costs = adaptation_costs(report, word_price=word_price, review_cost=review_cost)
    print(f"constructed word price ${word_price:.3f}; human redo ${review_cost:.2f}; context words stand in for tokens")
    print("label agreement stands in for acceptance as written; every rejected draft is redone once")
    print("arm | accepted / cases | mean words | word cost $ | rework cost $ | cost per accepted draft $")
    for row in costs["arms"]:
        print(
            f"{row['arm']} | {row['accepted']}/{row['cases']} | {row['mean_words']:.6f} | "
            f"{row['word_cost']:.6f} | {row['rework_cost']:.6f} | {row['total_cost']:.6f}"
        )
    price = costs["break_even_word_price"]
    print(f"adapter saves {costs['saved_words']:.6f} words/draft; break-even word price ${price:.9f} ({100 * price:.6f} cents)")


def print_lowrank(report):
    print(
        f"seed {report['seed']}; frozen {report['width']}-by-{report['width']} layer; needed rank {report['needed_rank']}"
    )
    print(
        f"training {report['n_train']}; held-out {report['n_test']}; noise SD {report['noise_sd']:g}; noise floor {report['noise_floor']:.4f}"
    )
    print(
        f"all methods: {report['steps']} steps, step size {report['step_size']:g}; adapter constant {report['nu']}"
    )
    print("update | trainable numbers | training MSE | held-out MSE")
    for row in report["results"]:
        print(
            f"{row['method']} | {row['trainable']} | {row['train_mse']:.6f} | {row['test_mse']:.6f}"
        )
    print(f"rank 1 at least twice noise floor: {report['rank_1_misses']}")
    print(f"ranks 2, 4, 8 within 10% of full: {report['within_full_10_percent']}")


def build_experiment(seed=7):
    original = generate(seed)
    windows = time_split(original.requests, CUTS, timedelta(days=LABEL_POLICY.window_days))
    golden = freeze(original, windows.pop("golden"), LABEL_POLICY)
    development = windows["training"] + windows["validation"]
    by_id = {d.request_id: d for d in original.correct}
    training = [by_id[r.request_id] for r in windows["training"]]
    drafts = [by_id[r.request_id] for r in development]
    data = original.model_copy(update={"requests": development, "correct": drafts})
    del original, by_id

    # Replay the documented historical access, never present the set as untouched.
    probe = {
        "entry": 1,
        "set": golden.manifest["name"],
        "reader": "brief-writer probe",
        "purpose": "read golden labels, outcomes and policy versions for brief expectations",
        "date": "2026-10-10",
        "brief": BRIEF,
    }
    golden.log.append(probe)
    events = [{"event": "access", **probe}]

    # Chapter 11's retrospective policy-text corpus is independent of request labels.
    encoder = CountEncoder().fit([clause.text for clause in policy_clauses(data.policies, CUTS[0])])
    index = dated_index(data.policies, encoder)
    prompt = StructuredPrompt()
    commission = AdaptationCommission(
        label_policy=LABEL_POLICY.version,
        seed=seed,
        prompt_sha256=prompt.fingerprint,
        index_sha256=fingerprint(encoder.manifest()),
        development_sha256=data_fingerprint(development, drafts),
        golden_sha256=golden.manifest["sha256"],
    )
    events.append({"event": "commission", "date": "2026-10-10", "sha256": commission.sha256})

    june_terms = fit_adapter_terms(training, data.policies)
    july = windows["validation"]
    july_labels = [d for d in drafts if d.request_id in {r.request_id for r in july}]
    july_states = read_evidence(data, july)
    prompt_july = run_arm(prompt_arm(prompt, index=index), data, july, states=july_states)
    adapter_july = run_arm(
        Arm("adapter", prompt, june_terms, index=index), data, july, states=july_states
    )
    trials = []
    for wording, k in (("policy", 1), ("policy", 2), ("customer", 1), ("customer", 2)):
        candidate = retrieval_arm(index, k, wording, prompt)
        runs = run_arm(candidate, data, july, states=july_states)
        trials.append({"wording": wording, "k": k, "correct": sum(agreement(runs, july_labels))})
    chosen = max(trials, key=lambda row: row["correct"])  # order fixes ties on July
    terms = fit_adapter_terms(drafts, data.policies)
    settings = {
        "prompt": {"template": prompt.version},
        "retrieval": {**chosen, "chosen_on": "July"},
        "adapter": {"training_drafts": len(drafts), "version": terms.version},
    }
    arms = {
        "prompt": prompt_arm(prompt, index=index),
        "retrieval": retrieval_arm(index, chosen["k"], chosen["wording"], prompt),
        "adapter": Arm(
            "adapter", prompt, terms, index=index, training_ids=tuple(d.request_id for d in drafts)
        ),
    }

    # Exactly one new consultation; the historical probe is entry 1.
    spent = golden.consult(
        "adaptation-demo", "apply the declared rule; reuse spent cases as regression checks"
    )
    events.append({"event": "access", **golden.log[-1]})
    requests, labels = [r for r, _ in spent], [d for _, d in spent]
    states = read_evidence(data, requests)
    runs = {name: run_arm(arm, data, requests, states=states) for name, arm in arms.items()}
    matches = {name: agreement(rows, labels) for name, rows in runs.items()}
    record = selection_record(commission, golden, events, runs, labels, settings)
    suite = RegressionSuite.build(data, spent, golden.manifest, index)
    suite_runs = {name: suite.run(arms[name]) for name in ("prompt", "adapter")}
    model_choice = {
        "family": "decoder",
        "budget": {
            "window": WINDOW,
            "shares": list(commission.shares),
            "margin": WINDOW - sum(commission.shares),
            "over_budget": "refer; never drop shipment events",
        },
        "cost": {
            "count_unit": "words, standing in for tokens",
            "context_words_per_draft": words_per_draft(runs["prompt"])["context"],
            "draft_share": DRAFT_SHARE,
            "latency_and_price": "not measured by a scripted drafter",
        },
        "adaptation_selection_record": record.sha256,
    }
    return {
        "seed": seed,
        "data": data,
        "windows": windows,
        "golden": golden,
        "spent_rows": spent,
        "access_log": list(golden.log),
        "events": events,
        "commission": commission,
        "record": record,
        "model_choice_record": model_choice,
        "encoder": encoder,
        "index": index,
        "arms": arms,
        "runs": runs,
        "agreements": matches,
        "correct": {name: sum(values) for name, values in matches.items()},
        "july_correct": {
            "prompt": sum(agreement(prompt_july, july_labels)),
            "retrieval": chosen["correct"],
            "adapter": sum(agreement(adapter_july, july_labels)),
        },
        "july_trials": trials,
        "june_terms": june_terms,
        "adapter_terms": terms,
        "training_drafts": drafts,
        "drafts_needed": drafts_needed(drafts, data.policies),
        "suite": suite,
        "suite_runs": suite_runs,
    }


def print_results(record, compact=False):
    """The full experiment table, or the book's compact quality and safety table.

    compact=True prints the same rows under short headings, with the exact
    chance to two significant digits, so that the table fits a printed page.
    """
    from demos.output_wrap import print_wrapped

    output = print_wrapped if compact else print
    if compact:
        output(
            "| Arm | Agree. 95% | Discordant | Lead 97.5% | Chance | Forbidden | Bound |"
        )
    else:
        output(
            "| Arm | Agreement (95% Wilson interval) | Only challenger / only prompt | Lead (97.5% interval) | Exact chance | Forbidden / cases | Clean-run upper bound | Context words | Draft words |"
        )
    output("|---|---|---|---|---|---|---|" if compact else "|---|---|---|---|---|---|---|---|---|")
    for row in record.evidence:
        low, high = row["agreement_interval"]
        card = row["paired"]
        discordant = "--" if card is None else f"{card['n_plus']} / {card['n_minus']}"
        difference = (
            "--"
            if card is None or card["n_plus"] + card["n_minus"] == 0
            else f"{card['lead']:.3f} [{card['interval'][0]:.3f}, {card['interval'][1]:.3f}]"
        )
        digits = ".2g" if compact else ".6g"
        chance = "--" if card is None else format(card["exact_chance"], digits)
        bound = "--" if row["zero_failure_bound"] is None else f"{row['zero_failure_bound']:.4f}"
        resources = "" if compact else f" {row['words']['context']:.2f} | {row['words']['draft']:.2f} |"
        output(f"| {row['arm']} | {row['correct']}/{row['cases']} [{low:.3f}, {high:.3f}] | {discordant} | {difference} | {chance} | {row['forbidden']}/{row['cases']} | {bound} |" + resources)


def print_resources(record):
    print("| Arm | Context words | Draft words |")
    print("|---|---|---|")
    for row in record.evidence:
        print(f"| {row['arm']} | {row['words']['context']:.2f} | {row['words']['draft']:.2f} |")


def print_experiment(report):
    commission, record = report["commission"], report["record"]
    print("chronological record:", json.dumps(report["events"], sort_keys=True))
    print(
        f"golden consultation: {report['access_log'][-1]['entry']}; prior probe {report['access_log'][0]['date']}; result already known"
    )
    print(
        f"windows: training {len(report['windows']['training'])}, July {len(report['windows']['validation'])}, golden {len(report['spent_rows'])}"
    )
    print(
        f"window {commission.window}; shares {', '.join(map(str, commission.shares))}; settings budget {commission.settings_budget} per arm"
    )
    print(
        f"family alpha {commission.alpha:.3f}; two challengers; split alpha {commission.split_alpha:.3f}"
    )
    print(
        f"July agreement: prompt {report['july_correct']['prompt']}/81; retrieval {report['july_correct']['retrieval']}/81; adapter through June {report['july_correct']['adapter']}/81"
    )
    print("July retrieval settings:", json.dumps(report["july_trials"], sort_keys=True))
    print(
        f"adapter refit: {len(report['training_drafts'])} development drafts; terms {report['adapter_terms'].version}"
    )
    print(
        f"September-labeled development drafts: {report['drafts_needed']['2026-09-01']['available']}"
    )
    print_results(record)
    cases = report["spent_rows"]
    adapter = report["runs"]["adapter"]
    for month in (8, 9):
        positions = [i for i, (r, _) in enumerate(cases) if r.at.month == month]
        print(
            f"adapter month {month}: {sum(report['agreements']['adapter'][i] for i in positions)}/{len(positions)}"
        )
    wrong_denials = sum(
        label.outcome == c.Outcome.DENY and not ok
        for (_, label), ok in zip(cases, report["agreements"]["adapter"])
    )
    window_misses = [
        r.request_id
        for (r, label), run in zip(cases, adapter)
        if label.outcome != c.Outcome.DENY
        and isinstance(run.reply, c.Decision)
        and run.reply.reason == "outside_return_window"
    ]
    print(
        f"wrong denials citing July: {wrong_denials}; newly eligible window misses: {len(window_misses)}"
    )
    print("window-miss ids:", ", ".join(window_misses))
    print(
        f"selected: {record.selected}; retrieval {record.decisions['retrieval']}; adapter {record.decisions['adapter']}"
    )
    print("selection record:", record.sha256)
    print("golden manifest:", json.dumps(report["golden"].manifest, sort_keys=True))
    print(
        "words, standing in for tokens; adapter terms looked up whole from the latest version cited by training drafts; no latency or price measured"
    )


def print_suite(report):
    suite = report["suite"]
    print(
        f"spent golden cases {len(suite.golden_ids)}; injected failures {len(suite.injected_ids)}; policy-change cases {len(suite.policy_change_ids)}"
    )
    by_window = {
        "training": {r.request_id for r in report["windows"]["training"]},
        "validation": {r.request_id for r in report["windows"]["validation"]},
        "golden": set(suite.golden_ids),
    }
    print(
        "injected by window:",
        json.dumps(
            {name: len(ids & set(suite.injected_ids)) for name, ids in by_window.items()},
            sort_keys=True,
        ),
    )
    by_id = {r.request_id: d for r, d in zip(suite.requests, suite.labels)}
    changes = {}
    for rid in suite.policy_change_ids:
        version = by_id[rid].policy_citation.split("@")[1].split("#")[0]
        changes[version] = changes.get(version, 0) + 1
    print("policy-change cases by version:", json.dumps(changes, sort_keys=True))
    print(
        f"distinct drafting cases {len(suite.requests)}; frozen retrieval questions {len(suite.questions)}"
    )
    print(
        "checks: schema; outcome/amount/tier; denial's dated citation; no write tool; no forbidden draft"
    )
    print("retrieval check: no previously passing question loses its relevant top-1 clause")
    for name, run in report["suite_runs"].items():
        print(
            f"{name}: {run.verdict}; newly failing {len(run.newly_failing)}; forbidden {len(run.forbidden_ids)}; clean-run bound {run.forbidden_bound:.4f}"
        )
        print(
            f"{name} failures by check:",
            json.dumps({k: len(v) for k, v in run.failures.items()}, sort_keys=True),
        )
        print(
            f"{name} recall@1:",
            "; ".join(f"{wording} {hits}/{n}" for wording, (hits, n) in run.recall.items()),
        )
    september_changes = set(suite.policy_change_ids) & set(suite.golden_ids)
    print(
        f"adapter misses golden policy-change cases: {len(september_changes & set(report['suite_runs']['adapter'].newly_failing))}/{len(september_changes)}"
    )
    print("adapter failing ids:", ", ".join(report["suite_runs"]["adapter"].newly_failing))
    print("suite fingerprint:", suite.fingerprint)
    print("rule: no passing case becomes failing; no forbidden draft; retrieval passes unchanged")
    print("spent by the probe, then reused; next adaptation decision needs untouched cases")


def exercise_fixture(report):
    """Constructed claims on spent ids: omit the three failed parses from scoring."""
    rows = []
    for i, (request, label) in enumerate(report["spent_rows"]):
        wrong = label.model_copy(
            update={
                "outcome": c.Outcome.APPROVE if label.outcome == c.Outcome.DENY else c.Outcome.DENY,
                "reason": "constructed_wrong_draft",
            }
        )
        prompt = label if i < 116 else wrong
        adapter = label if i < 113 or i in (116, 117) else wrong
        rows.append(
            {
                "request": request.model_dump(mode="json"),
                "label": label.model_dump(mode="json"),
                "prompt": json.dumps(draft_fields(prompt)),
                "adapter": "malformed reply" if i >= 118 else json.dumps(draft_fields(adapter)),
            }
        )
    return {
        "scenario": "constructed adaptation report; three parse failures dropped from adapter denominator",
        "reported": {
            "adapter": {"correct": 115, "total": 118},
            "prompt": {"correct": 116, "total": 121},
        },
        "conclusion": "so the adapter wins",
        "cases": rows,
    }


def rescore_exercise_report(fixture):
    from refund_agent.adaptation.experiment import DraftRun

    labels, predictions, parse_failures = [], {"prompt": [], "adapter": []}, 0
    for row in fixture["cases"]:
        request = c.Request.model_validate_json(json.dumps(row["request"]))
        labels.append(c.Decision.model_validate(row["label"]))
        for name in predictions:
            try:
                reply, error = parse_draft(row[name], request), None
            except DraftFormatError as failure:
                reply, error = None, str(failure)
                parse_failures += name == "adapter"
            predictions[name].append(
                DraftRun(request.request_id, reply, row[name], None, "", error)
            )
    matches = {name: agreement(rows, labels) for name, rows in predictions.items()}
    card = paired(matches["prompt"], matches["adapter"])
    return {
        "correct": {name: sum(values) for name, values in matches.items()},
        "total": len(labels),
        "parse_failures": parse_failures,
        "paired": asdict(card),
        "decision": decide(True, card, 0.05),
    }


def exercise_acceptance(report):
    latest = report["data"].policies[-1]
    november = latest.model_copy(
        update={
            "version": "2026-11-01",
            "effective_at": datetime(2026, 11, 1, tzinfo=timezone.utc),
            "return_days": 21,
            "text": latest.text.replace("2026-09-01", "2026-11-01").replace(
                "within 45 days", "within 21 days"
            ),
        }
    )
    future = constructed_requests(november, 20, report["seed"])
    index = dated_index(report["data"].policies + [november], report["encoder"])
    extended = report["suite"].extend(future, index)
    prompt = replace(report["arms"]["prompt"], index=index)
    adapter = replace(report["arms"]["adapter"], index=index)
    prompt_run, adapter_run = extended.run(prompt), extended.run(adapter)
    runs = run_arm(adapter, extended.data, future.requests)
    ids = [r.request_id for r, label in zip(runs, future.correct) if forbidden(r.reply, label)]
    prompt_future = run_arm(prompt, extended.data, future.requests)
    return {
        "requests": len(future.requests),
        "request_ids": [r.request_id for r in future.requests],
        "prompt_correct": sum(agreement(prompt_future, future.correct)),
        "prompt_verdict": prompt_run.verdict,
        "adapter_verdict": adapter_run.verdict,
        "forbidden_ids": ids,
        "forbidden_bound": zero_failure_bound(len(future.requests)) if not ids else None,
        "forbidden_interval": wilson_interval(len(ids), len(future.requests)),
        "zero_failure_bound_if_clean": zero_failure_bound(len(future.requests)),
        "fingerprint_changed": extended.fingerprint != report["suite"].fingerprint,
        "extended_fingerprint": extended.fingerprint,
        "golden_unchanged": extended.golden_manifest == report["golden"].manifest,
        "new_policy_change_cases": len(
            set(extended.policy_change_ids) - set(report["suite"].policy_change_ids)
        ),
        "new_cases_enter_training": 0,
    }


def print_keys(report):
    d_in, d_out, rank = 1024, 512, 4
    count = trainable_numbers(d_in, d_out, rank)
    print(
        f"exercise 3a: {count} numbers; {100 * count / (d_in * d_out):.6f}%; {adapter_bytes(d_in, d_out, rank)} bytes"
    )
    count = 4 * trainable_numbers(512, 512, 16)
    print(f"exercise 3b: {count} of {4 * 512**2}; {100 * count / (4 * 512**2):.2f}%")
    draft_p, context_p = [0.5, 0.8, 0.9], [0.1] * 5
    print(
        f"exercise 3c: masked loss {objective(draft_p):.3f}; with context {objective(context_p + draft_p):.3f} nats"
    )
    base = [True] * 241 + [False] * 20 + [True] * 8 + [False] * 31
    challenger = [True] * 261 + [False] * 39
    card = paired(base, challenger)
    print(
        f"exercise 4: lead {card.lead:.2f}; paired [{card.interval[0]:.4f}, {card.interval[1]:.4f}]; independent [{card.independent_interval[0]:.4f}, {card.independent_interval[1]:.4f}]; exact {card.exact_chance:.4f}; {decide(True, card, 0.05)}"
    )
    rescored = rescore_exercise_report(exercise_fixture(report))
    split = rescored["paired"]
    print(
        f"exercise 5: adapter {rescored['correct']['adapter']}/{rescored['total']}; prompt {rescored['correct']['prompt']}/{rescored['total']}; parse failures {rescored['parse_failures']}; discordant {split['n_plus']}/{split['n_minus']}; lead {split['lead']:.4f}; exact {split['exact_chance']:.4f}; {rescored['decision']}"
    )
    check = exercise_acceptance(report)
    print(
        f"exercise 6: prompt {check['prompt_correct']}/{check['requests']}, extended suite {check['prompt_verdict']}; adapter extended suite {check['adapter_verdict']}; forbidden {len(check['forbidden_ids'])}/{check['requests']}"
    )
    print("exercise 6 forbidden ids:", ", ".join(check["forbidden_ids"]))
    print(
        f"exercise 6 forbidden rate Wilson interval: [{check['forbidden_interval'][0]:.4f}, {check['forbidden_interval'][1]:.4f}]; zero-failure bound inapplicable to this failing run"
    )
    print(
        f"exercise 6 if 20 were clean, upper bound {check['zero_failure_bound_if_clean']:.4f}; policy-change cases added {check['new_policy_change_cases']}; new training cases {check['new_cases_enter_training']}"
    )
    print(
        f"exercise 6 suite fingerprint changed: {check['fingerprint_changed']}; golden manifest unchanged: {check['golden_unchanged']}"
    )
    print("exercise 6 suite fingerprint:", check["extended_fingerprint"])


def figure_data(report):
    cases = [
        {
            "request_id": request.request_id,
            "at": request.at.isoformat(),
            "agreement": {name: matches[i] for name, matches in report["agreements"].items()},
        }
        for i, (request, _) in enumerate(report["spent_rows"])
    ]
    hospital = paired(
        [True] * 166 + [False] * 14 + [True] * 4 + [False] * 16, [True] * 180 + [False] * 20
    )
    return {
        "seed": report["seed"],
        "golden": {
            "change": "2026-09-01",
            "cases": cases,
            "words": {name: words_per_draft(runs) for name, runs in report["runs"].items()},
        },
        "lowrank": lowrank_experiment(report["seed"]),
        "hospital": {
            "lead": hospital.lead,
            "paired": list(hospital.interval),
            "independent": list(hospital.independent_interval),
        },
    }


def write_data(report, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("adaptation_figures.json", figure_data(report)),
        ("adaptation_report_05.json", exercise_fixture(report)),
        (
            "adaptation_selection_record.json",
            {**report["record"].payload(), "sha256": report["record"].sha256},
        ),
        ("adaptation_model_choice_record.json", report["model_choice_record"]),
    ):
        (directory / name).write_text(
            json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )


# region book:fnd.prompting-finetuning-rag.prompt-01 output
import json

import numpy as np

from demos.output_wrap import print_wrapped
from refund_agent.adaptation.arms import prompt_arm
from refund_agent.adaptation.experiment import read_evidence, run_arm
from refund_agent.adaptation.prompt import (
    DraftFormatError,
    StructuredPrompt,
    parse_draft,
)
from refund_agent.data.generate import generate

if __name__ == "__main__":
    data = generate(7)
    request = next(r for r in data.requests if r.request_id == "r-7-0484")
    state = read_evidence(data, [request])[0]
    template = StructuredPrompt()
    run = run_arm(prompt_arm(template), data, [request], states=[state])[0]
    print_wrapped("prompt fingerprint:", template.fingerprint)
    for section in run.rendered.sections:
        shared = " shared" if section.spec.budget == "instructions" else ""
        print_wrapped(
            f"{section.name}: {section.words} words / "
            f"{section.spec.share}{shared}"
        )
        for line in section.text.splitlines():
            if line.startswith("{"):
                line = json.dumps(json.loads(line), indent=2)
            elif section.name == "Order":
                prefix, items = line.split("items ", 1)
                array, end = json.JSONDecoder().raw_decode(items)
                pretty = json.dumps(array, indent=2)
                line = prefix + "items " + pretty + items[end:]
            print_wrapped(line)
    print_wrapped(
        "section totals:",
        json.dumps(run.rendered.words_by_budget(), sort_keys=True),
    )
    print_wrapped("within shares and window:", run.rendered.within_budget())
    print_wrapped("draft:", json.dumps(json.loads(run.text), indent=2))
    parsed = parse_draft(run.text, request)
    malformed = json.loads(run.text)
    malformed["citation"] = "refund_cap_v4#eligibility"
    try:
        parse_draft(json.dumps(malformed), request)
    except DraftFormatError as error:
        print_wrapped("malformed reply refused:", error)
# endregion book:fnd.prompting-finetuning-rag.prompt-01


# region book:fnd.prompting-finetuning-rag.objective-01 output
if __name__ == "__main__":
    print_objective()
# endregion book:fnd.prompting-finetuning-rag.objective-01


# region book:fnd.prompting-finetuning-rag.lora-01 output
from demos.output_wrap import print_wrapped
from refund_agent.adaptation.lora import (
    LowRankAdapter,
    adapter_bytes,
    trainable_numbers,
)

if __name__ == "__main__":
    for width in (512, 4096):
        count = trainable_numbers(width, width, 8)
        print_wrapped(
            f"width {width}, rank 8: {count} of {width**2}; "
            f"{100 * count / width**2:.6f}%; {adapter_bytes(width, width, 8)} "
            f"bytes"
        )
    count = 24 * 4 * trainable_numbers(512, 512, 8)
    full = 24 * (4 * 512**2 + 2 * 512 * 2048)
    print_wrapped(
        f"24 blocks, four projections, rank 8: {count} numbers; {2 * count} "
        f"bytes ({2 * count / 2**20:g} MiB)"
    )
    print_wrapped(
        f"full block weights: {full} numbers; {2 * full} bytes "
        f"({2 * full / 2**20:g} MiB); adapter share {100 * count / full:.6f}%"
    )
    rng = np.random.default_rng(7)
    w0, x = rng.normal(size=(16, 16)), rng.normal(size=(3, 16))
    adapter = LowRankAdapter(w0, r=8, nu=32, seed=7)
    print_wrapped(
        f"rank {adapter.r}, constant {adapter.nu}: scale {adapter.scale:g}"
    )
    print_wrapped(
        "starts at pretrained output:",
        np.array_equal(adapter.forward(x), x @ w0),
    )
    adapter.up[:] = rng.normal(0, 0.1, adapter.up.shape)
    print_wrapped(
        "numerical update rank:", np.linalg.matrix_rank(adapter.merged() - w0)
    )
    print_wrapped(
        "merged equals unmerged within 1e-12:",
        np.allclose(
            adapter.forward(x), x @ adapter.merged(), rtol=0, atol=1e-12
        ),
    )
# endregion book:fnd.prompting-finetuning-rag.lora-01


# region book:fnd.prompting-finetuning-rag.lowrank-01 output
if __name__ == "__main__":
    print_lowrank(lowrank_experiment(7))
# endregion book:fnd.prompting-finetuning-rag.lowrank-01


# region book:fnd.prompting-finetuning-rag.paired-01 output
from refund_agent.adaptation.experiment import paired

if __name__ == "__main__":
    base = [True] * 166 + [False] * 14 + [True] * 4 + [False] * 16
    challenger = [True] * 180 + [False] * 20
    hospital = paired(base, challenger)
    print(
        f"hospital: {hospital.n} notes; baseline {sum(base)}; challenger "
        f"{sum(challenger)}"
    )
    both = sum(a and b for a, b in zip(base, challenger))
    neither = sum(not a and not b for a, b in zip(base, challenger))
    print(f"both complete {both}; neither {neither}")
    print(
        f"discordant: only challenger {hospital.n_plus}; only baseline "
        f"{hospital.n_minus}"
    )
    print(
        f"lead {hospital.lead:.4f}; paired SE {hospital.standard_error:.4f}; "
        f"95% interval [{hospital.interval[0]:.4f}, "
        f"{hospital.interval[1]:.4f}]"
    )
    print(
        f"independent SE {hospital.independent_error:.4f}; 95% interval "
        f"[{hospital.independent_interval[0]:.4f}, "
        f"{hospital.independent_interval[1]:.4f}]"
    )
    print(f"exact two-sided chance: {hospital.exact_chance:.4f}")
    for wins in (5, 6, 7):
        card = paired(
            [False] * 121, [True] * wins + [False] * (121 - wins), alpha=0.025
        )
        print(
            f"{wins} discordant to none in 121: exact "
            f"{card.exact_chance:.4f}; 97.5% interval "
            f"[{card.interval[0]:.4f}, {card.interval[1]:.4f}]"
        )
# endregion book:fnd.prompting-finetuning-rag.paired-01


if __name__ == "__main__":
    report = build_experiment(7)

# region book:fnd.prompting-finetuning-rag.experiment-01 output
if __name__ == "__main__":
    print_experiment(report)
# endregion book:fnd.prompting-finetuning-rag.experiment-01

# region book:fnd.prompting-finetuning-rag.results-01 output
if __name__ == "__main__":
    print_results(report["record"], compact=True)
# endregion book:fnd.prompting-finetuning-rag.results-01

# region book:fnd.prompting-finetuning-rag.resources-01 output
if __name__ == "__main__":
    print_resources(report["record"])
# endregion book:fnd.prompting-finetuning-rag.resources-01

# region book:fnd.prompting-finetuning-rag.suite-01 output
if __name__ == "__main__":
    print_suite(report)
# endregion book:fnd.prompting-finetuning-rag.suite-01

# region book:fnd.prompting-finetuning-rag.keys-01 output
if __name__ == "__main__":
    print_keys(report)
# endregion book:fnd.prompting-finetuning-rag.keys-01


# region book:fnd.prompting-finetuning-rag.quantize-01 output
from demos import adaptation_demo as adaptation

if __name__ == "__main__":
    adaptation.print_quantization(adaptation.quantization_experiment(7))
# endregion book:fnd.prompting-finetuning-rag.quantize-01


# region book:fnd.prompting-finetuning-rag.cost-01 output
from demos import adaptation_demo as adaptation

if __name__ == "__main__":
    adaptation.print_costs(adaptation.build_experiment(7))
# endregion book:fnd.prompting-finetuning-rag.cost-01


# region book:fnd.prompting-finetuning-rag.keys-02 output
from demos import adaptation_demo as adaptation

if __name__ == "__main__":
    word_price, redo = 0.001, 4.0
    print(f"exercise 7 constructed: ${word_price:.3f}/word; ${redo:g}/redo")
    for name, words, accepted in (("prompt", 500, 90), ("adapter", 400, 80)):
        cost = adaptation.cost_per_accepted_draft(
            words, accepted / 100, word_price, redo
        )
        print(f"{name}: {words} words; {accepted}/100 accepted; ${cost:.3f}")
    threshold = redo * ((1 - 80 / 100) - (1 - 90 / 100)) / (500 - 400)
    print(f"exercise 7 break-even word price: ${threshold:.6f}")
# endregion book:fnd.prompting-finetuning-rag.keys-02


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-data",
        action="store_true",
        help="regenerate plot inputs and exercise/selection records",
    )
    args = parser.parse_args()
    if args.write_data:
        write_data(report, Path(__file__).resolve().parent / "data")
