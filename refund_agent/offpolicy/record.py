"""The offline evaluation record, serialized before any simulation truth read."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from refund_agent.offpolicy.estimators import (
    direct_method,
    effective_sample_size,
    improvement_terms,
    ips,
    normal_interval,
    per_record_terms,
    policy_ratios,
    self_normalized_interval,
    self_normalized_ips,
)
from refund_agent.offpolicy.reward_model import CellModel
from refund_agent.offpolicy.rule import decide, exact_check
from refund_agent.offpolicy.support import support_table, supported

LIMITS = (
    "this window's customers, season and logger",
    "human-changed offers excluded; no known action chance",
    "no interference assumed",
    "future drift and delayed outcomes can change value",
    "normal intervals can mislead on heavy ratios",
    "seeded redraws approximate the exact law",
)


def _header(data):
    rule = data["rule"]
    lines = [
        "estimand: " + rule["estimand"],
        f"rule: {rule['estimator']}; interval {rule['level']:.2f}; effective floor {rule['n_eff_floor']}; exact level {rule['exact_level']:.3f}",
        "rule sha256: " + data["rule_fingerprint"],
        "learning: "
        + " to ".join(data["learning_window"])
        + f"; records {data['learning_records']}",
        "evaluation: "
        + " to ".join(data["window"].get("dates", ["constructed", "constructed"]))
        + f"; records {data['records']}; randomized {data['randomized_records']}; overrides apart {data['overrides']}",
        "window sha256: " + data["window"]["sha256"],
        "logger: " + ", ".join(data["logger_versions"]) + f"; seed {data['logger_seeds']}",
        "logger sha256: " + ", ".join(data["logger_fingerprints"]),
        "reward model sha256: " + data["model_fingerprint"],
    ]
    for name, card in data["policies"].items():
        lines.append(
            f"policy {name} sha256: {card['fingerprint']}; upstream {card['source_fingerprint'] or 'constructed'}"
        )
    return lines


def _support_lines(data):
    lines = [
        "support: cell offer records positive_share min_positive matches "
        + " ".join(data["policies"])
    ]
    tables = {
        name: {(r["cell"], r["offer"]): r for r in card["support"]}
        for name, card in data["policies"].items()
    }
    for row in data["policies"]["fixed"]["support"]:
        flags = " ".join(
            str(int(tables[name][(row["cell"], row["offer"])]["acts"]))
            for name in data["policies"]
            if name != "fixed"
        )
        minimum = "undefined" if row["min_positive"] is None else f"{row['min_positive']:.6f}"
        lines.append(
            f"{row['cell']} {row['offer']} {row['records']} {row['positive_share']:.6f} {minimum} {row['matches']} {flags} {int(row['acts'])}"
        )
    return lines


def _estimate_lines(data):
    lines = ["policy judged IPS SN DM DR n_eff"]
    for name, card in data["policies"].items():
        values = " ".join(
            "undefined" if card[key] is None else f"{card[key]:.6f}"
            for key in ("ips", "sn", "dm", "dr", "n_eff")
        )
        lines.append(f"{name} {card['judged']} {values}")
    return lines


def _decision_lines(data):
    lines = []
    for name, card in data["policies"].items():
        if name == "fixed":
            continue
        interval = (
            "undefined"
            if card["interval"] is None
            else "[" + ", ".join(f"{x:.6f}" for x in card["interval"]) + "]"
        )
        check = card["exact"]
        p = "undefined" if check["p_value"] is None else f"{check['p_value']:.6f}"
        improvement = "undefined" if card["improvement"] is None else f"{card['improvement']:.6f}"
        lines.append(
            f"{name} improvement {improvement}; interval {interval}; exact p {p}; {check['method']}; draws {check.get('draws', 0)}; seed {check.get('seed', data['rule']['seed'])}"
        )
        lines.append(
            f"{name}: {card['decision']}; "
            + ("; ".join(card["reasons"]) or "all predeclared checks met")
        )
    return lines


@dataclass
class EvaluationRecord:
    data: dict

    def __post_init__(self):
        self.data = json.loads(json.dumps(self.data, allow_nan=False))

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.data, sort_keys=True, indent=2, allow_nan=False) + "\n")
        return path

    @classmethod
    def read(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def render(self):
        data = self.data
        lines = _header(data) + _support_lines(data) + _estimate_lines(data) + _decision_lines(data)
        lines.append(
            f"access: rule fingerprint before outcomes; outcome reads {data['outcome_reads']}"
        )
        lines.append("autonomy: " + data["autonomy"])
        lines.append("limits: " + "; ".join(data["limits"]))
        return "\n".join(lines)


def _estimates(policy, model, records):
    table = support_table(records, policy)
    if not records or any(row.reward is None for row in records):
        return dict(
            judged=False,
            ips=None,
            sn=None,
            sn_reason="outcomes unresolved",
            dm=None,
            dr=None,
            n_eff=0.0,
            support=table,
        )
    w = policy_ratios(policy, records)
    rewards = [row.reward for row in records]
    return dict(
        judged=supported(table),
        ips=ips(w, rewards),
        sn=self_normalized_ips(w, rewards),
        sn_reason="" if w.sum() else "no positive importance ratios",
        dm=direct_method(policy, model, [r.context for r in records]),
        dr=float(np.mean(per_record_terms(policy, model, records))),
        n_eff=effective_sample_size(w),
        support=table,
        ips_interval=normal_interval(per_record_terms(policy, model, records, "ips")),
        sn_interval=self_normalized_interval(w, rewards),
    )


def _cards(policies, model, usable):
    return {
        name: dict(
            _estimates(policy, model, usable),
            fingerprint=policy.fingerprint,
            source_fingerprint=policy.source_fingerprint,
            policy=policy.record(),
        )
        for name, policy in policies.items()
    }


def _comparison(policy, baseline, model, usable, rule):
    computable = bool(usable) and all(row.reward is not None for row in usable)
    terms = (
        improvement_terms(policy, baseline, model, usable, rule.estimator) if computable else None
    )
    return dict(
        improvement=float(np.mean(terms)) if terms is not None else None,
        interval=normal_interval(terms, rule.level) if terms is not None else None,
        exact=exact_check(
            usable,
            policy,
            baseline,
            rule.seed,
            rule.draws,
            model=model,
            estimator=rule.estimator,
        ),
    )


def _assess_candidates(cards, candidates, baseline, model, usable, window, rule):
    one_logger = len({(r.logger_version, r.logger_fingerprint, r.logger_seed) for r in usable}) == 1
    final = bool(usable) and all(row.final and row.reward is not None for row in usable)
    for name, policy in candidates.items():
        card = cards[name]
        card.update(
            _comparison(policy, baseline, model, usable, rule),
            judged=card["judged"] and cards["fixed"]["judged"],
            baseline_n_eff=cards["fixed"]["n_eff"],
            outcomes_final=final,
            one_logging_policy=one_logger,
            intact=window.intact(),
        )
        card["decision"], card["reasons"] = decide(card, rule)


def _assumptions(cards, model):
    assumptions = []
    for name, card in cards.items():
        for row in card["support"]:
            if row["acts"] and (row["positive_share"] < 1 or not row["eligible"]):
                assumptions.append(
                    dict(
                        policy=name,
                        cell=row["cell"],
                        offer=row["offer"],
                        **model.prediction(row["cell"], row["offer"], supported=False),
                    )
                )
    return assumptions


def _metadata(rows, usable, window, rule, model, learning_window, learning_records):
    return dict(
        version="offpolicy-record-1",
        rule=asdict(rule),
        rule_fingerprint=rule.fingerprint,
        window=window.manifest,
        learning_window=list(learning_window),
        learning_records=learning_records,
        records=len(rows),
        randomized_records=len(usable),
        overrides=sum(row.override for row in rows),
        override_ids=[r.decision_id or r.request_id for r in rows if r.override],
        logger_versions=sorted({r.logger_version for r in usable}),
        logger_fingerprints=sorted({r.logger_fingerprint for r in usable}),
        logger_seeds=sorted({r.logger_seed for r in usable}),
        model_fingerprint=model.fingerprint,
        model_source_fingerprint=model.source_fingerprint,
        access=list(window.log),
        outcome_reads=window.reads,
        autonomy="drafts only; a person executes",
        limits=list(LIMITS),
    )


def evaluate(
    candidates,
    baseline,
    window,
    rule,
    *,
    model=None,
    reader="chapter-21",
    learning_window=("2025-04-01", "2026-02-01"),
    learning_records=0,
):
    model = CellModel() if model is None else model
    policies = {**candidates, "fixed": baseline}
    if "fixed" in candidates:
        raise ValueError("fixed is reserved for the baseline")
    window.journal(
        "rule",
        fingerprint=rule.fingerprint,
        candidates={name: p.fingerprint for name, p in policies.items()},
        model=model.fingerprint,
    )
    rows = window.consult(reader, f"evaluate under predeclared rule {rule.fingerprint}")
    usable = [row for row in rows if not row.override]
    cards = _cards(policies, model, usable)
    _assess_candidates(cards, candidates, baseline, model, usable, window, rule)
    data = _metadata(rows, usable, window, rule, model, learning_window, learning_records)
    data.update(policies=cards, assumptions=_assumptions(cards, model))
    return EvaluationRecord(data)
