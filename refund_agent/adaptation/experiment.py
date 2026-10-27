"""Matched drafts, a declared adoption rule and chapter 8's selection record."""

import json
import math
from dataclasses import asdict, dataclass, field
from statistics import NormalDist

from demos.data_science_strategy_demo import zero_failure_bound
from demos.ml_data_evaluation_demo import wilson_interval
from refund_agent import contracts as c
from refund_agent.adaptation.prompt import (
    DRAFT_SHARE,
    DraftFormatError,
    count_words,
    draft_fields,
    parse_draft,
)
from refund_agent.rules import decision
from refund_agent.selection import SelectionRecord, fingerprint
from refund_agent.tools import Tools

WRITE_TOOLS = {
    "refund.propose",
    "approval.request",
    "refund.execute",
    "audit.append",
    "payment.send",
}
BRIEF = "docs/plans/world-class-roadmap-2026-10-01/phase2/briefs/prompting-finetuning-rag-brief.md"


@dataclass(frozen=True)
class AdaptationCommission:
    label_policy: str = "refund-label-policy-v1"
    baseline: str = "prompt"
    challengers: tuple[str, ...] = ("retrieval", "adapter")
    windows: tuple[str, ...] = ("training before July", "settings on July", "golden from August")
    seed: int = 7
    model: str = "ScriptedDrafter through ModelClient; identical deterministic decoding"
    measure: str = "outcome, amount and tier; a denial's dated citation"
    window: int = 2048
    shares: tuple[int, ...] = (300, 600, 150, 250, 200, 400)
    settings_budget: int = 4
    alpha: float = 0.05
    adapter_refit: str = "all development drafts before August, after the July check"
    prior_access: str = "brief-writer probe on 2026-10-10; outcome already known"
    acceptance: str = (
        "no forbidden draft; within shares; lead's split-level interval above zero "
        "and exact two-sided discordant chance at most split alpha; tie keeps baseline"
    )
    prompt_sha256: str = ""
    index_sha256: str = ""
    development_sha256: str = ""
    golden_sha256: str = ""
    sha256: str = field(init=False)

    def __post_init__(self):
        if not 0 < self.alpha < 1 or not self.challengers or self.settings_budget < 1:
            raise ValueError(
                "probability alpha, challengers and a positive settings budget required"
            )
        if self.baseline in self.challengers or len(set(self.challengers)) != len(self.challengers):
            raise ValueError("distinct challengers must differ from the baseline")
        object.__setattr__(self, "sha256", "")
        object.__setattr__(self, "sha256", fingerprint(self.payload()))

    @property
    def split_alpha(self):
        return self.alpha / len(self.challengers)

    def payload(self):
        return {k: v for k, v in asdict(self).items() if k != "sha256"}

    def intact(self):
        return self.sha256 == fingerprint(self.payload())


def read_evidence(data, requests):
    """The same three read tools as the baseline, outside Agent; never read labels."""
    orders_by_reference = {}
    for order in data.orders:
        for reference in {order.order_id, order.reference}:
            orders_by_reference.setdefault(reference, []).append(order)
    shipments = {s.order_id: s for s in data.shipments}
    snapshots = {s.request_id: s for s in data.retrievals}
    states = []
    for request in requests:
        # Keep every alias match, including ambiguity, without unrelated payment ledgers.
        orders = orders_by_reference.get(request.order_reference, [])
        scope = {
            "orders": orders,
            "shipments": [shipments[o.order_id] for o in orders if o.order_id in shipments],
            "requests": [request],
            "correct": [],
            "retrievals": [snapshots[request.request_id]]
            if request.request_id in snapshots
            else [],
            "public_scenarios": [],
            "instructor_scenarios": [],
        }
        tools = Tools(data.model_copy(update=scope))
        state = c.State(request=request, trace_id=f"baseline-{request.request_id}")
        for name in ("order.lookup", "shipment.status", "policy.retrieve"):
            result = tools.call(c.ToolCall(name=name), state)
            if not result.ok:
                break
        state.phase = c.Phase.ASSESS
        states.append(state)
    return states


@dataclass(frozen=True)
class DraftRun:
    request_id: str
    reply: object
    text: str
    rendered: object
    evidence_sha256: str
    parse_error: str | None = None


def run_arm(arm, data, requests, *, states=None):
    requests = list(requests)
    states = read_evidence(data, requests) if states is None else states
    if [s.request.request_id for s in states] != [r.request_id for r in requests]:
        raise ValueError("one shared evidence packet per request, in the same order")
    runs = []
    for original in states:
        state = original.model_copy(deep=True)
        evidence_hash = fingerprint(state.model_dump(mode="json"))
        rendered = arm.render(state)
        text, error, reply = "", None, None
        try:
            if not rendered.within_budget():
                reply = decision(state, c.Outcome.REFER, "context_over_budget")
            else:
                reply = arm.client(data, rendered.text).reply(state)
            if isinstance(reply, c.Decision):
                text = json.dumps(draft_fields(reply), separators=(",", ":"))
                reply = parse_draft(text, state.request)
            elif isinstance(reply, str):
                text = reply
                reply = parse_draft(text, state.request)
            elif not isinstance(reply, c.ToolCall):
                raise DraftFormatError("decision or tool call required")
        except (DraftFormatError, TimeoutError) as failure:
            reply, error = None, str(failure)
        runs.append(DraftRun(state.request.request_id, reply, text, rendered, evidence_hash, error))
    return runs


def agreement(predictions, labels):
    """The exact per-case match used by baseline.score_decisions; malformed is wrong."""
    by_id = {}
    for row in predictions:
        rid = row.request_id
        if rid in by_id:
            raise ValueError("duplicate prediction")
        by_id[rid] = row.reply if isinstance(row, DraftRun) else row
    if len({d.request_id for d in labels}) != len(labels) or by_id.keys() != {
        d.request_id for d in labels
    }:
        raise ValueError("score requires exactly one prediction per labelled request")
    return [
        isinstance(by_id[d.request_id], c.Decision)
        and (
            by_id[d.request_id].outcome,
            by_id[d.request_id].amount_cents,
            by_id[d.request_id].tier,
        )
        == (d.outcome, d.amount_cents, d.tier)
        and (
            d.outcome != c.Outcome.DENY or by_id[d.request_id].policy_citation == d.policy_citation
        )
        for d in labels
    ]


def exact_discordant_chance(n_plus, n_minus):
    if type(n_plus) is not int or type(n_minus) is not int or min(n_plus, n_minus) < 0:
        raise ValueError("nonnegative integer discordant counts required")
    n = n_plus + n_minus
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(n_plus, n_minus) + 1)) / 2**n)


@dataclass(frozen=True)
class PairedResult:
    n: int
    n_plus: int
    n_minus: int
    lead: float
    standard_error: float
    interval: tuple[float, float]
    exact_chance: float
    alpha: float
    baseline_rate: float
    challenger_rate: float

    @property
    def independent_error(self):
        a, b = self.baseline_rate, self.challenger_rate
        return math.sqrt((a * (1 - a) + b * (1 - b)) / self.n)

    @property
    def independent_interval(self):
        half = NormalDist().inv_cdf(1 - self.alpha / 2) * self.independent_error
        return self.lead - half, self.lead + half


def paired(baseline, challenger, alpha=0.05):
    baseline, challenger = list(baseline), list(challenger)
    n = len(baseline)
    if not n or len(challenger) != n or not 0 < alpha < 1:
        raise ValueError("matched nonempty cases and probability alpha required")
    if any(value not in (0, 1) for value in baseline + challenger):
        raise ValueError("agreement must be 0 or 1")
    plus = sum(bool(b) and not a for a, b in zip(baseline, challenger))
    minus = sum(bool(a) and not b for a, b in zip(baseline, challenger))
    difference = plus - minus
    lead = difference / n
    se = math.sqrt(plus + minus - difference**2 / n) / n
    half = NormalDist().inv_cdf(1 - alpha / 2) * se
    return PairedResult(
        n,
        plus,
        minus,
        lead,
        se,
        (lead - half, lead + half),
        exact_discordant_chance(plus, minus),
        alpha,
        sum(baseline) / n,
        sum(challenger) / n,
    )


def forbidden(reply, label):
    if isinstance(reply, c.ToolCall):
        return reply.name in WRITE_TOOLS
    return (
        isinstance(reply, c.Decision)
        and reply.outcome == c.Outcome.APPROVE
        and label.outcome == c.Outcome.DENY
        and label.reason in ("amount_over_policy", "outside_return_window", "already_delivered")
    )


def words_per_draft(runs):
    if not runs:
        raise ValueError("draft runs required")
    counts = [r.rendered.words_by_budget() for r in runs]
    average = {key: sum(row[key] for row in counts) / len(runs) for key in counts[0]}
    average["context"] = sum(average.values())
    average["draft"] = sum(count_words(r.text) for r in runs) / len(runs)
    return average


def drafts_needed(drafts, policies):
    """Training availability, not a claimed sample complexity for a language model."""
    versions = [d.policy_citation.split("@")[1].split("#")[0] for d in drafts if d.policy_citation]
    return {
        p.version: {"available": versions.count(p.version), "minimum_for_scripted_upper_bound": 1}
        for p in policies
    }


def decide(eligible, card, alpha=0.025):
    if not math.isclose(card.alpha, alpha):
        raise ValueError("the interval must use the same split alpha as the exact check")
    if not eligible:
        return "sent back"
    if card.lead > 0 and card.interval[0] > 0 and card.exact_chance <= alpha:
        return "accepted"
    return "not adopted"


def _arm_eligibility(runs, labels, baseline):
    safety = {
        name: sum(forbidden(r.reply, label) for r, label in zip(rows, labels))
        for name, rows in runs.items()
    }
    # Within the shares: each context section's, and the draft's own. The context check
    # reserves the draft's share in the window, so a draft within it keeps the total inside.
    eligible = {
        name: safety[name] == 0
        and all(
            isinstance(r.reply, c.Decision)
            and r.parse_error is None
            and r.rendered.within_budget()
            and count_words(r.text) <= DRAFT_SHARE
            for r in rows
        )
        for name, rows in runs.items()
    }
    if not eligible[baseline]:
        raise ValueError("the baseline is ineligible; no arm can be adopted against it")
    return safety, eligible


def _selection_evidence(runs, labels, matches, cards, safety):
    return tuple(
        {
            "arm": name,
            "correct": sum(matches[name]),
            "cases": len(labels),
            "agreement_interval": wilson_interval(sum(matches[name]), len(labels)),
            "forbidden": safety[name],
            "zero_failure_bound": zero_failure_bound(len(labels)) if safety[name] == 0 else None,
            "paired": asdict(cards[name]) if name in cards else None,
            "words": words_per_draft(rows),
        }
        for name, rows in runs.items()
    )


def selection_record(commission, golden, events, runs, labels, selected_settings):
    if not commission.intact() or not golden.intact():
        raise ValueError("commission and frozen set must be intact")
    matches = {name: agreement(rows, labels) for name, rows in runs.items()}
    cards = {
        name: paired(matches[commission.baseline], matches[name], commission.split_alpha)
        for name in commission.challengers
    }
    safety, eligible = _arm_eligibility(runs, labels, commission.baseline)
    decisions = {
        name: decide(eligible[name], card, commission.split_alpha) for name, card in cards.items()
    }
    accepted = [name for name in commission.challengers if decisions[name] == "accepted"]
    selected = max(accepted, key=lambda name: cards[name].lead) if accepted else commission.baseline
    evidence = _selection_evidence(runs, labels, matches, cards, safety)
    return SelectionRecord(
        commission={**commission.payload(), "sha256": commission.sha256},
        receipt="offline scripted adaptation demonstration, 2026-10-10; outcome known in brief",
        deliveries=tuple(
            {"arm": name, "settings": selected_settings.get(name, {}), "eligible": eligible[name]}
            for name in runs
        ),
        checks={"commission_intact": True, "matched_ids": True, "eligible": eligible},
        frozen=golden.manifest,
        log=tuple(events),
        evidence=evidence,
        decisions=decisions,
        selected=selected,
        spent="golden-1: spent by the probe; suite core, no fresh quality estimate",
        dependencies=("model choice record", "structured prompt", "dated policy index"),
        signer="author approved brief defaults on 2026-10-10; offline demonstration only",
    )
