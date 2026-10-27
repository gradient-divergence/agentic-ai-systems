"""Decision-order replay using only final 14-day answers, never false-claim labels."""

import heapq
import math
from dataclasses import dataclass, replace
from datetime import datetime, timezone

import numpy as np

from refund_agent.offers.log import OfferLog, OfferRow
from refund_agent.offers.notices import BASELINE, OFFERS, OfferRequest, eligible
from refund_agent.offers.policies import Capped, EstimateTable, choose
from refund_agent.offers.response import NO_CONTACT_RATE, answers, true_rate

TRAIN_START = datetime(2025, 4, 1, tzinfo=timezone.utc)
TRAIN_END = datetime(2026, 6, 1, tzinfo=timezone.utc)


def training_rows(source=None):
    from demos.supervised_learning_demo import windows
    from refund_agent.data.history import generate_history

    history = generate_history() if source is None else source
    # Inspect dates before even eligibility. Neither held-out inputs nor outcomes
    # reach chapter 5's window rule; only label-final timestamps select this window.
    paid = [row for row in history if TRAIN_START <= row.at < TRAIN_END and eligible(row)]
    selected = windows(paid)["training"]
    return [
        OfferRequest(
            row.request_id,
            row.requester_id,
            row.at,
            row.reason,
            row.amount_cents,
            row.policy_version,
        )
        for row in selected
    ]


@dataclass
class ReplayResult:
    log: OfferLog
    tables: dict[str, EstimateTable]
    regret: tuple[float, ...]
    extra_contacts: tuple[float, ...]
    finished_at: datetime


def _settle(log, tables, pending, as_of):
    while pending and pending[0][0] <= as_of:
        _, _, key, decision, answer = heapq.heappop(pending)
        tables[key].observe(answer, as_of)
        log.append(
            replace(decision, contact_at=answer.contact_at, reward=answer.reward, observed_at=as_of)
        )


def _draw_offer(policy, table, t, offers, probabilities, rng):
    if isinstance(policy, Capped):
        coin = bool(rng.random() < policy.cap)
        index = (
            choose(rng, policy.inner.probabilities(table, t, offers))[0]
            if coin
            else OFFERS.index(policy.baseline)
        )
    else:
        coin, index = True, choose(rng, probabilities)[0]
    return coin, OFFERS[index]


def _draft(row, policy, seed, cap, coin, offers, probabilities, offer, key, table):
    return OfferRow.draft(
        row,
        policy,
        seed,
        offers,
        probabilities,
        offer,
        cap=cap,
        coin=coin,
        table_key=key,
        table_fingerprint=table.fingerprint,
        table_answers=tuple(map(int, table.answers)),
        table_no_contacts=tuple(map(int, table.no_contacts)),
    )


def replay(rows, policy, seed, table_key=lambda row: row.reason):
    rows = sorted(rows, key=lambda row: (row.at, row.request_id))
    if not rows or any(not TRAIN_START <= row.at < TRAIN_END or not eligible(row) for row in rows):
        raise ValueError("replay requires eligible requests decided before 1 June 2026")
    if len({row.request_id for row in rows}) != len(rows):
        raise ValueError("one decision per request required")
    rng = np.random.default_rng(seed)
    log, tables, pending, regrets, extras = OfferLog(), {}, [], [], []
    total_regret = total_extra = 0.0
    cap = policy.cap if isinstance(policy, Capped) else 1.0

    for t, row in enumerate(rows, 1):
        _settle(log, tables, pending, row.at)
        key = table_key(row)
        table = tables.setdefault(key, EstimateTable())
        offers = eligible(row)
        probabilities = policy.probabilities(table, t, offers)
        coin, offer = _draw_offer(policy, table, t, offers, probabilities, rng)
        answer = answers(row)[offer]
        decision = _draft(row, policy, seed, cap, coin, offers, probabilities, offer, key, table)
        log.append(decision)
        heapq.heappush(pending, (answer.matures_at, t, key, decision, answer))
        total_regret += max(NO_CONTACT_RATE[row.reason]) - true_rate(row.reason, offer)
        total_extra += true_rate(row.reason, BASELINE) - true_rate(row.reason, offer)
        regrets.append(total_regret)
        extras.append(total_extra)
    finished_at = max(item[0] for item in pending) if pending else rows[-1].at
    _settle(log, tables, pending, finished_at)
    return ReplayResult(log, tables, tuple(regrets), tuple(extras), finished_at)


def coin_comparison(rows):
    groups = [
        [row for row in rows if row.coin == flag and row.reward is not None]
        for flag in (False, True)
    ]
    counts = [len(group) for group in groups]
    successes = [sum(row.reward for row in group) for group in groups]
    result = {
        "baseline_count": counts[0],
        "learner_count": counts[1],
        "baseline_no_contacts": successes[0],
        "learner_no_contacts": successes[1],
        "approximate": True,
        "difference": None,
        "interval": None,
    }
    if all(counts):
        rates = [k / n for k, n in zip(successes, counts)]
        delta = rates[1] - rates[0]
        half = 1.96 * math.sqrt(sum(p * (1 - p) / n for p, n in zip(rates, counts)))
        result.update(difference=delta, interval=(delta - half, delta + half))
    return result


def _cell(rows, reason, offer):
    from demos.ml_data_evaluation_demo import wilson_interval

    sent = [r for r in rows if r.reason == reason and r.offer == offer]
    matured = [r for r in sent if r.reward is not None]
    k, n = sum(r.reward for r in matured), len(matured)
    return {
        "reason": reason,
        "offer": offer,
        "sent": len(sent),
        "matured": n,
        "no_contacts": k,
        "rate": k / n if n else None,
        "wilson": wilson_interval(k, n) if n else (0.0, 1.0),
        "true_rate": true_rate(reason, offer),
    }


def report(log):
    from demos.causal_inference_demo import requests_per_arm

    rows = log.rows
    if not rows:
        raise ValueError("an offer log is required")
    cells = [_cell(rows, reason, offer) for reason in NO_CONTACT_RATE for offer in OFFERS]
    first, last = min(r.decided_at for r in rows), max(r.decided_at for r in rows)
    months = (last.year - first.year) * 12 + last.month - first.month + 1
    target = requests_per_arm(math.sqrt((0.82 * 0.18 + 0.88 * 0.12) / 2), 0.06)
    waits = {}
    for reason in NO_CONTACT_RATE:
        waits[reason] = {}
        for cell in (cell for cell in cells if cell["reason"] == reason):
            monthly = cell["sent"] / months
            waits[reason][cell["offer"]] = {
                "sent_per_month": monthly,
                "months_to_550": 550 / monthly if monthly else None,
            }
    return {
        "cells": cells,
        "coin": coin_comparison(rows),
        "months": months,
        "sample_size_per_offer": target,
        "waits": waits,
        "regret": sum(max(NO_CONTACT_RATE[r.reason]) - true_rate(r.reason, r.offer) for r in rows),
        "expected_extra_contacts": sum(
            true_rate(r.reason, BASELINE) - true_rate(r.reason, r.offer) for r in rows
        ),
        "budget": 0.01 * len(rows),
    }
