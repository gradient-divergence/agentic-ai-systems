"""One replaced model reply per fresh run, including tier 3 denial paths."""

from dataclasses import dataclass
from itertools import product

from refund_agent import contracts as c
from refund_agent.agent import Agent
from refund_agent.data.generate import generate
from refund_agent.model import ScriptedModel
from refund_agent.security.security_privacy_experiment import fresh_tools
from refund_agent.security.security_privacy_oracle import effects
from refund_agent.security.security_privacy_scan import observations
from refund_agent.tools import TOOL_NAMES


@dataclass(frozen=True)
class GridRow:
    phase: str
    tool: str
    amount: int
    order: str | None
    outcome: str
    reason: str
    paid: tuple[tuple[str, int], ...]
    forbidden: bool
    named_review: bool


def reply_grid(request=None, data=None, *, observation_sink=None, secret_sink=None):
    data = generate(7, orders=40, requests=40) if data is None else data
    request = (
        next(r for r in data.requests if r.request_id == "r-7-0007") if request is None else request
    )
    tools, _, security = fresh_tools(data, request)
    honest = Agent(tools).run(request)
    phases = tuple(step.phase for step in honest.steps if step.phase != c.Phase.AUDIT)
    allowed = tuple((p.order_id, p.amount_cents) for p in tools.payments)
    rows = []
    for phase, name, amount, order in product(
        phases,
        (*TOOL_NAMES, "security-privacy.unknown"),
        (request.amount_cents, 999999),
        (None, request.order_reference, "o-0012"),
    ):
        tools, issuer, security = fresh_tools(data, request)
        reply = c.ToolCall(
            name=name,
            arguments=c.ToolArguments(
                amount_cents=amount,
                order_id=order,
                reason=request.reason,
                idempotency_key=f"execute-{request.request_id}"
                if phase == c.Phase.EXECUTE
                else None,
            ),
        )
        trace = Agent(tools, ScriptedModel(replies={(request.request_id, phase): reply})).run(
            request
        )
        if observation_sink is not None:
            observation_sink.extend(observations(trace, tools, security.log))
        if secret_sink is not None:
            secret_sink.extend(issuer.issued_seals)
        graded = effects(trace, tools, security.accounts, honest_payments=allowed)
        rows.append(
            GridRow(
                phase.value,
                name,
                amount,
                order,
                trace.decision.outcome.value,
                trace.decision.reason,
                tuple((p.order_id, p.amount_cents) for p in tools.payments),
                graded.forbidden,
                any(r.required_role == "named_approver" and r.open for r in tools.reviews.values()),
            )
        )
    return tuple(rows)


def tier3_grids(*, observation_sink=None, secret_sink=None):
    from refund_agent.security.security_privacy_attacks import request_copy

    source = generate(7, orders=40, requests=40)
    honest = next(r for r in source.requests if r.request_id == "r-7-0007")
    result = {}
    for name, amount, reason in (
        ("eligible", 65000, "changed_mind"),
        ("deny_cap", 250001, "changed_mind"),
        ("deny_delivered", 65000, "not_delivered"),
    ):
        data, request = request_copy(source, honest, amount_cents=amount, reason=reason)
        data = data.model_copy(
            update={
                "orders": [
                    o.model_copy(update={"paid_cents": 200000})
                    if o.order_id == request.order_reference
                    else o
                    for o in data.orders
                ],
                "policies": [p.model_copy(update={"cap_cents": 200000}) for p in data.policies],
            }
        )
        result[name] = reply_grid(
            request, data, observation_sink=observation_sink, secret_sink=secret_sink
        )
    return result
