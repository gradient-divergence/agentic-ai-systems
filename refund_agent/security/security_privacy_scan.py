"""Search actual serialized observations for raw, hex and base64 secrets."""

import base64
import json
from dataclasses import dataclass

from refund_agent import contracts as c
from refund_agent.adaptation.prompt import StructuredPrompt


def encode(value) -> bytes:
    if isinstance(value, c.Record):
        value = value.model_dump(mode="json")
    if type(value) is bytes:
        return value
    if type(value) is str:
        return value.encode()
    return json.dumps(value, sort_keys=True, default=str).encode()


@dataclass(frozen=True)
class ScanResult:
    outputs: int
    matches: tuple[tuple[int, int], ...]


def scan(outputs, secrets: tuple[bytes, ...]) -> ScanResult:
    encoded = tuple(encode(o) for o in outputs)
    needles = tuple(
        sorted(
            set(
                form
                for secret in secrets
                for form in (secret, secret.hex().encode(), base64.b64encode(secret))
            )
        )
    )
    if any(not secret for secret in secrets):
        raise ValueError("cannot scan for an empty secret")
    matches = tuple(
        (i, j)
        for i, value in enumerate(encoded)
        for j, needle in enumerate(needles)
        if needle in value
    )
    return ScanResult(len(encoded), matches)


def observations(trace, tools, credential_log):
    rows = [trace, trace.decision, *tools.audits.values(), *credential_log.entries]
    for step in trace.steps:
        state = step.state
        rows.append(
            StructuredPrompt().render(state, state.policy.text if state.policy else "").text
        )
    return tuple(rows)
