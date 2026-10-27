"""A scripted model that obeys every ACT instruction its context contains.

The fixture grammar is one ACT line per instruction. This does not measure a
live model, and says nothing about channels not represented in its input state.
"""

import shlex

from refund_agent import contracts as c
from refund_agent.model import ScriptedModel


class ObedientModel(ScriptedModel):
    def reply(self, state):
        texts = [state.request.text]
        texts.extend(event.status for event in state.shipment_events)
        if state.policy:
            texts.append(state.policy.text)
        if state.order:
            texts.extend(item.name for item in state.order.items)
        for text in texts:
            for line in text.splitlines():
                if not line.startswith("ACT "):
                    continue
                words = shlex.split(line)
                fields = dict(word.split("=", 1) for word in words[2:])
                if fields["at"] != state.phase.value:
                    continue
                return c.ToolCall(
                    name=words[1],
                    arguments=c.ToolArguments(
                        amount_cents=int(fields["amount"]) if "amount" in fields else None,
                        order_id=fields.get("order"),
                        approval_id=fields.get("approval"),
                        idempotency_key=f"security-privacy-execute-{state.request.request_id}",
                        reason=state.request.reason,
                    ),
                )
        return super().reply(state)
