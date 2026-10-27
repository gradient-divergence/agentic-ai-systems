"""The measured rules replacement; all non-assessment tool calls stay the same."""

from refund_agent import contracts as c
from refund_agent.model import Reply, ScriptedModel
from refund_agent.release.extensions import Component, register
from refund_agent.tools import Tools


class RulesClient(ScriptedModel):
    def __init__(self, tools: Tools):
        super().__init__()
        self.tools = tools

    def reply(self, state: c.State) -> Reply:
        state = c._validated(c.State, state)
        if state.phase == c.Phase.ASSESS:
            # The same three reads already populated state, as in baseline().
            # Preserve the supplied risk, policy, authority and scope compositions.
            return self.tools.assess(state)
        return super().reply(state)


NAME = "capstone-blueprint.rules"
SELECTOR = {
    "name": NAME,
    "module": __name__,
    "settings": {},
}


def wrap_model(*, value, tools, candidate, settings, **context):
    if settings:
        raise ValueError(
            "RulesClient has no configurable alternative behavior"
        )
    return RulesClient(tools)


register(
    Component(
        name=NAME,
        version="capstone-blueprint-rules-1",
        module=__name__,
        order=39,
        hooks={"wrap_model": wrap_model},
    )
)
