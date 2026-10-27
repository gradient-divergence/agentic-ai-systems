"""Record calls at the supplied tools boundary; inject one identical timeout per run."""


class FaultTools:
    def __init__(self, tools, tool=None, after=False):
        self.inner = tools
        self.tool = tool
        self.after = after
        self.fired = False
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def call(self, call, state):
        self.calls.append(call.model_dump(mode="json"))
        fire = call.name == self.tool and not self.fired
        if fire:
            self.fired = True
            if not self.after:
                raise TimeoutError("injected before tool action")
        result = self.inner.call(call, state)
        if fire:
            raise TimeoutError("injected after tool action")
        return result

    def record_decision(self, state):
        from refund_agent.contracts import ToolCall

        return self.call(ToolCall(name="audit.append"), state)
