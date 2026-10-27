"""A third offline runtime: several named reads run concurrently by default."""

from concurrent.futures import ThreadPoolExecutor

from refund_agent.runtimes.runner import Runner, RuntimeSettings


class RunnerC(Runner):
    def __init__(self, settings=None, read_tools=None):
        defaults = RuntimeSettings(
            stop_on_no_calls=False, retries=0, errors_to_model=False, max_turns=6
        )
        super().__init__(settings or defaults)
        self.read_tools = (
            set(read_tools)
            if read_tools is not None
            else {"order.lookup", "shipment.status", "policy.retrieve"}
        )

    def _dispatch(self, context, reply, registry, session, fatal_errors):
        if len(reply.calls) > 1 and all(call.name in self.read_tools for call in reply.calls):
            with ThreadPoolExecutor(max_workers=len(reply.calls)) as pool:
                futures = [
                    pool.submit(self.invoke, context, call, registry, session, fatal_errors)
                    for call in reply.calls
                ]
                return [future.result() for future in futures]
        return super()._dispatch(context, reply, registry, session, fatal_errors)
