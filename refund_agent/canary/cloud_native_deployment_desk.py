"""The generated person's desk shares the request-derived payment key."""

from refund_agent.agent import Agent
from refund_agent.batch_stream.generate import read_sources
from refund_agent.batch_stream.tools import EventTimeTools
from refund_agent.batch_stream.view import ALLOWED, first_complete, shipment_as_of
from refund_agent.canary.cloud_native_deployment_store import PersistentTools, ReadContracts
from refund_agent.sources import ContractedTools


def desk(data, store, requests, *, event_log, candidate, contracts=ReadContracts()):
    kind = type("CloudNativeDeploymentDesk", (PersistentTools, ContractedTools, EventTimeTools), {})
    tools = kind(data, canary_store=store, read_contracts=contracts, event_log=event_log,
                 candidate_record=candidate.record())
    traces = []
    for request in requests:
        trace = Agent(tools).run(request)
        if trace.decision.reason == "shipment_incomplete":
            at = first_complete(event_log, request.at + ALLOWED)
            tools.data = read_sources(data, at).model_copy(update={
                "shipments": [shipment_as_of(event_log, request.order_reference, at)],
            })
            trace = Agent(tools).run(request, at=at)
        traces.append(trace)
    return tools, traces
