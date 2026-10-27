#!/usr/bin/env python3
from __future__ import annotations

# Chapter 32 retires the scalar-score listing; see cloud_native_deployment_demo.
from dataclasses import dataclass


@dataclass(frozen=True)
class CanaryMetrics:
    error_rate: float
    latency_p95_ms: int
    cost_per_request: float


@dataclass(frozen=True)
class CanaryGate:
    max_error_rate: float
    max_latency_p95_ms: int
    max_cost_per_request: float


def evaluate_canary(metrics: CanaryMetrics, gate: CanaryGate) -> bool:
    return (
        metrics.error_rate <= gate.max_error_rate
        and metrics.latency_p95_ms <= gate.max_latency_p95_ms
        and metrics.cost_per_request <= gate.max_cost_per_request
    )




def run_canary_demo() -> None:
    gate = CanaryGate(max_error_rate=0.02, max_latency_p95_ms=450, max_cost_per_request=0.12)
    canary_metrics = CanaryMetrics(error_rate=0.008, latency_p95_ms=410, cost_per_request=0.09)
    promote = evaluate_canary(canary_metrics, gate)
    decision = "promote" if promote else "rollback"
    print({"decision": decision, "metrics": canary_metrics})


if __name__ == "__main__":
    run_canary_demo()
