"""Constructed candidate whose only fault is approval reuse after proposal change."""

from refund_agent.release.candidate import StaleApprovalTools, build_candidate


def candidate(suite):
    return build_candidate(suite, name="stale-approval")


__all__ = ["StaleApprovalTools", "candidate"]
