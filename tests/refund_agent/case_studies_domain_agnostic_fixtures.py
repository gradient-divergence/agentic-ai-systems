"""The merged chapter-39 requirement supplies chapter-40 publication tests."""

from refund_agent.launch.capstone_blueprint_requirement import refund_requirement
from refund_agent.transfer.case_studies_domain_agnostic_data import packets


def test_requirement():
    return refund_requirement("B6")


def test_packets():
    return packets(test_requirement())
