"""Choose the starter or reference file explicitly; never substitute on failure."""

import importlib.util
import os
import sys
from fractions import Fraction as F
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests/refund_agent"))
from case_studies_domain_agnostic_fixtures import test_packets as packet_inputs

HERE = Path(__file__).resolve().parent
SOURCE = HERE / os.environ.get("CASE_STUDIES_DOMAIN_AGNOSTIC_UNITS", "units_06.py")
SPEC = importlib.util.spec_from_file_location("case_studies_domain_agnostic_units", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    "domain,form,expected",
    [
        ("refund", "refund", False),
        ("triage", "copy", True),
        ("triage", "redesign", False),
        ("hospital", "copy", True),
        ("hospital", "redesign", False),
        ("bank", "copy", False),
        ("bank", "redesign", False),
    ],
)
def test_caps_match_the_action_harm_dimension(domain, form, expected):
    packet, facts = next((p, f) for p, f in packet_inputs() if (p.domain, p.form) == (domain, form))
    failures = MODULE.check_units(packet, facts)
    assert bool(failures) == expected, (
        f"{domain} {form}: the cap's dimension must be compared with the harm"
    )
    if failures:
        assert failures[0].numbers["cap_unit"] == "cents"
        assert failures[0].numbers["harm_dimension"] in ("sessions", "bookings")


@pytest.mark.parametrize(
    "domain,value,cap_unit,cap_dimension,harm_unit,harm_dimension,inclusive,expected",
    [
        ("refund", 5000, "cents", "money", "dollars", "money", False, False),
        ("refund", 50, "dollars", "money", "cents", "money", False, False),
        ("triage", 50, "count", "sessions", "count", "bookings", False, True),
        ("hospital", 1, "count", "bookings", "count", "sessions", True, True),
        # This dimensional exercise has its own at-most-50 cap.
        ("triage", 50, "sessions dropped", "sessions", "sessions dropped", "sessions", True, False),
    ],
)
def test_dimensions_win_over_unit_spelling(
    domain, value, cap_unit, cap_dimension, harm_unit, harm_dimension, inclusive, expected
):
    form = "refund" if domain == "refund" else "redesign"
    packet, facts = next((p, f) for p, f in packet_inputs() if (p.domain, p.form) == (domain, form))
    cap = packet.caps[0].model_copy(
        update={
            "value": F(value),
            "unit": cap_unit,
            "dimension": cap_dimension,
            "inclusive": inclusive,
        }
    )
    facts = facts.model_copy(
        update={
            "actions": tuple(
                action.model_copy(update={"harm_unit": harm_unit, "harm_dimension": harm_dimension})
                if action.name == cap.action
                else action
                for action in facts.actions
            )
        }
    )
    packet = packet.model_copy(update={"caps": (cap,)})
    failures = MODULE.check_units(packet, facts)
    assert bool(failures) == expected
    if expected:
        assert failures[0].numbers["cap_dimension"] == cap_dimension
        assert failures[0].numbers["harm_dimension"] == harm_dimension
