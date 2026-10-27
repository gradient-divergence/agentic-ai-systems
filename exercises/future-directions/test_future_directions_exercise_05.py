import json
from pathlib import Path

from future_directions_reference_05 import faults


def test_only_the_reversal_signal_is_wrong():
    directory = Path(__file__).parent
    starter = json.loads((directory / "decision_05.json").read_text())
    reference = json.loads((directory / "future_directions_reference_05.json").read_text())
    assert faults(starter) == ["post-adoption paired cost unobservable"]
    assert not faults(reference)
    starter["reversal"]["signal"] = reference["reversal"]["signal"]
    assert starter == reference
