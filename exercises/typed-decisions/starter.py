"""Exercise 6: accept or refuse a scripted stand-in, using redrawn July only.

Return a JSON-serializable report with decision, reasons, seed, requests,
brier, reliability, resolution, uncertainty, residual and negation_violations.
The illustrative rule requires REL <= 0.0005 and no negation gap above 1e-9.
Do not read August or September. No decision authorizes a denial or a payment.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from demos.typed_decisions_demo import july_rows  # noqa: E402
from refund_agent.typed import ScriptedVendorModel  # noqa: E402


def accept(rows, model):
    """Measure proper-score parts, probe negation, then apply the declared rule."""
    raise NotImplementedError("write the acceptance test on July and name both reasons for refusal")


if __name__ == "__main__":
    print(json.dumps(accept(july_rows(), ScriptedVendorModel()), sort_keys=True, indent=2))
