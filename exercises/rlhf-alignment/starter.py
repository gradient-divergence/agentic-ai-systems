"""Exercise 6: tune at beta 0.5, fit a Platt map, then check a later window.

Use exercise_windows(seed) for three separate windows. Fit the reference only
on training['labeled'], DPO only on training['pairs'], and the map only on the
calibration window. The later window measures calibration after every fit.
Return the fields the acceptance tests name; never change the most likely cause.
Acceptance targets the fitted-sample identity: trained log-odds within 0.02 of
reference log-odds plus the fitted comparison gap divided by beta. Report the
population discrepancy as a diagnostic, not a pass condition: this seed's
independent samples do not meet the population formula's 0.02 logit tolerance.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from demos.preference_demo import exercise_windows  # noqa: E402


def exercise_six(seed=7):
    training, calibration, later = exercise_windows(seed)
    raise NotImplementedError("train DPO, fit the map on calibration, and report on later")


if __name__ == "__main__":
    print(json.dumps(exercise_six(7), indent=2, allow_nan=False))
