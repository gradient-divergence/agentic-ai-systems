"""Regenerate exercise 5's constructed run, including its one bad conclusion."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from demos.training_control_demo import exercise_record  # noqa: E402

if __name__ == "__main__":
    path = Path(__file__).with_name("run-05.json")
    path.write_text(json.dumps(exercise_record(), indent=2) + "\n")
