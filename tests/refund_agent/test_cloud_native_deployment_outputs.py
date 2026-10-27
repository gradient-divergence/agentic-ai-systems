"""Stored publication artifacts must identify the code and actual cadence."""

from datetime import datetime, timedelta
from pathlib import Path

from demos.cloud_native_deployment_demo import candidates

OUTPUTS = Path(__file__).resolve().parents[2] / "book_outputs"
NAMES = ("window-01", "plan-01", "start-01", "exposure-01",
         "rehearsal-01", "rehearsal-02", "keys-01")


def output(name):
    return (OUTPUTS / f"_fnd.cloud-native-deployment.{name}.txt").read_text()


def test_stored_window_identifies_current_selected_implementation():
    suite, candidate = candidates()
    text = output("window-01")
    assert suite.fingerprint in text and candidate.fingerprint in text
    assert "Frozen eligible request identifiers in arrival order: 299" in text
    assert "Terminal decisions awaited: 299" in text


def test_every_saved_output_has_printable_paths_and_width():
    for name in NAMES:
        text = output(name)
        assert text.endswith("\n") and "companion/" not in text
        assert max(map(len, text.splitlines())) <= 79


def test_daily_arm_uses_actual_four_hour_pause_not_the_counterfactual():
    plan = output("plan-01")
    signed = datetime.fromisoformat(plan.split("Signed: morgan ")[1].strip())
    planted = signed + timedelta(seconds=1, days=2) + timedelta(seconds=1)
    daily = output("rehearsal-01").split("cadence: daily\n")[1]
    at = datetime.fromisoformat(daily.split("Exposure stops on: recheck_overdue ")[1].splitlines()[0])
    assert at - planted == timedelta(hours=4)
    assert "Further payments before stop (trigger excluded): 0 ; cents: 0" in daily
    detected = datetime.fromisoformat(daily.split("Fault detected at: ")[1].splitlines()[0])
    assert detected > at
    assert "24h counterfactual" in daily


def test_rehearsal_verdicts_require_every_observed_check():
    names = {"intents_closed_unpaid", "next_cohort_to_person", "post_restart",
             "readable_records", "receipts_confirmed_at_restart", "receipts_kept",
             "reconcile_idempotent", "components_restored", "no_request_paid_twice"}
    for block in output("rehearsal-01").split("Constructed version rollback; ")[1:]:
        checks = {key: value for line in block.splitlines() if ": " in line
                  for key, value in [line.split(": ", 1)] if key in names}
        assert set(checks) == names
        verdict = block.split("Verdict: ")[1].strip()
        assert (verdict == "pass") == all(value == "pass" for value in checks.values())
        assert "Requests paid twice: 0" in block and "Fresh restart: True" in block
