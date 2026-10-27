"""Reference solution diagnosis of the constructed decision; not an admission API."""


def faults(record):
    failures = []
    if not record["signed_day"] < record["first_request_day"]:
        failures.append("requirement not signed before evidence")
    if record["rebuild"] != {"allowed": 1, "after": "calibration alone", "judged_on": "window 2"}:
        failures.append("rebuild not permitted by the requirement")
    evidence = record["evidence"]
    if evidence["cost_mean"] >= 0 or evidence["cost_upper_95"] >= 0:
        failures.append("cost criterion failed")
    if any(evidence[k] != "pass" for k in ("calibration", "capacity")) or (
        evidence["release"] != "signed pass" or evidence["wrongful_denials"] != [0, 0]
    ):
        failures.append("counter-measure or release failed")
    reversal = record["reversal"]
    if (
        reversal["target"] != "B6 admitted at stage 6"
        or reversal["owner"] != "decision_owner"
        or (
            reversal["mechanism"] != "watch and rehearsed rollback"
            or reversal["confirmation"]
            != "next 2,000 requests, persisted boundaries and paid-only outcomes"
            or reversal["budget"]
            != "immediate withdrawal to shadow/person, independent of rollback"
        )
    ):
        failures.append("reversal ownership, mechanism or target invalid")
    if reversal["signal"] != "dated paid-only calibration with persisted confirmation":
        failures.append("post-adoption paired cost unobservable")
    return failures
