"""Exercise 6: complete audit() on your own late-version register instance."""

from datetime import timedelta

from demos.memory_rag_demo import demonstration_register, training_data, utc
from refund_agent.contracts import Policy
from refund_agent.memory.register import clause_text


def late_register():
    original = demonstration_register()
    # Replay the source announcements so the late insert is appended in recording order.
    register = type(original)()
    data, _ = training_data()
    late = Policy(
        version="2026-06-10",
        effective_at=utc(6, 10),
        return_days=25,
        cap_cents=90000,
        text="refund_cap_v4@2026-06-10: physical goods. "
        f"#eligibility: {clause_text('eligibility', 25)} #cap: {clause_text('cap', 90000)}",
    )
    announcements = [(p.effective_at - timedelta(days=7), p) for p in data.policies]
    announcements.append((utc(6, 17), late))
    for recorded_at, policy in sorted(announcements, key=lambda row: row[0]):
        register.announce(policy, recorded_at)
    return register


def audit(register):
    """Return only training drafts whose decisions differ between knowledge times.

    Each row: request_id, version_then, version_now, right_then, wrong_now.
    right_then tests the draft against the register known at that request's time.
    wrong_now compares outcome and policy reason against the register known now.
    Do not return unchanged drafts; the prescribed late policy is in late_policy_06.json.
    """
    raise NotImplementedError("Query both knowledge times and compare the structured decisions")


if __name__ == "__main__":
    import json

    print(json.dumps(audit(late_register()), indent=2, sort_keys=True))
