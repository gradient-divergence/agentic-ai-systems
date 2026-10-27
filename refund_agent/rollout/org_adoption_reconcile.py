"""Match money out and requests in; imported history is not governed execution."""


# region book:fnd.org-adoption.reconcile-01
def reconcile(intake, decisions, ledger, provider, window):
    from refund_agent.rollout.org_adoption_matching import (
        governed_payment,
        index_window,
        window_count,
    )
    from refund_agent.rollout.org_adoption_records import Window

    window = Window.model_validate(window)
    received, entries, receipts, paid, exceptions = index_window(
        intake,
        decisions,
        ledger,
        provider,
        window,
    )
    around_pairs = tuple(
        (p, key)
        for p, key in paid
        if not governed_payment(p, key, receipts, entries)
    )
    affected = {key for _, key in around_pairs}
    governed = sum(
        key not in affected
        and any(
            d.path == "governed" and d.terminal for d in entries.get(key, ())
        )
        for key in received
    )
    around = tuple(p for p, _ in around_pairs)
    return window_count(
        window,
        received,
        entries,
        paid,
        around,
        governed,
        exceptions,
    )


# endregion book:fnd.org-adoption.reconcile-01
