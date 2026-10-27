"""Executable capture regions, separate from the importable run producers."""

from demos.audits_explanations_integrated_demo import (
    integrated_run,
    print_audit,
    print_explanation,
    print_month,
    print_power,
    print_replays,
)

record = integrated_run()

# region book:fnd.audits-explanations.month-01 output
print_month(record)
# endregion book:fnd.audits-explanations.month-01

# region book:fnd.audits-explanations.audit-01 output
print_audit(record)
# endregion book:fnd.audits-explanations.audit-01

# region book:fnd.audits-explanations.power-01 output
print_power(record)
# endregion book:fnd.audits-explanations.power-01

# region book:fnd.audits-explanations.explain-01 output
print_explanation(record)
# endregion book:fnd.audits-explanations.explain-01

# region book:fnd.audits-explanations.explain-02 output
print_replays(record)
# endregion book:fnd.audits-explanations.explain-02
