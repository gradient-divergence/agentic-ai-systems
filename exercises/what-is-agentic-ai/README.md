# What Is Agentic AI?, exercise 6

Run `python companion/exercises/run_reference.py what-is-agentic-ai` with the
companion environment's Python. It verifies the original seed-7 public baseline
(4 approvals, 22 denials, 14 referrals, no execution), applies `reference.patch`
in a temporary package copy and writes `run-06.json`. The changed submission
must produce 2 approvals, 22 denials and 16 referrals without any write tool.

For a student submission, run `test_acceptance.py` from the patched companion
directory with `PYTHONPATH` pointing there and pytest's `pythonpath` option empty.
These acceptance tests are outside the default companion suite.
