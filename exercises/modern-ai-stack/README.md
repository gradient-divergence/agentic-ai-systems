# Architecture, Components and Contracts, exercise 6

Run `python companion/exercises/run_reference.py modern-ai-stack` with the
companion environment's Python. The runner patches a temporary package copy,
tests the submitted validator and writes `run-06.json`; the runnable agent's
Order contract stays unchanged.

Acceptance covers all generated orders, rejection of the dollar release and
the other three incompatible releases, multi-item quantities, all forty public
baseline decisions and absence of any write-tool call. For a student submission,
run this acceptance file against the patched package as described in chapter 1's
exercise README. These tests are outside the default companion suite.
