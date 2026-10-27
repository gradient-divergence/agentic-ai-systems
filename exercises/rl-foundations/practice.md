# Reinforcement Learning Foundations: optional practice

Run from the repository root with the companion environment and both the root
and `companion/` on `PYTHONPATH`. These exercises use constructed models only.
The answers come from `companion/book_outputs/_fnd.rl-foundations.keys-01.txt`.

1. A parcel is equally likely to use carrier X or carrier Y. X loses 0.10 of
   parcels and delivers a parcel that is not lost with probability 0.60 each
   day. Y loses 0.02 and delivers with probability 0.25. A claim is $49.99, a
   waiting day costs $3, and day 10 forces payment. Compute the belief after
   five days without a scan. Compare a rule using the carrier and day with the
   best rule using the day alone. Explain why a worse rule can remain stable.
   Use `two_carriers()` in `companion/demos/rl_foundations_demo.py`.

2. The $24.99 parcel is now exercise 6 in the chapter itself.

Numerical answers are in `book_outputs/_fnd.rl-foundations.keys-01.txt`;
exercise 8's public reference record is `reference_solution.py`.
