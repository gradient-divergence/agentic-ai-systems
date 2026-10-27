# Decision Strategy and Evidence Requirements, exercise 6

Run `python -m pytest -p no:cacheprovider companion/exercises/data-science-strategy/test_acceptance.py`
from the repository root with the companion environment's Python and the root
and companion directories on `PYTHONPATH`. Set `STRATEGY_REQUIREMENT` to check a
student's JSON file. The default `requirement-06.json` is the reference record.

The fixed run has 598 cases, bound 0.004997 (0.00499705869425 before rounding),
a signature before day 10, a referral threshold saying "no higher," and no promise
to undo paid refunds. Its third measure counts wrongful denials: requests the agent
denied that the person paid, with the payment not found false within 30 days. The
threshold says "no more than the baseline's" on the same requests, and the baseline
is the only comparator. These are exercise acceptance checks.
