# Chapter 40 exercises

The cases are constructed. Correct `case_studies_domain_agnostic_starter_01_05.py`
and `units_06.py`, then run each acceptance file separately from `companion/`:

```sh
python -m pytest exercises/case-studies-domain-agnostic/test_exercises_01_05.py -p no:cacheprovider
python -m pytest exercises/case-studies-domain-agnostic/test_exercise_06.py -p no:cacheprovider
```

The untouched starters fail for the copied answers and the missing comparison
of cap and harm dimensions. To check the reference solutions explicitly:

```sh
CASE_STUDIES_DOMAIN_AGNOSTIC_ANSWERS=case_studies_domain_agnostic_reference_01_05.py \
  python -m pytest exercises/case-studies-domain-agnostic/test_exercises_01_05.py -p no:cacheprovider
CASE_STUDIES_DOMAIN_AGNOSTIC_UNITS=case_studies_domain_agnostic_reference_06.py \
  python -m pytest exercises/case-studies-domain-agnostic/test_exercise_06.py -p no:cacheprovider
```

Exercise 6's bank dimensional check passes: cents and dollars both measure
money. Its numeric cap is still copied, which a dimensional check cannot see.
Acceptance also checks equivalent money in different units and different
dimensions with the same unit label. Comparing unit strings fails these cases.
The chapter's main incident packet uses fewer than 50 sessions.
The exercise's separate constructed ceiling says at most 50 (line 489); the
dimensional check does not decide whether that numeric ceiling is appropriate.
The bank rule still holds on its two cost intervals. The combined four-check
verdict requires chapter 39's actual rows and validator after merge.

`insurer_05.json` supplies the independent diagnosis fixture. Forty percent of
fraud findings mature after the copied 30-day window; all mature within 90 days.
The other given answers pass their rate, cost, authority and resource checks.
Serializing this as chapter 39's row type and proving its complete four-check
packet verdict remain integration work; no substitute row validator is shipped.

The data-facts note and captured `keys-01` give the numerical keys. All printed
paths start at `exercises/`, `demos/` or `refund_agent/`.
