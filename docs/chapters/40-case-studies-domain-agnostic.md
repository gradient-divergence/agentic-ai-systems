# Chapter 40: Domain-Agnostic Transfer Cases

[Companion README](../../README.md)

The chapter carries the refund agent's launch packet to three other domains, an
incident desk, a hospital's follow-up scheduling and a bank's card holds, and
checks each copy against four transfer checks: clock, coupling, rule and
authority. The code holds typed facts, packet construction, the checks, exact
cost intervals and a constructed hospital allocation. These are requirements and
demonstrations: the companion builds no incident-desk, hospital or banking agent.

## Files

| What | Where |
|---|---|
| Demo | `demos/case_studies_domain_agnostic_demo.py` |
| Code | `refund_agent/transfer/` |
| Data | `demos/data/transfer/` (facts and table cells for each domain, clock figures) |
| Exercises | `exercises/case-studies-domain-agnostic/` |
| Stored outputs | `book_outputs/_fnd.case-studies-domain-agnostic.*.txt` (7 files) |
| Tests | `tests/refund_agent/test_case_studies_domain_agnostic.py` |

## Run it

From the companion folder, after the install in the README. The last output,
`verdicts-01`, judges each packet against chapter 39's launch requirement, read
from the file named in `CASE_STUDIES_DOMAIN_AGNOSTIC_REQUIREMENT`. Chapter 39's
constructed refund requirement reproduces it:

```sh
python -c "from refund_agent.launch.capstone_blueprint_requirement import refund_requirement; print(refund_requirement('B6').model_dump_json())" > requirement.json
CASE_STUDIES_DOMAIN_AGNOSTIC_REQUIREMENT=requirement.json python -m demos.case_studies_domain_agnostic_demo
python -m pytest tests/refund_agent/test_case_studies_domain_agnostic.py
```

Without the variable, the demo prints the six other outputs and then stops with
a `KeyError`. `--write-data` rewrites the files in `demos/data/transfer/`.

## What the output shows

`packets-01` is the chapter's table: the seven launch questions answered for
the refund agent, the incident desk, the hospital and the bank, each answer
marked as refilled, redesigned or the same. The table is read from the
serialized cells, which must agree with each domain's facts before printing.

`triage-01` is the incident desk's clock. With a constant clearing rate of
4.158883 alarms an hour, an alarm read is out of date with chance 0.75 at 20
minutes and 0.875 at 30 minutes; a 1% tolerance allows 8.699742 seconds. The
redesign probes the alarm at restart with an 8-second limit (chance 0.009199).
The refund agent's shipment read, at 0.02 an hour, has a 30.151008-minute budget,
207.944154 times the alarm's.

`hospital-01` books 18 constructed discharges (6 urgent, 12 routine) into 21
slots, 6 of them near. Booking request by request places only 1 of the 6 urgent
patients in time and refers 5; booking the day as a batch places all 6, with a
first routine notification wait of 8 hours. Over 10,000 random orders (seed 7),
request-by-request booking places a mean of 2 urgent patients in time; all six
in time has exact probability 1/18564 and was observed 0 times.

`bank-01` scores a card-level probability of fraudulent use within a constructed
60-day window. The M/M/1 review queue (`mm1(12, 15)` from chapter 37) opens a
review in 20 minutes, during which fraud spends $50. Referral costs $4 a card,
plus $150 for each hour of review if the card is fraudulent; an unstable queue
is refused. The cost-minimizing edges stay exact fractions: clear below
2/175 = 0.0114, hold above 1/6 = 0.1667, refer between, and ties refer. The
copied refund edge, 400/4999 = 0.0800, is the wrong rule for the bank. A hold
stops 8.25 legitimate cards for each fraudulent card stopped; closing an account
needs a level 2 named approver.

`balance-01` sends two 3,000-cent requests against one 4,000-cent order through
the existing refund gate: the first is paid; the second passes assessment but is
refused at execution for balance, so 3,000 cents are paid and 1,000 remain. The
order balance is a shared resource that needs a limit at execution time.

`verdicts-01` gives each packet's verdict: the refund packet passes; the copied
triage, hospital and bank packets hold (on clock, coupling and rule), and each
redesign passes. `keys-01` gives the exercise keys.

## Exercises

`exercises/case-studies-domain-agnostic/` has one starter for exercises 1 to 5
and `units_06.py` for exercise 6, with reference solutions and
`insurer_05.json`. Run each acceptance file separately:

```sh
python -m pytest exercises/case-studies-domain-agnostic/test_exercises_01_05.py
python -m pytest exercises/case-studies-domain-agnostic/test_exercise_06.py
CASE_STUDIES_DOMAIN_AGNOSTIC_ANSWERS=case_studies_domain_agnostic_reference_01_05.py python -m pytest exercises/case-studies-domain-agnostic/test_exercises_01_05.py
CASE_STUDIES_DOMAIN_AGNOSTIC_UNITS=case_studies_domain_agnostic_reference_06.py python -m pytest exercises/case-studies-domain-agnostic/test_exercise_06.py
```

The untouched starters fail on the copied answers and on the missing comparison
of cap and harm dimensions (exercise 6's untouched `units_06.py` fails four
dimensional checks). In exercise 6 the bank's dimensional check passes, since
cents and dollars both measure money, while the bank rule still holds: a units
check cannot validate a copied numeric cap. Its acceptance cases include cents
against dollars and equal unit labels for different dimensions. In
`insurer_05.json`, 40% of fraud findings mature after the copied 30-day label
window; all mature within the required 90 days.

## What it leaves out

The serialized redesign packets and their full launch-row integration with
chapter 39's validator for every domain; the insurer fixture's four-check verdict
is not computed. The main incident scenario uses fewer than 50 sessions, while
exercise 6 separately specifies at most 50; the units check compares dimensions,
not that numeric boundary.
