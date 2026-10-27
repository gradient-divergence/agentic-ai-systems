# Data Systems exercises

Work from the companion directory. The six exercises follow the chapter's check,
apply, analyze and build sequence. All settings and records here are constructed.

For exercises 1–5, edit `data_systems_starter.py`. Run one acceptance file:

```bash
python -m pytest exercises/data-systems/test_data_systems_acceptance.py \
  -p no:cacheprovider
```

Each response initially fails its own check. Use the chapter's questions and
the supplied records to explain and correct each response.

For exercise 5, inspect `lineage-05.json`, which supplies the payment's lineage
and stored inputs. Diagnose the planted error and write a reply to the source's
owner, as the chapter asks.

For exercise 6, edit `contract_06.py` against `feed-06.json` and run:

```bash
python -m pytest exercises/data-systems/test_exercise_06.py -p no:cacheprovider
```

The starter fails two tests: its vocabulary is open and its version is unpinned.
It already demonstrates the sign invariant and the exact budget calculation.
Acceptance also checks the budget's inclusive boundary, expired and
future-observed reads, and the new version's effective-date boundary.
The new version is announced seven days before 1 October 2026. The budget uses
30 refunds in 600 hours, a one-sided 95% rate bound, and a 1% tolerance.

Reference solutions are `data_systems_reference.py` and
`data_systems_reference_06.py`. To verify them, run the same acceptance files
separately with `DATA_SYSTEMS_SOLUTION=data_systems_reference.py` and
`DATA_SYSTEMS_CONTRACT=data_systems_reference_06.py`, respectively. The reference
files are companion teaching material, not printed answer files.
