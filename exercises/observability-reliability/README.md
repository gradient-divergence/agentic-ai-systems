# Observability and Reliability exercises

These inputs are constructed. `trace-01.json` comes from the chapter's real
cohort rehearsal through chapter 32, with payments in the simulated ledger.
The remaining fixtures are teaching cases.

Run from this directory with the companion root on `PYTHONPATH`:

```sh
python -m pytest -p no:cacheprovider test_acceptance.py
```

The unmodified `detector_06.py` fails at a market close and misses the twenty
second requirement. Change `rate_at` to supply an hourly rate, with no expected
quotes outside the market session, and rerun. `reference.py` is public.

Starters 1–5 deliberately omit records, confuse policy referrals, use the wrong
traffic window/population, or trust a frozen audit age. The companion's chapter
test checks each starter's designed failure and its passing reference.

Exercise 6 requires both `rate_at` and `quiet_hours` in `detector_06.py`: an hourly
open-market rate, zero outside the session, and the threshold for one expected
false page in 325 open hours. Set `OBSERVABILITY_RELIABILITY_SOLUTION` to your
submission, or to the public `reference.py`, when running `test_acceptance.py`.
