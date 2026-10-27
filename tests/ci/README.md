# What CI runs

The suite has about 7,300 tests in 130 files, which take hours in one process, so CI runs them file by file
([`run-files.sh`](run-files.sh)), on macOS with Python 3.11, the setup the stored outputs were produced with.

| List | Used by | What it holds |
|---|---|---|
| [`quick.txt`](quick.txt) | the CI workflow, on every push and pull request; `make quick` | the 78 test files that each run in under 30 seconds |
| [`shard-1.txt`](shard-1.txt) to [`shard-5.txt`](shard-5.txt) | the full-suite workflow | all 124 test files CI runs, in five shards of about the same length |
| [`needs-book-repo.txt`](needs-book-repo.txt) | [`conftest.py`](../../conftest.py) | 12 tests that check the code against the book's own repository: its manuscript, figure scripts or git history. In a standalone clone they're skipped, with the reason "requires the book repository" |
| [`known-failures.txt`](known-failures.txt) | [`conftest.py`](../../conftest.py) | 37 tests that fail as of the first edition, in a standalone clone and in the book's repository alike. In a standalone clone they run as expected failures, so they stay visible, and one that starts passing shows as XPASS. Fixes are welcome |

Six test files that touch chapters 17 and 21 run only locally, not in CI: `tests/test_exploration_bandits_demo.py`,
`tests/test_practical_rl_demo.py`, `tests/test_memory_rag_demo.py`, `tests/test_sealed_package_guard.py`,
`tests/refund_agent/test_offers.py` and `tests/refund_agent/test_offpolicy.py`.

The lists were measured on 26 October 2026 from a fresh clone. When a test file is added, add it to a shard, and to
`quick.txt` if it runs in under 30 seconds.
