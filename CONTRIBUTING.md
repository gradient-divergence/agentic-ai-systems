# Contributing

Thank you for reading closely. This repository holds the code that *Foundations of Agentic AI Systems* prints, and every output the book quotes is stored in [`book_outputs/`](book_outputs/). Two rules follow from that, and they shape every change:

1. **The printed outputs must still reproduce.** A demo's output is the book's text. A change that alters a stored output changes what the book says, so it is an erratum, not a refactor.
2. **The dependencies stay pinned.** `uv.lock` and `.python-version` (Python 3.11) are what reproduce the printed numbers, so please don't upgrade packages in a fix.

## Report an error in the book

Check [`ERRATA.md`](ERRATA.md) first; if the error isn't there, [use the erratum form](https://github.com/gradient-divergence/agentic-ai-systems/issues/new?template=erratum.yml). Give the volume, the edition, the chapter and the page, what the book says and what it should say. A source, a calculation or a run of the code helps.

## Report a problem with the code

[Use the code form](https://github.com/gradient-divergence/agentic-ai-systems/issues/new?template=code-problem.yml), with the exact commands from a fresh clone, what happened, and your operating system and Python version.

## Propose a fix

```sh
git clone https://github.com/<you>/agentic-ai-systems.git
cd agentic-ai-systems
uv sync --locked --extra dev
source .venv/bin/activate
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
```

Then, before you open the pull request:

- run the tests of the chapters you touched, for example `python -m pytest tests/test_agentic_loop_demo.py`;
- run `make lint`;
- if a stored output in `book_outputs/` changes, say which one and why in the pull request.

Keep a pull request to one fix. Changes that only restyle code are hard to review against the printed listings and are unlikely to be merged.

## Conduct

Everyone taking part follows the [code of conduct](CODE_OF_CONDUCT.md).
