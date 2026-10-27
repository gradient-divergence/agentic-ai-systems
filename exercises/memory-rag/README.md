# Exercise 6: audit a late policy

Complete `audit(register)` in a copy of `starter_06.py`. Keep the prescribed
late-policy register and return only training drafts whose structured decisions
change between the request's knowledge time and 1 October 2026.

Run from the repository root, using the companion Python environment:

```sh
MEMORY_RAG_SOLUTION=/absolute/path/to/solution.py \
PYTHONPATH="$PWD:$PWD/companion" OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
python -m pytest companion/exercises/memory-rag/test_exercise_06.py -p no:cacheprovider
```

The acceptance harness runs your file in a fresh subprocess. Before importing it,
the worker installs a `sys.meta_path` finder that exposes a training-only view of
`demos.memory_rag_demo`. The same view is available as
`companion.demos.memory_rag_demo`, including through relative imports. The guard
stays active during module import, `late_register()` and `audit()`; catching a
guard assertion still rejects the submission.

`training_data()` returns `(data, requests)`: iterate **`requests`**, the 398
training requests before 1 July 2026. Pass `data` to `request_state()`; it provides
their orders, shipments, retrieval snapshots and labels, plus the policy versions
needed to build the register. Copies and serialization contain no held-out rows.
Direct `data.requests` access raises a held-out assertion; its serialized field
is empty. `demonstration_register()`, `utc()`, the starter's contract and register
imports, and `refund_agent.rules` remain available, under either package prefix.

The generator, data, split and golden-set modules are refused under every import
name, as are other companion APIs that could expose rows. Generator calls such as
`demos.memory_rag_demo.generate()` are refused too. File reads cannot open shipped
or cached companion data; `late_policy_06.json` is the public policy-only exception.
The worker also refuses native data-reader imports and new processes. Only the
training payload enters the worker; the parent checks the returned audit and
register state. The chapter demo outside this harness keeps its existing interface
and outputs.

This check is a teaching guard, not a security sandbox: reimplementing the generator
from its source, using `ctypes`, or editing the guard is outside the exercise.

Without `MEMORY_RAG_SOLUTION`, the tests accept an independent computed audit,
reject the unfinished starter and exercise the held-out-access regressions.
