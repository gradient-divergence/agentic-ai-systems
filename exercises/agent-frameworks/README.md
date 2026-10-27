# Agent Frameworks, exercises 6 and 7

Exercise 6 mounts the refund agent on the third runtime in `runner_c.py`.
Complete `mount(tools, model=None, settings=None)` in `starter.py`, preserving
the supplied tools, one call per phase, timeouts and the no-retry rule.

From the companion root:

```sh
AGENT_FRAMEWORKS_SOLUTION="$PWD/exercises/agent-frameworks/starter.py" \
  OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest exercises/agent-frameworks/test_exercise_06.py -p no:cacheprovider
```

The variable must name the file to check. The untouched starter leaves retries
enabled and fails the complete-suite comparison. The other checks cover
refusing two reads before dispatch, actual parallel reads and the runtime's
different defaults. Select `reference.py` instead to check the reference
solution, including both cross-runtime resumes and at most one receipt per
request.

Exercise 7's `harness_07.json` records a repair run with one planted decision
error. Identify the invalid event, explain its effect on acceptance and repair
the rule. Its reference solution is in
`book_outputs/_fnd.agent-frameworks.keys-01.txt`.
