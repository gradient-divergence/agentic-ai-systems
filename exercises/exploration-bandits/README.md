# Exploration, Bandits and Exposure Budgets: Exercise 6

Use the supplied `amount_context` in `demos/exploration_bandits_demo.py` to complete
the starter, then compare the runs at caps of 5% and 20%. Deliver the diff, both
reports and logs, and a judgment for each reason and amount band, including whether
ten months support the choice. The function is given; the runs and judgment earn
the grade.

From the companion root, select your Python submission:

```sh
EXPLORATION_BANDITS_SOLUTION="$PWD/exercises/exploration-bandits/starter.py" \
  python -m pytest exercises/exploration-bandits/test_acceptance.py -p no:cacheprovider
```

Set `EXPLORATION_BANDITS_SOLUTION` to another submission's path to check it. The public
reference solution is `demos/exploration_bandits_demo.py` from the companion root; select its absolute path to check the reference.
The acceptance file checks the chapter's stated numerical and behavioral conditions;
a person must also read the submitted judgment where the exercise asks for one.

## Reference judgment

The constructed response rates in `refund_agent/offers/response.py` depend on
reason and notice alone. Splitting by amount therefore leaves the planted best
notice the same in both bands:

| Reason | Under $25 | $25 and over |
|---|---|---|
| `damaged` | `dated` (0.86) | `dated` (0.86) |
| `changed_mind` | `dated` (0.83) | `dated` (0.83) |
| `not_delivered` | `referenced` (0.88) | `referenced` (0.88) |

Those planted rates describe the constructed world. A learned recommendation
needs evidence from the logs. Group matured answers by reason, amount band and
notice; the supplied report pools the two bands. Compare the choices with their
logged probabilities and uncertainty, keeping the fourteen-day delay intact.

Ten months cannot establish all six choices at either cap. The stored learning
run, `book_outputs/_fnd.exploration-bandits.run-01.txt`, has 9,837 requests, only
647 of them `not_delivered` before splitting by amount. Even a 20% learner coin
would expose only about 129 of those requests to all learner choices on average;
5% supplies fewer. That is already far short of the chapter's roughly 550 answers
per notice for distinguishing 0.82 from 0.88. The smaller 0.02 gaps for the other
reasons need still more evidence. The larger cap supplies evidence faster, but
neither cap justifies adopting a winner in every cell after ten months. Keep the
standard notice while the comparison is unresolved; a replay's highest observed
average alone does not authorize a change.
