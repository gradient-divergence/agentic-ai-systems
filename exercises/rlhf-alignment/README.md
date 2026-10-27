# RLHF and Preference Alignment: Exercise 6

Run the preference experiment, check its fitted-sample identity and distinguish population claims.

From the companion root, select your Python submission:

```sh
RLHF_ALIGNMENT_SOLUTION="$PWD/exercises/rlhf-alignment/starter.py" \
  python -m pytest exercises/rlhf-alignment/test_acceptance.py -p no:cacheprovider
```

Set `RLHF_ALIGNMENT_SOLUTION` to another submission's path to check it. The public
reference solution is `demos/preference_demo.py` from the companion root; select its absolute path to check the reference.
The acceptance file checks the chapter's stated numerical and behavioral conditions;
a person must also read the submitted judgment where the exercise asks for one.
