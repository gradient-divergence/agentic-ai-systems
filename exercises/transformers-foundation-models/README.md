# Sequence Processing, Transformers and Foundation Models: Exercise 12

Implement generation with and without a key/value cache and count the actual attention work.

From the companion root, select your Python submission:

```sh
TRANSFORMERS_SOLUTION="$PWD/exercises/transformers-foundation-models/starter.py" \
  python -m pytest exercises/transformers-foundation-models/test_acceptance.py -p no:cacheprovider
```

Set `TRANSFORMERS_SOLUTION` to another submission's path to check it. The public
reference solution is `reference.py`; select that path to check the reference.
The acceptance file checks the chapter's stated numerical and behavioral conditions;
a person must also read the submitted judgment where the exercise asks for one.
