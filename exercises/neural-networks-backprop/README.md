# Neural Networks and Backpropagation: Exercises 7 and 8

Exercise 7's supplied report and code contain one planted gradient-check error.
For exercise 8, implement `forward` and `backward` in `starter.py` for two hidden
ReLU layers; the expected array names and returned fields are in their docstrings.
The check uses all 300 constructed requests at smooth weights for its table of
gaps, and checks the named corner separately at the hand-set first layer.

From the companion root:

```sh
BACKPROP_SOLUTION="$PWD/exercises/neural-networks-backprop/starter.py" \
  python -m pytest exercises/neural-networks-backprop/test_acceptance.py -p no:cacheprovider
```

Set `BACKPROP_SOLUTION` to your submission's path, or to the public
`reference_solution.py` in this folder to check the reference. The checks cover
every array in double precision at epsilon 0.00001, mask switches, the named unit
at the corner, and the first-order loss prediction at a step of 0.001.
