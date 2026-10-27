# Chapter 33 exercises

`record-05.json` is a constructed, deliberately faulty promotion record. Compare
`candidate-05.json` with `incumbent-05.json`, recompute their fingerprints and
name the evidence its incomplete diff omitted. Its changed suite, listed owner
and correct rollback target are distractors. The index uses a rarity-weighted
encoder. No registry admits this planted record.

Exercise 6 assumes that the global prerequisites have already been judged and
signed in the constructed scenario. Its command checks the additional evidence
the table asks for. It creates no release evidence, promotion signature, payment
or runtime admission. The real production command additionally checks current
code and the actual release journal; its inherited fixture gap is reported by
the chapter demo.

```bash
python exercises/mlops-mle/mlops_mle_check_06.py
python exercises/mlops-mle/mlops_mle_check_06.py --candidate exercises/mlops-mle/b1_06.json
python -m pytest exercises/mlops-mle/test_acceptance.py -p no:cacheprovider
```

Complete `required(changes)` in `requirements_06.py`: match each changed component
and item, including wildcard rows, and return each evidence name once beside the
always-required evidence; refuse a changed item with no matching row. Add the missing runtime/adapter row. The command must
exit 1 with exactly `runtime_swap` unmet for the changed candidate, and 0 for B1;
every other row stays the same. Set `MLOPS_MLE_SOLUTION` to your Python file to
check another submission, or to this folder's public `reference_06.py`.

Exercise 3 uses exact rational collision arithmetic; exercise 4 counts the
three days before the June sign-off. Their captured facts are in
`book_outputs/_fnd.mlops-mle.keys-01.txt`, with the corresponding collision
and 17-version/288-bundle arithmetic in `collision-01.txt`.
