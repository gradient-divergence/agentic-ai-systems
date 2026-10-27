# Semi- and Self-Supervised Learning, exercise 6

Keep `self_train`'s signature. Return the fitted model and per-round dictionaries
with `accepted` (candidate count), `share_one`, the original labeled `base_rate`,
`guard` (`stop` or `continue`) and `total` (labels actually added). Print the
candidate count, share, base rate and guard decision for each round. Stop before
adding labels or refitting if the share leaves the inclusive band from 0.5 to 1.5
times the original labeled base rate.

Run the acceptance file explicitly with the companion Python and root/companion
on `PYTHONPATH`. Set `SEMI_SELF_SUPERVISED_SOLUTION` to the student's file, or to
this directory's `reference.py` for the reference solution. The default starter
fails deliberately. Tests cover both band edges, rejection above and below the
band before refitting, the original labeled base rate in later rounds, the
9,277-candidate seed-7 stopped round, repeatability and zero frozen-window
consultations. These tests are outside the default suite.
