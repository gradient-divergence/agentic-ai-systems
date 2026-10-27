# Additional practice, constructed

1. A constructed suite has nine scenarios that always pass and one with pass
   chance 0.5. Compute pass@4 and pass^4 by averaging over scenarios, then from
   the pooled chance 0.95. Explain which two directions pooling errs in.

   Key: scenario means 0.99375 and 0.90625; pooled values 0.99999375 and
   0.81450625. Pooling overstates pass@4 and understates pass^4. Neither is a
   live failure-rate bound.

2. Inspect `shadow-practice.json`. What is wrong with reporting 107 of 120 as
   agreement with an independent person's decision? Which requests may supply a
   blind comparison? What must happen to the forty contaminated records?

   Key: the person saw the agent's decision on forty requests. Agreement on that
   subset, 39/40, cannot be a blind comparator. Report 68/80 on the blind subset
   and the contaminated subset separately. Preserve the screen log, route the
   contaminated records for blind adjudication and do not count a reread as a new
   independent request. The original pooled result is 107/120.
