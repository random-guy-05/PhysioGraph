# Descriptive serial-troponin amendment

This is an explicit departure from the original instruction to stop all
directional analysis after the 100-per-database feasibility gate failed.
That primary gate remains failed (133 eICU / 76 MIMIC complete encounters
after the documented linkage amendment). The prior protocol and its failure
are preserved. This amendment is written after those counts and other project
outcomes were known, but before the pre/post concentration directions were
examined. It does not convert the failed study into a successful primary test.

## Reason and scope

A minimum-size rule limits strong inference, but it does not make an observed
case series uninformative. Describe all available trajectories to challenge
the simple assumption that troponin changes begin only after SpO2 instability.
The observation is concentration change, not injury onset: assay reference
limits, release kinetics, clearance, and selective repeat testing prevent that
stronger interpretation. No hypothesis test, mortality association, treatment
effect, confidence interval or population generalization will be reported.
This is not a rescue of the CVP timing specification, which has only six
complete MIMIC patients and remains closed.

## Fixed computation before reading concentrations

1. Use the completed `preicu_linkage_amendment.duckdb` tables, read-only,
   including the exact 133 eICU and 76 MIMIC complete trajectory encounters.
   Do not add patients or reselect assays based on concentration values.
2. Preserve the original first SpO2 episode, hospital boundary, assay/unit
   choice and time windows. The selected assay is the one with the earliest
   pre-episode measurement, with T before I and lexical unit for ties.
   Pre-episode measurements occupy [episode-720,episode), within the same
   hospital encounter. Their first-to-last separation is at least 60 minutes.
   Post-episode measurements occupy [episode+120,episode+720].
3. Pre ratio = last pre concentration / first pre concentration. Post ratio =
   maximum delayed post concentration / last pre concentration. Retain positive
   values and same-time medians already constructed by the original workflow.
   A rise uses the unchanged 1.5-fold threshold, evaluated with decimal
   cross-multiplication after rounding concentrations to ten decimal places.
   First assert that this representation differs from each input by <1e-12.
   This specifies boundary arithmetic before the first directional calculation.
4. Report the complete 2x2 distribution: neither rise, pre only, post only,
   both. Include all four cells even when zero, and both databases regardless
   of direction. Also report marginal pre/post rise counts and percentages.
   Do not select an interesting subgroup, alternative cutoff, or horizon.
5. Primary reporting is finite-sample description of all complete encounters.
   Report distinct people. A deterministic one-encounter-per-person sensitivity
   retains the complete encounter with smallest SHA256 of
   `20260905:dataset:person_id:stay_id`; this selection is made from identifiers
   before joining concentrations. Repeat the full 2x2 table without inference.
6. Report timing spans and the fraction of last pre specimens whose results
   were available by the episode. Specimen time is not result availability;
   neither retrospectively establishes a deployable real-time warning rule.
7. Independently recompute every selected first/last/peak concentration and
   both decimal rise labels in Python from the same assay/time-filtered records;
   compare with SQL. Preserve private encounter-level evidence under /Data.
   Check input sizes and modification times before and after; hash the source
   laboratory database and this amendment, and retain actual notebook outputs.

Any observed pre-episode rise contradicts an exclusive *measured-concentration*
sequence in those cases. It does not prove when myocardial injury began, why
SpO2 changed, that the measured rise caused it, or what would reduce mortality.
An absent pre-rise does not establish an uninjured myocardium. The selected
repeat-tested series is a small fraction of the original episode cohort.
No result here alone can satisfy the paradigm-shifting biological goal.
