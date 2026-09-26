# Cardiovascular recovery after saturation returns

This is an exploratory, physiologically motivated extension of the original
SpO2 project, frozen before the new recovery classifications or their outcomes.
Earlier project outcomes and failed hypotheses are known. Its statistical
family does not erase the wider program's multiple hypothesis searches.

Hypothesis: persistent HR elevation after a recorded desaturation has resolved
marks a different subsequent risk state from the same HR drift without a
recorded desaturation. A positive desaturation-specific association would
justify mechanistic follow-up; generic tachycardia prognosis would not.
This is not a breath-level reflex measurement, proof of oxygen debt, or a
claim that normalizing HR or oxygen would reduce mortality.

Prior biology motivates the question but limits novelty. Postexercise HR
recovery already predicts mortality ([Cole et al.](https://www.nejm.org/doi/abs/10.1056/NEJM199910283411804)),
and chemoreflex/baroreflex measurements already have prognostic associations
in HF ([HF chemoreflex/baroreflex study](https://www.jacc.org/doi/10.1016/j.jchf.2022.02.006)).
Neither establishes this ICU phenotype; a new HR coefficient alone would
not meet the user's biological-discovery goal.

## Phenotype fixed before outcomes

Use only the original 11,452 eICU/4,711 MIMIC eligible encounters. Choose the
lowest original stay_id per person before assessing exposure direction or
recovery. Reconstruct the original first absolute ≥4-point SpO2 transition,
with positive elapsed time ≤30 minutes. Cases are people whose first original
transition is downward. Do not replace an earlier upward transition with a
later downward one.

For each source, create the array of case (pre-bin, post-bin) pairs, sorted by
stay_id, before looking at recovery or mortality. Assign each original
unexposed person's sham (pre-bin, post-bin) from that array using the integer
SHA256 digest of `PhysioGraph recovery sham v1|source|person_id`, modulo array
length. This fixes comparable observation times without inspecting outcomes.
These are sham times, not matching on clinical severity or random treatment.

The recovery bin is post-bin plus four (about one hour later). Require native
SpO2 observations in all three bins, pre-to-post elapsed time in (0,30], and
post-to-recovery elapsed time in [45,75] minutes, within the original 240-minute
landmark. Require recovery SpO2 within one percentage point of pre-event SpO2.
For sham controls, additionally require post-bin SpO2 within one point of
pre-bin SpO2. No interpolation, alternate bins, carry-forward or later episodes.

Match HR in the pre and recovery bins, requiring its representative timestamp
within five minutes of the corresponding SpO2 timestamp. HR displacement is
(recovery HR − pre HR)/10, a continuous exposure. A recovered saturation bin
does not establish sustained recovery or independently measured arterial
oxygenation. Requiring recovery can itself select on treatment and physiology.

## Prespecified mortality screen and falsifier

Hospital mortality after the original four-hour landmark is the outcome.
Missing mortality remains missing. Report all selection stages and unknown
outcomes. Fit each database separately only if recovered desaturation cases
include ≥50 people, ≥10 deaths and ≥10 survivors, and sham controls include
≥100 people, ≥10 deaths and ≥10 survivors. eICU additionally requires complete
hospital IDs and ≥20 hospitals. These are minimum feasibility floors, not
power calculations or evidence of a clinically meaningful effect.

Use binomial logistic regression with intercept, desaturation indicator,
HR displacement, their interaction, baseline HR/10, (baseline SpO2−95)/5,
(age−65)/10, and male sex. Restrict to known sex and finite age. No variable
selection, nonlinear searches or alternate definitions. This is a minimally
adjusted screen; treatment and illness-severity confounding remain.

The two primary estimands per source are the odds ratio per 10-bpm HR
displacement among desaturation cases, and its ratio to the corresponding
odds ratio among sham controls (the interaction). Use a hospital-cluster
sandwich with finite-cluster correction in eICU and person-level HC0 in MIMIC.
Apply Holm correction to all four primary tests, assigning p=1 to any
non-estimable member. Report nominal 95% intervals separately from Holm p-values.

Advance only if all four effects are positive with Holm p<0.05, and both
within-desaturation point odds ratios are ≥1.5. Failure is not equivalence.
If successful, require a separately frozen severity/treatment adjustment,
sampling and recovery-selection analyses, functional-form checks, and an
independent biological endpoint before considering any discovery claim.
No subgroup or threshold rescue after failure.

Independently verify first episodes, sham hashing, recovery and HR timing in
Python and SQL. Check fit convergence, design and information matrix rank,
score equations, and independently reconstruct sandwich covariance. No patient
rows leave private Drive Data. Execute in the single primary notebook and
label local execution correctly.
