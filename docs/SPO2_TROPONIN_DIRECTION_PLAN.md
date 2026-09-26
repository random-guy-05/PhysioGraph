# Biological direction of the SpO2–troponin association

Prospective specification for a new mechanistic falsification analysis,
2026-09-05 UTC. Patient data are restricted to MIMIC and eICU and the
original PhysioGraph HF cohorts. This is not a replacement threshold for
the failed MAP-synchrony candidate.

The current review's strongest two-source secondary signal is troponin
rise, but its advanced estimates disagree. The biological ordering of
myocardial injury and the first SpO2 episode remains unresolved. Delayed
troponin detection can follow injury that began before the respiratory
signal. Troponin kinetics do not establish infarction mechanism:
[primary clinical kinetics study](https://pmc.ncbi.nlm.nih.gov/articles/8360674/).

## First gate: can the biological ordering be observed?

Retain the production first absolute SpO2 change of at least four percentage
points, in 15-minute medians over [0,240) ICU minutes, with at least three
bins and two transitions separated by no more than 30 minutes. Use the
warning-filtered MIMIC replay and the current eICU periodic data. Do not
condition this question on having MAP measurements.

Collect troponin T and I measurements from the same hospital encounter from
24 hours before ICU admission through 28 hours afterward. Do not include
samples before the hospital encounter began. Preserve assay, unit, specimen
time and result/revision time; the latter are not interchangeable. Reuse
verified extracted post-admission samples and obtain only the missing
pre-admission-to-ICU segment from the original laboratory sources.

At the first SpO2 episode, examine the preceding 12 hours and following
12 hours. A serial pre-episode trajectory requires at least two positive
values of the same assay and unit separated by at least 60 minutes.
Choose the assay/unit with the earliest pre-episode measurement, breaking
ties by troponin T before I and then lexical unit. This selection uses
pre-episode data only. Define pre-episode rise as last/first >=1.5, preserving
the project's existing relative-rise threshold. Record the continuous
ratio as well. A post-episode trajectory requires a specimen at least
120 minutes after the episode and no later than 12 hours after it. The
post-episode ratio uses its maximum and the last pre-episode concentration.

First report only the numbers with an episode, pre-episode assays, serial
pre-episode measurements, and delayed post-episode measurements. Do not
inspect trajectory direction or join mortality in this first gate. At
least 100 exposed encounters with complete pre/post trajectories in EACH
database are required before advancing to direction estimates. If this
gate fails, retain the counts and stop this mechanistic candidate; do not
relax the separation, extend the window, or call one pre-episode sample a
stable trajectory.

If the gate passes, freeze the final directional contrast, uncertainty,
clinical controls, and mortality analysis before opening those outcomes.
Within-person temporal ordering alone cannot establish a treatment effect
or a paradigm-shifting biological discovery. An absent observed pre-episode
rise cannot rule out injury because biomarker appearance is delayed.

## Execution and reporting

Keep the runnable workflow in the existing single primary notebook.
Private patient-level tables remain under Data outside the repository.
Record source fingerprints, exact time/assay rules, exclusions and all failed
support gates. Label local runs accurately. No mortality improvement or
novel biological mechanism is presumed.
