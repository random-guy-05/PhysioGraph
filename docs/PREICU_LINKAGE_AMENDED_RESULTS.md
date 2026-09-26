# Linkage-only amendment: Hb hypothesis is now testable

The local amended feasibility run completed on 2026-09-05 at 06:53 UTC. It
retrieved actual values for the 2,681 CBC and 2,009 troponin records identified
by the validated, fixed linkage audit. It changed no biological window, assay
choice, value threshold or support gate. Original patient databases and
explicit-ID-only results were preserved.

## Hemoglobin susceptibility

| Database | Original paired support | Amended paired support | Amended with / without SpO2 instability | Minimum screen |
|---|---:|---:|---:|---|
| eICU | 659 | 659 | 250 / 409 | Pass |
| MIMIC | 133 | 516 | 176 / 340 | Pass |

Pre-ICU Hb is now available in 3,450 MIMIC encounters, compared with 1,113 under
explicit-ID-only linkage. eICU remains at 3,984. The minimum of 200 observed
troponin endpoints, with at least 50 exposed and 50 unexposed encounters in
each database, is met. This permits specification of the biological interaction
analysis. It is not a power demonstration or evidence that Hb modifies risk.

## Myocardial-injury ordering

| Database | Original complete trajectories | Amended complete trajectories | Frozen minimum of 100 |
|---|---:|---:|---|
| eICU | 133 | 133 | Pass |
| MIMIC | 23 | 76 | Fail |

MIMIC now has 608 episodes with a pre-episode assay and 132 with serial
pre-episode observations, but only 76 with the required delayed post-episode
measurement. The two-source ordering gate still fails. Troponin trajectory
directions and mortality associations for this candidate remain unexamined.

## Interpretation and next analysis

No Hb-by-outcome association has been computed. Before doing so, freeze the
interaction estimand and functional form, confounder handling, outcome-observation
strategy, multiplicity family, information/precision criteria and sensitivity
analyses. The required clinical context includes absolute saturation, hypoxemic
burden, renal and circulatory state, respiratory support and vasoactive therapy.
Missing pressure and treatment information must remain explicit.

The original ICU-baseline troponin endpoint has not been broadened to include
an earlier ED baseline. eICU normalization is unchanged. Blood-gas Hb remains
excluded. Any positive subsequent result must retain the explicit-ID-only
sensitivity and acknowledge uncertainty in inferred links.

These are overlapping analyses of the same two databases, not independent
replications. No novel biological mechanism, transfusion benefit, or mortality
improvement is established. The full research goal remains unfulfilled.
