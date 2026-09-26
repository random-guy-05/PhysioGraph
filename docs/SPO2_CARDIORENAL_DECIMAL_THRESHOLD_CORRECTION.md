# Creatinine threshold arithmetic correction

After the preliminary paired contrast was fitted, a direct comparison against
decimal arithmetic found that binary floating-point subtraction missed some
creatinine increases exactly at the frozen 0.3 mg/dL boundary. There were six
and nine discrepant labels in the eICU 12/24-hour samples, and 22 and 22 in
MIMIC. Initial models and outputs are preserved under
`research/spo2_cardiorenal/preliminary_float_threshold`.

Correct the new creatinine endpoint using DECIMAL(24,10): peak minus baseline
must be at least 0.3, or peak must be at least 1.5 times baseline. Cross-multiplication
avoids a second division boundary. The inputs in these actual samples differ
from their ten-decimal representations by at most 8.89e-16 mg/dL, far below
their recorded laboratory precision. Validate the corrected labels independently
with Python Decimal on those same input values.

The clinical thresholds, baseline and follow-up windows, paired patient sets,
covariates, models and multiplicity rules do not change. Troponin retains the
existing project's ratio calculation and its labels are checked against the
existing endpoint. This correction repairs arithmetic for the newly introduced
creatinine comparator; it is not a threshold or subgroup selected for significance.
The initial estimates were already seen, so this is explicitly a post-estimate
numerical correction. Only the corrected rerun should be used as the final
paired-contrast result. No corrected significance would by itself establish
cardiac specificity or the requested discovery.
