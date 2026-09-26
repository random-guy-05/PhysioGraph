# Paired cardiorenal biomarker contrast

This is a post-result falsification analysis of the existing SpO2 interpretation,
not a new confirmatory trial. The original review already contains a secondary
troponin association and inconsistent creatinine associations. It does not
establish cardiac specificity because the observed samples differ. Only existing
MIMIC/eICU HF cohorts and their original four-hour SpO2 exposure are used.
Freeze this specification before examining the new paired-marker contrast.

## Biological question and limits

Is the instability-associated increase in a troponin-rise proxy larger than
the increase in creatinine worsening when both markers are measured in the
same patients over the same window? A positive differential association would
justify closer cardiac investigation. It would not prove ischemia, direct
myocardial oxygen-delivery failure, renal independence, or treatment benefit.

A [2024 multicenter study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11365000/)
found dynamic troponin T changes accompanying creatinine changes in AKI without
an MI diagnosis. This does not show that such troponin changes are solely due
to clearance. Stable serum creatinine cannot exclude early kidney dysfunction;
biomarker kinetics differ. Our assay-matched 1.5-fold troponin rise also does
not require an assay's 99th-percentile reference limit and is not an adjudicated
myocardial-injury or MI diagnosis. Do not relabel this screen as a novel discovery.

## Fixed endpoints and shared population

Use the original dynamics-eligible cohort and absolute adjacent SpO2 jump>=4
percentage points, preserving its 15-minute bins, gap and quality rules.
For each assay and unit, use the last positive troponin collected in ICU
[0,240] minutes and the maximum in (240,960]. Collapse multiple matched assays
by the maximum ratio exactly as in the current endpoint. Troponin rise T is
ratio>=1.5. Values must satisfy the existing 0–1,000,000 plausibility rule;
the baseline is strictly positive. Same-time values are collapsed by median.

Creatinine worsening K uses the last value in [0,240] and maximum in (240,960].
Require explicit mg/dL and values 0.1–30 mg/dL. Define K as an increase>=0.3
mg/dL OR a ratio>=1.5. This is a measured creatinine-worsening proxy, not a
complete KDIGO adjudication or proof of renal hypoperfusion. The source unit
inventory was inspected before locking and contains mg/dL in both databases.

Require both marker pairs and follow-up through minute 960 for every analyzed
encounter, including positive cases. This conditions on continued observation
and cannot represent early deaths/discharges. Do not label incompletely observed
markers as negative. Keep all eligible encounters and account for repeat patients
in uncertainty. Repeat the identical construction through minute 1680 only as
the specified 24-hour sensitivity. Do not search other lags or thresholds.
No mortality outcome is selected in this experiment.

## Paired contrast and adjustment

Define D=T-K, taking values -1,0,1. In each database, first report the full
four-category joint distribution by SpO2 exposure and the crude difference
in mean D. This equals the exposure-associated troponin risk difference minus
the corresponding creatinine risk difference in this shared population.
Do not stratify or adjust on post-landmark creatinine worsening.

The primary estimate is the exposure coefficient in an additive linear model
for D, with patient-clustered sandwich uncertainty and finite-sample correction.
The model is an adjusted association, not a treatment-effect model. Covariates
are fixed: age, male indicator and unknown-sex flag; first-four-hour mean SpO2,
fraction of bins below 90%, log(1+SpO2 readings), mean HR and MAP; log baseline
creatinine and lactate; log baseline troponin and a troponin-T indicator; maximum
documented FiO2, and documented ventilation and vasoactive treatment in [0,240).
The lab covariates use only values collected and available by minute 240.
For troponin covariates choose the last available positive baseline among
assays independently of the post-landmark maximum. Use ng/mL, the units verified
in both source inventories. Do not use post-window test counts as covariates;
report them as observation-process limitations.

Median-impute continuous covariates and add missingness indicators; all-missing
variables and constant columns may be omitted based only on covariates. Scale
nonconstant continuous covariates by their sample SD for numerical stability.
Do not remove a nonconstant covariate to rescue a favorable fit. Require full
design rank, finite covariance, at least 50 distinct patients per exposure arm,
and nonconstant D; otherwise report non-estimability. Linear predictions are
not clinical risks and must not be presented as a clinical risk calculator.

Apply Holm correction to the two primary database exposure coefficients. A
positive differential-association candidate requires positive 95% intervals,
Holm p<0.05, and at least a five-percentage-point point estimate in both databases.
The 24-hour sensitivity cannot rescue a failed primary result. This local
multiplicity rule does not erase earlier exploratory testing across the project.
Even passing is not completion of the user's biological-discovery goal.

Reconstruct troponin labels against the existing endpoint, check baseline and
post-window times, verify the risk-difference identity, and independently
reconstruct the clustered covariance. Publish all failures and missingness.
Keep patient data in private Drive Data and the executable workflow in the
single primary notebook. Label actual execution local unless it runs in Colab.
