# Hemoglobin modification of the existing SpO2 association

Frozen before hemoglobin-by-outcome inspection. Only MIMIC and eICU; both
databases have been explored previously. This is an observational biological
susceptibility test, not an estimate of transfusion benefit or causal oxygen
delivery. The broad anemia–hypoxemia hypothesis is already established in
prior literature; exact statistical novelty would not establish a mechanism.

## Population, exposure and endpoints

Use the unchanged dynamics-eligible HF risk set and latest pre-ICU CBC Hb
selected under the original feasibility protocol plus the documented,
validated missing-hospital-ID linkage amendment. Preserve the explicit-ID-only
selection for sensitivity. Never change source windows or endpoint definitions.
X is the existing first-four-hour absolute SpO2 jump indicator, including rises.
H=(12−Hb[g/dL])/2 is continuous, with no optimized threshold or spline search.
The two primary endpoints are the existing assay-matched 12-hour troponin rise
indicator and hospital mortality, fitted separately in each database. Missing
Hb and outcomes are not imputed. The primary population is patients with
observed pre-ICU Hb, not all HF patients. Troponin is a myocardial-injury proxy,
not adjudicated infarction; discharge mortality is not a fixed-time endpoint.

## Frozen models and scale

Fit source-specific binomial-logit GEE with independence working correlation
and patient-clustered robust covariance. Include intercept, X, H and X*H.
The primary interaction estimand is exp(beta[X*H]): the ratio of the SpO2
exposure odds ratios for each 2 g/dL lower Hb. Do not call it a risk ratio.
Use two-sided Wald inference and Holm correction over exactly four primary
interaction tests (two endpoints by two databases). Non-estimable tests occupy
their family slot with p=1. No pooled rescue of discordant results.

Covariates, all fixed: age/10, recorded male sex and unknown-sex flag;
(mean SpO2−95)/5; proportion of observed SpO2 bins below 90%; log(1+number of
source SpO2 readings); mean HR/20; mean MAP/20; log(creatinine), log(lactate),
maximum documented FiO2/20; documented invasive ventilation and documented
vasoactive/inotropic infusion in ICU minutes [0,240). Also include H times
centered mean SpO2 and H times hypoxemic-bin proportion to distinguish a
dynamics interaction from modification of absolute oxygenation. Do not
substitute discharge diagnoses as prospectively measured illness severity.

For continuous missing covariates use source-and-analysis-population median
plus a missingness indicator, including age if necessary. All-missing columns
are assigned zero; omit only constant columns (other than the intercept),
never X, H, or X*H. No collinearity-driven variable selection; rank deficiency
or invalid covariance makes a model non-estimable. Primary model requires
100 events, 100 non-events, 100 distinct persons and Hb SD >=0.5 g/dL;
report counts, full design dimension, events per parameter, convergence,
rank, condition number, robust SE and uncertainty. These are minimal
estimability checks, not a claim of adequate interaction power. A wide CI
that includes important amplification is inconclusive, not evidence of none.

At Hb=8 and 12 g/dL, standardize fitted risks over the same observed covariate
distribution at X=0 and X=1, updating all H interactions. Report risk
differences and their difference, with delta-method robust 95% CIs. Compute
these only if each of the four actual Hb neighborhoods (8±1 and 12±1, by X)
contains >=20 patients. These conditional model predictions are associative;
setting Hb in a regression is not an intervention on anemia or transfusion.

## Treatment and physiology source rules

Use existing production-quality raw SpO2 bins and unchanged first-four-hour
MAP, HR, creatinine and lactate definitions. Existing ICU laboratory covariates
have substantial missingness; no unvalidated new admission linkage for them.
FiO2 is a documented inspired oxygen fraction, converted from 0.20–1 to percent
and retained only at 21–100%. MIMIC dictionary labels must match inspired O2
fraction or FiO2, excluding ECMO/TandemHeart contexts and warning=1 records.
eICU respiratory labels must match FiO2/fraction inspired oxygen/inspired O2/
oxygen concentration. Preserve and audit raw labels and invalid-value counts.

MIMIC invasive ventilation requires procedure item 225792 overlapping [0,240).
eICU requires a literal pipe-delimited mechanical ventilation treatment
segment, excluding noninvasive ventilation and ventilator weaning. Generic
respiratory modes alone cannot establish an invasive airway.
MIMIC vasoactive drugs: 221906,221289,221662,221653,221749,222315,221986,
positive rate, overlapping [0,240), excluding Rewritten records. eICU uses
positive drugrate (or infusionrate if drugrate is missing) with exact drug/
brand word matches for norepinephrine/Levophed, epinephrine/adrenaline,
dopamine, dobutamine/Dobutrex, phenylephrine/Neo-Synephrine, vasopressin,
milrinone/Primacor at infusionoffset in [0,240). No dose-equivalence inference.
These are documentation indicators; absence of documentation is not confirmed
absence of treatment. eICU infusionoffset is entry timing. Context overlaps
the exposure window, so adjustment can include responses to instability.

## Sensitivity and interpretation

Repeat the fixed model using explicit-ID-only Hb selection. Repeat without
care-context covariates to reveal adjustment sensitivity, without promoting
that reduced model to primary. Report Hb measurement/store/revision timing;
exclude records entered after ICU admission as a separate sensitivity, while
acknowledging that eICU revision time is not necessarily first availability.
Compare first-drop and first-rise interactions descriptively if primary
replication warrants follow-up; recovery rises are not a negative control.

For troponin ascertainment, prespecify a sensitivity observation model using
the same design among all Hb-observed patients: five folds grouped by person,
L2 logistic regression C=1, training-fold median imputation and scaling,
random seed 20260905. Use inverse probability weights with probability floor
0.02 and report fraction below the floor, calibration, and Kish ESS overall
and by X. Weighted inference is not credible if >10% target probabilities
are below 0.02 or either exposure group has ESS<50. Missing-at-random remains
untestable and weighting does not repair unavailable confounders.

Advance this candidate only if both databases have positive primary
interactions with Holm p<0.05 for the same endpoint, directionally consistent
sensitivities, and evidence of a substantial association (interaction odds
ratio >=1.5 per 2 g/dL lower Hb, or >=5 percentage-point mortality difference
in differences). This is a research-screen threshold, not proof of clinical
utility. A positive result requires further specification checks and a
prospective validation/intervention strategy. Residual confounding by bleeding,
transfusion, illness severity, oxygen treatment and pulse-oximetry error
precludes a mortality-reduction claim. A negative or imprecise result is
reported unchanged; no thresholds/endpoints are tuned to obtain significance.

All code is embedded in the single primary notebook. Patient-level tables
remain in private Drive Data; publish aggregate outputs only. Record protocol
hash before association computation, source provenance and actual execution
environment. No result from this experiment by itself completes the user's
paradigm-shifting biological-discovery goal.
