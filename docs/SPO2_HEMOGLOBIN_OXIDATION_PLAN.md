# Hemoglobin oxidation after early SpO2 instability

Frozen before patient-level MetHb counts, distributions, changes or exposure associations. Earlier SpO2 findings and negative experiments are known. This is exploratory hypothesis development; its statistical family does not erase previous searches.

Hypothesis: the original early SpO2 instability precedes a larger subsequent increase in recorded methemoglobin fraction than occurs among original unexposed patients. This would be a biochemical association to investigate, not proof of causation, specific ROS/NO flux, or therapeutic benefit.

An [endothelial/cell-free hemoglobin experiment](https://www.sciencedirect.com/science/article/pii/S0167488999001639) observed hemoglobin oxidation during reoxygenation. Its modified cell-free preparation is not intact adult erythrocytes in HF. A [newborn erythrocyte experiment](https://www.nature.com/articles/pr2005720) studied hypoxia/reoxygenation and MetHb formation; age, conditions and timescale limit extrapolation. These are plausibility arguments, not evidence that mild charted ICU desaturations produce the same reaction.

A [2024 sepsis cohort](https://pmc.ncbi.nlm.nih.gov/articles/PMC11497947/) already reports prognostic MetHb associations. A static MetHb–mortality model with a new HF cutoff would not meet the goal. [Nitroglycerin](https://www.sciencedirect.com/science/article/pii/0002914985908860) can alter MetHb; NO donors, other oxidant drugs, transfusions, illness severity and selection for co-oximetry compete with the proposed explanation. [Hydroxocobalamin](https://academic.oup.com/labmed/article/55/1/50/7179524) can interfere with measurements. No endogenous-redox or therapeutic claim is permitted without resolving these alternatives.

## Source and pair selection

Use the original 11,452 eICU/4,711 MIMIC encounters and original first-four-hour SpO2 exposure. Select the lowest original stay_id per person before assay availability. Baseline is the last timestamp in [max(−1440,hospital start),0); follow-up is the first in (240,min(1680,observed hospital follow-up end)]. No replacement if a selected timestamp fails qualification. This orders samples around the recorded ICU window, not around the first lifetime hypoxic episode. Follow-up testing is conditional on survival and clinical care.

Primary eICU assay: native Methemoglobin, labtypeid=7. Primary MIMIC assay: 50814, blood-gas Methemoglobin. MIMIC 52144 (hematology Methemoglobin) is catalogued separately for support only and cannot replace a failed primary. Methemalbumin and carboxyhemoglobin are not substitutes.

The MIMIC dictionary and eICU native assay/unit labels were inspected before this plan, without patient counts or numeric results. eICU system units are percent, but one interface label is g/dL; reject that discordance. Accept explicit percentage labels (%; % total Hgb; % THB; % of Hgb; % of total Hb; % of total), ignoring case/whitespace. Missing eICU interface units may accompany explicit system %. MIMIC requires explicit percent units. Do not infer units from magnitude.

MIMIC linkage uses exact subject/hospital IDs. Recover missing hospital IDs only when exactly one supplied admission contains charttime, including a valid ED registration preceding admission. Establish uniqueness across all admissions before selecting original ICU encounters. Do not reassign discordant explicit IDs. Retain the original eligible-cohort lower bounds.

Copy raw values, result text and comments privately during the single source scan. Do not output numeric distributions or compute changes before timing support is established. At each selected timestamp use the latest recorded entry time; missing entry/revision times fail qualification. Reject conflicting tied latest records and MIMIC timestamps containing multiple distinct nonmissing specimen IDs. Storetime is a recording timestamp, not proof that separate specimens are revisions.

Reject invalid units, explicit inequality/censor markers in the result field, nonfinite numeric values and values outside [0,100] percent. Never replace rejected samples with another timestamp. MIMIC's deidentified result text does not prove uncensored results: the analysis concerns recorded numeric values, with hidden reporting-limit and rounding uncertainty retained as limitations.

## Prespecified biochemical screen

Require at least 100 qualified paired people in each source, including at least 25 exposed and 25 unexposed. eICU also requires complete hospital IDs and at least 20 hospitals. These are feasibility floors, not power calculations or definitions of a clinically meaningful effect. If the joint floor fails, report support/qualification only; do not compute exposure-specific changes.

Primary outcome: within-person follow-up minus baseline recorded MetHb, in percentage points. Primary contrast: mean change in exposed minus mean change in unexposed. The two source contrasts form the statistical family. Report all four group means, both contrasts, nominal 95% intervals and Bonferroni 97.5% intervals from 20,000 bootstrap draws. Resample entire hospitals in eICU (seed 2026090561) and people in MIMIC (seed 2026090562). Require nonzero group denominators in every draw; independently reconstruct 100 draws.

Advance only if both simultaneous intervals are wholly positive. No effect size is asserted here to be clinically meaningful. A positive screen requires separately frozen baseline/time/severity adjustment, medication/transfusion reconstruction, assay/rounding sensitivity and independent clinical-injury validation before a significant-biological-finding claim. No trimming, threshold search, alternate horizon, assay replacement or mortality model may rescue failure.

Independently validate original identities/exposure, admission assignments, pair times, unit/entry qualification and bootstrap reconstruction in SQL/Python. Patient rows stay in private Drive Data. Publish aggregate evidence and actual local execution in the single primary notebook; do not claim Google Colab execution.
