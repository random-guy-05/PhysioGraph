# Locked mortality screen for the acid–base candidate

Freeze before selecting mortality for this candidate. The completed outcome-blind
support stage found 1,767 eICU and 757 MIMIC encounters with preceding complete
gases. Joint instability/respiratory-alkalemia cells contain 51 and 13 encounters,
respectively. These counts were known when choosing the following conservative
screen; mortality counts and risk contrasts were not known.

Keep the original SpO2 exposure and the frozen preceding-state definitions.
Let X denote the original instability and A denote pH>7.45 AND PaCO2<35 mmHg
in the last qualifying pre-ICU gas. The endpoint is the existing in-hospital
mortality after the original four-hour cohort landmark. No troponin result is
selected in this screen. No cutoffs, gas windows or patient selections may be
changed after seeing mortality.

The scientific question is whether the observed mortality risk associated with
instability is substantially greater in the alkalemic state in both datasets.
The interaction on the risk-difference scale is
I=(risk[X=1,A=1]-risk[X=0,A=1])-(risk[X=1,A=0]-risk[X=0,A=0]).
This is a crude observational contrast, not an adjusted effect or a treatment
effect. A positive screen only warrants further investigation; it cannot
establish that correcting alkalemia, CO2, or SpO2 improves survival.

Use one encounter per person so that repeated admissions are not treated as
independent binomial observations. Before joining mortality, select among each
person's eligible pre-gas encounters by the smallest SHA256 of
`20260905:{dataset}:{person_id}:{stay_id}`. This is deterministic selection
independent of outcome; it is not a claim that identifiers are chronological.
Keep the selected patient set fixed even if mortality is missing. Exclude only
unknown mortality from the risk denominators and report it. Verify binary
outcome coding and source cohort linkage.

Report all eight cell counts and observed risks (two databases × X × A).
Construct two-sided Clopper–Pearson intervals for each cell with alpha=0.05/8.
Bonferroni therefore provides at least 95% simultaneous coverage of the eight
conditional binomial risks under the independent-patient model. Propagate
these bounds conservatively through the linear contrast above: subtract upper
bounds for negatively signed terms and lower bounds for positively signed terms
for the lower interaction bound, and reverse for its upper bound.
Do not relabel these conservative bounds as a precision-adjusted model.
No p-value or uncorrected significance claim is needed.

The hypothesis points toward positive I. Advance this particular screen only
if the simultaneous lower bound for I exceeds 0.05 (five absolute percentage
points) in both databases. If it does not, report the estimates and uncertainty;
do not interpret non-advancement as proof of no effect. Even passing would
require adequate confounding control, measurement validation, myocardial-injury
evidence, and independent replication before a biological claim. Both project
datasets have previously been explored, so neither is a pristine holdout.
The significance/precision screen is not the user's discovery completion rule.

Do not add covariates or strata after seeing the four cells. The 13-encounter
MIMIC joint cell does not support an elaborate confirmatory interaction model.
Record this as a limited falsification screen in the single notebook, with
actual local execution provenance. No non-MIMIC/eICU patient data, simulated
patients, or clinical intervention is introduced. The goal remains active.
