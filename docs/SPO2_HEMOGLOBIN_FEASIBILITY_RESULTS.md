# Hemoglobin susceptibility: linked-record feasibility failed

**Updated status:** the separately documented linkage-only amendment raises
MIMIC support to 516 and passes the same screen in both databases. The original
explicit-ID-only result below remains unchanged. See
PREICU_LINKAGE_AMENDED_RESULTS.md. No biological interaction has been computed.

The frozen feasibility experiment completed locally on 2026-09-05 at 06:17 UTC.
It used only the canonical MIMIC and eICU HF cohorts and pre-ICU CBC hemoglobin.
No Hb-by-troponin-rise or Hb-by-mortality interaction was computed.

| Database | Dynamics eligible | Pre-ICU Hb available | Hb plus observed 12-hour troponin endpoint | With instability / without instability |
|---|---:|---:|---:|---:|
| eICU | 11,452 | 3,984 | 659 | 250 / 409 |
| MIMIC | 4,711 | 1,113 | 133 | 53 / 80 |

The minimum required 200 paired observations in each database, including 50
with and 50 without the original SpO2 instability exposure. MIMIC failed the
total-count requirement. The planned Hb interaction analysis therefore did
not proceed. This does not establish presence or absence of susceptibility
to myocardial injury from low oxygen-carrying capacity.

The SQL and independent Pandas latest-sample selections agreed for all 3,984
eICU and 1,113 MIMIC encounters. MIMIC had 1,441 candidate CBC records, with no
unit exclusions and three exclusions under the frozen 3–22 g/dL quality rule.
eICU excluded non-Hb analytes and unknown/conflicting unit labels under the
prespecified whitelist. Its audit exposes g/dL versus gm/dL spelling differences
that the strict rule treated as conflicts; this conservative loss does not
explain the MIMIC failure and was not changed after the feasibility count.

These are specimen-time baselines. The selected result entry time was after
ICU admission for 220 eICU and 89 MIMIC encounters, so they must not all be
described as prospectively available to clinicians before ICU admission.

A separate outcome-blind encounter-linkage audit is being undertaken to test
whether explicit-ID-only extraction misses laboratories from documented ED
encounters. It does not change this frozen result or authorize silent reuse
of newly assigned records. See MIMIC_PREICU_LAB_LINKAGE_AUDIT.md.

All outputs are real local computations. No Colab run, novel biological
mechanism, transfusion benefit, or mortality improvement is established.
