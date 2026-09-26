# CareVue STORETIME availability decision

**Locked branch: `AVAILABILITY_PARITY_APPLIED`**

This decision was made from field semantics and measurement timing before any corrected SpO2–outcome association was accessed. MIMIC-III documents `CHARTEVENTS.CHARTTIME` as the time the observation was made and `STORETIME` as the time it was manually input or validated. MIMIC-IV gives the same operational definitions for its `chartevents` fields. Those definitions support using CareVue `STORETIME` as the closest available source-era analogue of the frozen MIMIC-IV availability field.

Sources:

- MIMIC-III `CHARTEVENTS`: https://mimic.mit.edu/docs/iii/tables/chartevents.html
- MIMIC-IV `chartevents`: https://mimic.mit.edu/docs/iv/modules/icu/chartevents.html

## Locked implementation

- Exposure source: CareVue `CHARTEVENTS.ITEMID = 646` only.
- Observation time: ICU-relative `CHARTTIME`, required to be within minutes 0–240 inclusive.
- Availability time: the later of ICU-relative `CHARTTIME` and `STORETIME`.
- Eligibility: availability time must be nonmissing and no later than minute 240.
- No alternative availability branch will be computed or compared.

## Source audit

- Candidate item-646 rows with charttime in minutes 0–240: **30,193**.
- Missing STORETIME: **0** (0.00%).
- Stored after minute 240: **2,726** (9.03%).
- Valid item-646 rows available by minute 240: **27,349**.
- Median chart-to-store delay: **10.0 minutes**.
- 95th-percentile chart-to-store delay: **84.0 minutes**.

## Qualification

The fields are sufficiently comparable for the specified landmark-availability restriction, but they are not proof of identical real-world latency or capture workflows across CareVue and MetaVision. The corrected analysis therefore treats exposure availability parity as supportable and applied while retaining cross-era measurement-process limitations. This choice is not based on an association estimate.
