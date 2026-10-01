# MIMIC-III respiratory-support source-mapping audit

Audit type: outcome-blind D_ITEMS codebook review plus the small, dedicated procedure-events table. Exposure, endpoint, threshold, horizons, and episode windows are unchanged.

## Prior zero result retained

The original cached `early_mechanical_ventilation` field is missing for all 10,230 eligible stays under its original extractor. That result is retained as the output of the earlier operational mapping; it is not interpreted as true absence of ventilation.

## Objective mapping defect

The selected D_ITEMS codebook maps the available affirmative channels as follows:

```text
 ITEMID                    LABEL   DBSOURCE            LINKSTO      CATEGORY PARAM_TYPE
 225792     Invasive Ventilation metavision procedureevents_mv 2-Ventilation    Process
 225794 Non-invasive Ventilation metavision procedureevents_mv 2-Ventilation    Process
```

The prior extractor searched CHARTEVENTS only, did not read PROCEDUREEVENTS_MV, and interpreted only a small set of text values. It therefore missed both procedure-event items. It also ignored numeric Checkbox values for ITEMID 226260 (Mechanically Ventilated), which links to CHARTEVENTS. The codebook identifies 225792 and 225794 as `Process` items linked to procedureevents_mv and 226260 as a Checkbox linked to chartevents.

The targeted first-four-hour PROCEDUREEVENTS_MV review found 1,192 / 10,230 stays with affirmative procedure evidence (1,014 invasive and 184 non-invasive; categories may overlap). The other 9,038 stays have no procedure record in this source and are unknown, not documented negatives. The numeric checkbox rows were not re-read because the validated cache contains only a four-hour summary with the old text-only ventilation parser; the full 330M-row CHARTEVENTS table was not rescanned in this analysis-only finalization.

## Corrected interpretation and limits

This source-mapping correction is based on codebook links/field types and the targeted procedure-events source before any exposure/outcome comparison. The source reconstruction is a lower-bound documentation audit, not a complete support phenotype. Original cached covariates remain unchanged. FiO2 remains summarized from the saved pre-landmark table (availability approximately 24%). No respiratory-adjusted outcome model is fit: support-negative status is not adequately ascertained, and the delayed episode sample has only 35 events (<10 events per parameter for the existing multivariable specification). The results do not support a claim of independence from respiratory failure.
