# Raw troponin reporting audit

Actual execution: local. Original MIMIC/eICU source files rescanned; existing endpoint labels preserved.

| Source | Horizon | Exposure | Legacy rise | Encounters | Any text uncertainty | Explicit bound in used samples |
|---|---|---:|---:|---:|---:|---:|
| eicu | 12h | 0 | 0 | 1122 | 0 | 0 |
| eicu | 12h | 0 | 1 | 314 | 0 | 0 |
| eicu | 12h | 1 | 0 | 620 | 0 | 0 |
| eicu | 12h | 1 | 1 | 238 | 0 | 0 |
| eicu | 24h | 0 | 0 | 965 | 0 | 0 |
| eicu | 24h | 0 | 1 | 304 | 0 | 0 |
| eicu | 24h | 1 | 0 | 531 | 0 | 0 |
| eicu | 24h | 1 | 1 | 242 | 0 | 0 |
| mimic | 12h | 0 | 0 | 398 | 398 | 0 |
| mimic | 12h | 0 | 1 | 118 | 118 | 0 |
| mimic | 12h | 1 | 0 | 167 | 167 | 0 |
| mimic | 12h | 1 | 1 | 91 | 91 | 0 |
| mimic | 24h | 0 | 0 | 346 | 346 | 0 |
| mimic | 24h | 0 | 1 | 129 | 129 | 0 |
| mimic | 24h | 1 | 0 | 144 | 144 | 0 |
| mimic | 24h | 1 | 1 | 99 | 99 | 0 |

Source grouping, cached numeric values/result times, original ratios and labels, and SQL/Python used-sample flags all matched. Source inventories are in text_inventory.json; baseline/follow-up flags and person counts are in label_impact.json.

Post-audit text-shape clarification: MIMIC has 4137 literal three-underscore placeholder records among 4137 unparsed records. This does not establish why that field was replaced or recover its original qualifiers. Classification and label-impact rules were unchanged by this clarification.

eICU contains 1800 explicitly bounded source records; 0 enter the accepted numeric pipeline. Missing numeric values are reported separately from text/numeric disagreements. Clean text among retained values does not resolve selection caused by omitted bounded values.

An explicit bound describes an interval. Missing or unparsed text is an uncertainty flag, not proven censoring. A flagged encounter is not automatically a false rise: baseline and follow-up intervals require separate adjudication. The audit includes all matched follow-up samples, not only the maximum. A numeric upper reference range in MIMIC does not establish its equivalence to an assay-specific 99th percentile; eICU lacks that field.

These results validate or qualify reporting provenance only. No endpoint was replaced, no new association was fitted, and no myocardial injury diagnosis, biological discovery or mortality benefit is established.
