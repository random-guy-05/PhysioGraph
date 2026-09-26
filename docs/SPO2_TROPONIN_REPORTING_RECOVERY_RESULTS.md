# Reporting-limit recovery and selection audit

Actual execution: local. MIMIC comments rescanned for the exact previously audited source IDs; eICU source rows reused.

| MIMIC whole-comment class | Numeric accepted | Source records | Matching exact numeric comment |
|---|---|---:|---:|
| unparsed | False | 621 | 0 |
| unparsed | True | 4137 | 0 |

| eICU horizon | Exposure | Full follow-up encounters | Existing numeric pairs | Later omitted baseline bound | Omitted follow-up bound | Additional potential timed pairs |
|---|---:|---:|---:|---:|---:|---:|
| 12h | 0 | 6461 | 1436 | 0 | 9 | 244 |
| 12h | 1 | 3929 | 858 | 0 | 5 | 187 |
| 24h | 0 | 5218 | 1269 | 0 | 12 | 193 |
| 24h | 1 | 3248 | 773 | 0 | 7 | 144 |

All source IDs and original numeric/text/item identities matched. SQL/Python patient flags and original numeric-pair sets matched. Baseline and follow-up bound counts overlap and must not be added. Person counts and unit exclusions are in the aggregate JSON outputs.

Whole-comment parsing is deliberately strict; narrative numbers are not interpreted as assay limits. A recoverable bound still requires interval-aware adjudication. Additional potential pairs are an optimistic timing ceiling: a positive upper bound does not prove a positive baseline or a valid rise. No bound was substituted by its limit, no numeric label was changed, and no new association, biological discovery or mortality benefit was estimated.
