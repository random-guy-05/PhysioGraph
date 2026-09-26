# Secondary eICU validation across disjoint hospitals

Actual execution: local incremental notebook cells. The failed MIMIC temporal-support experiment is preserved; this is not cross-database replication.

| Split | Hospitals | Patients | Intervals | Instability switchers | Support gate |
|---|---:|---:|---:|---:|---|
| discovery | 68 | 608 | 1363 | 259 | pass |
| validation | 70 | 460 | 1034 | 160 | pass |

| Model | Split | Troponin fold-change ratio | 95% CI | Holm p |
|---|---|---:|---|---:|
| primary | discovery | 0.9503 | 0.8916–1.0129 | 0.231272 |
| primary | validation | 0.9627 | 0.8557–1.0830 | 0.52174 |
| omit_lag | discovery | 0.9591 | 0.8513–1.0805 | 0.972583 |
| omit_lag | validation | 1.0225 | 0.8586–1.2178 | 0.972583 |

The prespecified positive internal-validation criterion is not met. A favorable discovery estimate or sensitivity cannot rescue failed primary validation. This is not equivalence or proof of no relationship.

The ratio compares modeled troponin fold changes during unstable versus stable windows within patients; it is not a mortality risk ratio or an infarction diagnosis. Short-panel dynamic bias, treatment, renal clearance, selective testing and delayed release remain unresolved. Omitting the lag is a diagnostic, not a proven bias correction.

Original patient-to-hospital linkage and SHA256 assignment match independent implementations. Numerical validation uses uncentered sufficient statistics and a synthetic full-dummy model with patients nested in hospitals. Hospital-clustered uncertainty accounts for absorbed patient degrees of freedom. No mortality endpoint was accessed. No mortality benefit or paradigm-shifting biological finding is established. The goal remains unmet.

Hemodynamic limitation (discovery): MAP was missing in 1155/1363 windows (84.7%) and median-imputed with a missingness indicator. This does not establish independence from changing blood pressure or perfusion.

Hemodynamic limitation (validation): MAP was missing in 933/1034 windows (90.2%) and median-imputed with a missingness indicator. This does not establish independence from changing blood pressure or perfusion.
