#!/usr/bin/env python3
"""Rerun the locked full adjusted lactate and in-hospital mortality models.

Uses saved cohort, SpO2 exposure, and lactate endpoint artifacts. It rereads only
MIMIC-III admission/diagnosis tables needed for mortality outcome and prior-code
covariate recoding; it does not rebuild the cohort, waveform features, or lactate
endpoints.
"""
from __future__ import annotations

import ast
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path('/Users/admin/Downloads/physiograph_dense_waveform_analysis')
CLIN = Path('/Users/admin/Downloads/mimic-iii-clinical-database-1.4')
EXT = Path('/Users/admin/Downloads/physiograph_age_subgroup_extension')
OUT = EXT / 'ckd_corrected_full_adjusted_models'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SRC / 'scripts'))
import run_dense_waveform_analysis as core  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def norm_code(value: object) -> str:
    if pd.isna(value):
        return ''
    return ''.join(ch for ch in str(value).strip().upper() if ch.isalnum())


def add_row(rows: list[dict], *, analysis: str, spec: str, outcome: str,
            ckd_col: str, frame: pd.DataFrame, covariates: list[str],
            fit: dict, endpoint: str) -> None:
    complete = frame.dropna(subset=[outcome, 'exposure', 'SUBJECT_ID'] + covariates)
    design = [np.ones(len(complete)), complete.exposure.astype(float).to_numpy()]
    for column in covariates:
        values = pd.to_numeric(complete[column], errors='coerce').to_numpy(dtype=float)
        sd = float(np.nanstd(values))
        if np.isfinite(sd) and sd >= 1e-12:
            design.append((values - float(np.nanmean(values))) / sd)
    design_rank = int(np.linalg.matrix_rank(np.column_stack(design)))
    exposed = complete.loc[complete.exposure.eq(1)]
    unexposed = complete.loc[complete.exposure.eq(0)]
    exposed_risk = float(exposed[outcome].mean()) if len(exposed) else np.nan
    unexposed_risk = float(unexposed[outcome].mean()) if len(unexposed) else np.nan
    crude_rr = exposed_risk / unexposed_risk if unexposed_risk > 0 else np.nan
    crude_rd = exposed_risk - unexposed_risk
    rows.append({
        'analysis': analysis,
        'specification': spec,
        'endpoint': endpoint,
        'n': fit.get('n'),
        'events': fit.get('events'),
        'non_events': fit.get('non_events'),
        'patients': fit.get('patients'),
        'exposed_events': int(exposed[outcome].sum()),
        'exposed_n': int(len(exposed)),
        'unexposed_events': int(unexposed[outcome].sum()),
        'unexposed_n': int(len(unexposed)),
        'exposed_risk': exposed_risk,
        'unexposed_risk': unexposed_risk,
        'crude_RR': crude_rr,
        'crude_RD': crude_rd,
        'ckd_prior_code_positive_n': int(complete[ckd_col].sum()),
        'parameters_including_intercept': fit.get('parameters'),
        'iterations': fit.get('iterations'),
        'converged': fit.get('converged'),
        'design_rank': design_rank,
        'full_rank': design_rank == fit.get('parameters'),
        'events_per_parameter': (fit.get('events') / fit.get('parameters')
                                 if fit.get('events') is not None and fit.get('parameters') else np.nan),
        'adjusted_RR': fit.get('adjusted_RR'),
        'robust_CI_low': fit.get('adjusted_RR_95CI_low'),
        'robust_CI_high': fit.get('adjusted_RR_95CI_high'),
        'robust_SE_log_RR': fit.get('robust_SE_log_RR'),
        'p_value': fit.get('p_value'),
        'status': fit.get('status'),
        'reason': fit.get('reason', ''),
        'covariates': '|'.join(covariates),
    })


# These are the exact saved analysis inputs; waveform processing and endpoint
# construction are deliberately not rerun here.
signal_path = SRC / '02_signal_qc/signal_features_recomputed.csv'
episode_path = SRC / '04_episode_lactate/episode_anchor_records.csv'
outcome_path = SRC / '03_landmark_lactate/lactate_outcomes.csv'
dx_path = CLIN / 'DIAGNOSES_ICD.csv.gz'
adm_path = CLIN / 'ADMISSIONS.csv.gz'
dict_path = CLIN / 'D_ICD_DIAGNOSES.csv.gz'
old_lactate_path = EXT / 'clinical_core_plus_prior_comorbidities_adjusted_rr.csv'
saved_core_path = SRC / '03_landmark_lactate/adjusted_modified_poisson.csv'

sig = pd.read_csv(signal_path, usecols=[
    'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'dynamics_eligible',
    'exposure_abs_jump_ge4', 'mean_binned_spo2'
])
sig = sig.loc[sig.dynamics_eligible.astype(bool), [
    'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'exposure_abs_jump_ge4',
    'mean_binned_spo2'
]].copy()
epi = pd.read_csv(episode_path, usecols=[
    'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'anchor_group', 'age_years', 'sex_male'
])
df = sig.merge(epi, on=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID'], validate='one_to_one')
df['exposure'] = df.exposure_abs_jump_ge4.astype(int)
assert len(df) == 1597 and int(df.exposure.sum()) == 539
assert df.mean_binned_spo2.notna().all()
assert (df.exposure == df.anchor_group.eq('episode').astype(int)).all()

outcomes = pd.read_csv(outcome_path, usecols=[
    'SUBJECT_ID', 'ICUSTAY_ID', 'baseline_lactate_mmol_l',
    'outcome_available_12h', 'lactate_rise_ge0_5_12h',
    'outcome_available_24h', 'lactate_rise_ge0_5_24h'
])
df = df.merge(outcomes, on=['SUBJECT_ID', 'ICUSTAY_ID'], validate='one_to_one')
for horizon in ('12h', '24h'):
    available = df[f'outcome_available_{horizon}'].fillna(False).astype(bool)
    y = df[f'lactate_rise_ge0_5_{horizon}']
    assert y.loc[available].notna().all()
    assert y.loc[available].isin([0, 1, False, True]).all()

# Check each expanded code family against the downloaded MIMIC-III diagnosis
# dictionary before applying it to the prior-admission diagnosis history.
dict_df = pd.read_csv(dict_path, usecols=['ICD9_CODE', 'LONG_TITLE'],
                       dtype={'ICD9_CODE': 'string'})
dict_df['code'] = dict_df.ICD9_CODE.map(norm_code)
dictionary_families = {
    '585.x': (lambda c: c.startswith('585'), 'chronic kidney disease'),
    '403.x': (lambda c: c.startswith('403'), 'hypertensive chronic kidney disease'),
    '404.x': (lambda c: c.startswith('404'), 'chronic kidney disease'),
    '582.x': (lambda c: c.startswith('582'), 'chronic glomerulonephritis'),
    '250.4x': (lambda c: c.startswith('2504'), 'renal manifestations'),
    '753.12-.14': (lambda c: c in {'75312', '75313', '75314'}, 'polycystic kidney'),
}
dictionary_validation = []
for family, (matches, title_phrase) in dictionary_families.items():
    found = dict_df.loc[dict_df.code.map(matches)]
    assert len(found) > 0, f'No MIMIC diagnosis dictionary entries for {family}'
    assert found.LONG_TITLE.str.lower().str.contains(title_phrase).any(), (
        f'No expected MIMIC code title found for {family}: {found.LONG_TITLE.tolist()}'
    )
    dictionary_validation.append({
        'family': family, 'dictionary_code_count': int(len(found)),
        'example_code': found.iloc[0].ICD9_CODE,
        'example_long_title': found.iloc[0].LONG_TITLE,
    })
dictionary_validation_path = OUT / 'ckd_code_dictionary_validation.csv'
pd.DataFrame(dictionary_validation).to_csv(dictionary_validation_path, index=False)

# Define both the prior and expanded CKD flags. All other model-specific
# covariate code rules are preserved exactly from the prior saved analyses.
dx = pd.read_csv(dx_path, usecols=['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE'],
                 dtype={'ICD9_CODE': 'string'})
dx['code'] = dx.ICD9_CODE.map(norm_code)
p3 = pd.to_numeric(dx.code.str.slice(0, 3), errors='coerce')
dx['ckd_old_585'] = dx.code.str.startswith('585')
dx['ckd_expanded'] = (
    p3.isin([585, 403, 404, 582])
    | dx.code.str.startswith('2504')
    | dx.code.isin({'75312', '75313', '75314'})
)
# Lactate model's existing non-CKD definitions (ESRD name only is changed).
dx['esrd_lactate'] = dx.code.str.startswith('5856') | dx.code.eq('V4511')
dx['lung_lactate'] = p3.between(490, 496) | p3.between(500, 508)
# Mortality report's existing non-CKD definitions, retained unchanged.
dx['esrd_mortality'] = dx.code.str.startswith('5856') | dx.code.isin(
    {'V4511', 'V4512', 'V560', 'V5631', 'V5632', 'V568'}
)
dx['lung_mortality'] = p3.between(490, 505) | dx.code.isin(
    {'4168', '4169', '5064', '5081', '5088', '515', '51883', '51884'}
)
dx['liver_prior'] = p3.eq(571)
dx['diabetes_prior'] = p3.eq(250)
flag_cols = [
    'ckd_old_585', 'ckd_expanded', 'esrd_lactate', 'lung_lactate',
    'esrd_mortality', 'lung_mortality', 'liver_prior', 'diabetes_prior'
]
per_admission = dx.groupby(['SUBJECT_ID', 'HADM_ID'], as_index=False)[flag_cols].max()

adm = pd.read_csv(adm_path, usecols=[
    'SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'HOSPITAL_EXPIRE_FLAG'
], parse_dates=['ADMITTIME'])
assert not adm.HADM_ID.duplicated().any()
adm = adm.merge(per_admission, on=['SUBJECT_ID', 'HADM_ID'], how='left', validate='one_to_one')
adm[flag_cols] = adm[flag_cols].fillna(False).astype(bool)
# Strictly earlier admissions only. Group equal admission timestamps together so
# a tied timestamp cannot leak a code into another admission at the same time.
by_time = (adm.groupby(['SUBJECT_ID', 'ADMITTIME'], as_index=False)[flag_cols]
           .max().sort_values(['SUBJECT_ID', 'ADMITTIME'], kind='mergesort'))
for col in flag_cols:
    cumulative = by_time.groupby('SUBJECT_ID', sort=False)[col].cummax()
    by_time[col] = cumulative.groupby(by_time.SUBJECT_ID, sort=False).shift(1).fillna(False).astype(bool)
prior = adm[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME']].merge(
    by_time, on=['SUBJECT_ID', 'ADMITTIME'], how='left', validate='many_to_one'
)
df = df.merge(adm[['SUBJECT_ID', 'HADM_ID', 'HOSPITAL_EXPIRE_FLAG']],
              on=['SUBJECT_ID', 'HADM_ID'], how='left', validate='many_to_one')
df = df.merge(prior[['SUBJECT_ID', 'HADM_ID'] + flag_cols],
              on=['SUBJECT_ID', 'HADM_ID'], how='left', validate='many_to_one')
assert len(df) == 1597 and df.HOSPITAL_EXPIRE_FLAG.notna().all()
assert df[flag_cols].notna().all().all()
assert (df.ckd_expanded | ~df.ckd_old_585).all()
assert df.esrd_lactate.isin([False, True]).all() and df.esrd_mortality.isin([False, True]).all()

core_covars = ['age_years', 'sex_male', 'baseline_lactate_mmol_l']
lactate_covars_old = core_covars + [
    'ckd_old_585', 'esrd_lactate', 'lung_lactate', 'liver_prior', 'diabetes_prior'
]
lactate_covars_new = core_covars + [
    'ckd_expanded', 'esrd_lactate', 'lung_lactate', 'liver_prior', 'diabetes_prior'
]
mortality_covars_old = core_covars + [
    'ckd_old_585', 'esrd_mortality', 'lung_mortality', 'liver_prior', 'diabetes_prior'
]
mortality_covars_new = core_covars + [
    'ckd_expanded', 'esrd_mortality', 'lung_mortality', 'liver_prior', 'diabetes_prior'
]
lactate_covars_new_mean_spo2 = core_covars + [
    'mean_binned_spo2', 'ckd_expanded', 'esrd_lactate', 'lung_lactate',
    'liver_prior', 'diabetes_prior'
]
mortality_covars_new_mean_spo2 = core_covars + [
    'mean_binned_spo2', 'ckd_expanded', 'esrd_mortality', 'lung_mortality',
    'liver_prior', 'diabetes_prior'
]

rows: list[dict] = []
reconciliation: list[dict] = []
saved_core = pd.read_csv(saved_core_path)
saved_old_lactate = pd.read_csv(old_lactate_path)
for horizon in ('12h', '24h'):
    y = f'lactate_rise_ge0_5_{horizon}'
    available = f'outcome_available_{horizon}'
    use = df.loc[df[available].fillna(False).astype(bool)].copy()
    core_fit = core.modified_poisson(use, y, 'exposure', core_covars, cluster_col='SUBJECT_ID')
    old_fit = core.modified_poisson(use, y, 'exposure', lactate_covars_old, cluster_col='SUBJECT_ID')
    new_fit = core.modified_poisson(use, y, 'exposure', lactate_covars_new, cluster_col='SUBJECT_ID')
    mean_spo2_fit = core.modified_poisson(
        use, y, 'exposure', lactate_covars_new_mean_spo2, cluster_col='SUBJECT_ID'
    )
    saved = saved_core.loc[saved_core.horizon.eq(horizon)
                           & saved_core.model.eq('clinical_core')].iloc[0]
    old_saved = saved_old_lactate.loc[saved_old_lactate.horizon.eq(horizon)].iloc[0]
    assert int(core_fit['n']) == int(saved['n'])
    assert int(core_fit['events']) == int(saved['events'])
    assert np.isclose(core_fit['adjusted_RR'], saved['adjusted_RR'], rtol=1e-10, atol=1e-10)
    assert int(old_fit['n']) == int(old_saved['n'])
    assert int(old_fit['events']) == int(old_saved['events'])
    assert np.isclose(old_fit['adjusted_RR'], old_saved['core_plus_comorbidity_RR'], rtol=1e-10, atol=1e-10)
    assert int(use.ckd_old_585.sum()) == int(ast.literal_eval(old_saved['prior_code_positive_n'])['prior_any_CKD'])
    for spec, ckd_col, covars, fit in [
        ('before_CKD_expansion', 'ckd_old_585', lactate_covars_old, old_fit),
        ('after_CKD_expansion', 'ckd_expanded', lactate_covars_new, new_fit),
        ('after_CKD_plus_mean_SpO2', 'ckd_expanded', lactate_covars_new_mean_spo2, mean_spo2_fit),
    ]:
        add_row(rows, analysis=f'primary_lactate_{horizon}', spec=spec, outcome=y,
                ckd_col=ckd_col, frame=use, covariates=covars, fit=fit,
                endpoint='complete-window last-value lactate rise >=0.5 mmol/L')
    reconciliation.append({
        'analysis': f'primary_lactate_{horizon}',
        'eligible_endpoint_n': int(len(use)),
        'events': int(use[y].sum()),
        'non_events': int(len(use) - use[y].sum()),
        'patients': int(use.SUBJECT_ID.nunique()),
        'prior_CKD_585_only_n': int(use.ckd_old_585.sum()),
        'prior_CKD_expanded_n': int(use.ckd_expanded.sum()),
        'additional_prior_CKD_flags': int((use.ckd_expanded & ~use.ckd_old_585).sum()),
        'prior_ESRD_lactate_codes_n': int(use.esrd_lactate.sum()),
        'ESRD_flags_unchanged': True,
        'old_adjusted_RR': old_fit['adjusted_RR'],
        'new_adjusted_RR': new_fit['adjusted_RR'],
        'CKD_plus_mean_SpO2_n': int(mean_spo2_fit['n']),
        'CKD_plus_mean_SpO2_adjusted_RR': mean_spo2_fit['adjusted_RR'],
        'RR_change': new_fit['adjusted_RR'] - old_fit['adjusted_RR'],
        'RR_change_after_mean_SpO2': mean_spo2_fit['adjusted_RR'] - new_fit['adjusted_RR'],
    })

# In-hospital mortality is the previously specified admission-level endpoint;
# it is not recast as a 12-/24-hour death endpoint.
df['in_hospital_death'] = pd.to_numeric(df.HOSPITAL_EXPIRE_FLAG, errors='coerce')
mort_old = core.modified_poisson(df, 'in_hospital_death', 'exposure',
                                 mortality_covars_old, cluster_col='SUBJECT_ID')
mort_new = core.modified_poisson(df, 'in_hospital_death', 'exposure',
                                 mortality_covars_new, cluster_col='SUBJECT_ID')
mort_mean_spo2 = core.modified_poisson(
    df, 'in_hospital_death', 'exposure', mortality_covars_new_mean_spo2,
    cluster_col='SUBJECT_ID'
)
for spec, ckd_col, covars, fit in [
    ('before_CKD_expansion', 'ckd_old_585', mortality_covars_old, mort_old),
    ('after_CKD_expansion', 'ckd_expanded', mortality_covars_new, mort_new),
    ('after_CKD_plus_mean_SpO2', 'ckd_expanded', mortality_covars_new_mean_spo2, mort_mean_spo2),
]:
    add_row(rows, analysis='in_hospital_mortality', spec=spec, outcome='in_hospital_death',
            ckd_col=ckd_col, frame=df, covariates=covars, fit=fit,
            endpoint='index-admission HOSPITAL_EXPIRE_FLAG')
# Confirm exact reproduction of the archived mortality model's published
# rounded result before interpreting the revised CKD-only change.
old_mort_reproduced = (
    mort_old.get('n') == 911 and mort_old.get('events') == 159
    and mort_old.get('patients') == 831
    and round(mort_old.get('adjusted_RR', np.nan), 4) == 1.5127
    and round(mort_old.get('adjusted_RR_95CI_low', np.nan), 4) == 1.1497
    and round(mort_old.get('adjusted_RR_95CI_high', np.nan), 4) == 1.9902
    and round(mort_old.get('p_value', np.nan), 5) == 0.00311
)
assert old_mort_reproduced, f'Prior mortality model did not reproduce: {mort_old}'
reconciliation.append({
    'analysis': 'in_hospital_mortality',
    'eligible_endpoint_n': int(mort_old['n']),
    'events': int(mort_old['events']),
    'non_events': int(mort_old['non_events']),
    'patients': int(mort_old['patients']),
    'prior_CKD_585_only_n': int(df.dropna(subset=mortality_covars_old + ['in_hospital_death']).ckd_old_585.sum()),
    'prior_CKD_expanded_n': int(df.dropna(subset=mortality_covars_new + ['in_hospital_death']).ckd_expanded.sum()),
    'additional_prior_CKD_flags': int((df.dropna(subset=mortality_covars_old + ['in_hospital_death']).ckd_expanded & ~df.dropna(subset=mortality_covars_old + ['in_hospital_death']).ckd_old_585).sum()),
    'prior_ESRD_mortality_codes_n': int(df.dropna(subset=mortality_covars_old + ['in_hospital_death']).esrd_mortality.sum()),
    'ESRD_flags_unchanged': True,
    'old_adjusted_RR': mort_old['adjusted_RR'],
    'new_adjusted_RR': mort_new['adjusted_RR'],
    'CKD_plus_mean_SpO2_n': int(mort_mean_spo2['n']),
    'CKD_plus_mean_SpO2_adjusted_RR': mort_mean_spo2['adjusted_RR'],
    'RR_change': mort_new['adjusted_RR'] - mort_old['adjusted_RR'],
    'RR_change_after_mean_SpO2': mort_mean_spo2['adjusted_RR'] - mort_new['adjusted_RR'],
})

result = pd.DataFrame(rows)
recon = pd.DataFrame(reconciliation)
result_path = OUT / 'full_adjusted_models_before_after.csv'
recon_path = OUT / 'ckd_coding_reconciliation.csv'
assert result.loc[result.specification.isin(['after_CKD_expansion', 'after_CKD_plus_mean_SpO2']), 'status'].eq('available').all()
assert result.full_rank.all()
for analysis in ['primary_lactate_12h', 'primary_lactate_24h', 'in_hospital_mortality']:
    ckd_only = result.loc[result.analysis.eq(analysis) & result.specification.eq('after_CKD_expansion')].iloc[0]
    mean_spo2 = result.loc[result.analysis.eq(analysis) & result.specification.eq('after_CKD_plus_mean_SpO2')].iloc[0]
    assert int(ckd_only['n']) == int(mean_spo2['n'])
    assert int(ckd_only['events']) == int(mean_spo2['events'])
result.to_csv(result_path, index=False)
recon.to_csv(recon_path, index=False)

# Compact report generated directly from saved model outputs.
def fmt(value: object, digits: int = 3) -> str:
    return 'NA' if pd.isna(value) else f'{float(value):.{digits}f}'

def show_model(analysis: str, spec: str) -> dict:
    return result.loc[result.analysis.eq(analysis) & result.specification.eq(spec)].iloc[0].to_dict()

lines = [
    '# CKD-corrected full adjusted lactate and mortality models with mean-SpO2 sensitivity',
    '',
    f"Local/precomputed MIMIC-III analysis; run completed {datetime.now(timezone.utc).isoformat()}.",
    '',
    '## Results',
    '',
    '| Outcome | CKD specification | N | Events | Patients | Exposed events / N | Unexposed events / N | Prior CKD positive | Crude RR | Crude RD | Adjusted RR (patient-cluster robust 95% CI) | p | EPV |',
    '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
]
for analysis in ['primary_lactate_12h', 'primary_lactate_24h', 'in_hospital_mortality']:
    for spec, label in [('before_CKD_expansion', 'Prior 585.x only'),
                        ('after_CKD_expansion', 'Expanded prior-coded CKD'),
                        ('after_CKD_plus_mean_SpO2', 'Expanded CKD + 0–4h mean SpO2')]:
        row = show_model(analysis, spec)
        lines.append(
            f"| {analysis} | {label} | {row['n']} | {row['events']} | {row['patients']} | "
            f"{row['exposed_events']} / {row['exposed_n']} | {row['unexposed_events']} / {row['unexposed_n']} | "
            f"{row['ckd_prior_code_positive_n']} | {fmt(row['crude_RR'])} | {fmt(row['crude_RD'])} | "
            f"{fmt(row['adjusted_RR'])} ({fmt(row['robust_CI_low'])}–{fmt(row['robust_CI_high'])}) | "
            f"{fmt(row['p_value'], 4)} | {fmt(row['events_per_parameter'], 1)} |"
        )
lines += [
    '',
    'The lactate outcomes are the locked complete-window last-value rise endpoints: follow-up through the 12-hour or 24-hour horizon is required for both events and non-events. Mortality is the prior report’s index-admission in-hospital death endpoint (`HOSPITAL_EXPIRE_FLAG`), not a fixed 12-/24-hour death endpoint.',
    '',
    '## Reconciliation',
    '',
    '| Analysis | N | Events / non-events | Prior CKD 585.x only | Expanded prior CKD | Additional flags | Old RR | Revised RR | RR after mean SpO2 | Change after mean SpO2 |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
]
for _, row in recon.iterrows():
    lines.append(
        f"| {row['analysis']} | {row['eligible_endpoint_n']} | {row['events']} / {row['non_events']} | "
        f"{row['prior_CKD_585_only_n']} | {row['prior_CKD_expanded_n']} | {row['additional_prior_CKD_flags']} | "
        f"{fmt(row['old_adjusted_RR'])} | {fmt(row['new_adjusted_RR'])} | {fmt(row['CKD_plus_mean_SpO2_adjusted_RR'])} | {fmt(row['RR_change_after_mean_SpO2'], 4)} |"
    )
mean12 = show_model('primary_lactate_12h', 'after_CKD_plus_mean_SpO2')
mean24 = show_model('primary_lactate_24h', 'after_CKD_plus_mean_SpO2')
mean_mortality = show_model('in_hospital_mortality', 'after_CKD_plus_mean_SpO2')
if mean12['robust_CI_low'] > 1 and mean_mortality['robust_CI_low'] > 1:
    conclusion = (
        f"After adding the 0–4h mean SpO2 covariate, the 12h lactate association "
        f"(RR {fmt(mean12['adjusted_RR'])}, 95% CI {fmt(mean12['robust_CI_low'])}–{fmt(mean12['robust_CI_high'])}) "
        f"and in-hospital mortality association (RR {fmt(mean_mortality['adjusted_RR'])}, 95% CI "
        f"{fmt(mean_mortality['robust_CI_low'])}–{fmt(mean_mortality['robust_CI_high'])}) remained above the null. "
        f"These associations persisted independently of measured early mean SpO2 level and baseline lactate in these adjusted models. "
        f"The 24h lactate interval ({fmt(mean24['robust_CI_low'])}–{fmt(mean24['robust_CI_high'])}) crosses 1, so this conclusion does not extend to the 24h endpoint."
    )
else:
    conclusion = (
        'After adding the 0–4h mean SpO2 covariate, at least one of the 12h lactate or mortality intervals includes 1. '
        'Keep the prior conclusion and treat this sensitivity as an exploratory result; do not claim independence from SpO2 level for both endpoints.'
    )
lines += [
    '',
    '## Conclusion',
    '',
    conclusion,
    '',
    '## Fixed model and code definitions',
    '',
    '- Exposure: saved first-four-hour indicator for any qualifying absolute SpO₂ transition ≥4 percentage points; same dynamics-eligible cohort (1,597 stays, 539 exposed).',
    '- Lactate outcomes: the saved corrected primary endpoint, last lactate in the fixed post-landmark window minus the last pre-landmark lactate ≥0.5 mmol/L; no endpoint or window was rebuilt or changed.',
    '- Added SpO2 level covariate: `mean_binned_spo2` is the mean of valid 15-minute SpO2 medians during ICU minutes [0, 240); it is available for all 1,597 dynamics-eligible stays.',
    '- Lactate model adjustment: age, sex, baseline lactate, prior-coded CKD, prior-coded ESRD, prior-coded chronic pulmonary disease, prior-coded chronic liver disease, and prior-coded diabetes.',
    '- Full adjusted models: age, sex, baseline lactate, revised prior-coded CKD, prior-coded ESRD, chronic pulmonary disease, chronic liver disease, and diabetes. Added-level sensitivity adds the saved 0–4h mean of valid 15-minute SpO2 medians (`mean_binned_spo2`) to each model.',
    '- Revised prior-coded CKD proxy: ICD-9-CM 585.x, 403.x, 404.x, 582.x, 250.4x, and 753.12/.13/.14, in any strictly earlier admission. This expands the prior 585.x-only flag; no eGFR verification or repeated-code requirement is imposed, so it is a broader diagnosis-code proxy, not a chart-validated CKD phenotype.',
    '- Code-family basis: the EHR CKD registry reports CKD, diabetic nephropathy, glomerulonephritis, and polycystic-kidney code families ([registry code table](https://pmc.ncbi.nlm.nih.gov/articles/PMC3022247/)); a separate electronic CKD phenotype includes hypertensive CKD codes 403.xx/404.xx ([eMERGE phenotype](https://pmc.ncbi.nlm.nih.gov/articles/PMC4419875/)). The downloaded MIMIC-III diagnosis dictionary confirmed the expected descriptions for all six families; the registry/phenotype inclusion algorithms are not reproduced here.',
    '- ESRD is a label-only change. Existing ESRD-associated code sets were retained for each model; these include dialysis-status codes (V45.11 in lactate; V45.11, V45.12, V56.0, V56.31, V56.32, V56.8 in mortality) and were not redefined.',
    '- Other lactate comorbidity code rules were unchanged: prior ICD-9 490–496 or 500–508 for chronic pulmonary disease, 571.x for liver disease, and 250.x for diabetes.',
    '- Other mortality comorbidity code rules were unchanged from the mortality report: prior 416.8/416.9, 490–505, 506.4, 508.1, 508.8, 515, 518.83, or 518.84 for chronic pulmonary disease; 571.x for liver disease; 250.x for diabetes.',
    '- Prior means diagnosis codes from strictly earlier hospital admission time(s); index-admission diagnoses are not used. Same-timestamp admissions are excluded from each other’s history.',
    '- Modified Poisson log link; patient-cluster sandwich covariance by SUBJECT_ID with finite-sample correction; two-sided Wald p-values and 95% CIs. Convergence status and iteration counts are saved in the result table.',
    '- Before-change validation: corrected 12-/24-hour lactate core and old full-adjusted estimates reproduce their saved results; old mortality reproduces the prior report (n=911, 159 deaths, 831 patients, RR 1.5127, 95% CI 1.1497–1.9902, p=0.00311). All rerun models converged and all design matrices were full rank; fit details are in the result table.',
    '- The mean-SpO2 model is a supplemental covariate sensitivity; it does not replace the original full-adjusted estimate.',
    '',
    'These are observational associations. “No prior code” is not confirmed absence of disease. The lactate extended models have low events per parameter, so their estimates are imprecise and vulnerable to overfitting; do not interpret p-values as proof of clinical utility or causality.',
    '',
    'All outputs are local/precomputed; this run was not executed in Google Colab.',
]
report_path = OUT / 'REPORT.md'
report_path.write_text('\n'.join(lines) + '\n')

inputs = [signal_path, episode_path, outcome_path, dx_path, adm_path, dict_path,
          saved_core_path, old_lactate_path]
audit = {
    'status': 'complete',
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'python': platform.python_version(),
    'scope': 'Reran only the adjusted lactate and in-hospital mortality models using saved processed cohort/exposure/lactate endpoint artifacts; no waveform, cohort, or lactate endpoint preprocessing was rerun.',
    'cohort_n': int(len(df)),
    'exposed_n': int(df.exposure.sum()),
    'lactate_core_reproduction': True,
    'lactate_old_full_model_reproduction': True,
    'mortality_old_full_model_reproduction': bool(old_mort_reproduced),
    'mortality_old_model': {k: mort_old.get(k) for k in ['n', 'events', 'patients', 'adjusted_RR', 'adjusted_RR_95CI_low', 'adjusted_RR_95CI_high', 'p_value', 'status']},
    'revised_models': result.to_dict(orient='records'),
    'ckd_reconciliation': recon.to_dict(orient='records'),
    'ckd_definition': 'ICD-9-CM 585.x, 403.x, 404.x, 582.x, 250.4x, 753.12/.13/.14 on strictly earlier admissions; adapted broad code proxy, not repeated-code/eGFR algorithm.',
    'ckd_dictionary_validation': dictionary_validation,
    'esrd_label': 'prior_ESRD; code definitions preserved from each existing model, including dialysis-status codes.',
    'inputs_sha256': {str(p): sha256(p) for p in inputs},
    'outputs_sha256': {str(p): sha256(p) for p in [result_path, recon_path, dictionary_validation_path, report_path]},
}
(OUT / 'analysis_audit.json').write_text(json.dumps(audit, indent=2, default=str) + '\n')
print(result.to_string(index=False))
print('\nCKD coding reconciliation:')
print(recon.to_string(index=False))
print(f'\nOld mortality model reproduced: {old_mort_reproduced}')
print(f'Report: {report_path}')
