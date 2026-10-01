#!/usr/bin/env python3
"""Add early SpO2 level and organ support to the frozen primary models.

Only new support covariates are extracted. Cohort, exposure, baseline lactate,
corrected complete-window lactate labels, and prior-diagnosis rules are reused.
Patient-level caches are written only to the explicitly supplied private cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

import run_dense_waveform_analysis as core

VENT_SETTINGS = {445,448,449,450,1340,1486,1600,224687,639,654,681,682,
    683,684,224685,224684,224686,218,436,535,444,224697,224695,224696,
    224746,224747,221,1,1211,1655,2000,226873,224738,224419,224750,
    227187,543,5865,5866,224707,224709,224705,224706,60,437,505,506,
    686,220339,224700,3459,501,502,503,224702,223,667,668,669,670,
    671,672,224701}
RESP_ITEMS = VENT_SETTINGS | {720,223848,223849,467,226260,640,468,469,
    470,471,227287,226732,223834}
VENT_SOURCE_URL = 'https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/durations/ventilation_classification.sql'


def sha256(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def sql_path(p):
    return "'" + str(p).replace("'", "''") + "'"


def idlist(ids):
    return ','.join(map(str, sorted(ids)))


def locked_frame(src, clin):
    sig = pd.read_csv(src/'02_signal_qc/signal_features_recomputed.csv')
    sig = sig.loc[sig.dynamics_eligible.astype(bool), ['SUBJECT_ID','HADM_ID',
        'ICUSTAY_ID','exposure_abs_jump_ge4','mean_binned_spo2']].copy()
    epi = pd.read_csv(src/'04_episode_lactate/episode_anchor_records.csv',
        usecols=['SUBJECT_ID','HADM_ID','ICUSTAY_ID','age_years','sex_male'])
    frame = sig.merge(epi, on=['SUBJECT_ID','HADM_ID','ICUSTAY_ID'], validate='one_to_one')
    frame['exposure'] = frame.exposure_abs_jump_ge4.astype(int)
    outcomes = pd.read_csv(src/'03_landmark_lactate/lactate_outcomes.csv',
        usecols=['SUBJECT_ID','ICUSTAY_ID','baseline_lactate_mmol_l',
                 'outcome_available_12h','lactate_rise_ge0_5_12h'])
    frame = frame.merge(outcomes, on=['SUBJECT_ID','ICUSTAY_ID'], validate='one_to_one')
    dx = pd.read_csv(clin/'DIAGNOSES_ICD.csv.gz', usecols=['SUBJECT_ID','HADM_ID','ICD9_CODE'],
                     dtype={'ICD9_CODE':'string'})
    code = dx.ICD9_CODE.fillna('').str.strip().str.upper().str.replace(r'[^A-Z0-9]','',regex=True)
    p3 = pd.to_numeric(code.str[:3], errors='coerce')
    dx['ckd_expanded'] = p3.isin([585,403,404,582]) | code.str.startswith('2504') | code.isin(['75312','75313','75314'])
    dx['esrd_lactate'] = code.str.startswith('5856') | code.eq('V4511')
    dx['esrd_mortality'] = code.str.startswith('5856') | code.isin(['V4511','V4512','V560','V5631','V5632','V568'])
    dx['lung_lactate'] = p3.between(490,496) | p3.between(500,508)
    dx['lung_mortality'] = p3.between(490,505) | code.isin(['4168','4169','5064','5081','5088','515','51883','51884'])
    dx['liver_prior'] = p3.eq(571)
    dx['diabetes_prior'] = p3.eq(250)
    flags = ['ckd_expanded','esrd_lactate','esrd_mortality','lung_lactate',
             'lung_mortality','liver_prior','diabetes_prior']
    adm = pd.read_csv(clin/'ADMISSIONS.csv.gz', usecols=['SUBJECT_ID','HADM_ID','ADMITTIME','HOSPITAL_EXPIRE_FLAG'],
                      parse_dates=['ADMITTIME'])
    assert not adm.HADM_ID.duplicated().any()
    adm = adm.merge(dx.groupby(['SUBJECT_ID','HADM_ID'],as_index=False)[flags].max(),
                    on=['SUBJECT_ID','HADM_ID'],how='left',validate='one_to_one')
    adm[flags] = adm[flags].fillna(False).astype(bool)
    history = adm.groupby(['SUBJECT_ID','ADMITTIME'],as_index=False)[flags].max().sort_values(['SUBJECT_ID','ADMITTIME'],kind='mergesort')
    for col in flags:
        cum = history.groupby('SUBJECT_ID',sort=False)[col].cummax()
        history[col] = cum.groupby(history.SUBJECT_ID,sort=False).shift(1).fillna(False).astype(bool)
    prior = adm[['SUBJECT_ID','HADM_ID','ADMITTIME','HOSPITAL_EXPIRE_FLAG']].merge(
        history,on=['SUBJECT_ID','ADMITTIME'],how='left',validate='many_to_one')
    frame = frame.merge(prior.drop(columns='ADMITTIME'),on=['SUBJECT_ID','HADM_ID'],validate='many_to_one')
    frame['in_hospital_death'] = frame.HOSPITAL_EXPIRE_FLAG.astype(float)
    assert len(frame)==1597 and frame.exposure.sum()==539 and frame.mean_binned_spo2.notna().all()
    assert frame[flags].notna().all().all()
    return frame


def extract_support(frame, clin, cache, fingerprints):
    cache.mkdir(parents=True,exist_ok=True)
    table=cache/'support_by_stay.csv'; audit_path=cache/'support_source_audit.json'
    if table.exists() and audit_path.exists():
        audit=json.loads(audit_path.read_text())
        if audit.get('input_hashes')==fingerprints and audit.get('extractor_sha256')==sha256(__file__):
            print('Reusing validated support cache.',flush=True)
            return pd.read_csv(table),audit
    stays=pd.read_csv(clin/'ICUSTAYS.csv.gz', usecols=['SUBJECT_ID','HADM_ID','ICUSTAY_ID','INTIME','DBSOURCE'],parse_dates=['INTIME'])
    stays=frame[['SUBJECT_ID','HADM_ID','ICUSTAY_ID']].merge(stays,on=['SUBJECT_ID','HADM_ID','ICUSTAY_ID'],validate='one_to_one')
    assert len(stays)==1597 and stays.INTIME.notna().all()
    items=pd.read_csv(clin/'D_ITEMS.csv.gz')
    labels=items.LABEL.fillna('')
    pressors=items.loc[labels.str.contains(r'norepinephrine|noradrenaline|epinephrine|adrenaline|dopamine|vasopressin|phenylephrine|levophed|neosynephrine|neo-synephrine',case=False,regex=True) & items.LINKSTO.fillna('').str.startswith('inputevents')]
    inotropes=items.loc[labels.str.contains(r'dobutamine|milrinone',case=False,regex=True) & items.LINKSTO.fillna('').str.startswith('inputevents')]
    press_ids=set(pressors.ITEMID.astype(int)); all_ids=press_ids|set(inotropes.ITEMID.astype(int))
    assert {221906,221289,221662,221749,222315,30047,30127}.issubset(press_ids)
    for item,label in {225792:'Invasive Ventilation',225794:'Non-invasive Ventilation',226260:'Mechanically Ventilated'}.items():
        assert items.loc[items.ITEMID.eq(item),'LABEL'].eq(label).any()
    con=duckdb.connect()
    con.execute("SET threads=4; SET memory_limit='2GB'; SET preserve_insertion_order=false;")
    con.register('stays',stays)
    chart=sql_path(clin/'CHARTEVENTS.csv.gz')
    print('Extracting early ventilation evidence from the complete CHARTEVENTS source...',flush=True)
    t=time.time()
    early_path=cache/'early_chart_support.parquet'
    early_meta=cache/'early_chart_support_metadata.json'
    extraction_query=f"""
          SELECT s.ICUSTAY_ID, try_cast(c.ITEMID AS BIGINT) AS itemid, c.VALUE AS value,
                 try_cast(c.VALUENUM AS DOUBLE) AS valuenum,
                 try_cast(c.SUBJECT_ID AS BIGINT) AS source_subject,
                 try_cast(c.HADM_ID AS BIGINT) AS source_admission,
                 s.SUBJECT_ID AS expected_subject, s.HADM_ID AS expected_admission
          FROM read_csv({chart},header=true,all_varchar=true) c
          JOIN stays s ON try_cast(c.ICUSTAY_ID AS BIGINT)=s.ICUSTAY_ID
          WHERE try_cast(c.CHARTTIME AS TIMESTAMP)>=s.INTIME
            AND try_cast(c.CHARTTIME AS TIMESTAMP)<s.INTIME+INTERVAL '4 hours'
            AND try_cast(c.ERROR AS DOUBLE) IS DISTINCT FROM 1
    """
    early_key={'source_hashes':fingerprints,'query_sha256':hashlib.sha256(extraction_query.encode()).hexdigest(),
               'stay_map_sha256':hashlib.sha256(stays.to_csv(index=False).encode()).hexdigest()}
    if not (early_path.exists() and early_meta.exists() and json.loads(early_meta.read_text())==early_key):
        con.execute(f"COPY ({extraction_query}) TO {sql_path(early_path)} (FORMAT PARQUET)")
        early_meta.write_text(json.dumps(early_key,indent=2))
    linkage=con.execute(f"""SELECT count(*) AS candidate_rows,
       count(*) FILTER(WHERE source_subject IS DISTINCT FROM expected_subject) AS subject_mismatch_rows,
       count(*) FILTER(WHERE source_admission IS NOT NULL AND source_admission!=expected_admission) AS admission_mismatch_rows,
       count(*) FILTER(WHERE source_admission IS NULL) AS missing_admission_rows,
       count(*) FILTER(WHERE source_subject IS DISTINCT FROM expected_subject OR
          (source_admission IS NOT NULL AND source_admission!=expected_admission)) AS excluded_inconsistent_rows
       FROM read_parquet({sql_path(early_path)})""").df().iloc[0].to_dict()
    ce=con.execute(f"""
        WITH early AS (SELECT * FROM read_parquet({sql_path(early_path)})
          WHERE source_subject=expected_subject
            AND (source_admission IS NULL OR source_admission=expected_admission))
        SELECT ICUSTAY_ID, count(*) AS early_chart_rows,
          count(*) FILTER(WHERE itemid IN ({idlist(RESP_ITEMS)})) AS early_respiratory_rows,
          max(CASE WHEN value IS NULL THEN 0
            WHEN itemid IN ({idlist(VENT_SETTINGS)}) THEN 1
            WHEN itemid=720 AND value!='Other/Remarks' THEN 1
            WHEN itemid=223848 AND value!='Other' THEN 1
            WHEN itemid=223849 THEN 1
            WHEN itemid=467 AND value='Ventilator' THEN 1
            WHEN itemid=226260 AND (valuenum>0 OR lower(trim(value)) IN ('yes','true','1')) THEN 1
            ELSE 0 END) AS mechanical_chart_evidence
        FROM early GROUP BY ICUSTAY_ID
    """).df()
    print(f'CHARTEVENTS extraction completed in {time.time()-t:.1f}s.',flush=True)
    print('Chart linkage audit: '+json.dumps(linkage,default=int),flush=True)
    proc=sql_path(clin/'PROCEDUREEVENTS_MV.csv.gz')
    pe=con.execute(f"""
        SELECT s.ICUSTAY_ID,
          max((try_cast(p.ITEMID AS BIGINT)=225792)::INTEGER) AS mechanical_procedure_evidence,
          max((try_cast(p.ITEMID AS BIGINT)=225794)::INTEGER) AS niv_documented_0_4h
        FROM read_csv({proc},header=true,all_varchar=true) p JOIN stays s
          ON try_cast(p.ICUSTAY_ID AS BIGINT)=s.ICUSTAY_ID
        WHERE try_cast(p.ITEMID AS BIGINT) IN (225792,225794)
          AND try_cast(p.SUBJECT_ID AS BIGINT)=s.SUBJECT_ID
          AND (p.HADM_ID IS NULL OR try_cast(p.HADM_ID AS BIGINT)=s.HADM_ID)
          AND try_cast(p.STARTTIME AS TIMESTAMP)<s.INTIME+INTERVAL '4 hours'
          AND try_cast(p.ENDTIME AS TIMESTAMP)>s.INTIME
          AND coalesce(try_cast(p.CANCELREASON AS INTEGER),0)=0
          AND coalesce(p.STATUSDESCRIPTION,'')!='Rewritten'
        GROUP BY s.ICUSTAY_ID
    """).df()
    result=stays[['ICUSTAY_ID','DBSOURCE']].merge(ce,on='ICUSTAY_ID',how='left',validate='one_to_one').merge(pe,on='ICUSTAY_ID',how='left',validate='one_to_one')
    result[result.columns.difference(['ICUSTAY_ID','DBSOURCE'])]=result[result.columns.difference(['ICUSTAY_ID','DBSOURCE'])].fillna(0)
    result['mechanical_ventilation_0_4h']=result[['mechanical_chart_evidence','mechanical_procedure_evidence']].max(axis=1)
    # No procedure record alone is not sufficient to assign a support negative.
    result.loc[(result.early_chart_rows.eq(0)) & result.mechanical_ventilation_0_4h.eq(0),'mechanical_ventilation_0_4h']=np.nan
    inf=[]
    for name in ('INPUTEVENTS_CV','INPUTEVENTS_MV'):
        print('Extracting '+name+' early vasoactive infusions...',flush=True)
        path=sql_path(clin/(name+'.csv.gz'))
        timing=("try_cast(p.CHARTTIME AS TIMESTAMP)>=s.INTIME AND try_cast(p.CHARTTIME AS TIMESTAMP)<s.INTIME+INTERVAL '4 hours'" if name.endswith('CV') else
                "try_cast(p.STARTTIME AS TIMESTAMP)<s.INTIME+INTERVAL '4 hours' AND try_cast(p.ENDTIME AS TIMESTAMP)>s.INTIME AND coalesce(p.STATUSDESCRIPTION,'')!='Rewritten'")
        inf.append(con.execute(f"""
          SELECT s.ICUSTAY_ID,
            max((try_cast(p.ITEMID AS BIGINT) IN ({idlist(press_ids)}))::INTEGER) AS vasopressor_0_4h,
            1 AS vasoactive_0_4h
          FROM read_csv({path},header=true,all_varchar=true) p JOIN stays s
            ON try_cast(p.ICUSTAY_ID AS BIGINT)=s.ICUSTAY_ID
          WHERE try_cast(p.ITEMID AS BIGINT) IN ({idlist(all_ids)})
            AND try_cast(p.SUBJECT_ID AS BIGINT)=s.SUBJECT_ID
            AND (p.HADM_ID IS NULL OR try_cast(p.HADM_ID AS BIGINT)=s.HADM_ID)
            AND try_cast(p.RATE AS DOUBLE)>0 AND {timing}
          GROUP BY s.ICUSTAY_ID
        """).df())
    meds=pd.concat(inf,ignore_index=True).groupby('ICUSTAY_ID',as_index=False).max()
    result=result.merge(meds,on='ICUSTAY_ID',how='left',validate='one_to_one')
    result[['vasopressor_0_4h','vasoactive_0_4h']]=result[['vasopressor_0_4h','vasoactive_0_4h']].fillna(0)
    con.close()
    audit={'input_hashes':fingerprints,'extractor_sha256':sha256(__file__),
           'ventilation_mapping_source':VENT_SOURCE_URL,'chart_record_linkage_audit':linkage,
           'pressor_item_mapping':pressors[['ITEMID','LABEL','LINKSTO']].to_dict(orient='records'),
           'inotrope_item_mapping':inotropes[['ITEMID','LABEL','LINKSTO']].to_dict(orient='records'),
           'ventilation_setting_itemids':sorted(VENT_SETTINGS),
           'stays':len(result),'stays_without_early_chart_rows':int(result.early_chart_rows.eq(0).sum())}
    result.to_csv(table,index=False)
    audit_path.write_text(json.dumps(audit,indent=2))
    return result,audit


def checked_fit(frame,y,covariates):
    fit=core.modified_poisson(frame,y,'exposure',covariates,cluster_col='SUBJECT_ID')
    assert fit['status']=='available' and fit['converged']
    use=frame.dropna(subset=[y,'exposure','SUBJECT_ID']+covariates)
    X=np.column_stack([np.ones(len(use)),use.exposure.to_numpy(float)] +
        [(use[c].to_numpy(float)-use[c].to_numpy(float).mean())/use[c].to_numpy(float).std() for c in covariates])
    assert np.isfinite(X).all() and np.linalg.matrix_rank(X)==X.shape[1]
    assert fit['parameters']==X.shape[1] and set(fit['covariates'].split('|'))==set(covariates)
    yy=use[y].to_numpy(float)
    opt=minimize(lambda b:np.exp(X@b).sum()-yy@(X@b),
        np.r_[np.log(yy.mean()),np.zeros(X.shape[1]-1)],method='trust-exact',
        jac=lambda b:X.T@(np.exp(X@b)-yy),
        hess=lambda b:X.T@(X*np.exp(X@b)[:,None]),options={'gtol':1e-8})
    assert np.max(np.abs(X.T@(np.exp(X@opt.x)-yy)))<1e-5
    mu=np.exp(X@opt.x); bread=np.linalg.inv(X.T@(X*mu[:,None]))
    scores=X*(yy-mu)[:,None]; groups,unique=pd.factorize(use.SUBJECT_ID,sort=True)
    cluster=np.zeros((len(unique),X.shape[1]));np.add.at(cluster,groups,scores)
    cov=bread@(cluster.T@cluster)@bread*(len(unique)/(len(unique)-1))*((len(use)-1)/(len(use)-X.shape[1]))
    se=np.sqrt(np.diag(cov));ci=np.exp(opt.x[1]+np.array([-1,1])*1.96*se[1])
    assert np.allclose([np.exp(opt.x[1]),*ci],[fit['adjusted_RR'],fit['adjusted_RR_95CI_low'],fit['adjusted_RR_95CI_high']],rtol=1e-6,atol=1e-8)
    fit.update(design_rank=int(np.linalg.matrix_rank(X)),independent_likelihood_and_CI_check=True,
               max_absolute_score=float(np.max(np.abs(X.T@(mu-yy)))),predicted_mean_max=float(mu.max()))
    return fit,use


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-output-root',type=Path,required=True)
    parser.add_argument('--clinical-root',type=Path,required=True)
    parser.add_argument('--private-cache-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--reference-results',type=Path)
    parser.add_argument('--input-manifest',type=Path)
    args=parser.parse_args();started=time.time();out=args.output_root;out.mkdir(parents=True,exist_ok=True)
    frame=locked_frame(args.source_output_root,args.clinical_root)
    base=['age_years','sex_male','baseline_lactate_mmol_l','ckd_expanded']
    specs=[('primary_lactate_12h','lactate_rise_ge0_5_12h',base+['esrd_lactate','lung_lactate','liver_prior','diabetes_prior']),
           ('in_hospital_mortality','in_hospital_death',base+['esrd_mortality','lung_mortality','liver_prior','diabetes_prior'])]
    refs=pd.read_csv(args.reference_results) if args.reference_results else None
    fits=[]; baseline={}
    for label,y,covars in specs:
        use=frame.loc[frame.outcome_available_12h.fillna(False).astype(bool)].copy() if label=='primary_lactate_12h' else frame.copy()
        fit,sample=checked_fit(use,y,covars)
        expected=(388,50,27,155,23,233) if label=='primary_lactate_12h' else (911,159,79,345,80,566)
        observed=(len(sample),int(sample[y].sum()),int(sample.loc[sample.exposure.eq(1),y].sum()),int(sample.exposure.eq(1).sum()),int(sample.loc[sample.exposure.eq(0),y].sum()),int(sample.exposure.eq(0).sum()))
        assert observed==expected,(label,observed,expected)
        if refs is not None:
            old=refs.loc[refs.analysis.eq(label)&refs.specification.eq('after_CKD_expansion')].iloc[0]
            assert np.allclose([fit['adjusted_RR'],fit['adjusted_RR_95CI_low'],fit['adjusted_RR_95CI_high']],old[['adjusted_RR','robust_CI_low','robust_CI_high']].to_numpy(float),rtol=1e-8)
        baseline[label]=fit
    print('Both quoted baseline models and their exact denominators reproduced.',flush=True)
    names=['ADMISSIONS','DIAGNOSES_ICD','D_ITEMS','ICUSTAYS','CHARTEVENTS','PROCEDUREEVENTS_MV','INPUTEVENTS_CV','INPUTEVENTS_MV']
    expected_hashes=pd.read_csv(args.input_manifest).set_index('filename') if args.input_manifest else None
    fingerprints={}
    for name in names:
        p=args.clinical_root/(name+'.csv.gz');print('Verifying source '+p.name,flush=True)
        h=sha256(p);fingerprints[p.name]=h
        if expected_hashes is not None: assert h==expected_hashes.loc[p.name,'sha256'],p.name+' source hash differs'
    support,source_audit=extract_support(frame,args.clinical_root,args.private_cache_root,fingerprints)
    frame=frame.merge(support,on='ICUSTAY_ID',how='left',validate='one_to_one')
    counts=[];rows=[];reconciliation=[]
    for label,y,covars in specs:
        use=frame.loc[frame.outcome_available_12h.fillna(False).astype(bool)].copy() if label=='primary_lactate_12h' else frame.copy()
        expanded_sample=use.dropna(subset=[y,'exposure','SUBJECT_ID']+covars+['mean_binned_spo2','vasopressor_0_4h','mechanical_ventilation_0_4h'])
        for model,added in [('original_full_adjusted',[]),('plus_mean_SpO2',['mean_binned_spo2']),
            ('original_full_adjusted_matched_support_sample',[]),
            ('plus_mean_SpO2_matched_support_sample',['mean_binned_spo2']),
            ('plus_mean_SpO2_vasopressors_ventilation',['mean_binned_spo2','vasopressor_0_4h','mechanical_ventilation_0_4h']),
            ('sensitivity_plus_mean_SpO2_any_vasoactive_ventilation',['mean_binned_spo2','vasoactive_0_4h','mechanical_ventilation_0_4h'])]:
            model_frame=expanded_sample if 'matched_support_sample' in model else use
            fit,sample=checked_fit(model_frame,y,covars+added)
            exposure_groups={}
            for value in [0,1]:
                group=sample.loc[sample.exposure.eq(value)]
                exposure_groups[f'group_{value}_n']=len(group);exposure_groups[f'group_{value}_events']=int(group[y].sum())
            fit.update(analysis=label,specification=model,requested_covariates='|'.join(covars+added),
                removed_vs_original_n=baseline[label]['n']-fit['n'],**exposure_groups)
            rows.append(fit)
            if model=='plus_mean_SpO2_vasopressors_ventilation':
                original_sample=use.dropna(subset=[y,'exposure','SUBJECT_ID']+covars)
                removed=original_sample.loc[~original_sample.ICUSTAY_ID.isin(sample.ICUSTAY_ID)]
                assert removed.mechanical_ventilation_0_4h.isna().all()
                reconciliation.append({'analysis':label,'original_n':int(baseline[label]['n']),
                    'original_events':int(baseline[label]['events']),'expanded_n':len(sample),
                    'expanded_events':int(sample[y].sum()),'removed_n':len(removed),
                    'removed_events':int(removed[y].sum()),'removed_non_events':int(len(removed)-removed[y].sum()),
                    'removed_exposed_n':int(removed.exposure.eq(1).sum()),
                    'removed_unexposed_n':int(removed.exposure.eq(0).sum()),
                    'removal_reason':'no valid chart documentation in hours 0-4 and no positive invasive-ventilation procedure; ventilation status unavailable'})
                for exposure,group in sample.groupby('exposure'):
                    counts.append({'analysis':label,'exposure':int(exposure),'n':len(group),
                      'events':int(group[y].sum()),'vasopressor_n':int(group.vasopressor_0_4h.sum()),
                      'mechanical_ventilation_n':int(group.mechanical_ventilation_0_4h.sum()),
                      'any_vasoactive_n':int(group.vasoactive_0_4h.sum()),
                      'mean_SpO2_mean':float(group.mean_binned_spo2.mean()),
                      'early_chart_rows_min':int(group.early_chart_rows.min())})
    results=pd.DataFrame(rows);results.to_csv(out/'adjusted_models.csv',index=False)
    pd.DataFrame(counts).to_csv(out/'support_counts_by_exposure.csv',index=False)
    pd.DataFrame(reconciliation).to_csv(out/'denominator_reconciliation.csv',index=False)
    audit={'execution':'local; not Google Colab','completed_utc':pd.Timestamp.now(tz='UTC').isoformat(),
      'runtime_seconds':time.time()-started,'duckdb_version':duckdb.__version__,
      'source_audit':source_audit,'baseline_models_reproduced':baseline,
      'fixed_cohort_n':1597,'fixed_exposed_n':539,'denominator_reconciliation':reconciliation,
      'saved_endpoint_and_signal_hashes':{name:sha256(args.source_output_root/name) for name in
         ['02_signal_qc/signal_features_recomputed.csv','03_landmark_lactate/lactate_outcomes.csv','04_episode_lactate/episode_anchor_records.csv']},
      'no_source_cohort_exposure_or_endpoint_overwrite':True,'code_sha256':sha256(__file__)}
    (out/'analysis_audit.json').write_text(json.dumps(audit,indent=2,default=str))
    report=['# Primary models with early SpO2 level and organ support','',
      'Local analysis-only model rerun. Cohort, exposure, endpoint labels, windows, and prior-comorbidity definitions are fixed. New support covariates alone were reconstructed from the validated original clinical sources.','',
      '| Outcome | Model | N | Events | Adjusted RR | Patient-cluster robust 95% CI | p |',
      '|---|---|---:|---:|---:|---|---:|']
    for row in rows:
        report.append(f"| {row['analysis']} | {row['specification']} | {row['n']} | {row['events']} | {row['adjusted_RR']:.6f} | {row['adjusted_RR_95CI_low']:.6f}–{row['adjusted_RR_95CI_high']:.6f} | {row['p_value']:.6f} |")
    report += ['', '## Definitions and validation','',
      '- Original adjustment: age, sex, baseline lactate, expanded prior-coded CKD, prior-coded ESRD, chronic pulmonary disease, chronic liver disease, and diabetes. The existing model-specific ESRD and lung code sets are preserved.',
      '- SpO2 level: mean of valid 15-minute SpO2 medians during ICU minutes [0,240), identical to the previously reported mean-SpO2 sensitivity; not a single admission reading.',
      '- Vasopressor: any positive-rate norepinephrine/Levophed, epinephrine, dopamine, vasopressin, or phenylephrine/Neosynephrine infusion in [0,4) hours. CareVue uses chart times; MetaVision uses infusion-interval overlap, including infusions already running at ICU entry. Rewritten MetaVision rows are excluded. Pure dobutamine/milrinone are excluded from the main pressor indicator and included only in the any-vasoactive sensitivity.',
      '- Mechanical ventilation: any first-four-hour evidence from the MIMIC-III ventilation-setting classification, explicit positive numeric/text mechanically-ventilated checkbox (226260), or an overlapping noncancelled invasive-ventilation procedure (225792). Error-marked chart records and rewritten/cancelled procedures are excluded. Noninvasive procedure (225794) alone does not set this flag. This measures documented ventilation evidence; no procedure record alone is not treated as a negative.',
      f'- Ventilation-setting mapping: {VENT_SOURCE_URL}',
      '- Both original RRs/CIs and exact outcome/exposure counts were reproduced before adding support. Stays without valid early chart documentation or positive invasive-procedure evidence retain unavailable ventilation status and are excluded from the expanded complete-case model. Matched-sample original and mean-SpO2-only models distinguish sample changes from the added adjustment. Every model has a full-rank design, converged, and passed an independent likelihood and patient-cluster sandwich-CI calculation.',
      '- Modified Poisson with log link; sandwich covariance clustered by SUBJECT_ID, same finite-sample correction and two-sided Wald intervals as the original models. The lactate outcome still requires complete follow-up through its fixed horizon for events and non-events. Mortality remains index-admission in-hospital mortality.',
      '- The expanded lactate model has 49 events and 13 parameters (3.77 events/parameter). Its uncertainty and risk of overfitting require caution. Support and instability are both measured in hours 0–4; this adjustment establishes a conditional association, not a causal effect or demonstrated predictive improvement.',
      '- Patient-level support caches remain outside the public project. All result tables here are aggregate.']
    report += ['', '## Complete-case denominator reconciliation','',
      '| Outcome | Original N/events | Expanded N/events | Removed events | Removed non-events | Removed exposed/unexposed |',
      '|---|---:|---:|---:|---:|---:|']
    for r in reconciliation:
        report.append(f"| {r['analysis']} | {r['original_n']}/{r['original_events']} | {r['expanded_n']}/{r['expanded_events']} | {r['removed_events']} | {r['removed_non_events']} | {r['removed_exposed_n']}/{r['removed_unexposed_n']} |")
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    print(results[['analysis','specification','n','events','adjusted_RR','adjusted_RR_95CI_low','adjusted_RR_95CI_high','p_value']].to_string(index=False),flush=True)


if __name__=='__main__':
    main()
