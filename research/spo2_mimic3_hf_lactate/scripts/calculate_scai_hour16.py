#!/usr/bin/env python3
"""Retrospective hour-16 Kapur/CSWG minimum evidenced SCAI staging.

Reuses the fixed SpO2 cohort. Patient-linked caches remain outside the repo.
No missing physiology, unobserved OHCA, or unspecified ECMO is called normal.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import time
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import rerun_support_adjusted_models as support

REFERENCE='https://www.jacc.org/doi/10.1016/j.jacc.2022.04.049'
DRUGS={
 'norepinephrine':r'norepinephrine|noradrenaline|levophed',
 'epinephrine':r'(?<!nor)epinephrine|adrenalin',
 'phenylephrine':r'phenylephrine|neosynephrine|neo-synephrine',
 'dopamine':r'dopamine','dobutamine':r'dobutamine|dobutrex',
 'milrinone':r'milrinone|primacor','vasopressin':r'vasopressin'}
SBP={51,442,455,3313,3315,3317,3319,3321,3323,6701,224167,227243,220050,220179,225309}
MAP={52,443,456,224,3312,3314,3316,3318,3320,3322,6590,6702,6927,224322,220052,220181,225312}
# Explicit functional/setting evidence; line-site/dressing entries are not support.
DEVICES={225:'iabp',224:'iabp',224322:'iabp',227980:'iabp',225982:'iabp',
 429:'lvad',600:'rvad',220128:'rvad',228154:'impella',228198:'tandem',228189:'tandem'}
DEVICE_PROCEDURES={224272:'iabp',228169:'impella',228201:'tandem',228202:'tandem'}
ECMO={2957,3265,5798,5931,5937,6758,6931,7015,7018,7449,224660,228193}
LABS={50813:'lactate',50861:'alt',50820:'ph'}

def sql(p): return "'"+str(p).replace("'","''")+"'"
def ids(values): return ','.join(str(v) for v in sorted(values))

def cached_query(con,query,path,key):
    metadata=path.with_suffix('.metadata.json')
    expected={'query_sha256':hashlib.sha256(query.encode()).hexdigest(),**key}
    if not (path.exists() and metadata.exists() and json.loads(metadata.read_text())==expected):
        con.execute(f'COPY ({query}) TO {sql(path)} (FORMAT PARQUET)')
        metadata.write_text(json.dumps(expected,indent=2))
    return con.execute(f'SELECT * FROM read_parquet({sql(path)})').df()

def extract(frame,clin,cache,fingerprints):
    stays=pd.read_csv(clin/'ICUSTAYS.csv.gz',usecols=['SUBJECT_ID','HADM_ID','ICUSTAY_ID','INTIME','OUTTIME','DBSOURCE'],parse_dates=['INTIME','OUTTIME'])
    stays=frame[['SUBJECT_ID','HADM_ID','ICUSTAY_ID']].merge(stays,on=['SUBJECT_ID','HADM_ID','ICUSTAY_ID'],validate='one_to_one')
    items=pd.read_csv(clin/'D_ITEMS.csv.gz')
    maps={}
    for drug,pattern in DRUGS.items():
        selected=items.loc[items.LINKSTO.fillna('').str.startswith('inputevents') & items.LABEL.fillna('').str.contains(pattern,case=False,regex=True)]
        for item in selected.ITEMID.astype(int):
            assert item not in maps,(item,drug,maps.get(item))
            maps[item]=drug
    assert maps[221906]=='norepinephrine' and maps[221289]=='epinephrine'
    con=duckdb.connect();con.execute("SET threads=4; SET memory_limit='2GB'; SET preserve_insertion_order=false;")
    con.register('stays',stays)
    key={'source_hashes':fingerprints,'stays_sha256':hashlib.sha256(stays.to_csv(index=False).encode()).hexdigest()}
    linkage="try_cast(p.SUBJECT_ID AS BIGINT)=s.SUBJECT_ID AND (p.HADM_ID IS NULL OR try_cast(p.HADM_ID AS BIGINT)=s.HADM_ID)"
    chart_ids=SBP|MAP|set(DEVICES)|ECMO
    query=f"""SELECT s.ICUSTAY_ID,try_cast(p.ITEMID AS BIGINT) AS itemid,
      date_diff('second',s.INTIME,try_cast(p.CHARTTIME AS TIMESTAMP))/60.0 AS minute,
      try_cast(p.VALUENUM AS DOUBLE) AS value,p.VALUE AS value_text,
      try_cast(p.SUBJECT_ID AS BIGINT) AS source_subject,
      try_cast(p.HADM_ID AS BIGINT) AS source_admission,
      s.SUBJECT_ID AS expected_subject,s.HADM_ID AS expected_admission
      FROM read_csv({sql(clin/'CHARTEVENTS.csv.gz')},header=true,all_varchar=true) p
      JOIN stays s ON try_cast(p.ICUSTAY_ID AS BIGINT)=s.ICUSTAY_ID
      WHERE try_cast(p.CHARTTIME AS TIMESTAMP)>=s.INTIME
        AND try_cast(p.CHARTTIME AS TIMESTAMP)<=s.INTIME+INTERVAL '16 hours'
        AND try_cast(p.ITEMID AS BIGINT) IN ({ids(chart_ids)})
        AND try_cast(p.ERROR AS DOUBLE) IS DISTINCT FROM 1"""
    print('Extracting selected BP and circulatory-support chart records through hour 16...',flush=True)
    chart=cached_query(con,query,cache/'chart_through16.parquet',key)
    mismatch=chart.source_subject.ne(chart.expected_subject)|(chart.source_admission.notna()&chart.source_admission.ne(chart.expected_admission))
    chart_audit={'selected_candidates':len(chart),'excluded_linkage_mismatch_rows':int(mismatch.sum())}
    chart=chart.loc[~mismatch].copy()
    query=f"""SELECT s.ICUSTAY_ID,try_cast(p.ITEMID AS BIGINT) AS itemid,
      date_diff('second',s.INTIME,try_cast(p.CHARTTIME AS TIMESTAMP))/60.0 AS minute,
      try_cast(p.VALUENUM AS DOUBLE) AS value,p.VALUEUOM AS unit
      FROM read_csv({sql(clin/'LABEVENTS.csv.gz')},header=true,all_varchar=true) p
      JOIN stays s ON try_cast(p.HADM_ID AS BIGINT)=s.HADM_ID
      WHERE {linkage} AND try_cast(p.ITEMID AS BIGINT) IN ({ids(LABS)})
        AND try_cast(p.CHARTTIME AS TIMESTAMP)>=s.INTIME
        AND try_cast(p.CHARTTIME AS TIMESTAMP)<=s.INTIME+INTERVAL '16 hours'"""
    print('Extracting lactate, ALT and blood pH through hour 16...',flush=True)
    labs=cached_query(con,query,cache/'labs_through16.parquet',key)
    units=labs.unit.fillna('').str.lower().str.replace(' ','',regex=False)
    mgdl=labs.itemid.eq(50813)&units.isin(['mg/dl','mgperdl'])
    labs.loc[mgdl,'value']/=9.008
    allowed=(labs.itemid.eq(50813)&units.isin(['','mmol/l','mmolperliter','mmoll-1','mg/dl','mgperdl']))|labs.itemid.eq(50820)|(labs.itemid.eq(50861)&units.isin(['','iu/l','u/l','units/l']))
    lab_audit={'selected_candidates':len(labs),'excluded_unrecognized_unit_rows':int((~allowed).sum()),'blood_ph_item':50820}
    labs=labs.loc[allowed].copy()
    drug_parts=[]
    for name in ['INPUTEVENTS_CV','INPUTEVENTS_MV']:
        mv=name.endswith('MV')
        timing=("try_cast(p.STARTTIME AS TIMESTAMP)<=s.INTIME+INTERVAL '16 hours' AND try_cast(p.ENDTIME AS TIMESTAMP)>s.INTIME" if mv else "try_cast(p.CHARTTIME AS TIMESTAMP)>=s.INTIME AND try_cast(p.CHARTTIME AS TIMESTAMP)<=s.INTIME+INTERVAL '16 hours'")
        start="STARTTIME" if mv else 'CHARTTIME'
        end="date_diff('second',s.INTIME,try_cast(p.ENDTIME AS TIMESTAMP))/60.0" if mv else 'NULL::DOUBLE'
        extra="AND coalesce(p.STATUSDESCRIPTION,'')!='Rewritten' AND coalesce(try_cast(p.CANCELREASON AS INTEGER),0)=0" if mv else ''
        query=f"""SELECT s.ICUSTAY_ID,try_cast(p.ITEMID AS BIGINT) AS itemid,
          date_diff('second',s.INTIME,try_cast(p.{start} AS TIMESTAMP))/60.0 AS start_minute,
          {end} AS end_minute,try_cast(p.RATE AS DOUBLE) AS rate
          FROM read_csv({sql(clin/(name+'.csv.gz'))},header=true,all_varchar=true) p
          JOIN stays s ON try_cast(p.ICUSTAY_ID AS BIGINT)=s.ICUSTAY_ID
          WHERE {linkage} AND try_cast(p.ITEMID AS BIGINT) IN ({ids(maps)})
          AND {timing} {extra}"""
        print('Extracting time-specific '+name+' vasoactive records...',flush=True)
        drugs=cached_query(con,query,cache/(name.lower()+'_through16.parquet'),key)
        drugs['drug']=drugs.itemid.map(maps);drugs['system']='MV' if mv else 'CV';drug_parts.append(drugs)
    query=f"""SELECT s.ICUSTAY_ID,try_cast(p.ITEMID AS BIGINT) AS itemid,
      date_diff('second',s.INTIME,try_cast(p.STARTTIME AS TIMESTAMP))/60.0 AS start_minute,
      date_diff('second',s.INTIME,try_cast(p.ENDTIME AS TIMESTAMP))/60.0 AS end_minute
      FROM read_csv({sql(clin/'PROCEDUREEVENTS_MV.csv.gz')},header=true,all_varchar=true) p
      JOIN stays s ON try_cast(p.ICUSTAY_ID AS BIGINT)=s.ICUSTAY_ID
      WHERE {linkage} AND try_cast(p.ITEMID AS BIGINT) IN ({ids(DEVICE_PROCEDURES)})
      AND try_cast(p.STARTTIME AS TIMESTAMP)<=s.INTIME+INTERVAL '16 hours'
      AND try_cast(p.ENDTIME AS TIMESTAMP)>s.INTIME
      AND coalesce(p.STATUSDESCRIPTION,'')!='Rewritten'
      AND coalesce(try_cast(p.CANCELREASON AS INTEGER),0)=0"""
    procedures=cached_query(con,query,cache/'mcs_procedures_through16.parquet',key)
    con.close()
    mapping={'reference':REFERENCE,'drug_item_map':{str(k):v for k,v in maps.items()},
      'sbp_items':sorted(SBP),'map_items':sorted(MAP),'device_chart_items':DEVICES,
      'device_procedure_items':DEVICE_PROCEDURES,'ecmo_unspecified_items':sorted(ECMO),
      'lab_items':LABS,'chart_audit':chart_audit,'lab_audit':lab_audit}
    return stays,chart,labs,pd.concat(drug_parts,ignore_index=True),procedures,mapping

def summarize_values(chart,labs,endpoint,lookback):
    measurements=[]
    for concept,itemset,low,high,src in [('sbp',SBP,20,300,chart),('map',MAP,10,250,chart),
      ('lactate',{50813},0,50,labs),('alt',{50861},0,100000,labs),('ph',{50820},6.5,8,labs)]:
        part=src.loc[src.itemid.isin(itemset)&src.minute.between(endpoint-lookback,endpoint,inclusive='both')&src.value.between(low,high)].copy()
        # Same-time item/source duplicates collapse before selecting latest time.
        part=part.groupby(['ICUSTAY_ID','minute'],as_index=False).value.median().sort_values('minute')
        latest=part.drop_duplicates('ICUSTAY_ID',keep='last')[['ICUSTAY_ID','minute','value']]
        latest=latest.rename(columns={'value':concept,'minute':concept+'_minute'})
        measurements.append(latest.set_index('ICUSTAY_ID'))
    return pd.concat(measurements,axis=1)

def treatment_state(stays,chart,drugs,procedures,endpoint):
    result=stays[['ICUSTAY_ID','DBSOURCE']].set_index('ICUSTAY_ID').copy()
    selected=drugs.loc[drugs.system.eq('MV')&drugs.start_minute.le(endpoint)&drugs.end_minute.gt(endpoint)&drugs.rate.gt(0)].copy()
    cv=drugs.loc[drugs.system.eq('CV')&drugs.start_minute.between(endpoint-60,endpoint,inclusive='both')].copy()
    cv=cv.groupby(['ICUSTAY_ID','drug','start_minute'],as_index=False).rate.max().sort_values('start_minute').drop_duplicates(['ICUSTAY_ID','drug'],keep='last')
    selected=pd.concat([selected[['ICUSTAY_ID','drug']],cv.loc[cv.rate.gt(0),['ICUSTAY_ID','drug']]],ignore_index=True).drop_duplicates()
    result['drug_count_documented']=selected.groupby('ICUSTAY_ID').drug.nunique().reindex(result.index,fill_value=0)
    result['drug_names']=selected.groupby('ICUSTAY_ID').drug.agg(lambda x:'|'.join(sorted(x))).reindex(result.index,fill_value='')
    ch=chart.loc[chart.minute.between(endpoint-240,endpoint,inclusive='both')].copy()
    # Chart evidence is recent documentation, not an exact device interval.
    dev=ch.loc[ch.itemid.isin(DEVICES)&ch.value.gt(0),['ICUSTAY_ID','itemid']].copy();dev['device']=dev.itemid.map(DEVICES)
    pe=procedures.loc[procedures.start_minute.le(endpoint)&procedures.end_minute.gt(endpoint),['ICUSTAY_ID','itemid']].copy();pe['device']=pe.itemid.map(DEVICE_PROCEDURES)
    documented=pd.concat([dev[['ICUSTAY_ID','device']],pe[['ICUSTAY_ID','device']]],ignore_index=True).drop_duplicates()
    result['device_count_documented']=documented.groupby('ICUSTAY_ID').device.nunique().reindex(result.index,fill_value=0)
    result['device_names']=documented.groupby('ICUSTAY_ID').device.agg(lambda x:'|'.join(sorted(x))).reindex(result.index,fill_value='')
    ecmo=ch.loc[ch.itemid.isin(ECMO)&(ch.value.gt(0)|ch.value_text.fillna('').str.lower().isin(['yes','on','active']))]
    result['unspecified_ecmo_evidence']=result.index.isin(ecmo.ICUSTAY_ID)
    result['recent_chart_observed']=result.index.isin(ch.ICUSTAY_ID)
    return result.drop(columns='DBSOURCE')

def abnormal(row):
    return any(pd.notna(row.get(k)) and fn(row[k]) for k,fn in {
      'sbp':lambda v:v<=90,'map':lambda v:v<=65,'lactate':lambda v:v>=2,
      'alt':lambda v:v>=200,'ph':lambda v:v<=7.2}.items())

def stage(row):
    def observed(k): return pd.notna(row.get(k))
    def between(k,a,b): return observed(k) and a<=row[k]<=b
    hypotension=between('sbp',60,90) or between('map',50,65)
    severe_bp=(observed('sbp') and row.sbp<60) or (observed('map') and row['map']<50)
    hypo=between('lactate',2,10) or (observed('alt') and row.alt>=200)
    worsened=between('lactate',5,10) or (observed('alt') and row.alt>500)
    severe_perf=(observed('lactate') and row.lactate>10) or (observed('ph') and row.ph<=7.2)
    drugs=int(row.drug_count_documented);devices=int(row.device_count_documented);total=drugs+devices
    if severe_bp or severe_perf or drugs>3 or devices>3: return 'E','extremis_threshold'
    if hypotension and worsened: return 'D','hypotension_and_worsened_hypoperfusion'
    if total>=2: return 'D','two_or_more_documented_drugs_or_devices'
    if total==1 and row.persistent_abnormality: return 'D','one_therapy_with_4h_and_16h_persistent_abnormality_proxy'
    if hypotension and hypo: return 'C','hypotension_and_hypoperfusion'
    if total==1: return 'C','one_documented_drug_or_device'
    if hypotension or hypo: return 'B','isolated_hypotension_or_hypoperfusion'
    # OHCA/physical examination are not established: never certify normal A.
    return 'unclassified','no_B_to_E_evidence_but_stage_A_not_confirmable'

def main():
    parser=argparse.ArgumentParser()
    for name in ['source-output-root','clinical-root','private-cache-root','output-root','input-manifest']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();start=time.time();args.private_cache_root.mkdir(parents=True,exist_ok=True);args.output_root.mkdir(parents=True,exist_ok=True)
    frame=support.locked_frame(args.source_output_root,args.clinical_root)
    old_support=pd.read_csv(args.source_output_root/'10_support_adjustment_cache/support_by_stay.csv',usecols=['ICUSTAY_ID','mechanical_ventilation_0_4h'])
    frame=frame.merge(old_support,on='ICUSTAY_ID',validate='one_to_one')
    frame['primary_lactate_sample']=frame.outcome_available_12h.fillna(False).astype(bool)&frame.mechanical_ventilation_0_4h.notna()
    frame['mortality_sample']=frame.baseline_lactate_mmol_l.notna()&frame.mechanical_ventilation_0_4h.notna()
    assert frame.primary_lactate_sample.sum()==380 and frame.mortality_sample.sum()==900
    manifest=pd.read_csv(args.input_manifest).set_index('filename');fingerprints={}
    for name in ['ICUSTAYS','ADMISSIONS','D_ITEMS','D_LABITEMS','CHARTEVENTS','LABEVENTS','INPUTEVENTS_CV','INPUTEVENTS_MV','PROCEDUREEVENTS_MV']:
        f=args.clinical_root/(name+'.csv.gz');print('Verifying '+f.name,flush=True);h=support.sha256(f)
        assert h==manifest.loc[f.name,'sha256'],f.name
        fingerprints[f.name]=h
    stays,chart,labs,drugs,procedures,mapping=extract(frame,args.clinical_root,args.private_cache_root,fingerprints)
    admissions=pd.read_csv(args.clinical_root/'ADMISSIONS.csv.gz',usecols=['SUBJECT_ID','HADM_ID','DISCHTIME','DEATHTIME'],parse_dates=['DISCHTIME','DEATHTIME'])
    frame=frame.merge(stays.drop(columns=['SUBJECT_ID','HADM_ID']),on='ICUSTAY_ID',validate='one_to_one').merge(admissions,on=['SUBJECT_ID','HADM_ID'],validate='many_to_one')
    deadline=frame.INTIME+pd.Timedelta(hours=16)
    frame['status_at16']=np.select([frame.DEATHTIME.notna()&frame.DEATHTIME.le(deadline),frame.DISCHTIME.notna()&frame.DISCHTIME.le(deadline),frame.OUTTIME.notna()&frame.OUTTIME.le(deadline)],['died_by_hour16','discharged_by_hour16','left_icu_by_hour16'],default='in_icu_at_hour16')
    early=summarize_values(chart,labs,240,240).join(treatment_state(stays,chart,drugs,procedures,240))
    early['early_abnormal']=early.apply(abnormal,axis=1)
    distributions=[];component_rows=[];all_frames=[]
    for lookback in [240,60,360]:
        current=frame.merge(summarize_values(chart,labs,960,lookback).reset_index(),on='ICUSTAY_ID',how='left',validate='one_to_one').merge(treatment_state(stays,chart,drugs,procedures,960).reset_index(),on='ICUSTAY_ID',validate='one_to_one')
        prior=current.ICUSTAY_ID.map(early.early_abnormal).fillna(False)
        early_treatment=current.ICUSTAY_ID.map(early.drug_count_documented+early.device_count_documented).fillna(0)
        current['persistent_abnormality']=prior&current.apply(abnormal,axis=1)&early_treatment.gt(0)
        assigned=current.apply(stage,axis=1);current['stage_minimum_evidenced']=[r[0] for r in assigned];current['stage_evidence']=[r[1] for r in assigned]
        current.loc[current.status_at16.ne('in_icu_at_hour16'),'stage_minimum_evidenced']=current.loc[current.status_at16.ne('in_icu_at_hour16'),'status_at16']
        current['measurement_lookback_minutes']=lookback;current['lower_bound_only']=True
        for population,mask in [('all_spo2_eligible',current.index==current.index),('primary_lactate_model',current.primary_lactate_sample),('mortality_model',current.mortality_sample)]:
            for group,exposure_mask in [('all',current.index==current.index),('instability',current.exposure.eq(1)),('no_instability',current.exposure.eq(0))]:
                sample=current.loc[mask&exposure_mask];inicu=sample.status_at16.eq('in_icu_at_hour16');classified=sample.stage_minimum_evidenced.isin(list('ABCDE'))
                for label in list('ABCDE')+['unclassified','died_by_hour16','discharged_by_hour16','left_icu_by_hour16']:
                    n=int(sample.stage_minimum_evidenced.eq(label).sum())
                    distributions.append({'population':population,'exposure_group':group,'measurement_lookback_minutes':lookback,'category':label,'n':n,'cohort_n':len(sample),'in_icu_at16_n':int(inicu.sum()),'classified_n':int(classified.sum()),'percent_of_cohort':100*n/len(sample),'percent_of_classified':100*n/classified.sum() if classified.sum() and label in 'ABCDE' else np.nan})
            sample=current.loc[mask&current.status_at16.eq('in_icu_at_hour16')]
            for concept in ['sbp','map','lactate','alt','ph']:
                component_rows.append({'population':population,'measurement_lookback_minutes':lookback,'component':concept,'in_icu_n':len(sample),'available_n':int(sample[concept].notna().sum())})
        all_frames.append(current)
    patient=pd.concat(all_frames,ignore_index=True);patient.to_csv(args.private_cache_root/'patient_scai_hour16.csv',index=False)
    distribution=pd.DataFrame(distributions);distribution.to_csv(args.output_root/'stage_distribution.csv',index=False)
    pd.DataFrame(component_rows).to_csv(args.output_root/'component_availability.csv',index=False)
    audit={'execution':'local retrospective descriptive analysis; not Google Colab','completed_utc':pd.Timestamp.now(tz='UTC').isoformat(),'runtime_seconds':time.time()-start,'code_sha256':support.sha256(__file__),'source_hashes':fingerprints,'mapping':mapping,'primary_time':'ICU minute 960; latest valid physiology in [720,960]; no measurements after 960','treatment_time':'MV intervals active at minute 960; CV latest per-drug chart in [900,960]; device intervals active at 960 or recent chart evidence in [720,960]','limitations':['minimum evidenced Kapur/CSWG stage, not clinician-adjudicated SCAI','OHCA unavailable; stage A cannot be confirmed; every assigned stage is a lower bound','Unspecified ECMO not counted as cardiac MCS because VA versus VV unavailable','MCS chart evidence and CareVue drug records are documentation proxies','Drug dose escalation and clinical treatment indication not adjudicated','Persistence proxy: abnormal at hour4 and hour16 with documented early treatment','Published >3 device threshold used for stage E; original comparator has >=3','Freshness sensitivity uses one-hour and six-hour physiology lookback; unchanged hour16 treatment definition'],'cohort_n':1597,'private_patient_cache':str(args.private_cache_root/'patient_scai_hour16.csv')}
    (args.output_root/'analysis_audit.json').write_text(json.dumps(audit,indent=2,default=str))
    primary=distribution.loc[distribution.measurement_lookback_minutes.eq(240)&distribution.exposure_group.eq('all')]
    report=['# Hour-16 minimum evidenced Kapur/CSWG SCAI distribution','','Local descriptive calculation on the fixed 1,597-stay SpO2 cohort; separate rows also show the 380-stay lactate and 900-stay mortality samples. Stage at hour 16 uses latest valid physiology during hours 12–16 and time-specific treatment evidence. Measurements after hour 16 are excluded.','','**This is an EHR-derived lower-bound classification. Stage A cannot be certified because OHCA and complete examination information are unavailable. Zero assigned A observations does not mean zero clinically stage-A patients.**','','| Population | Category | N | Cohort N | % of cohort |','|---|---|---:|---:|---:|']
    for r in primary.itertuples(index=False): report.append(f'| {r.population} | {r.category} | {r.n} | {r.cohort_n} | {r.percent_of_cohort:.2f}% |')
    report+=['','## Methods and limitations','',f'- Reference: {REFERENCE}','- Highest evidenced stage uses the existing project Kapur/CSWG thresholds; treatment count includes vasopressors and inotropes plus distinct documented cardiac-device types. Stage E uses the published >3 drugs or >3 devices threshold.','- Stage D persistence requires abnormal early and current physiology with early treatment; this remains a retrospective proxy for failure to stabilize.','- Same-time numeric duplicates are collapsed by their median. Blood pH uses item 50820; urine and other-fluid pH are excluded. Lactate uses item 50813, and ALT 50861.','- SBP plausible range 20–300 mmHg, MAP 10–250, lactate 0–50 mmol/L, ALT 0–100,000 U/L, blood pH 6.5–8.0. These filters and lookbacks are explicit operational choices.','- Missing components never become normal. Patients dead, discharged, or outside the index ICU by hour 16 appear separately.','- Stage A is unavailable under the strict completeness rule. B–E categories indicate minimum evidenced severity; missing components could imply a higher stage.','- Non-cardiac causes of hypotension, lactate elevation, or vasoactive use are not adjudicated. This does not prove cardiogenic shock in every classified patient.','- Full stage, exposure-stratified distributions and 1-/6-hour physiology-lookback sensitivities are in stage_distribution.csv. Component coverage is in component_availability.csv.','- Patient-linked results remain in the private cache; repository tables contain aggregate results only.']
    (args.output_root/'REPORT.md').write_text('\n'.join(report)+'\n')
    print(primary[['population','category','n','cohort_n','percent_of_cohort']].to_string(index=False),flush=True)

if __name__=='__main__': main()
