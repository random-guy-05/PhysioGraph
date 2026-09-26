"""Build and execute the single raw-data biological discovery notebook."""
import argparse
import contextlib
import hashlib
import io
import json
import textwrap
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'PhysioGraph_Biological_Discovery.ipynb'
PLAN=ROOT/'docs/BIOLOGICAL_DISCOVERY_PLAN.md'
CONTEXT_PLAN=ROOT/'docs/SPO2_CIRCULATORY_CONTEXT_PLAN.md'
CELLS=[]

def cell(kind,source):
    c={'cell_type': kind,'id': f'biology-{len(CELLS):02d}','metadata': {},'source': textwrap.dedent(source).strip()}
    if kind=='code': c.update(execution_count=None,outputs=[])
    CELLS.append(c)

cell('markdown','''
# PhysioGraph: biological follow-up of SpO2 instability

**Active scope: MIMIC and eICU only.** The active experiment is the final
section, "Circulatory context of SpO2 instability", which develops the
existing SpO2-myocardial injury/decompensation findings. Earlier phosphate
cells retain the completed, inconclusive experiment for auditability.

Run all cells in order. Uses the original credentialed MIMIC/eICU files in
**My Drive/Data**, never toy or simulated patients. In Colab, mount your own
Drive when prompted. No patient data are transmitted to third-party services.
The delivered execution, when present, is **local**, not Colab.

The first candidate tests phosphate rise during lactate normalization against
subsequent hospital mortality. A significant association is insufficient to
establish a new mechanism, treatment benefit, or a paradigm-shifting result.
This notebook deliberately reports inadequate support and rejected hypotheses.
''')
cell('markdown',PLAN.read_text())
cell('code','''
import importlib.util, subprocess, sys
requirements={'duckdb':'duckdb==1.4.3','numpy':'numpy==2.2.6',
    'pandas':'pandas==2.2.3','scipy':'scipy==1.15.3','statsmodels':'statsmodels==0.14.4'}
missing=[p for m,p in requirements.items() if importlib.util.find_spec(m) is None]
if missing: subprocess.check_call([sys.executable,'-m','pip','install','-q',*missing])
import os, json, hashlib, time, platform
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import duckdb
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE=Path('/content/drive/MyDrive')
else:
    DRIVE=Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
ENVIRONMENT='google_colab' if IN_COLAB else 'local'
RAW={d:DRIVE/'Data'/d/'Full' for d in ['eICU','MIMIC']}
assert all(p.is_dir() for p in RAW.values()), 'Expected original databases under My Drive/Data'
PRIVATE=DRIVE/'Data'/'PhysioGraph_Biological_Discovery_20260905'
PRIVATE.mkdir(exist_ok=True)
OUT=Path.cwd()/'research'/'biological_discovery'
OUT.mkdir(parents=True,exist_ok=True)
START=datetime.now(timezone.utc).isoformat()
SEED=20260905
def write_json(name,value):
    p=OUT/name
    p.write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\\n')
def status(stage,state='running',**kwargs):
    write_json('run_status.json',dict(status=state,stage=stage,environment=ENVIRONMENT,
        started_utc=START,updated_utc=datetime.now(timezone.utc).isoformat(),**kwargs))
    print(stage,state,flush=True)
status('setup')
print('Data source:',DRIVE/'Data')
print('Patient-level intermediates:',PRIVATE)
con=duckdb.connect(str(PRIVATE/'clinical.duckdb'))
con.execute("SET threads=2")
con.execute("SET memory_limit='2GB'")
source_manifest={}
def sqlstr(x): return "'"+str(x).replace("'","''")+"'"
def rawview(dataset,name):
    p=RAW[dataset]/(name+'.csv')
    assert p.is_file(), str(p)
    s=p.stat()
    source_manifest[dataset+'/'+name]=dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    return f"read_csv({sqlstr(p)},header=true,all_varchar=true,sample_size=10000,strict_mode=true)"
''')
cell('code',f"PROTOCOL_SHA256={hashlib.sha256(PLAN.read_bytes()).hexdigest()!r}\nwrite_json('protocol.json',dict(sha256=PROTOCOL_SHA256,frozen_before_analysis=True,seed=SEED))")
cell('markdown','''
## Cohorts and chronology

One ICU stay per hospital encounter. HF is explicitly coded; cardiomyopathy or
shock alone do not qualify. Cohort entry requires adult age, ICU admission
within 24 hours of hospital admission, and survival/in-ICU presence through
12 hours. MIMIC HF codes are retrospective discharge diagnoses. Repeat
hospitalizations retain patient identifiers for clustered uncertainty.
''')
cell('code','''
status('eICU cohort')
p=rawview('eICU','patient'); dx=rawview('eICU','diagnosis'); adx=rawview('eICU','admissionDx')
hf_regex=r'(^|[|/ ,])(congestive heart failure|heart failure|chf)([|/ ,]|$)'
con.execute(f"""
CREATE OR REPLACE TABLE eicu_cohort AS
WITH hf AS (
 SELECT DISTINCT patientunitstayid FROM {dx}
 WHERE regexp_matches(lower(diagnosisstring),{sqlstr(hf_regex)})
 UNION SELECT DISTINCT patientunitstayid FROM {adx}
 WHERE regexp_matches(lower(coalesce(admitdxname,'')||' '||coalesce(admitdxtext,'')),{sqlstr(hf_regex)})
), ranked AS (
 SELECT *,row_number() OVER(PARTITION BY patienthealthsystemstayid
 ORDER BY try_cast(hospitaladmitoffset AS DOUBLE) DESC,try_cast(patientunitstayid AS BIGINT)) rn FROM {p}
)
SELECT try_cast(patientunitstayid AS BIGINT) stay_id,uniquepid patient_id,
 patienthealthsystemstayid encounter_id,hospitalid site,
 try_cast(replace(age,'>','') AS DOUBLE) age,
 CASE lower(gender) WHEN 'male' THEN 1.0 WHEN 'female' THEN 0.0 END male,
 CASE lower(hospitaldischargestatus) WHEN 'expired' THEN 1 WHEN 'alive' THEN 0 END mortality,
 try_cast(unitdischargeoffset AS DOUBLE) unit_end,
 try_cast(hospitaldischargeoffset AS DOUBLE) hospital_end
FROM ranked WHERE rn=1 AND patientunitstayid IN(SELECT patientunitstayid FROM hf)
 AND try_cast(replace(age,'>','') AS DOUBLE)>=18
 AND try_cast(hospitaladmitoffset AS DOUBLE) BETWEEN -1440 AND 0
 AND try_cast(unitdischargeoffset AS DOUBLE)>720
 AND try_cast(hospitaldischargeoffset AS DOUBLE)>720
 AND uniquepid IS NOT NULL
""")
status('MIMIC cohort')
dx=rawview('MIMIC','diagnoses_icd'); p=rawview('MIMIC','patients')
a=rawview('MIMIC','admissions'); i=rawview('MIMIC','icustays')
con.execute(f"""
CREATE OR REPLACE TABLE mimic_cohort AS
WITH hf AS (
 SELECT DISTINCT hadm_id FROM {dx} WHERE
 (icd_version='9' AND starts_with(icd_code,'428')) OR
 (icd_version='10' AND starts_with(upper(icd_code),'I50'))
), ranked AS (
 SELECT *,row_number() OVER(PARTITION BY hadm_id ORDER BY intime,stay_id) rn FROM {i}
)
SELECT try_cast(i.stay_id AS BIGINT) stay_id,i.subject_id patient_id,i.hadm_id encounter_id,
 'MIMIC' site,try_cast(p.anchor_age AS DOUBLE)+year(try_cast(a.admittime AS TIMESTAMP))-
 try_cast(p.anchor_year AS DOUBLE) age,
 CASE p.gender WHEN 'M' THEN 1.0 WHEN 'F' THEN 0.0 END male,
 try_cast(a.hospital_expire_flag AS INTEGER) mortality,try_cast(i.intime AS TIMESTAMP) intime,
 epoch(try_cast(i.outtime AS TIMESTAMP)-try_cast(i.intime AS TIMESTAMP))/60 unit_end,
 epoch(try_cast(a.dischtime AS TIMESTAMP)-try_cast(i.intime AS TIMESTAMP))/60 hospital_end
FROM ranked i JOIN {a} a ON i.hadm_id=a.hadm_id AND i.subject_id=a.subject_id
 JOIN {p} p ON i.subject_id=p.subject_id
WHERE i.rn=1 AND i.hadm_id IN(SELECT hadm_id FROM hf)
 AND epoch(try_cast(i.intime AS TIMESTAMP)-try_cast(a.admittime AS TIMESTAMP))/60 BETWEEN 0 AND 1440
 AND age>=18 AND unit_end>720 AND hospital_end>720
 AND (a.deathtime IS NULL OR try_cast(a.deathtime AS TIMESTAMP)>try_cast(i.intime AS TIMESTAMP)+INTERVAL 12 HOUR)
""")
cohort_counts=[]
for d in ['eicu','mimic']:
    counts=con.execute(f'SELECT count(*) n,count(DISTINCT patient_id) patients,count(DISTINCT encounter_id) encounters FROM {d}_cohort').fetchone()
    assert counts[0]==counts[2] and counts[0]>100
    cohort_counts.append(dict(dataset=d,n=counts[0],patients=counts[1],encounters=counts[2]))
write_json('cohort_counts.json',cohort_counts)
print(pd.DataFrame(cohort_counts).to_string(index=False))
''')
cell('markdown','''
## Extract actual lab observations

Stream original CSV tables; retain only eligible encounters and measurements
before 12 hours. Dictionary-derived MIMIC assays must be blood assays.
Unknown units are excluded and counted. Phosphate and creatinine use mg/dL;
lactate uses mmol/L. No missing measurement becomes a normal value.
''')
cell('code','''
status('eICU raw lab scan')
l=rawview('eICU','lab')
con.execute(f"""
CREATE OR REPLACE TABLE eicu_labs AS
SELECT try_cast(l.patientunitstayid AS BIGINT) stay_id,
 CASE lower(l.labname) WHEN 'lactate' THEN 'lactate' WHEN 'phosphate' THEN 'phosphate'
 WHEN 'phosphorus' THEN 'phosphate' WHEN 'creatinine' THEN 'creatinine' END marker,
 try_cast(l.labresultoffset AS DOUBLE) AS event_minute,try_cast(l.labresult AS DOUBLE) AS value,
 l.labname assay,l.labmeasurenamesystem unit,
 try_cast(l.labresultrevisedoffset AS DOUBLE) available_minute
FROM {l} l JOIN eicu_cohort c ON try_cast(l.patientunitstayid AS BIGINT)=c.stay_id
WHERE lower(l.labname) IN ('lactate','phosphate','phosphorus','creatinine')
 AND try_cast(l.labresultoffset AS DOUBLE)>=0 AND try_cast(l.labresultoffset AS DOUBLE)<720
""")
status('MIMIC raw lab scan')
d=rawview('MIMIC','d_labitems')
mapping=con.execute(f"""SELECT itemid,label,CASE lower(label)
 WHEN 'lactate' THEN 'lactate' WHEN 'phosphate' THEN 'phosphate'
 WHEN 'creatinine' THEN 'creatinine' WHEN 'creatinine, whole blood' THEN 'creatinine' END marker
 FROM {d} WHERE lower(fluid)='blood' AND lower(label) IN
 ('lactate','phosphate','creatinine','creatinine, whole blood')""").df()
assert set(mapping.marker)=={'lactate','phosphate','creatinine'}
con.register('assay_mapping',mapping)
write_json('mimic_assay_mapping.json',mapping.to_dict(orient='records'))
l=rawview('MIMIC','labevents')
con.execute(f"""
CREATE OR REPLACE TABLE mimic_labs AS
SELECT c.stay_id,m.marker,epoch(try_cast(l.charttime AS TIMESTAMP)-c.intime)/60 AS event_minute,
 try_cast(l.valuenum AS DOUBLE) AS value,l.itemid assay,l.valueuom unit,
 epoch(try_cast(l.storetime AS TIMESTAMP)-c.intime)/60 available_minute
FROM {l} l JOIN assay_mapping m ON l.itemid=m.itemid
 JOIN mimic_cohort c ON l.hadm_id=c.encounter_id AND l.subject_id=c.patient_id
WHERE try_cast(l.charttime AS TIMESTAMP)>=c.intime
 AND try_cast(l.charttime AS TIMESTAMP)<c.intime+INTERVAL 12 HOUR
""")
unit_audit=[]
for d in ['eicu','mimic']:
    tab=con.execute(f'SELECT marker,unit,count(*) n FROM {d}_labs GROUP BY marker,unit ORDER BY marker,unit').df()
    tab['dataset']=d
    unit_audit.extend(tab.where(pd.notna(tab),None).to_dict(orient='records'))
write_json('unit_audit.json',unit_audit)
print(pd.DataFrame(unit_audit).to_string(index=False))
''')
cell('code','''
status('feature construction and support audit')
frames={}; support=[]
for d in ['eicu','mimic']:
    con.execute(f"""
    CREATE OR REPLACE TABLE {d}_clean_labs AS
    SELECT stay_id,marker,event_minute,median(value) AS value,max(available_minute) AS available_minute
    FROM {d}_labs WHERE
    ((marker='lactate' AND lower(replace(unit,' ',''))='mmol/l' AND value BETWEEN 0.1 AND 30)
    OR (marker='phosphate' AND lower(replace(unit,' ',''))='mg/dl' AND value BETWEEN 0.2 AND 20)
    OR (marker='creatinine' AND lower(replace(unit,' ',''))='mg/dl' AND value BETWEEN 0.1 AND 20))
    GROUP BY stay_id,marker,event_minute
    """)
    c=con.execute(f'SELECT * FROM {d}_cohort').df().set_index('stay_id')
    for marker in ['lactate','phosphate','creatinine']:
        for label,lo,hi,ordering in [('b',0,240,'ASC'),('f',480,720,'DESC')]:
            t=con.execute(f"""SELECT stay_id,value,event_minute,available_minute FROM {d}_clean_labs
            WHERE marker={sqlstr(marker)} AND event_minute>={lo} AND event_minute<{hi}
            QUALIFY row_number() OVER(PARTITION BY stay_id ORDER BY event_minute {ordering})=1""").df().set_index('stay_id')
            c=c.join(t.rename(columns={{k:f'{marker}_{label}_{k}' for k in t.columns}}))
    for marker in ['lactate','phosphate','creatinine']:
        c[marker+'_delta']=c[marker+'_f_value']-c[marker+'_b_value']
    c['normalizer']=(c.lactate_b_value>=2)&(c.lactate_f_value<2)
    c['phosphate_rise']=(c.phosphate_delta>=0.5).astype(float).where(c.phosphate_delta.notna())
    required=[f'{m}_{w}_value' for m in ['lactate','phosphate','creatinine'] for w in ['b','f']]
    c['complete_labs']=c[required].notna().all(axis=1)
    c['available_by_landmark']=c[[f'{m}_{w}_available_minute' for m in ['lactate','phosphate','creatinine'] for w in ['b','f']]].lt(720).all(axis=1)
    assert c[[f'{m}_{w}_event_minute' for m in ['lactate','phosphate','creatinine'] for w in ['b','f']]].max().max()<720
    assert c.index.is_unique and c.encounter_id.is_unique
    for name,mask in [('cohort',pd.Series(True,index=c.index)),('paired_lactate',c.lactate_b_value.notna()&c.lactate_f_value.notna()),
                      ('normalizers',c.normalizer),('primary_complete_labs',c.normalizer&c.complete_labs)]:
        sub=c[mask]
        support.append(dict(dataset=d,stage=name,n=len(sub),known_outcomes=int(sub.mortality.notna().sum()),
            deaths=int(sub.mortality.sum()),sites=int(sub.site.nunique()),patients=int(sub.patient_id.nunique())))
    con.register('features_frame',c.reset_index())
    con.execute(f'CREATE OR REPLACE TABLE {d}_features AS SELECT * FROM features_frame')
    frames[d]=c
write_json('support_audit.json',support)
print(pd.DataFrame(support).to_string(index=False))
'''.replace('{{k:', '{k:').replace('t.columns}}','t.columns}'))
cell('markdown','''
## Locked mortality comparison

Modified-Poisson regression with patient-clustered robust uncertainty. No
regression is graded when there are fewer than 100 deaths or fewer than 20
deaths in either phosphate arm. Descriptive risks remain visible. These
models do not yet adjust for SCAI, RRT or administered phosphate and cannot
support a new biological mechanism or a clinical action.
''')
cell('code','''
status('locked mortality models')
results=[]; risk_rows=[]
formula='mortality ~ phosphate_rise + age + male + np.log1p(lactate_b_value) + np.log1p(lactate_f_value) + np.log1p(phosphate_b_value) + np.log1p(creatinine_b_value) + creatinine_delta'
for d,c in frames.items():
    masks={'primary':c.normalizer&c.complete_labs,
           'renal_restriction':c.normalizer&c.complete_labs&(c.creatinine_b_value<2)&(c.creatinine_delta<.3),
           'available_by_landmark':c.normalizer&c.complete_labs&c.available_by_landmark}
    for label,mask in masks.items():
        s=c[mask].dropna(subset=['mortality','age','male','patient_id']).copy()
        arms=s.groupby('phosphate_rise').mortality.agg(['count','sum','mean'])
        for arm,row in arms.iterrows():
            risk_rows.append(dict(dataset=d,analysis=label,phosphate_rise=int(arm),n=int(row['count']),
                deaths=int(row['sum']),mortality_risk=float(row['mean'])))
        record=dict(dataset=d,analysis=label,n=len(s),patients=int(s.patient_id.nunique()),
            deaths=int(s.mortality.sum()),status='insufficient_support',biological_claim_ready=False)
        if len(arms)==2 and s.mortality.sum()>=100 and arms['sum'].min()>=20 and (arms['count']-arms['sum']).min()>=20:
            import statsmodels.api as sm
            import statsmodels.formula.api as smf
            model=smf.glm(formula,s,family=sm.families.Poisson()).fit(cov_type='cluster',cov_kwds={'groups':s.patient_id})
            b=float(model.params['phosphate_rise']); ci=model.conf_int().loc['phosphate_rise'].to_numpy()
            record.update(status='estimated',rr=float(np.exp(b)),ci95=np.exp(ci).tolist(),
                p=float(model.pvalues['phosphate_rise']),converged=bool(model.converged))
            record['support_note']='Requires treatment, acid-base, SCAI and multicenter controls before biological interpretation.'
        results.append(record)
write_json('locked_primary_results.json',results)
write_json('mortality_risks.json',risk_rows)
print(pd.DataFrame(risk_rows).to_string(index=False))
print(pd.DataFrame(results).to_string(index=False))
print('No significance or clinical claim is manufactured when data are sparse.')
''')
cell('code','''
status('provenance and output verification')
# Full source fingerprints are computed after extraction; source size/time
# stability is checked so changes during the scan fail the run.
for name,info in source_manifest.items():
    p=Path(info['path']); digest=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): digest.update(block)
    current=p.stat()
    assert current.st_size==info['bytes'] and current.st_mtime_ns==info['mtime_ns'], 'Source changed during run: '+name
    info['sha256']=digest.hexdigest()
    print('Fingerprinted',name,flush=True)
write_json('input_manifest.json',source_manifest)
outputs={p.name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
    for p in sorted(OUT.iterdir()) if p.suffix=='.json' and p.name not in ['manifest.json','run_status.json']}
write_json('manifest.json',dict(environment=ENVIRONMENT,started_utc=START,
    finished_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=PROTOCOL_SHA256,
    python=sys.version,software={m:__import__(m).__version__ for m in ['duckdb','numpy','pandas','scipy','statsmodels']},
    outputs=outputs,synthetic_patients=0,source='original MIMIC/eICU under My Drive/Data',
    paradigm_shifting_result=False,clinical_benefit_established=False))
status('initial locked experiment','complete',goal_complete=False)
con.close()
print('The discovery goal remains open. Review the actual results before advancing or rejecting this candidate.')
''')

CONTEXT_START=len(CELLS)
cell('markdown',CONTEXT_PLAN.read_text())
cell('code',f"CONTEXT_PROTOCOL_SHA256={hashlib.sha256(CONTEXT_PLAN.read_bytes()).hexdigest()!r}")
cell('code','''
import sys, importlib.util, subprocess, json, hashlib, os
from pathlib import Path
from datetime import datetime, timezone
requirements={'numpy':'numpy==2.2.6','pandas':'pandas==2.2.3','duckdb':'duckdb==1.4.3','yaml':'pyyaml'}
missing=[p for m,p in requirements.items() if importlib.util.find_spec(m) is None]
if missing:subprocess.check_call([sys.executable,'-m','pip','install','-q',*missing])
import numpy as np
import pandas as pd
import duckdb
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE=Path('/content/drive/MyDrive')
else:DRIVE=Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects'/'PhysioGraph'
assert (PROJECT/'src/physiograph').is_dir(), 'Keep the PhysioGraph project beside Data in Drive'
sys.path.insert(0,str(PROJECT/'src'))
from physiograph.etl.audit import AuditLogger
from physiograph.etl.mimic_extractor import _load_mimic_cohort, _mimic_chartevent_item_map, _mimic_lab_item_map
from physiograph.etl.eicu_extractor import _load_eicu_patients, _build_eicu_diagnosis_flags, _build_eicu_cohort
from physiograph.constants import EICU_LAB_NAME_MAP, EICU_VITAL_COLUMN_MAP
PRIVATE=DRIVE/'Data'/'PhysioGraph_Biological_Discovery_20260905'
PRIVATE.mkdir(exist_ok=True)
OUT=PROJECT/'research'/'spo2_circulatory_context';OUT.mkdir(parents=True,exist_ok=True)
ENVIRONMENT='google_colab' if IN_COLAB else 'local'
def context_json(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\\n')
def context_status(stage,state='running'):
    context_json('run_status.json',dict(stage=stage,status=state,environment=ENVIRONMENT,
        updated_utc=datetime.now(timezone.utc).isoformat(),goal_complete=False))
    print(stage,state,flush=True)
context_json('protocol.json',dict(sha256=CONTEXT_PROTOCOL_SHA256,frozen_before_feature_extraction=True))
con=duckdb.connect(str(PRIVATE/'spo2_context.duckdb'))
con.execute('SET threads=2');con.execute("SET memory_limit='2GB'")
RAW={d:DRIVE/'Data'/d/'Full' for d in ['MIMIC','eICU']}
sources={}
def qs(value):return "'"+str(value).replace("'","''")+"'"
def raw_csv(dataset,name):
    p=RAW[dataset]/(name+'.csv');s=p.stat()
    sources[dataset+'/'+name]=dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    return f'read_csv({qs(p)},header=true,all_varchar=true,sample_size=10000,strict_mode=true)'
context_status('production cohort reconstruction')
cohorts={};cohort_summary=[]
for d,source in [('eicu','eICU'),('mimic','MIMIC')]:
    audit=AuditLogger(d)
    if d=='eicu' and globals().get('RESUME_EICU',False):
        # Explicit recovery of the successfully committed step from this run.
        c=con.execute('SELECT * FROM eicu_cohort').df()
        assert c.stay_id.is_unique and c.age.ge(18).all() and c.cohort_hf_flag.eq(1).all()
        assert (c.early_icu_flag.eq(1)|c.shock_icd_flag.eq(1)).all()
        print('Resumed committed eICU cohort from the preceding local attempt:',len(c))
        for name in ['patient','diagnosis','admissionDx']:raw_csv(source,name)
    elif d=='eicu':
        patient=_load_eicu_patients(RAW[source],audit)
        flags=_build_eicu_diagnosis_flags(RAW[source],patient)
        c,_=_build_eicu_cohort(RAW[source],patient,flags,audit,max_stays=None)
        outcome=patient.set_index('patientunitstayid').hospitaldischargestatus.str.lower()
        c['hospital_mortality']=c.stay_id.map(outcome.map({'expired':1.,'alive':0.}))
        for name in ['patient','diagnosis','admissionDx']:raw_csv(source,name)
    else:
        c,_,full=_load_mimic_cohort(RAW[source],audit,max_stays=None)
        mortality=pd.read_csv(RAW[source]/'admissions.csv',usecols=['hadm_id','hospital_expire_flag']).set_index('hadm_id').hospital_expire_flag
        c['hospital_mortality']=c.hadm_id.map(mortality).astype(float)
        for name in ['patients','admissions','icustays','diagnoses_icd']:raw_csv(source,name)
    before=len(c)
    exclude=c.death_offset_minutes.between(0,240)|c.followup_end_offset_minutes.le(240)
    c=c.loc[~exclude].copy()
    assert c.stay_id.is_unique
    c['encounter_id']=c['hadm_id'] if d=='mimic' else c['hospital_encounter_id']
    con.register('cohort_frame',c)
    con.execute(f'CREATE OR REPLACE TABLE {d}_cohort AS SELECT * FROM cohort_frame')
    cohorts[d]=c
    cohort_summary.append(dict(dataset=d,before_exclusion=before,eligible=len(c),
        people=c.person_id.nunique(),known_hospital_outcomes=c.hospital_mortality.notna().sum()))
context_json('cohort_support.json',cohort_summary)
print(pd.DataFrame(cohort_summary).to_string(index=False))
''')
cell('code','''
context_status('eICU raw vital scan')
source=raw_csv('eICU','vitalPeriodic')
concepts={'sao2':'spo2','systemicmean':'map','heartrate':'hr','systemicsystolic':'sbp','respiration':'resp_rate'}
con.execute(f"""CREATE OR REPLACE TABLE eicu_vitals_wide AS
SELECT try_cast(v.patientunitstayid AS BIGINT) stay_id,
try_cast(v.observationoffset AS DOUBLE) event_minute,
{','.join('try_cast(v.'+col+' AS DOUBLE) AS '+concept for col,concept in concepts.items())}
FROM {source} v JOIN eicu_cohort c ON try_cast(v.patientunitstayid AS BIGINT)=c.stay_id
WHERE try_cast(v.observationoffset AS DOUBLE)>=0 AND try_cast(v.observationoffset AS DOUBLE)<240""")
con.execute("CREATE OR REPLACE TABLE eicu_vitals AS "+' UNION ALL '.join(
    f"SELECT stay_id,event_minute,{qs(concept)} concept,{concept} AS value,'periodic_invasive_context' source_type FROM eicu_vitals_wide WHERE {concept} IS NOT NULL"
    for concept in concepts.values()))
context_status('MIMIC raw vital scan')
itemmap=_mimic_chartevent_item_map(RAW['MIMIC'])
chosen={i:m for i,m in itemmap.items() if m in ['spo2','map','sbp','hr','resp_rate','fio2']}
mapping=pd.DataFrame([dict(itemid=i,concept=m) for i,m in chosen.items()]);con.register('vital_map',mapping)
source=raw_csv('MIMIC','chartevents');raw_csv('MIMIC','d_items')
con.execute(f"""CREATE OR REPLACE TABLE mimic_vitals AS
SELECT c.stay_id,date_diff('second',c.admit_time,try_cast(v.charttime AS TIMESTAMP))/60.0 event_minute,
m.concept,try_cast(v.valuenum AS DOUBLE) AS value,
CASE WHEN try_cast(v.itemid AS BIGINT) IN (220181,220179) THEN 'noninvasive'
WHEN m.concept IN ('map','sbp') THEN 'invasive' ELSE 'charted' END source_type
FROM {source} v JOIN mimic_cohort c ON try_cast(v.stay_id AS BIGINT)=c.stay_id
JOIN vital_map m ON try_cast(v.itemid AS BIGINT)=m.itemid
WHERE try_cast(v.charttime AS TIMESTAMP)>=c.admit_time AND try_cast(v.charttime AS TIMESTAMP)<c.admit_time+INTERVAL '4 hours'
AND coalesce(try_cast(v.warning AS DOUBLE),0)<>1
AND try_cast(v.valuenum AS DOUBLE) IS NOT NULL""")
con.execute('CREATE OR REPLACE TABLE mimic_context_quality_marker AS SELECT 1 AS version')
context_status('joint SpO2 and MAP feature construction')
coverage=[]
for d in ['eicu','mimic']:
    con.execute(f"""CREATE OR REPLACE TABLE {d}_bins AS
    WITH dedup AS (
      SELECT stay_id,event_minute,concept,median(value) AS value FROM {d}_vitals
      WHERE (concept='spo2' AND value BETWEEN 50 AND 100)
         OR (concept='map' AND value BETWEEN 20 AND 200)
         OR (concept='hr' AND value BETWEEN 20 AND 250)
         OR (concept='sbp' AND value BETWEEN 30 AND 300)
         OR (concept='resp_rate' AND value BETWEEN 2 AND 80)
      GROUP BY stay_id,event_minute,concept
    ) SELECT stay_id,floor(event_minute/15)::INTEGER bin,concept,median(value) AS value,
      median(event_minute) AS representative_minute,
      count(*) n_readings FROM dedup GROUP BY stay_id,bin,concept""")
    bins=con.execute(f'SELECT * FROM {d}_bins ORDER BY stay_id,bin').df()
    wide=bins.pivot(index=['stay_id','bin'],columns='concept',values='value').reset_index()
    clock=bins[bins.concept.eq('spo2')][['stay_id','bin','representative_minute']]
    wide=wide.merge(clock,on=['stay_id','bin'],how='left',validate='one_to_one')
    feature_rows=[]
    for stay,s in wide.groupby('stay_id',sort=False):
        s=s[s.spo2.notna()].sort_values('bin')
        gap=s.representative_minute.diff();ds=s.spo2.diff();dm=s['map'].diff()
        transition=gap.gt(0)&gap.le(30)
        joint=transition&dm.notna()
        jump=ds.abs().ge(4)&joint;fall=dm.le(-10)&joint
        joint_n=int(joint.sum());spo2_n=len(s)
        row=dict(stay_id=int(stay),spo2_bins=spo2_n,spo2_transitions=int(transition.sum()),
            joint_transitions=joint_n,joint_eligible=spo2_n>=3 and joint_n>=2,
            spo2_jump_any=int((ds.abs().ge(4)&transition).any()),
            joint_spo2_jump_any=int(jump.any()),map_fall_any=int(fall.any()),
            synchronous_any=int((jump&fall).any()),synchronous_drop=int((jump&fall&ds.lt(0)).any()),
            synchronous_rise=int((jump&fall&ds.gt(0)).any()),
            joint_spo2_jumps=int(jump.sum()),map_falls=int(fall.sum()),
            spo2_mean=float(s.spo2.mean()),spo2_min=float(s.spo2.min()),
            map_mean=float(s['map'].mean()),map_min=float(s['map'].min()),
            hr_mean=float(s.hr.mean()),map_observed_bins=int(s['map'].notna().sum()))
        feature_rows.append(row)
    features=pd.DataFrame(feature_rows)
    con.register('feature_frame',features)
    con.execute(f'CREATE OR REPLACE TABLE {d}_features AS SELECT * FROM feature_frame')
    eligible=features[features.joint_eligible]
    coverage.append(dict(dataset=d,cohort=len(cohorts[d]),spo2_observed=len(features),
        spo2_dynamics=int(((features.spo2_bins>=3)&(features.spo2_transitions>=2)).sum()),
        jointly_observed=len(eligible),synchronous=int(eligible.synchronous_any.sum()),
        simultaneous_drop=int(eligible.synchronous_drop.sum()),
        simultaneous_rise=int(eligible.synchronous_rise.sum())))
context_json('joint_feature_support.json',coverage)
print(pd.DataFrame(coverage).to_string(index=False))
# No feature/outcome association is computed in this feasibility stage.
context_json('source_metadata.json',sources)
context_status('cohort and joint-feature feasibility','complete')
con.close()
''')

CONTEXT_LABS_CELL=len(CELLS)
cell('code','''
# This cell can resume from the completed raw vital/cohort extraction.
# It does not regenerate or alter the locked SpO2/MAP exposure.
import sys, json, hashlib, importlib.util
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import duckdb
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
if IN_COLAB:DRIVE=Path('/content/drive/MyDrive')
else:DRIVE=Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects'/'PhysioGraph'
sys.path.insert(0,str(PROJECT/'src'))
from physiograph.etl.mimic_extractor import _mimic_lab_item_map
from physiograph.constants import EICU_LAB_NAME_MAP
PRIVATE=DRIVE/'Data'/'PhysioGraph_Biological_Discovery_20260905'
OUT=PROJECT/'research'/'spo2_circulatory_context'
RAW={d:DRIVE/'Data'/d/'Full' for d in ['MIMIC','eICU']}
con=duckdb.connect(str(PRIVATE/'spo2_context.duckdb'))
con.execute('SET threads=2');con.execute("SET memory_limit='2GB'")
sources=json.loads((OUT/'source_metadata.json').read_text())
def context_json(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\\n')
def context_status(stage,state='running'):
    context_json('run_status.json',dict(stage=stage,status=state,
        environment='google_colab' if IN_COLAB else 'local',
        updated_utc=datetime.now(timezone.utc).isoformat(),goal_complete=False))
    print(stage,state,flush=True)
def qs(value):return "'"+str(value).replace("'","''")+"'"
def raw_csv(dataset,name):
    p=RAW[dataset]/(name+'.csv');s=p.stat()
    sources[dataset+'/'+name]=dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    return f'read_csv({qs(p)},header=true,all_varchar=true,sample_size=10000,strict_mode=true)'
context_status('eICU myocardial-injury and baseline lab extraction')
chosen={k:v for k,v in EICU_LAB_NAME_MAP.items() if v in ['troponin_i','troponin_t','lactate','creatinine','ph']}
mapping=pd.DataFrame([dict(raw_name=k,concept=v) for k,v in chosen.items()]);con.register('lab_map',mapping)
source=raw_csv('eICU','lab')
con.execute(f"""CREATE OR REPLACE TABLE eicu_context_labs AS
SELECT c.stay_id,try_cast(l.labresultoffset AS DOUBLE) event_minute,
try_cast(l.labresultrevisedoffset AS DOUBLE) available_minute,
m.concept,l.labname raw_name,lower(trim(coalesce(l.labmeasurenamesystem,l.labmeasurenameinterface,''))) unit,
try_cast(l.labresult AS DOUBLE) AS value
FROM {source} l JOIN eicu_cohort c ON try_cast(l.patientunitstayid AS BIGINT)=c.stay_id
JOIN lab_map m ON lower(trim(l.labname))=m.raw_name
WHERE try_cast(l.labresultoffset AS DOUBLE) BETWEEN 0 AND least(1680,c.followup_end_offset_minutes)
AND try_cast(l.labresult AS DOUBLE) IS NOT NULL""")
context_status('MIMIC myocardial-injury and baseline lab extraction')
chosen={k:v for k,v in _mimic_lab_item_map(RAW['MIMIC']).items() if v in ['troponin_i','troponin_t','lactate','creatinine','ph']}
mapping=pd.DataFrame([dict(itemid=k,concept=v) for k,v in chosen.items()]);con.register('lab_map',mapping)
source=raw_csv('MIMIC','labevents');raw_csv('MIMIC','d_labitems')
con.execute(f"""CREATE OR REPLACE TABLE mimic_context_labs AS
SELECT c.stay_id,date_diff('second',c.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0 event_minute,
date_diff('second',c.admit_time,try_cast(l.storetime AS TIMESTAMP))/60.0 available_minute,
m.concept,l.itemid raw_name,lower(trim(coalesce(l.valueuom,''))) unit,try_cast(l.valuenum AS DOUBLE) AS value
FROM {source} l JOIN mimic_cohort c ON try_cast(l.hadm_id AS BIGINT)=c.hadm_id
JOIN lab_map m ON try_cast(l.itemid AS BIGINT)=m.itemid
WHERE try_cast(l.charttime AS TIMESTAMP)>=c.admit_time
AND date_diff('second',c.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0<=least(1680,c.followup_end_offset_minutes)
AND try_cast(l.valuenum AS DOUBLE) IS NOT NULL""")
support=[];unit_rows=[]
for d in ['eicu','mimic']:
    c=con.execute(f'SELECT * FROM {d}_cohort').df().set_index('stay_id')
    labs=con.execute(f'SELECT * FROM {d}_context_labs').df()
    bounds={'troponin_t':(0,1000000),'troponin_i':(0,1000000),'lactate':(.1,40),'creatinine':(.1,30),'ph':(6.5,8)}
    keep=np.zeros(len(labs),dtype=bool)
    for concept,(lo,hi) in bounds.items():keep|=labs.concept.eq(concept)&labs.value.between(lo,hi)
    labs=labs[keep].copy()
    unit_rows.extend(labs.groupby(['concept','raw_name','unit']).size().rename('records').reset_index().assign(dataset=d).to_dict('records'))
    labs=labs.groupby(['stay_id','event_minute','concept','unit'],as_index=False).agg(value=('value','median'),available_minute=('available_minute','max'))
    base=labs[labs.event_minute.le(240)].sort_values('event_minute')
    for marker in ['lactate','creatinine','ph']:
        last=base[base.concept.eq(marker)].drop_duplicates('stay_id',keep='last').set_index('stay_id')
        c['baseline_'+marker]=last.value
        c['baseline_'+marker+'_available_minute']=last.available_minute
    base_t=base[base.concept.str.startswith('troponin')&base.value.gt(0)].drop_duplicates(['stay_id','concept','unit'],keep='last')
    for end,label in [(960,'12h'),(1680,'24h')]:
        post=labs[labs.event_minute.gt(240)&labs.event_minute.le(end)&labs.concept.str.startswith('troponin')]
        peak=post.groupby(['stay_id','concept','unit'],as_index=False).value.max().rename(columns={'value':'post_peak'})
        pairs=base_t.merge(peak,on=['stay_id','concept','unit'],how='inner')
        pairs['ratio']=pairs.post_peak/pairs.value
        result=pairs.groupby('stay_id').ratio.max()
        c['troponin_ratio_'+label]=result
        observed=c['troponin_ratio_'+label].notna()
        positive=c['troponin_ratio_'+label].ge(1.5)
        full=c.followup_end_offset_minutes.ge(end)
        c['troponin_rise_'+label]=positive.astype(float).where(observed&(positive|full))
        support.append(dict(dataset=d,endpoint='troponin_'+label,cohort=len(c),
            baseline_positive_assay_people=int(base_t.stay_id.nunique()),
            paired=int(observed.sum()),classifiable=int(c['troponin_rise_'+label].notna().sum()),
            events=int(c['troponin_rise_'+label].sum())))
    con.register('endpoint_frame',c.reset_index())
    con.execute(f'CREATE OR REPLACE TABLE {d}_context_endpoints AS SELECT * FROM endpoint_frame')
context_json('lab_assay_unit_audit.json',unit_rows)
context_json('endpoint_support.json',support)
print(pd.DataFrame(support).to_string(index=False))
context_status('raw source fingerprint verification')
for key,info in sources.items():
    p=Path(info['path']);digest=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    s=p.stat()
    assert s.st_size==info['bytes'] and s.st_mtime_ns==info['mtime_ns'], 'Source changed: '+key
    info['sha256']=digest.hexdigest()
context_json('input_manifest.json',sources)
context_status('SpO2-context features and endpoints extracted','complete')
con.close()
print('Feasibility outputs only: no new feature-outcome association or biological discovery claimed.')
''')

CONTEXT_RECONCILE_CELL=len(CELLS)
cell('code','''
# Reconcile the historical recovery attempt to the exact production risk set.
# Raw scans can contain a superset; model tables must use the final cohort.
import json, importlib.util, sys
from pathlib import Path
from datetime import datetime, timezone
import duckdb
import pandas as pd
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects'/'PhysioGraph'
OUT=PROJECT/'research'/'spo2_circulatory_context'
sys.path.insert(0,str(PROJECT/'src'))
from physiograph.analysis.spo2_protocol import _first_instability
(OUT/'run_status.json').write_text(json.dumps(dict(stage='production cohort and SpO2 boundary reconciliation',
    status='running',environment='google_colab' if IN_COLAB else 'local',
    updated_utc=datetime.now(timezone.utc).isoformat(),goal_complete=False),indent=2)+'\\n')
con=duckdb.connect(str(DRIVE/'Data'/'PhysioGraph_Biological_Discovery_20260905'/'spo2_context.duckdb'))
con.execute('SET threads=2')
con.execute("SET memory_limit='2GB'")
# Recover the initial scan's omitted production warning filter exactly once.
# Fresh full-notebook extraction above already applies and records this rule.
quality_done=con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name='mimic_context_quality_marker'").fetchone()[0]>0
if not quality_done:
    from physiograph.etl.mimic_extractor import _mimic_chartevent_item_map
    root=DRIVE/'Data'/'MIMIC'/'Full';raw_path=root/'chartevents.csv'
    input_info=json.loads((OUT/'input_manifest.json').read_text())['MIMIC/chartevents']
    st=raw_path.stat()
    assert st.st_size==input_info['bytes'] and st.st_mtime_ns==input_info['mtime_ns'], 'Raw input changed before quality correction'
    (OUT/'run_status.json').write_text(json.dumps(dict(stage='MIMIC production warning-filter correction',
        status='running',environment='google_colab' if IN_COLAB else 'local',
        updated_utc=datetime.now(timezone.utc).isoformat(),goal_complete=False),indent=2)+'\\n')
    itemmap=_mimic_chartevent_item_map(root)
    mapping=pd.DataFrame([dict(itemid=i,concept=m) for i,m in itemmap.items()
                          if m in ['spo2','map','sbp','hr','resp_rate','fio2']])
    con.register('quality_vital_map',mapping)
    before_quality=con.execute('SELECT count(*) FROM mimic_vitals').fetchone()[0]
    con.execute("""CREATE OR REPLACE TABLE mimic_vitals AS
      SELECT c.stay_id,date_diff('second',c.admit_time,try_cast(v.charttime AS TIMESTAMP))/60.0 event_minute,
      m.concept,try_cast(v.valuenum AS DOUBLE) AS value,
      CASE WHEN try_cast(v.itemid AS BIGINT) IN (220181,220179) THEN 'noninvasive'
      WHEN m.concept IN ('map','sbp') THEN 'invasive' ELSE 'charted' END source_type
      FROM read_csv(?,header=true,all_varchar=true,sample_size=10000,strict_mode=true) v
      JOIN mimic_cohort c ON try_cast(v.stay_id AS BIGINT)=c.stay_id
      JOIN quality_vital_map m ON try_cast(v.itemid AS BIGINT)=m.itemid
      WHERE try_cast(v.charttime AS TIMESTAMP)>=c.admit_time
        AND try_cast(v.charttime AS TIMESTAMP)<c.admit_time+INTERVAL '4 hours'
        AND coalesce(try_cast(v.warning AS DOUBLE),0)<>1
        AND try_cast(v.valuenum AS DOUBLE) IS NOT NULL""",[str(raw_path)])
    con.execute("""CREATE OR REPLACE TABLE mimic_bins AS
      WITH dedup AS (
        SELECT stay_id,event_minute,concept,median(value) AS value FROM mimic_vitals
        WHERE (concept='spo2' AND value BETWEEN 50 AND 100)
          OR (concept='map' AND value BETWEEN 20 AND 200)
          OR (concept='hr' AND value BETWEEN 20 AND 250)
          OR (concept='sbp' AND value BETWEEN 30 AND 300)
          OR (concept='resp_rate' AND value BETWEEN 2 AND 80)
        GROUP BY stay_id,event_minute,concept)
      SELECT stay_id,floor(event_minute/15)::INTEGER bin,concept,median(value) AS value,
        median(event_minute) representative_minute,count(*) n_readings
        FROM dedup GROUP BY stay_id,bin,concept""")
    st=raw_path.stat()
    assert st.st_size==input_info['bytes'] and st.st_mtime_ns==input_info['mtime_ns'], 'Raw input changed during quality correction'
    con.execute('CREATE OR REPLACE TABLE mimic_context_quality_marker AS SELECT 1 AS version')
    (OUT/'mimic_quality_correction.json').write_text(json.dumps(dict(
        raw_rows_before=before_quality,raw_rows_after=con.execute('SELECT count(*) FROM mimic_vitals').fetchone()[0],
        correction='canonical cohort, [0,240) window, and warning !=1',
        production_filter='src/physiograph/etl/mimic_extractor.py:_stream_mimic_measurement_events',
        source_sha256=input_info['sha256'],cross_tabulations_previously_seen=True,
        no_effect_estimates_fitted=True),indent=2)+'\\n')
reconciliation=[];support=[]
for d in ['eicu','mimic']:
    before=con.execute(f'SELECT count(*) FROM {d}_cohort').fetchone()[0]
    con.execute(f"""CREATE TABLE IF NOT EXISTS {d}_canonical_exclusions AS
    SELECT * FROM {d}_cohort WHERE FALSE""")
    con.execute(f"""INSERT INTO {d}_canonical_exclusions
    SELECT * FROM {d}_cohort WHERE death_offset_minutes BETWEEN 0 AND 240
      OR followup_end_offset_minutes<=240""")
    con.execute(f"DELETE FROM {d}_cohort WHERE stay_id IN (SELECT stay_id FROM {d}_canonical_exclusions)")
    for suffix in ['features','context_endpoints']:
        con.execute(f'DELETE FROM {d}_{suffix} WHERE stay_id NOT IN (SELECT stay_id FROM {d}_cohort)')
    # Production SpO2 dynamics use [0,240), unlike baseline labs which include 240.
    # The initial raw scan was an inclusive superset; remove its terminal bin.
    boundary_records=con.execute(f'SELECT count(*) FROM {d}_bins WHERE bin>=16').fetchone()[0]
    before_dynamics=con.execute(f'SELECT count(*) FROM {d}_features WHERE spo2_bins>=3 AND spo2_transitions>=2').fetchone()[0]
    con.execute(f'DELETE FROM {d}_bins WHERE bin>=16')
    con.execute(f"""CREATE OR REPLACE TABLE {d}_features AS
      WITH wide AS (
        SELECT b.stay_id,bin,max(value) FILTER(WHERE concept='spo2') spo2,
        max(value) FILTER(WHERE concept='map') map_value,
        max(value) FILTER(WHERE concept='hr') hr,
        max(representative_minute) FILTER(WHERE concept='spo2') spo2_time
        FROM {d}_bins b JOIN {d}_cohort c USING(stay_id) GROUP BY b.stay_id,bin), lagged AS (
        SELECT *,spo2-lag(spo2) OVER w ds,map_value-lag(map_value) OVER w dm,
        spo2_time-lag(spo2_time) OVER w gap FROM wide WHERE spo2 IS NOT NULL
        WINDOW w AS (PARTITION BY stay_id ORDER BY bin)), marked AS (
        SELECT *,coalesce(gap>0 AND gap<=30,FALSE) AS valid_transition,
        coalesce(gap>0 AND gap<=30 AND dm IS NOT NULL,FALSE) joint FROM lagged)
      SELECT stay_id,count(*) spo2_bins,count(*) FILTER(WHERE valid_transition) spo2_transitions,
      count(*) FILTER(WHERE joint) joint_transitions,
      count(*)>=3 AND count(*) FILTER(WHERE joint)>=2 joint_eligible,
      max(CASE WHEN valid_transition AND abs(ds)>=4 THEN 1 ELSE 0 END) spo2_jump_any,
      max(CASE WHEN joint AND abs(ds)>=4 THEN 1 ELSE 0 END) joint_spo2_jump_any,
      max(CASE WHEN joint AND dm<=-10 THEN 1 ELSE 0 END) map_fall_any,
      max(CASE WHEN joint AND abs(ds)>=4 AND dm<=-10 THEN 1 ELSE 0 END) synchronous_any,
      max(CASE WHEN joint AND ds<=-4 AND dm<=-10 THEN 1 ELSE 0 END) synchronous_drop,
      max(CASE WHEN joint AND ds>=4 AND dm<=-10 THEN 1 ELSE 0 END) synchronous_rise,
      count(*) FILTER(WHERE joint AND abs(ds)>=4) joint_spo2_jumps,
      count(*) FILTER(WHERE joint AND dm<=-10) map_falls,
      avg(spo2) spo2_mean,min(spo2) spo2_min,avg(map_value) map_mean,min(map_value) map_min,
      avg(hr) hr_mean,count(map_value) map_observed_bins FROM marked GROUP BY stay_id""")
    c=con.execute(f'SELECT * FROM {d}_cohort').df()
    f=con.execute(f'SELECT * FROM {d}_features').df()
    assert c.stay_id.is_unique and c.age.ge(18).all() and c.cohort_hf_flag.eq(1).all()
    assert (c.early_icu_flag.eq(1)|c.shock_icd_flag.eq(1)).all()
    assert not (c.death_offset_minutes.between(0,240)|c.followup_end_offset_minutes.le(240)).any()
    assert set(f.stay_id).issubset(set(c.stay_id))
    raw_spo2=con.execute(f"""SELECT v.stay_id,'{d}' dataset,concept,
        event_minute offset_minutes,value value_numeric FROM {d}_vitals v
        JOIN {d}_cohort c USING(stay_id) WHERE concept='spo2'""").df()
    production_exposed=set(_first_instability(raw_spo2).stay_id)
    rebuilt_exposed=set(f.loc[f.spo2_bins.ge(3)&f.spo2_transitions.ge(2)&f.spo2_jump_any.eq(1),'stay_id'])
    assert production_exposed==rebuilt_exposed, 'Production SpO2 exposure mismatch: '+d
    eligible=f[f.joint_eligible]
    reconciliation.append(dict(dataset=d,before=before,canonical=len(c),removed_this_execution=before-len(c),
        terminal_bins_removed=boundary_records,dynamics_before_boundary_fix=before_dynamics,
        dynamics_after_boundary_fix=int(((f.spo2_bins>=3)&(f.spo2_transitions>=2)).sum()),
        production_exposure_patient_sets_match=True,production_exposed=len(production_exposed),
        cumulative_excluded=con.execute(f'SELECT count(*) FROM {d}_canonical_exclusions').fetchone()[0],
        historical_reference={'eicu':12028,'mimic':17758}[d],
        matches_historical_count=len(c)=={'eicu':12028,'mimic':17758}[d]))
    support.append(dict(dataset=d,cohort=len(c),spo2_observed=len(f),
        spo2_dynamics=int(((f.spo2_bins>=3)&(f.spo2_transitions>=2)).sum()),
        jointly_observed=len(eligible),synchronous=int(eligible.synchronous_any.sum()),
        simultaneous_drop=int(eligible.synchronous_drop.sum()),simultaneous_rise=int(eligible.synchronous_rise.sum())))
(OUT/'canonical_cohort_reconciliation.json').write_text(json.dumps(reconciliation,indent=2)+'\\n')
(OUT/'joint_feature_support_canonical.json').write_text(json.dumps(support,indent=2)+'\\n')
print(pd.DataFrame(reconciliation).to_string(index=False))
print(pd.DataFrame(support).to_string(index=False))
con.close()
''')

CONTEXT_SOURCE_CELL=len(CELLS)
cell('code','''
# Pressure-source sensitivity and timing support, without outcome associations.
import json, importlib.util
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects'/'PhysioGraph'
OUT=PROJECT/'research'/'spo2_circulatory_context'
con=duckdb.connect(str(DRIVE/'Data'/'PhysioGraph_Biological_Discovery_20260905'/'spo2_context.duckdb'))
con.execute('SET threads=2')
summaries=[];provenance=[]
for d in ['eicu','mimic']:
    provenance.extend(con.execute(f"""SELECT '{d}' dataset,source_type,count(*) records,
        count(DISTINCT v.stay_id) stays FROM {d}_vitals v JOIN {d}_cohort c USING(stay_id)
        WHERE concept='map' AND value BETWEEN 20 AND 200 GROUP BY source_type""").df().to_dict('records'))
    for variant in (['pooled','arterial'] if d=='mimic' else ['pooled']):
        if variant=='pooled':
            b=con.execute(f'SELECT b.* FROM {d}_bins b JOIN {d}_cohort c USING(stay_id)').df()
        else:
            b=con.execute(f"""WITH dedup AS (
              SELECT v.stay_id,event_minute,concept,median(value) AS value
              FROM {d}_vitals v JOIN {d}_cohort c USING(stay_id)
              WHERE event_minute<240 AND ((concept='spo2' AND value BETWEEN 50 AND 100)
                OR (concept='map' AND source_type='invasive' AND value BETWEEN 20 AND 200)
                OR (concept='hr' AND value BETWEEN 20 AND 250))
              GROUP BY v.stay_id,event_minute,concept)
              SELECT stay_id,floor(event_minute/15)::INTEGER bin,concept,
              median(value) AS value,median(event_minute) representative_minute,
              count(*) n_readings FROM dedup GROUP BY stay_id,bin,concept""").df()
        wide=b.pivot(index=['stay_id','bin'],columns='concept',values='value').reset_index()
        for concept in ['spo2','map']:
            clock=b[b.concept.eq(concept)][['stay_id','bin','representative_minute','n_readings']]
            clock=clock.rename(columns={'representative_minute':concept+'_time','n_readings':concept+'_readings'})
            wide=wide.merge(clock,on=['stay_id','bin'],how='left',validate='one_to_one')
        s=wide[wide.spo2.notna()].sort_values(['stay_id','bin']).copy()
        g=s.groupby('stay_id',sort=False)
        gap=g.spo2_time.diff();ds=g.spo2.diff();dm=g['map'].diff()
        s['valid']=gap.gt(0)&gap.le(30);s['joint']=s.valid&dm.notna()
        s['jump']=ds.abs().ge(4)&s.joint;s['fall']=dm.le(-10)&s.joint
        s['sync']=s.jump&s.fall;s['sync_drop']=s.sync&ds.lt(0);s['sync_rise']=s.sync&ds.gt(0)
        s['low_map']=s['map'].lt(65)
        f=s.groupby('stay_id',sort=False).agg(
            spo2_bins=('spo2','count'),spo2_transitions=('valid','sum'),joint_transitions=('joint','sum'),
            joint_spo2_jump_any=('jump','max'),map_fall_any=('fall','max'),
            synchronous_any=('sync','max'),synchronous_drop=('sync_drop','max'),synchronous_rise=('sync_rise','max'),
            joint_spo2_jumps=('jump','sum'),map_falls=('fall','sum'),
            spo2_mean=('spo2','mean'),spo2_min=('spo2','min'),map_mean=('map','mean'),map_min=('map','min'),
            hr_mean=('hr','mean'),spo2_readings=('spo2_readings','sum'),map_readings=('map_readings','sum'),
            low_map_bins=('low_map','sum'),map_observed_bins=('map','count')).reset_index()
        f['joint_eligible']=f.spo2_bins.ge(3)&f.joint_transitions.ge(2)
        for col in ['joint_spo2_jump_any','map_fall_any','synchronous_any','synchronous_drop','synchronous_rise']:
            f[col]=f[col].astype(int)
        t=pd.DataFrame(dict(stay_id=s.stay_id,bin=s.bin,event_minute=s.spo2_time,
            spo2_change=ds,map_change=dm,joint=s.joint,spo2_jump=s.jump,map_fall=s.fall,
            map_spo2_time_difference=s.map_time-s.spo2_time))
        t=t[t.joint].copy()
        # Independent SQL reconstruction on actual data checks the transition logic.
        con.register('check_bins',b)
        check=con.execute("""WITH wide AS (
          SELECT stay_id,bin,max(value) FILTER(WHERE concept='spo2') spo2,
          max(value) FILTER(WHERE concept='map') map_value,
          max(representative_minute) FILTER(WHERE concept='spo2') spo2_time
          FROM check_bins GROUP BY stay_id,bin), lagged AS (
          SELECT *,spo2-lag(spo2) OVER w ds,map_value-lag(map_value) OVER w dm,
          spo2_time-lag(spo2_time) OVER w gap FROM wide WHERE spo2 IS NOT NULL
          WINDOW w AS (PARTITION BY stay_id ORDER BY bin)), marked AS (
          SELECT *,coalesce(gap>0 AND gap<=30 AND dm IS NOT NULL,FALSE) joint
          FROM lagged)
          SELECT stay_id,count(*) spo2_bins,count(*) FILTER(WHERE joint) joint_transitions,
          count(*) FILTER(WHERE joint AND abs(ds)>=4) joint_spo2_jumps,
          count(*) FILTER(WHERE joint AND dm<=-10) map_falls,
          max(CASE WHEN joint AND abs(ds)>=4 AND dm<=-10 THEN 1 ELSE 0 END) synchronous_any
          FROM marked GROUP BY stay_id""").df().set_index('stay_id')
        actual=f.set_index('stay_id').reindex(check.index)
        for col in check.columns:
            assert actual[col].eq(check[col]).all(), 'Independent SQL feature mismatch: '+d+'/'+variant+'/'+col
        con.register('source_features',f);con.register('source_transitions',t)
        con.execute(f'CREATE OR REPLACE TABLE {d}_{variant}_features AS SELECT * FROM source_features')
        con.execute(f'CREATE OR REPLACE TABLE {d}_{variant}_transitions AS SELECT * FROM source_transitions')
        if variant=='pooled':
            old=con.execute(f'SELECT * FROM {d}_features WHERE spo2_bins>0').df().set_index('stay_id')
            new=f.set_index('stay_id').reindex(old.index)
            for col in ['joint_eligible','spo2_bins','spo2_transitions','joint_transitions',
                        'joint_spo2_jump_any','map_fall_any','synchronous_any','synchronous_drop','synchronous_rise']:
                assert new[col].eq(old[col]).all(), 'Feature definition drift: '+d+'/'+col
        a=f[f.joint_eligible];both=a.joint_spo2_jump_any.eq(1)&a.map_fall_any.eq(1)
        summaries.append(dict(dataset=d,variant=variant,jointly_observed=len(a),
            both_component_events=int(both.sum()),synchronous=int(a.synchronous_any.sum()),
            asynchronous_both=int((both&a.synchronous_any.eq(0)).sum()),
            joint_transitions=len(t),
            median_absolute_within_bin_time_difference=float(t.map_spo2_time_difference.abs().median())))
(OUT/'pressure_provenance.json').write_text(json.dumps(provenance,indent=2)+'\\n')
(OUT/'pressure_source_support.json').write_text(json.dumps(summaries,indent=2,allow_nan=False)+'\\n')
print(pd.DataFrame(summaries).to_string(index=False))
con.close()
''')

CONTEXT_ANALYSIS_CELL=len(CELLS)
cell('markdown',(ROOT/'docs/SPO2_CIRCULATORY_ANALYSIS_LOCK.md').read_text())
cell('code',f"ANALYSIS_LOCK_SHA256={hashlib.sha256((ROOT/'docs/SPO2_CIRCULATORY_ANALYSIS_LOCK.md').read_bytes()).hexdigest()!r}")
cell('code','''
# Apply the frozen support gate before estimating any association.
import json, importlib.util
from pathlib import Path
from datetime import datetime, timezone
import duckdb
import pandas as pd
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects'/'PhysioGraph'
OUT=PROJECT/'research'/'spo2_circulatory_context'
lock_path=OUT/'analysis_lock.json'
if lock_path.exists():
    assert json.loads(lock_path.read_text())['sha256']==ANALYSIS_LOCK_SHA256, 'Analysis specification changed after locking'
else:
    lock_path.write_text(json.dumps(dict(sha256=ANALYSIS_LOCK_SHA256,
        recorded_before_cross_tabulations=True,utc=datetime.now(timezone.utc).isoformat()),indent=2)+'\\n')
con=duckdb.connect(str(DRIVE/'Data'/'PhysioGraph_Biological_Discovery_20260905'/'spo2_context.duckdb'))
gates=[];states=[];endpoint_support=[]
for d in ['eicu','mimic']:
    endpoints=con.execute(f'SELECT * FROM {d}_context_endpoints').df()
    for endpoint in ['hospital_mortality','troponin_rise_12h','troponin_rise_24h']:
        endpoint_support.append(dict(dataset=d,endpoint=endpoint,cohort=len(endpoints),
            observed=int(endpoints[endpoint].notna().sum()),events=int(endpoints[endpoint].sum())))
    for variant in (['pooled','arterial'] if d=='mimic' else ['pooled']):
        f=con.execute(f'SELECT * FROM {d}_{variant}_features WHERE joint_eligible').df()
        a=f.merge(endpoints,on='stay_id',validate='one_to_one')
        a['component_state']=a.joint_spo2_jump_any.astype(int)+2*a.map_fall_any.astype(int)
        for endpoint in ['hospital_mortality','troponin_rise_12h','troponin_rise_24h']:
            observed=a[a[endpoint].notna()]
            for state,g in observed.groupby('component_state'):
                states.append(dict(dataset=d,variant=variant,endpoint=endpoint,
                    component_state={0:'neither',1:'spo2_only',2:'map_only',3:'both'}[state],
                    n=len(g),events=int(g[endpoint].sum())))
            for subgroup in ['all','all_spo2_ge90','measured_lactate_lt2']:
                b=observed[observed.component_state.eq(3)]
                if subgroup=='all_spo2_ge90':b=b[b.spo2_min.ge(90)]
                elif subgroup=='measured_lactate_lt2':b=b[b.baseline_lactate.lt(2)]
                sync=b[b.synchronous_any.eq(1)];async_group=b[b.synchronous_any.eq(0)]
                es=int(sync[endpoint].sum());ea=int(async_group[endpoint].sum())
                nes=len(sync)-es;nea=len(async_group)-ea
                sufficient=es+ea>=100 and min(es,ea,nes,nea)>=20
                gates.append(dict(dataset=d,variant=variant,endpoint=endpoint,subgroup=subgroup,
                    jointly_observed_with_outcome=len(observed),primary_comparison_n=len(b),
                    events=es+ea,synchronous_n=len(sync),synchronous_events=es,
                    asynchronous_n=len(async_group),asynchronous_events=ea,
                    passes_count_gate=bool(sufficient),
                    decision='eligible for model support checks' if sufficient else 'counts only; frozen event gate failed'))
for name,obj in [('timing_contrast_support.json',gates),('component_state_counts.json',states),
                 ('endpoint_support_canonical.json',endpoint_support)]:
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\\n')
print(pd.DataFrame(gates).query("subgroup == 'all'").to_string(index=False))
con.close()
print('Support gate evaluated locally; no model fit or biological discovery asserted by this cell.')
''')

cell('code','''
# Conditional association screen; it runs only when the frozen gates pass.
import json, importlib.util, subprocess, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
OUT=PROJECT/'research'/'spo2_circulatory_context'
gates=json.loads((OUT/'timing_contrast_support.json').read_text())
primary=[g for g in gates if g['variant']=='pooled' and g['subgroup']=='all'
         and g['endpoint'] in ['hospital_mortality','troponin_rise_12h']]
results=[]
if any(g['passes_count_gate'] for g in primary):
    if importlib.util.find_spec('statsmodels') is None:
        subprocess.check_call([sys.executable,'-m','pip','install','-q','statsmodels==0.14.4','scipy==1.15.2'])
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
    from statsmodels.stats.multitest import multipletests
    con=duckdb.connect(str(DRIVE/'Data'/'PhysioGraph_Biological_Discovery_20260905'/'spo2_context.duckdb'))
for gate in primary:
    result=dict(dataset=gate['dataset'],endpoint=gate['endpoint'],stage='conditional association screen',
                causal=False,treatment_adjusted=False,biological_discovery=False)
    if not gate['passes_count_gate']:
        result.update(status='not fitted',reason='frozen count gate failed');results.append(result);continue
    d=gate['dataset'];endpoint=gate['endpoint']
    f=con.execute(f'SELECT * FROM {d}_pooled_features WHERE joint_eligible AND joint_spo2_jump_any=1 AND map_fall_any=1').df()
    e=con.execute(f'SELECT * FROM {d}_context_endpoints').df()
    a=f.merge(e,on='stay_id',validate='one_to_one').dropna(subset=[endpoint]).copy()
    numeric=['age','hr_mean','spo2_mean','spo2_min','map_mean','map_min',
             'joint_spo2_jumps','map_falls','joint_transitions']
    a['hr_missing']=a.hr_mean.isna().astype(float)
    a['hr_mean']=a.hr_mean.fillna(a.hr_mean.median())
    a=a.dropna(subset=numeric+['spo2_readings','person_id'])
    x=a[numeric].astype(float).copy()
    x['log_spo2_readings']=np.log1p(a.spo2_readings)
    for col in x:
        sd=x[col].std(ddof=0)
        x[col]=(x[col]-x[col].mean())/sd if sd>0 else 0.0
    sex=a.is_male.map({0.0:'female',1.0:'male'}).fillna('unknown')
    x=pd.concat([x,pd.get_dummies(sex,prefix='sex',drop_first=True,dtype=float)],axis=1)
    x['hr_missing']=a.hr_missing
    x=x.loc[:,x.nunique()>1]
    x=sm.add_constant(x,has_constant='add');x['synchronous_any']=a.synchronous_any.astype(float)
    y=a[endpoint].astype(float)
    grouped=a.groupby('synchronous_any')[endpoint].agg(['count','sum'])
    event_gate=len(grouped)==2 and y.sum()>=100 and grouped['sum'].min()>=20 and (grouped['count']-grouped['sum']).min()>=20
    result.update(n=len(a),events=int(y.sum()),coefficients=x.shape[1],person_clusters=int(a.person_id.nunique()))
    reason=None
    if not event_gate:reason='count gate failed after required-covariate exclusions'
    elif min(y.sum(),len(y)-y.sum())<10*x.shape[1]:reason='fewer than 10 events or non-events per coefficient'
    elif a.person_id.nunique()<30:reason='fewer than 30 independent person clusters'
    elif np.linalg.matrix_rank(x.to_numpy())<x.shape[1]:reason='rank deficient design'
    if reason:
        result.update(status='not fitted',reason=reason);results.append(result);continue
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',PerfectSeparationWarning)
            base=sm.GLM(y,x.drop(columns='synchronous_any'),family=sm.families.Binomial()).fit(maxiter=100)
            model=sm.GLM(y,x,family=sm.families.Binomial()).fit(maxiter=100,
                cov_type='cluster',cov_kwds={'groups':a.person_id})
        if not base.converged or not model.converged:raise ValueError('Model did not converge')
        beta=float(model.params['synchronous_any']);se=float(model.bse['synchronous_any'])
        if not np.isfinite([beta,se]).all() or se<=0:raise ValueError('Invalid clustered uncertainty')
        if max(abs(beta-1.96*se),abs(beta+1.96*se))>700:raise ValueError('Unstable, unbounded effect interval')
        result.update(status='fitted',conditional_odds_ratio=float(np.exp(beta)),
            ci95=[float(np.exp(beta-1.96*se)),float(np.exp(beta+1.96*se))],
            p_value=float(model.pvalues['synchronous_any']),
            base_deviance=float(base.deviance),extended_deviance=float(model.deviance))
        if d=='eicu' and a.hospital_id.nunique()>=30:
            hospital=sm.GLM(y,x,family=sm.families.Binomial()).fit(maxiter=100,
                cov_type='cluster',cov_kwds={'groups':a.hospital_id})
            hs=float(hospital.bse['synchronous_any'])
            result['hospital_cluster_ci95']=[float(np.exp(beta-1.96*hs)),float(np.exp(beta+1.96*hs))]
    except (ValueError,np.linalg.LinAlgError,PerfectSeparationWarning) as exc:
        result.update(status='fit failed',reason=str(exc))
    results.append(result)
if any(g['passes_count_gate'] for g in primary):
    con.close()
    for d in ['eicu','mimic']:
        rows=[r for r in results if r['dataset']==d]
        adjusted=multipletests([r.get('p_value',1.0) if r['status']=='fitted' else 1.0 for r in rows],method='holm')[1]
        for r,p in zip(rows,adjusted):
            if r['status']=='fitted':r['holm_p_two_primary_endpoints']=float(p)
(OUT/'conditional_screen_results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\\n')
replicated=[]
for endpoint in ['hospital_mortality','troponin_rise_12h']:
    pair=[r for r in results if r['endpoint']==endpoint]
    if len(pair)==2 and all(r['status']=='fitted' and r['conditional_odds_ratio']>1
                          and r.get('holm_p_two_primary_endpoints',1)<.05 for r in pair):
        replicated.append(endpoint)
from datetime import datetime,timezone
import hashlib
finished=datetime.now(timezone.utc).isoformat()
manifest=dict(environment='google_colab' if IN_COLAB else 'local',finished_utc=finished,
    analysis_lock_sha256=ANALYSIS_LOCK_SHA256,synthetic_patients=0,
    raw_input_manifest_sha256=hashlib.sha256((OUT/'input_manifest.json').read_bytes()).hexdigest(),
    implementation_sha256=hashlib.sha256((PROJECT/'scripts/build_biological_notebook.py').read_bytes()).hexdigest(),
    production_spo2_helper_sha256=hashlib.sha256((PROJECT/'src/physiograph/analysis/spo2_protocol.py').read_bytes()).hexdigest(),
    software=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,duckdb=duckdb.__version__),
    cross_database_positive_conditional_screens=replicated,
    paradigm_shifting_result=False,clinical_benefit_established=False)
(OUT/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2)+'\\n')
(OUT/'run_status.json').write_text(json.dumps(dict(stage='SpO2 circulatory-context candidate analyzed',
    status='complete',environment=manifest['environment'],updated_utc=finished,goal_complete=False,
    cross_database_positive_conditional_screens=replicated),indent=2)+'\\n')
print(pd.DataFrame(results).to_string(index=False))
print('This is a conditional association screen. The biological discovery goal is not complete.')
''')

DIRECTION_START=len(CELLS)
cell('markdown',(ROOT/'docs/SPO2_TROPONIN_DIRECTION_PLAN.md').read_text()+'\n\n'+(ROOT/'docs/SPO2_TROPONIN_BOUNDARY_AMENDMENT.md').read_text())
cell('code',f"DIRECTION_PROTOCOL_SHA256={hashlib.sha256((ROOT/'docs/SPO2_TROPONIN_DIRECTION_PLAN.md').read_bytes()).hexdigest()!r}\nDIRECTION_BOUNDARY_SHA256={hashlib.sha256((ROOT/'docs/SPO2_TROPONIN_BOUNDARY_AMENDMENT.md').read_bytes()).hexdigest()!r}")
cell('code','''
# Biological-ordering feasibility; no trajectory ratios or mortality associations.
import sys,json,importlib.util
from pathlib import Path
from datetime import datetime,timezone
import duckdb
import pandas as pd
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects'/'PhysioGraph'
sys.path.insert(0,str(PROJECT/'src'))
from physiograph.analysis.spo2_protocol import _first_instability
from physiograph.etl.mimic_extractor import _mimic_lab_item_map
from physiograph.constants import EICU_LAB_NAME_MAP
PRIVATE=DRIVE/'Data'/'PhysioGraph_Biological_Discovery_20260905'
OUT=PROJECT/'research'/'spo2_troponin_direction';OUT.mkdir(exist_ok=True)
PRIOR=PROJECT/'research'/'spo2_circulatory_context'
def direction_json(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\\n')
def direction_status(stage,status='running'):
    direction_json('run_status.json',dict(stage=stage,status=status,
        environment='google_colab' if IN_COLAB else 'local',
        updated_utc=datetime.now(timezone.utc).isoformat(),goal_complete=False))
    print(stage,status,flush=True)
lock=OUT/'protocol_lock.json'
if lock.exists():assert json.loads(lock.read_text())['sha256']==DIRECTION_PROTOCOL_SHA256
else:direction_json('protocol_lock.json',dict(sha256=DIRECTION_PROTOCOL_SHA256,
    locked_before_support_extraction=True,utc=datetime.now(timezone.utc).isoformat()))
amendment=OUT/'boundary_amendment.json'
if amendment.exists():assert json.loads(amendment.read_text())['sha256']==DIRECTION_BOUNDARY_SHA256
else:direction_json('boundary_amendment.json',dict(sha256=DIRECTION_BOUNDARY_SHA256,
    initial_support_counts_already_seen=True,trajectory_directions_not_seen=True,
    utc=datetime.now(timezone.utc).isoformat()))
con=duckdb.connect(str(PRIVATE/'troponin_direction.duckdb'))
con.execute('SET threads=2');con.execute("SET memory_limit='2GB'")
def qs(value):return "'"+str(value).replace("'","''")+"'"
con.execute(f"ATTACH {qs(PRIVATE/'spo2_context.duckdb')} AS prior (READ_ONLY)")
assert con.execute("SELECT count(*) FROM information_schema.tables WHERE table_catalog='prior' AND table_name='mimic_context_quality_marker'").fetchone()[0]>0
manifest=json.loads((PRIOR/'input_manifest.json').read_text())
used={}
def verified_source(dataset,name):
    key=dataset+'/'+name;info=manifest[key];p=Path(info['path']);st=p.stat()
    assert st.st_size==info['bytes'] and st.st_mtime_ns==info['mtime_ns'], 'Source changed: '+key
    used[key]=info
    return p
direction_status('episode anchors and hospital boundaries')
coverage=[]
for d,source in [('eicu','eICU'),('mimic','MIMIC')]:
    fields='stay_id,followup_end_offset_minutes'+(',admit_time,hadm_id' if d=='mimic' else '')
    c=con.execute(f'SELECT {fields} FROM prior.{d}_cohort').df()
    raw=con.execute(f"""SELECT v.stay_id,'{d}' dataset,concept,
        event_minute offset_minutes,value value_numeric FROM prior.{d}_vitals v
        JOIN prior.{d}_cohort c USING(stay_id) WHERE concept='spo2'""").df()
    onset=_first_instability(raw).rename(columns={'instability_onset_minutes':'episode_minute'})
    a=c.merge(onset[['stay_id','episode_minute']],on='stay_id',validate='one_to_one')
    if d=='eicu':
        patient=pd.read_csv(verified_source(source,'patient'),usecols=['patientunitstayid','hospitaladmitoffset'])
        starts=patient.set_index('patientunitstayid').hospitaladmitoffset
        a['hospital_start_minute']=pd.to_numeric(a.stay_id.map(starts),errors='coerce')
    else:
        admission=pd.read_csv(verified_source(source,'admissions'),usecols=['hadm_id','admittime','edregtime']).set_index('hadm_id')
        clocks=admission[['admittime','edregtime']].apply(pd.to_datetime,errors='coerce')
        starts=clocks.min(axis=1)
        a['hospital_start_minute']=(a.hadm_id.map(starts)-a.admit_time).dt.total_seconds()/60.0
    missing=int(a.hospital_start_minute.isna().sum())
    a=a[a.hospital_start_minute.notna()].copy()
    con.register('episode_anchors',a)
    con.execute(f'CREATE OR REPLACE TABLE {d}_anchors AS SELECT * FROM episode_anchors')
    coverage.append(dict(dataset=d,cohort=len(c),episodes=len(onset),
        missing_hospital_start=missing,anchors=len(a)))
direction_json('episode_support.json',coverage)
direction_status('eICU pre-ICU troponin extraction')
mapping=pd.DataFrame([dict(raw_name=k,concept=v) for k,v in EICU_LAB_NAME_MAP.items() if v in ['troponin_t','troponin_i']])
con.register('assay_map',mapping)
lab=verified_source('eICU','lab')
con.execute(f"""CREATE OR REPLACE TABLE eicu_preicu_troponin AS
 SELECT a.stay_id,try_cast(l.labresultoffset AS DOUBLE) event_minute,
 try_cast(l.labresultrevisedoffset AS DOUBLE) available_minute,m.concept,
 lower(trim(coalesce(l.labmeasurenamesystem,l.labmeasurenameinterface,''))) unit,
 try_cast(l.labresult AS DOUBLE) AS value
 FROM read_csv({qs(lab)},header=true,all_varchar=true,sample_size=10000,strict_mode=true) l
 JOIN eicu_anchors a ON try_cast(l.patientunitstayid AS BIGINT)=a.stay_id
 JOIN assay_map m ON lower(trim(l.labname))=m.raw_name
 WHERE try_cast(l.labresultoffset AS DOUBLE)>=greatest(-1440,a.hospital_start_minute)
 AND try_cast(l.labresultoffset AS DOUBLE)<0
 AND try_cast(l.labresult AS DOUBLE) IS NOT NULL""")
direction_status('MIMIC pre-ICU troponin extraction')
root=DRIVE/'Data'/'MIMIC'/'Full'
mapping=pd.DataFrame([dict(itemid=k,concept=v) for k,v in _mimic_lab_item_map(root).items() if v in ['troponin_t','troponin_i']])
con.register('assay_map',mapping)
lab=verified_source('MIMIC','labevents');verified_source('MIMIC','d_labitems')
con.execute(f"""CREATE OR REPLACE TABLE mimic_preicu_troponin AS
 SELECT a.stay_id,date_diff('second',a.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0 event_minute,
 date_diff('second',a.admit_time,try_cast(l.storetime AS TIMESTAMP))/60.0 available_minute,
 m.concept,lower(trim(coalesce(l.valueuom,''))) unit,try_cast(l.valuenum AS DOUBLE) AS value
 FROM read_csv({qs(lab)},header=true,all_varchar=true,sample_size=10000,strict_mode=true) l
 JOIN mimic_anchors a ON try_cast(l.hadm_id AS BIGINT)=a.hadm_id
 JOIN assay_map m ON try_cast(l.itemid AS BIGINT)=m.itemid
 WHERE date_diff('second',a.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0>=greatest(-1440,a.hospital_start_minute)
 AND try_cast(l.charttime AS TIMESTAMP)<a.admit_time AND try_cast(l.valuenum AS DOUBLE) IS NOT NULL""")
for key,info in used.items():
    st=Path(info['path']).stat()
    assert st.st_size==info['bytes'] and st.st_mtime_ns==info['mtime_ns'], 'Source changed during scan: '+key
direction_json('source_manifest.json',dict(verified_fingerprints_reused_from=str(PRIOR/'input_manifest.json'),
    same_size_and_mtime_before_and_after=True,sources=used))
direction_status('serial troponin trajectory support')
support=[]
for d in ['eicu','mimic']:
    con.execute(f"""CREATE OR REPLACE TABLE {d}_troponin AS
      WITH combined AS (
        SELECT stay_id,event_minute,available_minute,concept,unit,value FROM {d}_preicu_troponin
        UNION ALL
        SELECT l.stay_id,event_minute,available_minute,concept,unit,value FROM prior.{d}_context_labs l
        JOIN {d}_anchors a USING(stay_id) WHERE concept IN ('troponin_t','troponin_i'))
      SELECT stay_id,event_minute,concept,unit,median(value) AS value,max(available_minute) available_minute
      FROM combined WHERE value>0 AND value<=1000000 GROUP BY stay_id,event_minute,concept,unit""")
    con.execute(f"""CREATE OR REPLACE TABLE {d}_trajectory_support AS
      WITH pre_assays AS (
        SELECT t.stay_id,t.concept,t.unit,count(*) pre_n,min(event_minute) pre_first_time,
        max(event_minute) pre_last_time FROM {d}_troponin t JOIN {d}_anchors a USING(stay_id)
        WHERE event_minute>=a.episode_minute-720 AND event_minute<a.episode_minute
          AND event_minute>=a.hospital_start_minute
        GROUP BY t.stay_id,t.concept,t.unit), chosen AS (
        SELECT * FROM pre_assays QUALIFY row_number() OVER(PARTITION BY stay_id
          ORDER BY pre_first_time,CASE WHEN concept='troponin_t' THEN 0 ELSE 1 END,unit)=1), post_assays AS (
        SELECT c.stay_id,count(t.event_minute) post_n FROM chosen c JOIN {d}_anchors a USING(stay_id)
        LEFT JOIN {d}_troponin t ON t.stay_id=c.stay_id AND t.concept=c.concept AND t.unit=c.unit
          AND t.event_minute>=a.episode_minute+120 AND t.event_minute<=a.episode_minute+720
        GROUP BY c.stay_id)
      SELECT a.stay_id,a.episode_minute,c.concept,c.unit,c.pre_n,c.pre_first_time,c.pre_last_time,
        coalesce(p.post_n,0) post_n,
        coalesce(c.pre_n>=2 AND c.pre_last_time-c.pre_first_time>=60,FALSE) serial_pre,
        coalesce(c.pre_n>=2 AND c.pre_last_time-c.pre_first_time>=60 AND p.post_n>=1,FALSE) complete_trajectory
      FROM {d}_anchors a LEFT JOIN chosen c USING(stay_id) LEFT JOIN post_assays p USING(stay_id)""")
    s=con.execute(f'SELECT * FROM {d}_trajectory_support').df()
    support.append(dict(dataset=d,episodes_with_hospital_boundary=len(s),
        any_pre_episode_assay=int(s.pre_n.notna().sum()),serial_pre_episode=int(s.serial_pre.sum()),
        complete_pre_post_trajectories=int(s.complete_trajectory.sum()),
        passes_count_gate=bool(s.complete_trajectory.sum()>=100)))
direction_json('trajectory_support.json',support)
direction_json('analysis_manifest.json',dict(environment='google_colab' if IN_COLAB else 'local',
    protocol_sha256=DIRECTION_PROTOCOL_SHA256,boundary_amendment_sha256=DIRECTION_BOUNDARY_SHA256,
    completed_utc=datetime.now(timezone.utc).isoformat(),
    all_sources_pass=all(s['passes_count_gate'] for s in support),
    trajectory_directions_inspected=False,mortality_joined=False,synthetic_patients=0,
    biological_discovery=False))
con.close()
print(pd.DataFrame(support).to_string(index=False))
direction_status('biological-ordering feasibility','complete')
''')


HEMOGLOBIN_START=len(CELLS)
cell('markdown',(ROOT/'docs/SPO2_HEMOGLOBIN_FEASIBILITY_PLAN.md').read_text())
cell('code',f"HEMOGLOBIN_PROTOCOL_SHA256={hashlib.sha256((ROOT/'docs/SPO2_HEMOGLOBIN_FEASIBILITY_PLAN.md').read_bytes()).hexdigest()!r}")
cell('code','''
# Pre-ICU hemoglobin feasibility only; no outcome interactions are computed.
import sys,json,importlib.util,hashlib
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
import numpy as np
import duckdb
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects'/'PhysioGraph'
PRIVATE=DRIVE/'Data'/'PhysioGraph_Biological_Discovery_20260905'
OUT=PROJECT/'research'/'spo2_hemoglobin';OUT.mkdir(exist_ok=True)
PRIOR=PROJECT/'research'/'spo2_circulatory_context'
def hb_json(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\\n')
def hb_status(stage,status='running'):
    hb_json('run_status.json',dict(stage=stage,status=status,
        environment='google_colab' if IN_COLAB else 'local',
        updated_utc=datetime.now(timezone.utc).isoformat(),goal_complete=False))
    print(stage,status,flush=True)
lock=OUT/'protocol_lock.json'
if lock.exists():assert json.loads(lock.read_text())['sha256']==HEMOGLOBIN_PROTOCOL_SHA256
else:hb_json('protocol_lock.json',dict(sha256=HEMOGLOBIN_PROTOCOL_SHA256,
    locked_before_hemoglobin_extraction=True,outcome_interactions_not_seen=True,
    utc=datetime.now(timezone.utc).isoformat()))
con=duckdb.connect(str(PRIVATE/'spo2_hemoglobin.duckdb'))
con.execute('SET threads=2');con.execute("SET memory_limit='2GB'")
def qs(value):return "'"+str(value).replace("'","''")+"'"
con.execute(f"ATTACH {qs(PRIVATE/'spo2_context.duckdb')} AS prior (READ_ONLY)")
manifest=json.loads((PRIOR/'input_manifest.json').read_text());used={}
def hb_source(dataset,name):
    key=dataset+'/'+name;info=manifest[key];p=Path(info['path']);s=p.stat()
    assert s.st_size==info['bytes'] and s.st_mtime_ns==info['mtime_ns'],key
    used[key]=info
    return p
hb_status('pre-ICU hemoglobin source extraction')
for d,source in [('eicu','eICU'),('mimic','MIMIC')]:
    extra=',c.admit_time,c.hadm_id' if d=='mimic' else ''
    eligible=con.execute(f"""SELECT c.stay_id,f.spo2_jump_any AS exposed,
        e.troponin_rise_12h IS NOT NULL AS troponin_observed {extra}
        FROM prior.{d}_cohort c JOIN prior.{d}_features f USING(stay_id)
        JOIN prior.{d}_context_endpoints e USING(stay_id)
        WHERE f.spo2_bins>=3 AND f.spo2_transitions>=2""").df()
    if d=='eicu':
        patient=pd.read_csv(hb_source(source,'patient'),
            usecols=['patientunitstayid','hospitaladmitoffset']).set_index('patientunitstayid')
        eligible['hospital_start_minute']=pd.to_numeric(
            eligible.stay_id.map(patient.hospitaladmitoffset),errors='coerce')
    else:
        admission=pd.read_csv(hb_source(source,'admissions'),
            usecols=['hadm_id','admittime','edregtime']).set_index('hadm_id')
        start=admission.apply(pd.to_datetime,errors='coerce').min(axis=1)
        eligible['hospital_start_minute']=(eligible.hadm_id.map(start)-eligible.admit_time).dt.total_seconds()/60
        dictionary=pd.read_csv(hb_source(source,'d_labitems'))
        selected=dictionary[dictionary.itemid.eq(51222)]
        assert len(selected)==1 and selected.iloc[0]['label']=='Hemoglobin'
        assert selected.iloc[0]['category']=='Hematology' and selected.iloc[0]['fluid']=='Blood'
    con.register('eligible_frame',eligible)
    con.execute(f'CREATE OR REPLACE TABLE {d}_eligible AS SELECT * FROM eligible_frame')
    con.execute(f"""CREATE OR REPLACE TABLE {d}_first_direction AS
      WITH b AS (SELECT stay_id,bin,value,representative_minute,
        value-lag(value) OVER(PARTITION BY stay_id ORDER BY bin) delta,
        representative_minute-lag(representative_minute) OVER(PARTITION BY stay_id ORDER BY bin) gap
        FROM prior.{d}_bins WHERE concept='spo2'),
      first_jump AS (SELECT stay_id,delta FROM b WHERE abs(delta)>=4 AND gap>0 AND gap<=30
        QUALIFY row_number() OVER(PARTITION BY stay_id ORDER BY bin)=1)
      SELECT e.stay_id,CASE WHEN j.delta<0 THEN 'drop' WHEN j.delta>0 THEN 'rise' ELSE 'none' END first_direction
      FROM {d}_eligible e LEFT JOIN first_jump j USING(stay_id)""")
    assert con.execute(f"""SELECT count(*) FROM {d}_eligible e JOIN {d}_first_direction f USING(stay_id)
      WHERE (e.exposed=1)<>(f.first_direction<>'none')""").fetchone()[0]==0
    lab=hb_source(source,'lab' if d=='eicu' else 'labevents')
    if d=='eicu':
        con.execute(f"""CREATE OR REPLACE TABLE eicu_hb_raw AS
          SELECT e.stay_id,try_cast(l.labresultoffset AS DOUBLE) event_minute,
            try_cast(l.labresultrevisedoffset AS DOUBLE) available_minute,
            l.labname raw_name,try_cast(l.labtypeid AS INTEGER) lab_type,
            l.labmeasurenamesystem unit_system,l.labmeasurenameinterface unit_interface,
            try_cast(l.labresult AS DOUBLE) raw_value,
            lower(trim(l.labname)) IN ('hgb','hemoglobin','haemoglobin')
              AND try_cast(l.labtypeid AS INTEGER)=3 AS source_approved
          FROM read_csv({qs(lab)},header=true,all_varchar=true,sample_size=10000,strict_mode=true) l
          JOIN eicu_eligible e ON try_cast(l.patientunitstayid AS BIGINT)=e.stay_id
          WHERE (lower(l.labname) LIKE '%hgb%' OR lower(l.labname) LIKE '%hemoglobin%'
            OR lower(l.labname) LIKE '%haemoglobin%')
          AND try_cast(l.labresultoffset AS DOUBLE) BETWEEN greatest(-1440,e.hospital_start_minute) AND 0
          AND e.hospital_start_minute IS NOT NULL""")
    else:
        con.execute(f"""CREATE OR REPLACE TABLE mimic_hb_raw AS
          SELECT e.stay_id,date_diff('second',e.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0 event_minute,
            date_diff('second',e.admit_time,try_cast(l.storetime AS TIMESTAMP))/60.0 available_minute,
            'Hemoglobin' raw_name,3 AS lab_type,l.valueuom unit_system,'' unit_interface,
            try_cast(l.valuenum AS DOUBLE) raw_value,TRUE AS source_approved
          FROM read_csv({qs(lab)},header=true,all_varchar=true,sample_size=10000,strict_mode=true) l
          JOIN mimic_eligible e ON try_cast(l.hadm_id AS BIGINT)=e.hadm_id
          WHERE try_cast(l.itemid AS INTEGER)=51222
          AND date_diff('second',e.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0
            BETWEEN greatest(-1440,e.hospital_start_minute) AND 0
          AND e.hospital_start_minute IS NOT NULL""")
for key,info in used.items():
    s=Path(info['path']).stat()
    assert s.st_size==info['bytes'] and s.st_mtime_ns==info['mtime_ns'],key
audit=[]
for d in ['eicu','mimic']:
    rows=con.execute(f"""SELECT raw_name,lab_type,unit_system,unit_interface,source_approved,count(*) records
      FROM {d}_hb_raw GROUP BY ALL ORDER BY records DESC""").df()
    rows['dataset']=d;rows=rows.astype(object).where(pd.notna(rows),None);audit.extend(rows.to_dict('records'))
hb_json('source_name_unit_audit.json',audit)
# Source name/type whitelist and accepted units were fixed before these values were normalized.
hb_status('hemoglobin normalization and endpoint availability')
coverage=[];quality=[];validation=[]
for d in ['eicu','mimic']:
    raw=con.execute(f'SELECT * FROM {d}_hb_raw').df()
    for key in ['unit_system','unit_interface']:
        raw[key+'_clean']=raw[key].fillna('').str.strip().str.lower().str.replace(' ','',regex=False)
    u=raw.unit_system_clean.where(raw.unit_system_clean.ne(''),raw.unit_interface_clean)
    conflict=raw.unit_system_clean.ne('')&raw.unit_interface_clean.ne('')&raw.unit_system_clean.ne(raw.unit_interface_clean)
    raw['unit_approved']=u.isin(['g/dl','g/l'])&~conflict
    raw['hemoglobin']=raw.raw_value/np.where(u.eq('g/l'),10.0,1.0)
    raw['quality_approved']=raw.source_approved.fillna(False)&raw.unit_approved&raw.hemoglobin.between(3,22)
    quality.append(dict(dataset=d,candidate_records=len(raw),
        wrong_source=int((~raw.source_approved.fillna(False)).sum()),
        unknown_or_conflicting_unit=int((~raw.unit_approved).sum()),
        outside_value_range_or_missing=int((~raw.hemoglobin.between(3,22)).sum()),
        accepted_records=int(raw.quality_approved.sum())))
    con.register('normalized_frame',raw)
    con.execute(f'CREATE OR REPLACE TABLE {d}_hb_normalized AS SELECT * FROM normalized_frame')
    con.execute(f"""CREATE OR REPLACE TABLE {d}_baseline_hb AS
      SELECT stay_id,event_minute,median(hemoglobin) hemoglobin,max(available_minute) available_minute
      FROM {d}_hb_normalized WHERE quality_approved GROUP BY stay_id,event_minute
      QUALIFY row_number() OVER(PARTITION BY stay_id ORDER BY event_minute DESC)=1""")
    independent=raw[raw.quality_approved].groupby(['stay_id','event_minute'],as_index=False).hemoglobin.median()
    independent=independent.sort_values('event_minute').drop_duplicates('stay_id',keep='last').set_index('stay_id')
    baseline=con.execute(f'SELECT * FROM {d}_baseline_hb').df().set_index('stay_id')
    assert set(independent.index)==set(baseline.index)
    np.testing.assert_allclose(independent.hemoglobin.sort_index(),baseline.hemoglobin.sort_index(),rtol=0,atol=1e-12)
    validation.append(dict(dataset=d,independent_latest_value_match=True,people=len(baseline)))
    support=con.execute(f"""SELECT e.exposed,count(*) eligible,count(b.hemoglobin) pre_icu_hemoglobin,
      sum(CASE WHEN b.hemoglobin IS NOT NULL AND e.troponin_observed THEN 1 ELSE 0 END) paired_endpoint_observed,
      sum(CASE WHEN b.hemoglobin IS NOT NULL AND b.available_minute>0 THEN 1 ELSE 0 END) result_entered_after_icu_admission
      FROM {d}_eligible e LEFT JOIN {d}_baseline_hb b USING(stay_id) GROUP BY e.exposed ORDER BY e.exposed""").df()
    observed={int(r.exposed):int(r.paired_endpoint_observed) for r in support.itertuples()}
    passes=sum(observed.values())>=200 and observed.get(0,0)>=50 and observed.get(1,0)>=50
    coverage.append(dict(dataset=d,passes_minimal_feasibility=passes,groups=support.to_dict('records')))
hb_json('coverage.json',coverage);hb_json('quality_audit.json',quality)
hb_json('independent_validation.json',validation)
hb_json('source_manifest.json',dict(verified_fingerprints_reused_from=str(PRIOR/'input_manifest.json'),
    same_size_and_mtime_before_and_after=True,sources=used))
hb_json('analysis_manifest.json',dict(protocol_sha256=HEMOGLOBIN_PROTOCOL_SHA256,
    builder_sha256=hashlib.sha256((PROJECT/'scripts/build_biological_notebook.py').read_bytes()).hexdigest(),
    environment='google_colab' if IN_COLAB else 'local',completed_utc=datetime.now(timezone.utc).isoformat(),
    all_sources_pass=all(s['passes_minimal_feasibility'] for s in coverage),
    outcome_interactions_inspected=False,mortality_joined=False,synthetic_patients=0,biological_discovery=False))
con.close()
print(json.dumps(coverage,indent=2))
hb_status('hemoglobin susceptibility feasibility','complete')
''')


LINKAGE_START=len(CELLS)
cell('markdown',(ROOT/'docs/MIMIC_PREICU_LAB_LINKAGE_AUDIT.md').read_text())
cell('code',f"LINKAGE_PROTOCOL_SHA256={hashlib.sha256((ROOT/'docs/MIMIC_PREICU_LAB_LINKAGE_AUDIT.md').read_bytes()).hexdigest()!r}")
cell('code','''
# Outcome-blind MIMIC encounter-linkage audit. No concentrations are selected.
import sys,json,hashlib,importlib.util
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
import duckdb
from scipy.stats import beta
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects'/'PhysioGraph'
PRIVATE=DRIVE/'Data'/'PhysioGraph_Biological_Discovery_20260905'
OUT=PROJECT/'research'/'mimic_preicu_lab_linkage';OUT.mkdir(exist_ok=True)
PRIOR=PROJECT/'research'/'spo2_circulatory_context'
sys.path.insert(0,str(PROJECT/'src'))
from physiograph.etl.mimic_extractor import _mimic_lab_item_map
def link_json(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\\n')
def link_status(stage,status='running'):
    link_json('run_status.json',dict(stage=stage,status=status,
        environment='google_colab' if IN_COLAB else 'local',
        updated_utc=datetime.now(timezone.utc).isoformat(),goal_complete=False))
    print(stage,status,flush=True)
lock=OUT/'protocol_lock.json'
if lock.exists():assert json.loads(lock.read_text())['sha256']==LINKAGE_PROTOCOL_SHA256
else:link_json('protocol_lock.json',dict(sha256=LINKAGE_PROTOCOL_SHA256,
    locked_before_linkage_audit=True,previous_biological_support_counts_seen=True,
    outcome_interactions_not_seen=True,utc=datetime.now(timezone.utc).isoformat()))
manifest=json.loads((PRIOR/'input_manifest.json').read_text());used={}
def link_source(name):
    key='MIMIC/'+name;info=manifest[key];p=Path(info['path']);s=p.stat()
    assert s.st_size==info['bytes'] and s.st_mtime_ns==info['mtime_ns'],key
    used[key]=info
    return p
def qs(value):return "'"+str(value).replace("'","''")+"'"
con=duckdb.connect(str(PRIVATE/'mimic_preicu_lab_linkage.duckdb'))
con.execute('SET threads=2');con.execute("SET memory_limit='2GB'")
con.execute(f"ATTACH {qs(PRIVATE/'spo2_context.duckdb')} AS prior (READ_ONLY)")
con.execute("""CREATE OR REPLACE TABLE eligible AS
  SELECT c.stay_id,c.person_id AS subject_id,c.hadm_id,c.admit_time
  FROM prior.mimic_cohort c JOIN prior.mimic_features f USING(stay_id)
  WHERE f.spo2_bins>=3 AND f.spo2_transitions>=2""")
admission=pd.read_csv(link_source('admissions'),
    usecols=['subject_id','hadm_id','admittime','edregtime','dischtime'])
assert admission.hadm_id.is_unique
for key in ['admittime','edregtime','dischtime']:
    admission[key]=pd.to_datetime(admission[key],errors='coerce')
admission['encounter_start']=admission[['admittime','edregtime']].min(axis=1)
admission['valid_boundary']=admission.encounter_start.notna()&admission.dischtime.notna()&admission.dischtime.ge(admission.encounter_start)
con.register('admission_frame',admission)
con.execute('CREATE OR REPLACE TABLE admissions AS SELECT * FROM admission_frame')
assert con.execute("""SELECT count(*) FROM eligible e JOIN admissions a USING(hadm_id)
    WHERE e.subject_id<>a.subject_id""").fetchone()[0]==0
link_source('d_labitems')
mapping={k:'troponin' for k,v in _mimic_lab_item_map(DRIVE/'Data/MIMIC/Full').items() if v in ['troponin_i','troponin_t']}
mapping.update({51222:'cbc_hemoglobin',50811:'blood_gas_hemoglobin',51640:'chemistry_hemoglobin'})
itemmap=pd.DataFrame([dict(itemid=k,family=v) for k,v in mapping.items()])
con.register('item_map',itemmap)
lab=link_source('labevents')
link_status('pre-ICU laboratory identifier scan')
con.execute(f"""CREATE OR REPLACE TABLE candidate_labs AS
 SELECT try_cast(l.labevent_id AS BIGINT) labevent_id,try_cast(l.subject_id AS BIGINT) subject_id,
   try_cast(l.hadm_id AS BIGINT) recorded_hadm_id,coalesce(trim(l.hadm_id),'')='' AS missing_hadm_id,
   try_cast(l.specimen_id AS BIGINT) specimen_id,try_cast(l.itemid AS INTEGER) itemid,m.family,
   try_cast(l.charttime AS TIMESTAMP) charttime,try_cast(l.storetime AS TIMESTAMP) storetime,
   l.valueuom unit,coalesce(try_cast(l.valuenum AS DOUBLE)>0,FALSE) positive_numeric_available
 FROM read_csv({qs(lab)},header=true,all_varchar=true,sample_size=10000,strict_mode=true) l
 JOIN item_map m ON try_cast(l.itemid AS INTEGER)=m.itemid
 WHERE EXISTS(SELECT 1 FROM eligible e WHERE e.subject_id=try_cast(l.subject_id AS BIGINT)
   AND try_cast(l.charttime AS TIMESTAMP) BETWEEN e.admit_time-INTERVAL '24 hours' AND e.admit_time)""")
assert con.execute('SELECT count(*)=count(DISTINCT labevent_id) FROM candidate_labs').fetchone()[0]
for key,info in used.items():
    s=Path(info['path']).stat()
    assert s.st_size==info['bytes'] and s.st_mtime_ns==info['mtime_ns'],key
link_status('fixed temporal assignment and known-ID validation')
con.execute("""CREATE OR REPLACE TABLE temporal_candidates AS
 SELECT l.labevent_id,a.hadm_id AS candidate_hadm_id,a.admittime
 FROM candidate_labs l JOIN admissions a ON l.subject_id=a.subject_id
   AND a.valid_boundary AND l.charttime BETWEEN a.encounter_start AND a.dischtime""")
con.execute("""CREATE OR REPLACE TABLE linkage AS
 SELECT l.*,coalesce(t.candidates,0) temporal_candidate_count,
   CASE WHEN t.candidates=1 THEN t.candidate_hadm_id ELSE NULL END unique_temporal_hadm_id
 FROM candidate_labs l LEFT JOIN (
   SELECT labevent_id,count(*) candidates,min(candidate_hadm_id) candidate_hadm_id
   FROM temporal_candidates GROUP BY labevent_id) t USING(labevent_id)""")
validation=[];inventory=[]
for family in itemmap.family.unique():
    counts=con.execute("""SELECT count(*) records,
      count(*) FILTER(WHERE recorded_hadm_id IS NOT NULL) known_id,
      count(*) FILTER(WHERE missing_hadm_id) missing_id,
      count(*) FILTER(WHERE NOT missing_hadm_id AND recorded_hadm_id IS NULL) invalid_id,
      count(*) FILTER(WHERE temporal_candidate_count=0) no_temporal_match,
      count(*) FILTER(WHERE temporal_candidate_count>1) ambiguous_temporal_match,
      count(*) FILTER(WHERE recorded_hadm_id IS NOT NULL AND temporal_candidate_count=1) uniquely_assigned_known,
      count(*) FILTER(WHERE recorded_hadm_id IS NOT NULL AND temporal_candidate_count=1
        AND recorded_hadm_id=unique_temporal_hadm_id) correctly_assigned_known,
      count(*) FILTER(WHERE missing_hadm_id AND temporal_candidate_count=1) uniquely_assigned_missing
      FROM linkage WHERE family=?""",[family]).df().iloc[0].to_dict()
    counts={k:int(v) for k,v in counts.items()}
    n=counts['uniquely_assigned_known'];k=counts['correctly_assigned_known']
    patients,correct_patients=con.execute("""SELECT count(*),coalesce(sum(all_correct),0) FROM (
      SELECT subject_id,bool_and(recorded_hadm_id=unique_temporal_hadm_id) all_correct
      FROM linkage WHERE family=? AND recorded_hadm_id IS NOT NULL AND temporal_candidate_count=1
      GROUP BY subject_id)""",[family]).fetchone()
    patients=int(patients);correct_patients=int(correct_patients)
    lower=float(beta.ppf(.05,correct_patients,patients-correct_patients+1)) if correct_patients else (0.0 if patients else None)
    validation.append(dict(family=family,**counts,
      known_id_accuracy=k/n if n else None,validation_patients=patients,all_correct_patients=correct_patients,
      patient_agreement_one_sided_95pct_lower=lower,
      passes_validation=bool(patients>=200 and lower is not None and lower>.99)))
inventory=con.execute("""SELECT family,itemid,unit,missing_hadm_id,count(*) records,
  count(*) FILTER(WHERE positive_numeric_available) positive_numeric_records
  FROM linkage GROUP BY ALL ORDER BY family,itemid,unit""").df()
inventory=inventory.astype(object).where(pd.notna(inventory),None).to_dict('records')
con.execute("""CREATE OR REPLACE TABLE eligible_candidate_additions AS
 SELECT e.stay_id,l.labevent_id,l.family,l.unique_temporal_hadm_id AS candidate_hadm_id,
   l.charttime<a.admittime AS before_inpatient_admission,l.positive_numeric_available
 FROM linkage l JOIN eligible e ON l.unique_temporal_hadm_id=e.hadm_id
   AND l.subject_id=e.subject_id AND l.charttime BETWEEN e.admit_time-INTERVAL '24 hours' AND e.admit_time
 JOIN admissions a ON a.hadm_id=e.hadm_id
 WHERE l.missing_hadm_id AND l.temporal_candidate_count=1""")
additions=con.execute("""SELECT family,count(*) candidate_records,count(DISTINCT stay_id) eligible_encounters,
 count(*) FILTER(WHERE before_inpatient_admission) records_before_inpatient_admission,
 count(DISTINCT stay_id) FILTER(WHERE before_inpatient_admission) encounters_with_pre_inpatient_records
 FROM eligible_candidate_additions GROUP BY family ORDER BY family""").df().to_dict('records')
link_json('known_id_validation.json',validation)
link_json('source_inventory.json',inventory)
link_json('candidate_additions.json',additions)
link_json('admission_boundaries.json',dict(admissions=len(admission),
    invalid_boundaries=int((~admission.valid_boundary).sum()),eligible_encounters=con.execute('SELECT count(*) FROM eligible').fetchone()[0]))
link_json('source_manifest.json',dict(verified_fingerprints_reused_from=str(PRIOR/'input_manifest.json'),
    same_size_and_mtime_before_and_after=True,sources=used))
link_json('analysis_manifest.json',dict(protocol_sha256=LINKAGE_PROTOCOL_SHA256,
    builder_sha256=hashlib.sha256((PROJECT/'scripts/build_biological_notebook.py').read_bytes()).hexdigest(),
    environment='google_colab' if IN_COLAB else 'local',completed_utc=datetime.now(timezone.utc).isoformat(),
    lab_concentrations_selected=False,mortality_joined=False,prior_patient_tables_modified=False,
    biological_discovery=False,amendment_applied=False))
con.close()
print(json.dumps(dict(validation=validation,candidate_additions=additions),indent=2))
link_status('MIMIC pre-ICU linkage audit','complete')
''')


AMENDMENT_START=len(CELLS)
cell('markdown',(ROOT/'docs/PREICU_LAB_LINKAGE_AMENDMENT.md').read_text())
cell('code',f"AMENDMENT_PROTOCOL_SHA256={hashlib.sha256((ROOT/'docs/PREICU_LAB_LINKAGE_AMENDMENT.md').read_bytes()).hexdigest()!r}\n"
    +CELLS[HEMOGLOBIN_START+1]['source']+'\n'+CELLS[DIRECTION_START+1]['source'])
cell('code','''
# Restore only the saved uniquely assigned missing-ID records in a separate database.
import sys,json,hashlib,importlib.util
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
import numpy as np
import duckdb
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects'/'PhysioGraph'
PRIVATE=DRIVE/'Data'/'PhysioGraph_Biological_Discovery_20260905'
AMEND_ROOT=PROJECT/'research'/'preicu_linkage_amendment';AMEND_ROOT.mkdir(exist_ok=True)
OUT=AMEND_ROOT
PRIOR=PROJECT/'research'/'spo2_circulatory_context'
sys.path.insert(0,str(PROJECT/'src'))
from physiograph.etl.mimic_extractor import _mimic_lab_item_map
def amend_json(name,obj):
    (AMEND_ROOT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\\n')
def amend_status(stage,status='running'):
    amend_json('run_status.json',dict(stage=stage,status=status,
        environment='google_colab' if IN_COLAB else 'local',
        updated_utc=datetime.now(timezone.utc).isoformat(),goal_complete=False))
    print(stage,status,flush=True)
lock=AMEND_ROOT/'protocol_lock.json'
if lock.exists():assert json.loads(lock.read_text())['sha256']==AMENDMENT_PROTOCOL_SHA256
else:amend_json('protocol_lock.json',dict(sha256=AMENDMENT_PROTOCOL_SHA256,
    locked_before_restored_values_extraction=True,original_feasibility_and_linkage_counts_seen=True,
    biological_interactions_and_directions_not_seen=True,utc=datetime.now(timezone.utc).isoformat()))
for folder,name,expected in [('spo2_hemoglobin','protocol_lock.json',HEMOGLOBIN_PROTOCOL_SHA256),
    ('spo2_troponin_direction','protocol_lock.json',DIRECTION_PROTOCOL_SHA256),
    ('spo2_troponin_direction','boundary_amendment.json',DIRECTION_BOUNDARY_SHA256)]:
    assert json.loads((PROJECT/'research'/folder/name).read_text())['sha256']==expected
validation=json.loads((PROJECT/'research/mimic_preicu_lab_linkage/known_id_validation.json').read_text())
for family in ['cbc_hemoglobin','troponin']:
    assert next(r for r in validation if r['family']==family)['passes_validation']
assert json.loads((PROJECT/'research/mimic_preicu_lab_linkage/independent_validation.json').read_text())['patient_time_assignment_matches_independent_pandas']
used=json.loads((PROJECT/'research/spo2_hemoglobin/source_manifest.json').read_text())['sources']
for key,info in used.items():
    s=Path(info['path']).stat()
    assert s.st_size==info['bytes'] and s.st_mtime_ns==info['mtime_ns'],key
checkpoints={}
for name in ['spo2_context','spo2_hemoglobin','troponin_direction','mimic_preicu_lab_linkage']:
    p=PRIVATE/(name+'.duckdb');s=p.stat();h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    checkpoints[name]=dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=h.hexdigest())
amend_json('input_database_fingerprints.json',checkpoints)
def qs(value):return "'"+str(value).replace("'","''")+"'"
con=duckdb.connect(str(PRIVATE/'preicu_linkage_amendment.duckdb'))
con.execute('SET threads=2');con.execute("SET memory_limit='2GB'")
for alias,name in [('prior','spo2_context'),('hb','spo2_hemoglobin'),('ordering','troponin_direction'),('linkage','mimic_preicu_lab_linkage')]:
    con.execute(f"ATTACH {qs(PRIVATE/(name+'.duckdb'))} AS {alias} (READ_ONLY)")
amend_status('retrieving actual values for audited missing-ID laboratories')
lab=Path(used['MIMIC/labevents']['path'])
con.execute(f"""CREATE OR REPLACE TABLE mimic_restored_labs AS
 SELECT a.stay_id,try_cast(l.labevent_id AS BIGINT) labevent_id,
   try_cast(l.hadm_id AS BIGINT) source_hadm_id,a.candidate_hadm_id assigned_hadm_id,
   try_cast(l.itemid AS INTEGER) itemid,a.family,
   date_diff('second',c.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0 event_minute,
   date_diff('second',c.admit_time,try_cast(l.storetime AS TIMESTAMP))/60.0 available_minute,
   l.valueuom raw_unit,try_cast(l.valuenum AS DOUBLE) raw_value,'validated_temporal' linkage_class
 FROM read_csv({qs(lab)},header=true,all_varchar=true,sample_size=10000,strict_mode=true) l
 JOIN linkage.eligible_candidate_additions a ON try_cast(l.labevent_id AS BIGINT)=a.labevent_id
 JOIN prior.mimic_cohort c ON a.stay_id=c.stay_id AND a.candidate_hadm_id=c.hadm_id
   AND try_cast(l.subject_id AS BIGINT)=c.person_id
 WHERE a.family IN ('cbc_hemoglobin','troponin') AND coalesce(trim(l.hadm_id),'')=''""")
expected=con.execute("""SELECT count(*) FROM linkage.eligible_candidate_additions
 WHERE family IN ('cbc_hemoglobin','troponin')""").fetchone()[0]
assert con.execute('SELECT count(*) FROM mimic_restored_labs').fetchone()[0]==expected
assert con.execute('SELECT count(*)=count(DISTINCT labevent_id) FROM mimic_restored_labs').fetchone()[0]
assert con.execute("""SELECT count(*) FROM mimic_restored_labs
 WHERE source_hadm_id IS NOT NULL OR event_minute < -1440 OR event_minute>0""").fetchone()[0]==0
for key,info in used.items():
    s=Path(info['path']).stat()
    assert s.st_size==info['bytes'] and s.st_mtime_ns==info['mtime_ns'],key
for key,info in checkpoints.items():
    s=Path(info['path']).stat()
    assert s.st_size==info['bytes'] and s.st_mtime_ns==info['mtime_ns'],key
for d in ['eicu','mimic']:
    con.execute(f'CREATE OR REPLACE TABLE {d}_eligible AS SELECT * FROM hb.{d}_eligible')
    con.execute(f'CREATE OR REPLACE TABLE {d}_first_direction AS SELECT * FROM hb.{d}_first_direction')
    con.execute(f'CREATE OR REPLACE TABLE {d}_anchors AS SELECT * FROM ordering.{d}_anchors')
con.execute("""CREATE OR REPLACE TABLE eicu_hb_raw AS
 SELECT *, 'recorded_id' linkage_class FROM hb.eicu_hb_raw""")
con.execute("""CREATE OR REPLACE TABLE mimic_hb_raw AS
 SELECT *, 'recorded_id' linkage_class FROM hb.mimic_hb_raw
 UNION ALL SELECT stay_id,event_minute,available_minute,'Hemoglobin',3,raw_unit,'',raw_value,TRUE,linkage_class
 FROM mimic_restored_labs WHERE family='cbc_hemoglobin' AND itemid=51222""")
con.execute('CREATE OR REPLACE TABLE eicu_preicu_troponin AS SELECT * FROM ordering.eicu_preicu_troponin')
mapping=pd.DataFrame([dict(itemid=k,concept=v) for k,v in _mimic_lab_item_map(DRIVE/'Data/MIMIC/Full').items() if v in ['troponin_i','troponin_t']])
con.register('troponin_map',mapping)
con.execute("""CREATE OR REPLACE TABLE mimic_preicu_troponin AS
 SELECT stay_id,event_minute,available_minute,concept,unit,value FROM ordering.mimic_preicu_troponin
 UNION ALL SELECT r.stay_id,r.event_minute,r.available_minute,m.concept,lower(trim(coalesce(r.raw_unit,''))),r.raw_value
 FROM mimic_restored_labs r JOIN mimic_anchors a USING(stay_id) JOIN troponin_map m USING(itemid)
 WHERE r.family='troponin'""")
restored=con.execute("""SELECT family,count(*) records,count(DISTINCT stay_id) encounters
 FROM mimic_restored_labs GROUP BY family ORDER BY family""").df().to_dict('records')
amend_json('restored_record_counts.json',restored)
amend_json('source_manifest.json',dict(verified_fingerprints_reused_from=str(PRIOR/'input_manifest.json'),
    same_size_and_mtime_before_and_after=True,sources=used))
con.close()
print(json.dumps(restored,indent=2))
amend_status('restoration complete; unchanged feasibility algorithms pending')
''')
_hb_marker="hb_status('hemoglobin normalization and endpoint availability')"
_hb_original=CELLS[HEMOGLOBIN_START+2]['source']
assert _hb_original.count(_hb_marker)==1
_hb_tail=_hb_marker+_hb_original.split(_hb_marker,1)[1]
_hb_tail=_hb_tail.replace('protocol_sha256=HEMOGLOBIN_PROTOCOL_SHA256,',
    'protocol_sha256=HEMOGLOBIN_PROTOCOL_SHA256,linkage_amendment_sha256=AMENDMENT_PROTOCOL_SHA256,')
assert _hb_tail.count('linkage_amendment_sha256')==1
cell('code','''
# Reuse the exact original Hb normalization and support algorithm on augmented records.
OUT=AMEND_ROOT/'hemoglobin';OUT.mkdir(exist_ok=True)
def hb_json(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\\n')
def hb_status(stage,status='running'):
    hb_json('run_status.json',dict(stage=stage,status=status,
        environment='google_colab' if IN_COLAB else 'local',
        updated_utc=datetime.now(timezone.utc).isoformat(),goal_complete=False))
    print(stage,status,flush=True)
con=duckdb.connect(str(PRIVATE/'preicu_linkage_amendment.duckdb'))
con.execute('SET threads=2');con.execute("SET memory_limit='2GB'")
'''+_hb_tail+'''
amend_status('hemoglobin support complete; injury-ordering support pending')
''')
_direction_marker="direction_status('serial troponin trajectory support')"
_direction_original=CELLS[DIRECTION_START+2]['source']
assert _direction_original.count(_direction_marker)==1
_direction_tail=_direction_marker+_direction_original.split(_direction_marker,1)[1]
_direction_tail=_direction_tail.replace('protocol_sha256=DIRECTION_PROTOCOL_SHA256,',
    'protocol_sha256=DIRECTION_PROTOCOL_SHA256,linkage_amendment_sha256=AMENDMENT_PROTOCOL_SHA256,')
assert _direction_tail.count('linkage_amendment_sha256')==1
cell('code','''
# Reuse the exact original ordering-availability algorithm; do not compute trajectory directions.
OUT=AMEND_ROOT/'troponin_direction';OUT.mkdir(exist_ok=True)
def direction_json(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\\n')
def direction_status(stage,status='running'):
    direction_json('run_status.json',dict(stage=stage,status=status,
        environment='google_colab' if IN_COLAB else 'local',
        updated_utc=datetime.now(timezone.utc).isoformat(),goal_complete=False))
    print(stage,status,flush=True)
con=duckdb.connect(str(PRIVATE/'preicu_linkage_amendment.duckdb'))
con.execute('SET threads=2');con.execute("SET memory_limit='2GB'")
con.execute(f"ATTACH {qs(PRIVATE/'spo2_context.duckdb')} AS prior (READ_ONLY)")
'''+_direction_tail+'''
for subdir,original,name in [('hemoglobin','spo2_hemoglobin','coverage.json'),('troponin_direction','spo2_troponin_direction','trajectory_support.json')]:
    before=json.loads((PROJECT/'research'/original/name).read_text())
    after=json.loads((AMEND_ROOT/subdir/name).read_text())
    assert next(r for r in before if r['dataset']=='eicu')==next(r for r in after if r['dataset']=='eicu')
amend_json('analysis_manifest.json',dict(amendment_sha256=AMENDMENT_PROTOCOL_SHA256,
    original_hemoglobin_protocol_sha256=HEMOGLOBIN_PROTOCOL_SHA256,
    original_direction_protocol_sha256=DIRECTION_PROTOCOL_SHA256,
    original_direction_boundary_sha256=DIRECTION_BOUNDARY_SHA256,
    environment='google_colab' if IN_COLAB else 'local',completed_utc=datetime.now(timezone.utc).isoformat(),
    hemoglobin_feasibility_pass=json.loads((AMEND_ROOT/'hemoglobin/analysis_manifest.json').read_text())['all_sources_pass'],
    ordering_feasibility_pass=json.loads((AMEND_ROOT/'troponin_direction/analysis_manifest.json').read_text())['all_sources_pass'],
    biological_associations_examined=False,troponin_directions_examined=False,biological_discovery=False))
amend_status('linkage-only amended feasibility','complete')
''')

INTERACTION_START=len(CELLS)
import runpy

interaction_cells=runpy.run_path(str(ROOT/'scripts/hemoglobin_interaction_cells.py'))
cell('markdown',(ROOT/'docs/SPO2_HEMOGLOBIN_INTERACTION_PLAN.md').read_text())
cell('code',f"HB_INTERACTION_PROTOCOL_SHA256={hashlib.sha256((ROOT/'docs/SPO2_HEMOGLOBIN_INTERACTION_PLAN.md').read_bytes()).hexdigest()!r}\n"+interaction_cells['SETUP'])
cell('code',interaction_cells['CONTEXT'])
cell('code',interaction_cells['MODELS'])
cell('code',interaction_cells['VALIDATION'])
cell('code',interaction_cells['REPORT'])

OXYGEN_RESPONSE_START=len(CELLS)
cell('markdown',(ROOT/'docs/SPO2_OXYGEN_RESPONSE_FEASIBILITY_PLAN.md').read_text())
oxygen_response_cells=runpy.run_path(str(ROOT/'scripts/oxygen_response_cells.py'))
cell('code',f"OXYGEN_RESPONSE_PROTOCOL_SHA256={hashlib.sha256((ROOT/'docs/SPO2_OXYGEN_RESPONSE_FEASIBILITY_PLAN.md').read_bytes()).hexdigest()!r}\n"+oxygen_response_cells['FEASIBILITY'])

WAVEFORM_OVERLAP_START=len(CELLS)
cell('markdown',(ROOT/'docs/SPO2_MECHANISM_REVIEW.md').read_text())
waveform_overlap_cells=runpy.run_path(str(ROOT/'scripts/waveform_overlap_cells.py'))
cell('code',waveform_overlap_cells['OVERLAP'])
cell('code',waveform_overlap_cells['VALIDATION'])

WAVEFORM_PILOT_START=len(CELLS)
cell('markdown',(ROOT/'docs/SPO2_WAVEFORM_PILOT_PLAN.md').read_text())
waveform_pilot_cells=runpy.run_path(str(ROOT/'scripts/waveform_pilot_cells.py'))
cell('code',f"WAVEFORM_PILOT_PROTOCOL_SHA256={hashlib.sha256((ROOT/'docs/SPO2_WAVEFORM_PILOT_PLAN.md').read_bytes()).hexdigest()!r}\n"+waveform_pilot_cells['SETUP'])
cell('code',waveform_pilot_cells['MATERIALIZE'])
cell('code',waveform_pilot_cells['ANALYSIS'])
cell('code',waveform_pilot_cells['VALIDATION'])

ACID_BASE_START=len(CELLS)
cell('markdown',(ROOT/'docs/SPO2_ACID_BASE_FEASIBILITY_PLAN.md').read_text())
acid_base_cells=runpy.run_path(str(ROOT/'scripts/acid_base_cells.py'))
cell('code',f"ACID_BASE_PROTOCOL_SHA256={hashlib.sha256((ROOT/'docs/SPO2_ACID_BASE_FEASIBILITY_PLAN.md').read_bytes()).hexdigest()!r}\n"+acid_base_cells['SETUP'])
cell('code',acid_base_cells['EXTRACT'])
cell('code',acid_base_cells['SUPPORT'])
cell('code',acid_base_cells['VALIDATION'])

ACID_BASE_MORTALITY_START=len(CELLS)
cell('markdown',(ROOT/'docs/SPO2_ACID_BASE_MORTALITY_SCREEN_PLAN.md').read_text())
cell('code',f"ACID_BASE_MORTALITY_PROTOCOL_SHA256={hashlib.sha256((ROOT/'docs/SPO2_ACID_BASE_MORTALITY_SCREEN_PLAN.md').read_bytes()).hexdigest()!r}\n"+acid_base_cells['MORTALITY'])

CARDIORENAL_START=len(CELLS)
cell('markdown',(ROOT/'docs/SPO2_CARDIORENAL_CONTRAST_PLAN.md').read_text()+'\n\n'+(ROOT/'docs/SPO2_CARDIORENAL_ASSAY_ENCODING_CLARIFICATION.md').read_text()+'\n\n'+(ROOT/'docs/SPO2_CARDIORENAL_DECIMAL_THRESHOLD_CORRECTION.md').read_text())
cardiorenal_cells=runpy.run_path(str(ROOT/'scripts/cardiorenal_cells.py'))
cell('code',f"CARDIORENAL_PROTOCOL_SHA256={hashlib.sha256((ROOT/'docs/SPO2_CARDIORENAL_CONTRAST_PLAN.md').read_bytes()).hexdigest()!r}\n"+cardiorenal_cells['ANALYSIS'])

VENOUS_PRESSURE_START=len(CELLS)
cell('markdown',(ROOT/'docs/SPO2_VENOUS_PRESSURE_PLAN.md').read_text())
venous_pressure_cells=runpy.run_path(str(ROOT/'scripts/venous_pressure_cells.py'))
cell('code',f"VENOUS_PRESSURE_PROTOCOL_SHA256={hashlib.sha256((ROOT/'docs/SPO2_VENOUS_PRESSURE_PLAN.md').read_bytes()).hexdigest()!r}\n"+venous_pressure_cells['SETUP'])
cell('code',venous_pressure_cells['EXTRACT'])
cell('code',venous_pressure_cells['SUPPORT'])

TROPONIN_DESCRIPTIVE_START=len(CELLS)
cell('markdown',(ROOT/'docs/SPO2_TROPONIN_DESCRIPTIVE_AMENDMENT.md').read_text())
troponin_descriptive_cells=runpy.run_path(str(ROOT/'scripts/troponin_descriptive_cells.py'))
cell('code',f"TROPONIN_DESCRIPTIVE_PROTOCOL_SHA256={hashlib.sha256((ROOT/'docs/SPO2_TROPONIN_DESCRIPTIVE_AMENDMENT.md').read_bytes()).hexdigest()!r}\n"+troponin_descriptive_cells['ANALYSIS'])

ARTERIAL_PAIRING_START=len(CELLS)
cell('markdown',(ROOT/'docs/SPO2_ARTERIAL_PAIRING_PLAN.md').read_text())
arterial_pairing_cells=runpy.run_path(str(ROOT/'scripts/arterial_pairing_cells.py'))
cell('code',f"ARTERIAL_PAIRING_PROTOCOL_SHA256={hashlib.sha256((ROOT/'docs/SPO2_ARTERIAL_PAIRING_PLAN.md').read_bytes()).hexdigest()!r}\n"+arterial_pairing_cells['ANALYSIS'])

TROPONIN_OPPORTUNITY_START=len(CELLS)
cell('markdown',(ROOT/'docs/SPO2_TROPONIN_OPPORTUNITY_PLAN.md').read_text())
troponin_opportunity_cells=runpy.run_path(str(ROOT/'scripts/troponin_opportunity_cells.py'))
cell('code',f"TROPONIN_OPPORTUNITY_PROTOCOL_SHA256={hashlib.sha256((ROOT/'docs/SPO2_TROPONIN_OPPORTUNITY_PLAN.md').read_bytes()).hexdigest()!r}\n"+troponin_opportunity_cells['ANALYSIS'])

TROPONIN_ASSAY_AUDIT_START=len(CELLS)
cell('markdown',(ROOT/'docs/SPO2_TROPONIN_ASSAY_AUDIT_PLAN.md').read_text())
troponin_assay_audit_cells=runpy.run_path(str(ROOT/'scripts/troponin_assay_audit_cells.py'))
cell('code',f"TROPONIN_ASSAY_AUDIT_PROTOCOL_SHA256={hashlib.sha256((ROOT/'docs/SPO2_TROPONIN_ASSAY_AUDIT_PLAN.md').read_bytes()).hexdigest()!r}\n"+troponin_assay_audit_cells['SETUP'])
cell('code',troponin_assay_audit_cells['EXTRACT'])
cell('code',troponin_assay_audit_cells['SUPPORT'])

TROPONIN_REPORTING_RECOVERY_START=len(CELLS)
cell('markdown',(ROOT/'docs/SPO2_TROPONIN_REPORTING_RECOVERY_PLAN.md').read_text())
troponin_reporting_recovery_cells=runpy.run_path(str(ROOT/'scripts/troponin_reporting_recovery_cells.py'))
cell('code',f"TROPONIN_REPORTING_RECOVERY_PROTOCOL_SHA256={hashlib.sha256((ROOT/'docs/SPO2_TROPONIN_REPORTING_RECOVERY_PLAN.md').read_bytes()).hexdigest()!r}\n"+troponin_reporting_recovery_cells['SETUP'])
cell('code',troponin_reporting_recovery_cells['EXTRACT'])
cell('code',troponin_reporting_recovery_cells['SUPPORT'])

TROPONIN_COMMENT_TEMPLATE_START=len(CELLS)
cell('markdown',(ROOT/'docs/SPO2_TROPONIN_COMMENT_TEMPLATE_AMENDMENT.md').read_text())
cell('code',f"TROPONIN_COMMENT_TEMPLATE_SHA256={hashlib.sha256((ROOT/'docs/SPO2_TROPONIN_COMMENT_TEMPLATE_AMENDMENT.md').read_bytes()).hexdigest()!r}\n"+troponin_reporting_recovery_cells['TEMPLATE_AUDIT'])

TROPONIN_EMERGENCE_START=len(CELLS)
cell('markdown',(ROOT/'docs/SPO2_TROPONIN_EMERGENCE_PLAN.md').read_text())
troponin_emergence_cells=runpy.run_path(str(ROOT/'scripts/troponin_emergence_cells.py'))
cell('code',f"TROPONIN_EMERGENCE_PROTOCOL_SHA256={hashlib.sha256((ROOT/'docs/SPO2_TROPONIN_EMERGENCE_PLAN.md').read_bytes()).hexdigest()!r}\n"+troponin_emergence_cells['SETUP'])
cell('code',troponin_emergence_cells['ANALYSIS'])

WITHIN_PERSON_TROPONIN_START=len(CELLS)
cell('markdown',(ROOT/'docs/SPO2_WITHIN_PERSON_TROPONIN_PLAN.md').read_text())
within_person_troponin_cells=runpy.run_path(str(ROOT/'scripts/within_person_troponin_cells.py'))
cell('code',f"WITHIN_PERSON_TROPONIN_PROTOCOL_SHA256={hashlib.sha256((ROOT/'docs/SPO2_WITHIN_PERSON_TROPONIN_PLAN.md').read_bytes()).hexdigest()!r}\n"+within_person_troponin_cells['SETUP'])
cell('code',within_person_troponin_cells['TIMING'])
cell('code',within_person_troponin_cells['VITALS'])
cell('code',within_person_troponin_cells['FEATURES'])
cell('code',within_person_troponin_cells['MODELS'])
cell('markdown',(ROOT/'docs/SPO2_WITHIN_PERSON_LAG_AMENDMENT.md').read_text())
cell('code',f"WITHIN_PERSON_LAG_AMENDMENT_SHA256={hashlib.sha256((ROOT/'docs/SPO2_WITHIN_PERSON_LAG_AMENDMENT.md').read_bytes()).hexdigest()!r}\n"+within_person_troponin_cells['LAG_SENSITIVITY'])

EICU_HOSPITAL_START=len(CELLS)
cell('markdown',(ROOT/'docs/SPO2_EICU_HOSPITAL_AMENDMENT.md').read_text())
eicu_hospital_cells=runpy.run_path(str(ROOT/'scripts/eicu_hospital_coupling_cells.py'))
cell('code',f"EICU_HOSPITAL_PROTOCOL_SHA256={hashlib.sha256((ROOT/'docs/SPO2_EICU_HOSPITAL_AMENDMENT.md').read_bytes()).hexdigest()!r}\n"+eicu_hospital_cells['ANALYSIS'])

ECG_LINKAGE_START=len(CELLS)
cell('markdown',(ROOT/'docs/SPO2_ECG_LINKAGE_PLAN.md').read_text())
ecg_linkage_cells=runpy.run_path(str(ROOT/'scripts/ecg_linkage_cells.py'))
cell('code',f"ECG_LINKAGE_PROTOCOL_SHA256={hashlib.sha256((ROOT/'docs/SPO2_ECG_LINKAGE_PLAN.md').read_bytes()).hexdigest()!r}\n"+ecg_linkage_cells['SETUP'])
cell('code',ecg_linkage_cells['DOWNLOAD'])
cell('code',ecg_linkage_cells['LINKAGE'])

ECG_CLOCK_START=len(CELLS)
cell('markdown',(ROOT/'docs/SPO2_ECG_CLOCK_PLAN.md').read_text())
ecg_clock_cells=runpy.run_path(str(ROOT/'scripts/ecg_clock_cells.py'))
cell('code',f"ECG_CLOCK_PROTOCOL_SHA256={hashlib.sha256((ROOT/'docs/SPO2_ECG_CLOCK_PLAN.md').read_bytes()).hexdigest()!r}\n"+ecg_clock_cells['ANALYSIS'])

ECG_REPOLARIZATION_START=len(CELLS)
cell('markdown',(ROOT/'docs/SPO2_ECG_REPOLARIZATION_PLAN.md').read_text()+'\n\n'+(ROOT/'docs/SPO2_ECG_TIMESTAMP_AMENDMENT.md').read_text())
ecg_repolarization_cells=runpy.run_path(str(ROOT/'scripts/ecg_repolarization_cells.py'))
cell('code',f"ECG_REPOLARIZATION_PROTOCOL_SHA256={hashlib.sha256((ROOT/'docs/SPO2_ECG_REPOLARIZATION_PLAN.md').read_bytes()).hexdigest()!r}\nECG_TIMESTAMP_AMENDMENT_SHA256={hashlib.sha256((ROOT/'docs/SPO2_ECG_TIMESTAMP_AMENDMENT.md').read_bytes()).hexdigest()!r}\n"+ecg_repolarization_cells['SETUP'])
cell('code',ecg_repolarization_cells['DOWNLOAD'])
cell('code',ecg_repolarization_cells['MEASURE'])
cell('code',ecg_repolarization_cells['MODELS'])
cell('markdown',(ROOT/'docs/SPO2_ECG_INTERPRETATION_LIMITS.md').read_text())

HEART_RATE_POLARITY_START=len(CELLS)
cell('markdown',(ROOT/'docs/SPO2_HEART_RATE_POLARITY_PLAN.md').read_text())
heart_rate_polarity_cells=runpy.run_path(str(ROOT/'scripts/heart_rate_polarity_cells.py'))
cell('code',f"HEART_RATE_POLARITY_PROTOCOL_SHA256={hashlib.sha256((ROOT/'docs/SPO2_HEART_RATE_POLARITY_PLAN.md').read_bytes()).hexdigest()!r}\n"+heart_rate_polarity_cells['SETUP'])
cell('code',heart_rate_polarity_cells['PHENOTYPE'])
cell('code',heart_rate_polarity_cells['SCREEN'])

ARTERIAL_RAW_RECOVERY_START=len(CELLS)
cell('markdown',(ROOT/'docs/SPO2_ARTERIAL_RAW_RECOVERY_PLAN.md').read_text())
arterial_raw_cells=runpy.run_path(str(ROOT/'scripts/arterial_raw_recovery_cells.py'))
cell('code',f"ARTERIAL_RAW_PROTOCOL_SHA256={hashlib.sha256((ROOT/'docs/SPO2_ARTERIAL_RAW_RECOVERY_PLAN.md').read_bytes()).hexdigest()!r}\n"+arterial_raw_cells['SETUP'])
cell('code',arterial_raw_cells['EXTRACT'])
cell('code',arterial_raw_cells['SUPPORT'])

DIRECT_FLOW_START=len(CELLS)
cell('markdown',(ROOT/'docs/SPO2_DIRECT_FLOW_PLAN.md').read_text())
direct_flow_cells=runpy.run_path(str(ROOT/'scripts/direct_flow_cells.py'))
cell('code',f"DIRECT_FLOW_PROTOCOL_SHA256={hashlib.sha256((ROOT/'docs/SPO2_DIRECT_FLOW_PLAN.md').read_bytes()).hexdigest()!r}\n"+direct_flow_cells['SETUP'])
cell('code',direct_flow_cells['EXTRACT'])
cell('code',direct_flow_cells['SUPPORT'])

RECOVERY_HYSTERESIS_START=len(CELLS)
cell('markdown',(ROOT/'docs/SPO2_RECOVERY_HYSTERESIS_PLAN.md').read_text())
recovery_cells=runpy.run_path(str(ROOT/'scripts/recovery_hysteresis_cells.py'))
cell('code',f"RECOVERY_PROTOCOL_SHA256={hashlib.sha256((ROOT/'docs/SPO2_RECOVERY_HYSTERESIS_PLAN.md').read_bytes()).hexdigest()!r}\n"+recovery_cells['SETUP'])
cell('code',recovery_cells['PHENOTYPE'])
cell('code',recovery_cells['SCREEN'])

HEMOGLOBIN_OXIDATION_START=len(CELLS)
cell('markdown',(ROOT/'docs/SPO2_HEMOGLOBIN_OXIDATION_PLAN.md').read_text())
oxidation_cells=runpy.run_path(str(ROOT/'scripts/hemoglobin_oxidation_cells.py'))
cell('code',f"OXIDATION_PROTOCOL_SHA256={hashlib.sha256((ROOT/'docs/SPO2_HEMOGLOBIN_OXIDATION_PLAN.md').read_bytes()).hexdigest()!r}\n"+oxidation_cells['SETUP'])
cell('code',oxidation_cells['EXTRACT'])
cell('code',oxidation_cells['TIMING'])
cell('code',oxidation_cells['SCREEN'])

OXIDATION_HOSPITAL_START=len(CELLS)
cell('markdown',(ROOT/'docs/SPO2_OXIDATION_HOSPITAL_PLAN.md').read_text())
oxidation_hospital_cells=runpy.run_path(str(ROOT/'scripts/oxidation_hospital_cells.py'))
cell('code',f"OXIDATION_HOSPITAL_PROTOCOL_SHA256={hashlib.sha256((ROOT/'docs/SPO2_OXIDATION_HOSPITAL_PLAN.md').read_bytes()).hexdigest()!r}\n"+oxidation_hospital_cells['SETUP'])
cell('code',oxidation_hospital_cells['QUALIFY'])
cell('code',oxidation_hospital_cells['SCREEN'])

IONIZED_CALCIUM_START=len(CELLS)
cell('markdown',(ROOT/'docs/SPO2_IONIZED_CALCIUM_PLAN.md').read_text())
calcium_cells=runpy.run_path(str(ROOT/'scripts/ionized_calcium_cells.py'))
cell('code',f"CALCIUM_PROTOCOL_SHA256={hashlib.sha256((ROOT/'docs/SPO2_IONIZED_CALCIUM_PLAN.md').read_bytes()).hexdigest()!r}\n"+calcium_cells['SETUP'])
cell('code',calcium_cells['EXTRACT'])
cell('code',calcium_cells['QUALIFY'])
cell('code',calcium_cells['SCREEN'])

CALCIUM_RELATIVE_START=len(CELLS)
cell('markdown',(ROOT/'docs/SPO2_CALCIUM_RELATIVE_PLAN.md').read_text())
calcium_relative_cells=runpy.run_path(str(ROOT/'scripts/calcium_relative_cells.py'))
cell('code',f"CALCIUM_RELATIVE_PROTOCOL_SHA256={hashlib.sha256((ROOT/'docs/SPO2_CALCIUM_RELATIVE_PLAN.md').read_bytes()).hexdigest()!r}\n"+calcium_relative_cells['SETUP'])
cell('code',calcium_relative_cells['QUALIFY'])
cell('code',calcium_relative_cells['SCREEN'])

ADRENERGIC_METABOLIC_START=len(CELLS)
cell('markdown',(ROOT/'docs/SPO2_ADRENERGIC_METABOLIC_PLAN.md').read_text())
adrenergic_cells=runpy.run_path(str(ROOT/'scripts/adrenergic_metabolic_cells.py'))
cell('code',f"ADRENERGIC_PROTOCOL_SHA256={hashlib.sha256((ROOT/'docs/SPO2_ADRENERGIC_METABOLIC_PLAN.md').read_bytes()).hexdigest()!r}\n"+adrenergic_cells['SETUP'])
cell('code',adrenergic_cells['EXTRACT'])
cell('code',adrenergic_cells['QUALIFY'])
cell('code',adrenergic_cells['SCREEN'])

CAPNOGRAPHY_START=len(CELLS)
cell('markdown',(ROOT/'docs/SPO2_CAPNOGRAPHY_FEASIBILITY_PLAN.md').read_text())
capnography_cells=runpy.run_path(str(ROOT/'scripts/capnography_feasibility_cells.py'))
cell('code',f"CAPNOGRAPHY_PROTOCOL_SHA256={hashlib.sha256((ROOT/'docs/SPO2_CAPNOGRAPHY_FEASIBILITY_PLAN.md').read_bytes()).hexdigest()!r}\n"+capnography_cells['SETUP'])
cell('code',capnography_cells['EXTRACT'])
cell('code',capnography_cells['VERIFY'])
cell('code',capnography_cells['SUPPORT'])

LACTATE_CO2_START=len(CELLS)
cell('markdown',(ROOT/'docs/SPO2_LACTATE_CO2_PLAN.md').read_text())
lactate_co2_cells=runpy.run_path(str(ROOT/'scripts/lactate_co2_cells.py'))
cell('code',f"LACTATE_CO2_PROTOCOL_SHA256={hashlib.sha256((ROOT/'docs/SPO2_LACTATE_CO2_PLAN.md').read_bytes()).hexdigest()!r}\n"+lactate_co2_cells['SETUP'])
cell('code',lactate_co2_cells['EXTRACT'])
cell('code',lactate_co2_cells['VERIFY'])
cell('code',lactate_co2_cells['QUALIFY'])
cell('code',lactate_co2_cells['SCREEN'])

VENOUS_O2_FEASIBILITY_START=len(CELLS)
cell('markdown',(ROOT/'docs/SPO2_VENOUS_O2_FEASIBILITY_PLAN.md').read_text())
venous_o2_cells=runpy.run_path(str(ROOT/'scripts/venous_o2_reserve_cells.py'))
cell('code',f"VENOUS_O2_FEASIBILITY_SHA256={hashlib.sha256((ROOT/'docs/SPO2_VENOUS_O2_FEASIBILITY_PLAN.md').read_bytes()).hexdigest()!r}\n"+venous_o2_cells['SETUP'])
cell('code',venous_o2_cells['EXTRACT'])
cell('code',venous_o2_cells['SUPPORT'])

MASKED_PULSATILITY_START=len(CELLS)
cell('markdown',(ROOT/'docs/MASKED_PULSATILITY_FEASIBILITY_PLAN.md').read_text())
masked_pulsatility_cells=runpy.run_path(str(ROOT/'scripts/masked_pulsatility_cells.py'))
cell('code',f"MASKED_PULSATILITY_FEASIBILITY_SHA256={hashlib.sha256((ROOT/'docs/MASKED_PULSATILITY_FEASIBILITY_PLAN.md').read_bytes()).hexdigest()!r}\n"+masked_pulsatility_cells['SETUP'])
cell('code',masked_pulsatility_cells['BP_EXTRACT'])
cell('code',masked_pulsatility_cells['BASELINE_SCREEN'])
cell('code',masked_pulsatility_cells['FOLLOWUP_GATE'])

MASKED_PULSATILITY_AUDIT_START=len(CELLS)
cell('markdown',(ROOT/'docs/MASKED_PULSATILITY_SOURCE_AUDIT_PLAN.md').read_text())
masked_pulsatility_audit_cells=runpy.run_path(str(ROOT/'scripts/masked_pulsatility_source_audit_cells.py'))
cell('code',f"MASKED_PULSATILITY_SOURCE_AUDIT_SHA256={hashlib.sha256((ROOT/'docs/MASKED_PULSATILITY_SOURCE_AUDIT_PLAN.md').read_bytes()).hexdigest()!r}\n"+masked_pulsatility_audit_cells['SETUP'])
cell('code',masked_pulsatility_audit_cells['EXTRACT'])
cell('code',masked_pulsatility_audit_cells['REPORT'])

MASKED_PULSATILITY_SOURCE_REPAIR_START=len(CELLS)
cell('markdown',(ROOT/'docs/MASKED_PULSATILITY_SOURCE_REPAIR_PLAN.md').read_text())
masked_pulsatility_source_repair_cells=runpy.run_path(str(ROOT/'scripts/masked_pulsatility_source_repair_cells.py'))
cell('code',f"MASKED_PULSATILITY_SOURCE_REPAIR_SHA256={hashlib.sha256((ROOT/'docs/MASKED_PULSATILITY_SOURCE_REPAIR_PLAN.md').read_bytes()).hexdigest()!r}\n"+masked_pulsatility_source_repair_cells['SETUP'])
cell('code',masked_pulsatility_source_repair_cells['LABEL_DISCOVERY'])
cell('code',masked_pulsatility_source_repair_cells['EXTRACT_AND_DEDUP'])
cell('code',masked_pulsatility_source_repair_cells['HARMONIZE'])
cell('code',masked_pulsatility_source_repair_cells['LACTATE_AVAILABILITY'])
cell('code',masked_pulsatility_source_repair_cells['REPORT'])

MASKED_PULSATILITY_DELAYED_SUPPORT_START=len(CELLS)
cell('markdown','''
# Masked Pulsatility and Delayed Hemodynamic Support

The outcome-blind source repair passed strongly. The locked primary exposure
has 1,123 compensated routine-cuff crossings and 9,878 controls in MIMIC, plus
577 crossings and 9,147 controls across 134 contributing eICU hospitals. This
notebook now tells one prespecified story: whether the first masked-pulsatility
crossing predicts new continuous vasoactive/inotropic support more than two
hours later, beyond current SBP/MAP and their preceding trajectories.

MIMIC is the development database. eICU is the locked external confirmation.
The first two hours are a concurrent-deterioration blanking period. The primary
window is strictly greater than two through twelve hours. Secondary outcomes
are not loaded until the primary analysis has completed.
''')
cell('markdown',(ROOT/'docs/MASKED_PULSATILITY_DELAYED_SUPPORT_PROTOCOL.md').read_text())
delayed_support_cells=runpy.run_path(str(ROOT/'scripts/masked_pulsatility_delayed_support_cells.py'))
cell('code',delayed_support_cells['SETUP'])
cell('code',delayed_support_cells['BASELINE'])
cell('code',delayed_support_cells['LOCK_ANALYSIS'])
cell('code',delayed_support_cells['MIMIC_OUTCOME'])
cell('code',delayed_support_cells['EICU_OUTCOME'])
cell('code',delayed_support_cells['PRIMARY'])
cell('code',delayed_support_cells['ROBUSTNESS'])
cell('code',delayed_support_cells['SECONDARY'])
cell('code',delayed_support_cells['REPORT'])

SHOCK_SIGNAL_DISCOVERY_START=len(CELLS)
cell('markdown','''
# One routine signal before objective shock progression

The masked-pulsatility hypothesis is closed as `NO_SIGNAL` and is not retuned
here. This section performs a finite MIMIC-only discovery screen of 36 frozen
level, slope, change, and variability features from nine routine physiological
signals. The target requires a new pressure/support abnormality plus a distinct
objective hypoperfusion abnormality; treatment alone is not the endpoint.

Candidate measurements stop at ICU hour 4, hours 4–6 are blanked, and the
primary outcome window is hours >6 through 16. eICU remains unopened unless
one MIMIC candidate clears multiplicity, effect-size, coverage, bootstrap,
several-hour-lead, anti-circularity, integrity, and literature-novelty gates.
''')
cell('markdown',(ROOT/'docs/OBJECTIVE_SHOCK_SINGLE_SIGNAL_DISCOVERY_PROTOCOL.md').read_text())
shock_discovery_cells=runpy.run_path(str(ROOT/'scripts/shock_signal_discovery_cells.py'))
cell('code',shock_discovery_cells['SETUP'])
cell('code',shock_discovery_cells['EXTRACT'])
cell('code',shock_discovery_cells['ENDPOINT'])
cell('code',shock_discovery_cells['LOCK'])
cell('code',shock_discovery_cells['SCREEN'])
cell('code',shock_discovery_cells['STABILITY'])
cell('code',shock_discovery_cells['REPORT'])
SPO2_VARIABILITY_VALIDATION_START=len(CELLS)
cell('markdown',"""# Targeted SpO2-variability replication\n\nThis is an explicitly post-hoc follow-up. It preserves the original failed 1.30 discovery gate, runs fixed MIMIC stability analyses, freezes the external specification, and performs one hospital-clustered eICU validation.""")
cell('markdown',(ROOT/'docs/SPO2_VARIABILITY_TARGETED_VALIDATION_PROTOCOL.md').read_text())
spo2_validation_cells=runpy.run_path(str(ROOT/'scripts/spo2_variability_validation_cells.py'))
cell('code',spo2_validation_cells['SETUP'])
cell('code',spo2_validation_cells['FREEZE'])
cell('code',spo2_validation_cells['VALIDATE'])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--context-only',action='store_true',help='Execute the active SpO2 extension and preserve archived outputs')
    parser.add_argument('--context-labs-only',action='store_true',help='Resume endpoint extraction after the completed vital/cohort build')
    parser.add_argument('--resume-eicu',action='store_true',help='Recover the eICU cohort committed by the prior interrupted local attempt')
    parser.add_argument('--context-reconcile-only',action='store_true',help='Reconcile extracted tables to the production risk set')
    parser.add_argument('--direction-only',action='store_true',help='Execute the new serial-troponin biological-ordering feasibility gate')
    parser.add_argument('--hemoglobin-only',action='store_true',help='Execute pre-ICU hemoglobin susceptibility feasibility')
    parser.add_argument('--linkage-only',action='store_true',help='Execute the outcome-blind MIMIC laboratory linkage audit')
    parser.add_argument('--amendment-only',action='store_true',help='Execute the documented linkage-only amendment and unchanged feasibility gates')
    parser.add_argument('--interaction-context-only',action='store_true',help='Extract frozen hemoglobin analysis care context without outcomes')
    parser.add_argument('--interaction-models-only',action='store_true',help='Run frozen hemoglobin models using completed audited care extraction')
    parser.add_argument('--oxygen-response-only',action='store_true',help='Run outcome-blind oxygen response timing feasibility')
    parser.add_argument('--waveform-overlap-only',action='store_true',help='Check public MIMIC waveform metadata against the existing HF cohort')
    parser.add_argument('--waveform-materialize-only',action='store_true',help='Retrieve the frozen 11-record MIMIC waveform pilot inputs')
    parser.add_argument('--waveform-analysis-only',action='store_true',help='Run the frozen respiratory pilot on the completed MIMIC waveform extraction')
    parser.add_argument('--waveform-validation-only',action='store_true',help='Validate completed pilot inputs without repeating episode analysis')
    parser.add_argument('--acid-base-only',action='store_true',help='Run outcome-blind arterial acid-base support in the original cohorts')
    parser.add_argument('--acid-base-validation-only',action='store_true',help='Validate completed acid-base support without repeating raw extraction')
    parser.add_argument('--acid-base-mortality-only',action='store_true',help='Run the frozen acid-base crude mortality interaction screen')
    parser.add_argument('--cardiorenal-only',action='store_true',help='Run the frozen paired troponin-creatinine association contrast')
    parser.add_argument('--venous-pressure-only',action='store_true',help='Run the frozen CVP timing feasibility extraction and validation')
    parser.add_argument('--venous-pressure-support-only',action='store_true',help='Validate completed CVP extraction without repeating raw scans')
    parser.add_argument('--troponin-descriptive-only',action='store_true',help='Run the explicit descriptive amendment without rescuing the failed primary gate')
    parser.add_argument('--arterial-pairing-only',action='store_true',help='Run the timing-only arterial oxygen pairing audit from existing caches')
    parser.add_argument('--troponin-opportunity-only',action='store_true',help='Run the paired troponin sampling-opportunity contrast')
    parser.add_argument('--assay-audit-only',action='store_true',help='Rescan and audit original troponin reporting text')
    parser.add_argument('--assay-audit-support-only',action='store_true',help='Validate a completed assay-text extraction without rescanning raw files')
    parser.add_argument('--reporting-recovery-only',action='store_true',help='Recover same-record MIMIC comments and audit excluded eICU bounds')
    parser.add_argument('--reporting-recovery-support-only',action='store_true',help='Validate completed comment extraction and bounded-result selection')
    parser.add_argument('--comment-template-only',action='store_true',help='Recover only the explicitly whitelisted MIMIC comment bounds')
    parser.add_argument('--troponin-emergence-only',action='store_true',help='Run frozen first-specimen emergence from explicit reporting bounds')
    parser.add_argument('--within-person-troponin-only',action='store_true',help='Run sequential within-person oxygen/troponin support and analysis')
    parser.add_argument('--within-person-models-only',action='store_true',help='Use completed longitudinal vital extraction for the frozen exposure gate and model')
    parser.add_argument('--eicu-hospital-only',action='store_true',help='Run the explicitly secondary eICU hospital-disjoint validation')
    parser.add_argument('--ecg-linkage-only',action='store_true',help='Fetch official MIMIC diagnostic ECG metadata and audit original-cohort overlap')
    parser.add_argument('--ecg-clock-only',action='store_true',help='Compare diagnostic ECG times with isolated charted EKG procedures without correcting clocks')
    parser.add_argument('--ecg-repolarization-only',action='store_true',help='Run the frozen paired ECG repolarization-versus-conduction experiment')
    parser.add_argument('--heart-rate-polarity-only',action='store_true',help='Run the frozen first-transition HR-direction mortality screen')
    parser.add_argument('--arterial-raw-recovery-only',action='store_true',help='Scan original raw lab sources for missing arterial timing support')
    parser.add_argument('--direct-flow-only',action='store_true',help='Run direct cardiac-output metadata support in the original MIMIC cohort')
    parser.add_argument('--recovery-hysteresis-only',action='store_true',help='Run fixed recovery and sham-anchor mortality screen in both original cohorts')
    parser.add_argument('--hemoglobin-oxidation-only',action='store_true',help='Run frozen temporal methemoglobin support and biochemical screen in both cohorts')
    parser.add_argument('--hemoglobin-oxidation-analysis-only',action='store_true',help='Use completed MetHb raw extraction for fixed support, screen and validation')
    parser.add_argument('--oxidation-hospitals-only',action='store_true',help='Run the separate fixed methemoglobin change screen across disjoint eICU hospitals')
    parser.add_argument('--ionized-calcium-only',action='store_true',help='Run frozen calcium/pH raw extraction, qualification and biochemical screen')
    parser.add_argument('--ionized-calcium-analysis-only',action='store_true',help='Run fixed calcium/pH qualification and screen from completed raw cache')
    parser.add_argument('--calcium-relative-only',action='store_true',help='Run the separately frozen native-ratio calcium/pH experiment')
    parser.add_argument('--adrenergic-metabolic-only',action='store_true',help='Run the frozen acute coupled potassium/glucose experiment')
    parser.add_argument('--adrenergic-metabolic-analysis-only',action='store_true',help='Use completed raw potassium/glucose extraction for qualification and joint-event analysis')
    parser.add_argument('--capnography-only',action='store_true',help='Run frozen original-event capnography metadata extraction and timing support')
    parser.add_argument('--capnography-support-only',action='store_true',help='Validate completed capnography metadata without repeating raw scans')
    parser.add_argument('--capnography-resume-validation-only',action='store_true',help='Resume interrupted hash verification and timing support from committed raw metadata tables')
    parser.add_argument('--lactate-co2-only',action='store_true',help='Run frozen coupled lactate/arterial CO2/pH experiment')
    parser.add_argument('--lactate-co2-analysis-only',action='store_true',help='Qualify and screen completed raw lactate/CO2/pH extraction')
    parser.add_argument('--venous-o2-feasibility-only',action='store_true',help='Run only the outcome-blind venous oxygen reserve Phase A audit')
    parser.add_argument('--masked-pulsatility-only',action='store_true',help='Run only the outcome-blind masked pulsatility Phase A gate')
    parser.add_argument('--masked-pulsatility-source-audit-only',action='store_true',help='Run only the outcome-blind masked pulsatility source audit')
    parser.add_argument('--masked-pulsatility-source-repair-only',action='store_true',help='Run only the final outcome-blind masked pulsatility source repair and feasibility audit')
    parser.add_argument('--masked-pulsatility-source-repair-report-only',action='store_true',help='Resume only source-repair setup and final reporting from completed cached tables')
    parser.add_argument('--masked-pulsatility-delayed-support-preoutcome-only',action='store_true',help='Build pre-anchor covariates and freeze the delayed-support analysis before outcome access')
    parser.add_argument('--masked-pulsatility-delayed-support-only',action='store_true',help='Run the complete locked delayed-support analysis in MIMIC then eICU')
    parser.add_argument('--shock-signal-discovery-preassociation-only',action='store_true',help='Extract MIMIC data and lock the 36-test screen before candidate associations')
    parser.add_argument('--shock-signal-discovery-only',action='store_true',help='Run the complete frozen MIMIC-only objective-shock signal discovery screen')
    parser.add_argument('--spo2-variability-validation-only',action='store_true',help='Run the frozen post-hoc MIMIC follow-up and one eICU external validation')
    args=parser.parse_args()
    if args.hemoglobin_oxidation_only: args.execute=True
    if args.hemoglobin_oxidation_analysis_only: args.execute=True
    if args.oxidation_hospitals_only: args.execute=True
    if args.ionized_calcium_only or args.ionized_calcium_analysis_only: args.execute=True
    if args.calcium_relative_only: args.execute=True
    if args.adrenergic_metabolic_only or args.adrenergic_metabolic_analysis_only: args.execute=True
    if args.capnography_only or args.capnography_support_only or args.capnography_resume_validation_only: args.execute=True
    if args.lactate_co2_only or args.lactate_co2_analysis_only: args.execute=True
    if args.venous_o2_feasibility_only: args.execute=True
    if args.masked_pulsatility_only: args.execute=True
    if args.masked_pulsatility_source_audit_only: args.execute=True
    if args.masked_pulsatility_source_repair_only: args.execute=True
    if args.masked_pulsatility_source_repair_report_only: args.execute=True
    if args.masked_pulsatility_delayed_support_preoutcome_only: args.execute=True
    if args.masked_pulsatility_delayed_support_only: args.execute=True
    if args.shock_signal_discovery_preassociation_only: args.execute=True
    if args.shock_signal_discovery_only: args.execute=True
    if args.spo2_variability_validation_only: args.execute=True
    if args.recovery_hysteresis_only: args.execute=True
    if args.direct_flow_only: args.execute=True
    if args.arterial_raw_recovery_only: args.execute=True
    if args.heart_rate_polarity_only: args.execute=True
    if args.ecg_repolarization_only: args.execute=True
    if args.ecg_clock_only: args.execute=True
    if args.arterial_pairing_only or args.troponin_opportunity_only or args.assay_audit_only or args.assay_audit_support_only or args.reporting_recovery_only or args.reporting_recovery_support_only or args.comment_template_only or args.troponin_emergence_only or args.within_person_troponin_only or args.within_person_models_only or args.eicu_hospital_only or args.ecg_linkage_only: args.execute=True
    # The primary runnable notebook follows the user's current project scope.
    # Preserve the superseded phosphate experiment as historical evidence only.
    active_start=SPO2_VARIABILITY_VALIDATION_START
    active_cells=CELLS[active_start:]
    nb={'nbformat': 4,'nbformat_minor': 5,'cells': active_cells,'metadata': {
        'kernelspec':{'display_name': 'Python 3','language': 'python','name': 'python3'},
        'language_info':{'name': 'python'},'colab':{'name': TARGET.name,'provenance': []}}}
    if TARGET.exists():
        old=json.loads(TARGET.read_text())
        archived_ids={c['id'] for c in CELLS[:CONTEXT_START]}
        archived=[c for c in old['cells'] if c.get('id') in archived_ids]
        if archived:
            archive=ROOT/'research/biological_discovery/archived_phosphate_notebook.ipynb'
            if not archive.exists():
                historical={**old,'cells':archived}
                archive.write_text(json.dumps(historical,indent=1)+'\n')
        known_ids={c['id'] for c in CELLS}
        if any(c.get('id') not in known_ids for c in old['cells']):
            raise RuntimeError('Refusing removal of an unrecognized notebook cell')
        previous={c.get('id'):c for c in old['cells']}
        for after in active_cells:
            before=previous.get(after['id'],{})
            if before.get('source')==after['source'] and before.get('cell_type')=='code':
                after['outputs']=before.get('outputs',[])
                after['execution_count']=before.get('execution_count')
    TARGET.write_text(json.dumps(nb,indent=1)+'\n')
    if not args.execute and not args.context_only and not args.context_labs_only and not args.context_reconcile_only and not args.direction_only and not args.hemoglobin_only and not args.linkage_only and not args.amendment_only and not args.interaction_context_only and not args.interaction_models_only and not args.oxygen_response_only and not args.waveform_overlap_only and not args.waveform_materialize_only and not args.waveform_analysis_only and not args.waveform_validation_only and not args.acid_base_only and not args.acid_base_validation_only and not args.acid_base_mortality_only and not args.cardiorenal_only and not args.venous_pressure_only and not args.venous_pressure_support_only and not args.troponin_descriptive_only and not args.venous_o2_feasibility_only:
        print(TARGET)
        return
    import nbformat
    nbformat.validate(nbformat.from_dict(nb))
    namespace={'__name__':'__main__','RESUME_EICU':args.resume_eicu,'COMMENT_TEMPLATE_ONLY':args.comment_template_only}
    count=0
    for cell_index,c in enumerate(nb['cells']):
        i=active_start+cell_index
        if c['cell_type']!='code': continue
        count+=1
        if args.interaction_context_only and not INTERACTION_START<=i<INTERACTION_START+3: continue
        if args.waveform_overlap_only and not WAVEFORM_OVERLAP_START<=i<WAVEFORM_PILOT_START: continue
        if args.waveform_materialize_only and not WAVEFORM_PILOT_START<=i<WAVEFORM_PILOT_START+3: continue
        if args.waveform_analysis_only and (not WAVEFORM_PILOT_START<=i<ACID_BASE_START or i==WAVEFORM_PILOT_START+2): continue
        if args.waveform_validation_only and i not in (WAVEFORM_PILOT_START+1,WAVEFORM_PILOT_START+4): continue
        if args.acid_base_only and not ACID_BASE_START<=i<ACID_BASE_MORTALITY_START: continue
        if args.acid_base_validation_only and i not in (ACID_BASE_START+1,ACID_BASE_START+4): continue
        if args.acid_base_mortality_only and not ACID_BASE_MORTALITY_START<=i<CARDIORENAL_START: continue
        if args.cardiorenal_only and not CARDIORENAL_START<=i<VENOUS_PRESSURE_START: continue
        if args.venous_pressure_only and not VENOUS_PRESSURE_START<=i<TROPONIN_DESCRIPTIVE_START: continue
        if args.venous_pressure_support_only and i not in (VENOUS_PRESSURE_START+1,VENOUS_PRESSURE_START+3): continue
        if args.troponin_descriptive_only and not TROPONIN_DESCRIPTIVE_START<=i<ARTERIAL_PAIRING_START: continue
        if args.arterial_pairing_only and not ARTERIAL_PAIRING_START<=i<TROPONIN_OPPORTUNITY_START: continue
        if args.troponin_opportunity_only and not TROPONIN_OPPORTUNITY_START<=i<TROPONIN_ASSAY_AUDIT_START: continue
        if args.assay_audit_only and not TROPONIN_ASSAY_AUDIT_START<=i<TROPONIN_REPORTING_RECOVERY_START: continue
        if args.assay_audit_support_only and i not in (TROPONIN_ASSAY_AUDIT_START+1,TROPONIN_ASSAY_AUDIT_START+3): continue
        if args.reporting_recovery_only and not TROPONIN_REPORTING_RECOVERY_START<=i<TROPONIN_COMMENT_TEMPLATE_START: continue
        if args.reporting_recovery_support_only and i not in (TROPONIN_REPORTING_RECOVERY_START+1,TROPONIN_REPORTING_RECOVERY_START+3): continue
        if args.comment_template_only and i not in (TROPONIN_REPORTING_RECOVERY_START+1,TROPONIN_COMMENT_TEMPLATE_START+1): continue
        if args.troponin_emergence_only and not TROPONIN_EMERGENCE_START<=i<WITHIN_PERSON_TROPONIN_START: continue
        if args.within_person_troponin_only and not WITHIN_PERSON_TROPONIN_START<=i<EICU_HOSPITAL_START: continue
        if args.within_person_models_only and i not in (WITHIN_PERSON_TROPONIN_START+1,WITHIN_PERSON_TROPONIN_START+2,WITHIN_PERSON_TROPONIN_START+4,WITHIN_PERSON_TROPONIN_START+5,WITHIN_PERSON_TROPONIN_START+7): continue
        if args.eicu_hospital_only and not EICU_HOSPITAL_START<=i<ECG_LINKAGE_START: continue
        if args.ecg_linkage_only and not ECG_LINKAGE_START<=i<ECG_CLOCK_START: continue
        if args.ecg_clock_only and not ECG_CLOCK_START<=i<ECG_REPOLARIZATION_START: continue
        if args.ecg_repolarization_only and not ECG_REPOLARIZATION_START<=i<HEART_RATE_POLARITY_START: continue
        if args.heart_rate_polarity_only and not HEART_RATE_POLARITY_START<=i<ARTERIAL_RAW_RECOVERY_START: continue
        if args.arterial_raw_recovery_only and not ARTERIAL_RAW_RECOVERY_START<=i<DIRECT_FLOW_START: continue
        if args.direct_flow_only and not DIRECT_FLOW_START<=i<RECOVERY_HYSTERESIS_START: continue
        if args.recovery_hysteresis_only and not RECOVERY_HYSTERESIS_START<=i<HEMOGLOBIN_OXIDATION_START: continue
        if args.hemoglobin_oxidation_only and not HEMOGLOBIN_OXIDATION_START<=i<OXIDATION_HOSPITAL_START: continue
        if args.hemoglobin_oxidation_analysis_only and i not in (HEMOGLOBIN_OXIDATION_START+1,HEMOGLOBIN_OXIDATION_START+3,HEMOGLOBIN_OXIDATION_START+4): continue
        if args.oxidation_hospitals_only and not OXIDATION_HOSPITAL_START<=i<IONIZED_CALCIUM_START: continue
        if args.ionized_calcium_only and not IONIZED_CALCIUM_START<=i<CALCIUM_RELATIVE_START: continue
        if args.ionized_calcium_analysis_only and i not in (IONIZED_CALCIUM_START+1,IONIZED_CALCIUM_START+3,IONIZED_CALCIUM_START+4): continue
        if args.calcium_relative_only and not CALCIUM_RELATIVE_START<=i<ADRENERGIC_METABOLIC_START: continue
        if args.adrenergic_metabolic_only and not ADRENERGIC_METABOLIC_START<=i<CAPNOGRAPHY_START: continue
        if args.adrenergic_metabolic_analysis_only and i not in (ADRENERGIC_METABOLIC_START+1,ADRENERGIC_METABOLIC_START+3,ADRENERGIC_METABOLIC_START+4): continue
        if args.capnography_only and not CAPNOGRAPHY_START<=i<LACTATE_CO2_START: continue
        if args.capnography_support_only and i not in (CAPNOGRAPHY_START+1,CAPNOGRAPHY_START+4): continue
        if args.capnography_resume_validation_only and i not in (CAPNOGRAPHY_START+1,CAPNOGRAPHY_START+3,CAPNOGRAPHY_START+4): continue
        if args.lactate_co2_only and not LACTATE_CO2_START<=i<VENOUS_O2_FEASIBILITY_START: continue
        if args.lactate_co2_analysis_only and i not in (LACTATE_CO2_START+1,LACTATE_CO2_START+4,LACTATE_CO2_START+5): continue
        if args.venous_o2_feasibility_only and not VENOUS_O2_FEASIBILITY_START<=i<MASKED_PULSATILITY_START: continue
        if args.masked_pulsatility_only and not MASKED_PULSATILITY_START<=i<MASKED_PULSATILITY_AUDIT_START: continue
        if args.masked_pulsatility_source_audit_only and not MASKED_PULSATILITY_AUDIT_START<=i<MASKED_PULSATILITY_SOURCE_REPAIR_START: continue
        if args.masked_pulsatility_source_repair_only and i<MASKED_PULSATILITY_SOURCE_REPAIR_START: continue
        if args.masked_pulsatility_source_repair_report_only and i not in (MASKED_PULSATILITY_SOURCE_REPAIR_START+1,MASKED_PULSATILITY_SOURCE_REPAIR_START+6): continue
        if args.masked_pulsatility_delayed_support_preoutcome_only and i not in (MASKED_PULSATILITY_DELAYED_SUPPORT_START+2,MASKED_PULSATILITY_DELAYED_SUPPORT_START+3,MASKED_PULSATILITY_DELAYED_SUPPORT_START+4): continue
        if args.masked_pulsatility_delayed_support_only and i<MASKED_PULSATILITY_DELAYED_SUPPORT_START: continue
        if args.shock_signal_discovery_preassociation_only and i not in range(SHOCK_SIGNAL_DISCOVERY_START+2,SHOCK_SIGNAL_DISCOVERY_START+6): continue
        if args.shock_signal_discovery_only and i<SHOCK_SIGNAL_DISCOVERY_START: continue
        if args.spo2_variability_validation_only and i<SPO2_VARIABILITY_VALIDATION_START: continue
        if args.oxygen_response_only and not OXYGEN_RESPONSE_START<=i<WAVEFORM_OVERLAP_START: continue
        if args.interaction_models_only and (not INTERACTION_START<=i<OXYGEN_RESPONSE_START or i==INTERACTION_START+2): continue
        if args.amendment_only and not AMENDMENT_START<=i<INTERACTION_START: continue
        if args.linkage_only and not LINKAGE_START<=i<AMENDMENT_START: continue
        if args.hemoglobin_only and not HEMOGLOBIN_START<=i<LINKAGE_START: continue
        if args.direction_only and not DIRECTION_START<=i<HEMOGLOBIN_START: continue
        if args.context_reconcile_only and i<CONTEXT_RECONCILE_CELL: continue
        if args.context_labs_only and i<CONTEXT_LABS_CELL: continue
        if args.context_only and i<CONTEXT_START: continue
        c['execution_count']=count
        c['outputs']=[]
        stream=io.StringIO()
        print('Executing notebook cell',i,flush=True)
        try:
            with contextlib.redirect_stdout(stream):
                exec(  # noqa: S102 - intentionally execute generated notebook cells
                    compile(c['source'],f'{TARGET.name}:cell{i}','exec'),namespace
                )
        except Exception as exc:
            c['outputs'].append({'output_type': 'error','ename': type(exc).__name__,'evalue': str(exc),'traceback': traceback.format_exc().splitlines()})
            out=Path(namespace.get('OUT',ROOT/'research/biological_discovery'))
            out.mkdir(parents=True,exist_ok=True)
            (out/'run_status.json').write_text(json.dumps({'status': 'failed','cell': i,'error_type': type(exc).__name__,'message': str(exc)}))
            raise
        finally:
            value=stream.getvalue()
            if value: c['outputs'].insert(0,{'output_type': 'stream','name': 'stdout','text': value})
            TARGET.write_text(json.dumps(nb,indent=1)+'\n')
            print(value,flush=True)
    print('Locally executed selected notebook cells:',TARGET,flush=True)

if __name__=='__main__':
    main()
