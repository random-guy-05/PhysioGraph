"""Build and execute the single raw-data biological discovery notebook."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import textwrap
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'PhysioGraph_Biological_Discovery.ipynb'
PLAN=ROOT/'docs/BIOLOGICAL_DISCOVERY_PLAN.md'
CONTEXT_PLAN=ROOT/'docs/SPO2_CIRCULATORY_CONTEXT_PLAN.md'
CELLS=[]

def cell(kind,source):
    c=dict(cell_type=kind,id=f'biology-{len(CELLS):02d}',metadata={},source=textwrap.dedent(source).strip())
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

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--context-only',action='store_true',help='Execute the active SpO2 extension and preserve archived outputs')
    parser.add_argument('--context-labs-only',action='store_true',help='Resume endpoint extraction after the completed vital/cohort build')
    parser.add_argument('--resume-eicu',action='store_true',help='Recover the eICU cohort committed by the prior interrupted local attempt')
    parser.add_argument('--context-reconcile-only',action='store_true',help='Reconcile extracted tables to the production risk set')
    parser.add_argument('--direction-only',action='store_true',help='Execute the new serial-troponin biological-ordering feasibility gate')
    args=parser.parse_args()
    # The primary runnable notebook follows the user's current project scope.
    # Preserve the superseded phosphate experiment as historical evidence only.
    active_cells=CELLS[CONTEXT_START:]
    nb=dict(nbformat=4,nbformat_minor=5,cells=active_cells,metadata={
        'kernelspec':dict(display_name='Python 3',language='python',name='python3'),
        'language_info':dict(name='python'),'colab':dict(name=TARGET.name,provenance=[])})
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
    if not args.execute and not args.context_only and not args.context_labs_only and not args.context_reconcile_only and not args.direction_only:
        print(TARGET)
        return
    import nbformat
    nbformat.validate(nbformat.from_dict(nb))
    namespace={'__name__':'__main__','RESUME_EICU':args.resume_eicu}
    import faulthandler
    faulthandler.dump_traceback_later(180,repeat=True)
    count=0
    for cell_index,c in enumerate(nb['cells']):
        i=CONTEXT_START+cell_index
        if c['cell_type']!='code': continue
        count+=1
        if args.direction_only and i<DIRECTION_START: continue
        if args.context_reconcile_only and i<CONTEXT_RECONCILE_CELL: continue
        if args.context_labs_only and i<CONTEXT_LABS_CELL: continue
        if args.context_only and i<CONTEXT_START: continue
        c['execution_count']=count
        c['outputs']=[]
        stream=io.StringIO()
        print('Executing notebook cell',i,flush=True)
        try:
            with contextlib.redirect_stdout(stream):
                exec(compile(c['source'],f'{TARGET.name}:cell{i}','exec'),namespace)
        except Exception as exc:
            c['outputs'].append(dict(output_type='error',ename=type(exc).__name__,evalue=str(exc),traceback=traceback.format_exc().splitlines()))
            out=Path(namespace.get('OUT',ROOT/'research/biological_discovery'))
            out.mkdir(parents=True,exist_ok=True)
            (out/'run_status.json').write_text(json.dumps(dict(status='failed',cell=i,error_type=type(exc).__name__,message=str(exc))))
            raise
        finally:
            value=stream.getvalue()
            if value: c['outputs'].insert(0,dict(output_type='stream',name='stdout',text=value))
            TARGET.write_text(json.dumps(nb,indent=1)+'\n')
            print(value,flush=True)
    print('Locally executed selected notebook cells:',TARGET,flush=True)

if __name__=='__main__': main()
