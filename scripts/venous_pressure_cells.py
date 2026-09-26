"""Outcome-blind CVP timing feasibility for the original SpO2 episodes."""

SETUP = r"""
import json,hashlib,importlib.util
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import duckdb
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph';PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905'
OUT=PROJECT/'research/spo2_venous_pressure';OUT.mkdir(parents=True,exist_ok=True)
def vj(name,value):(OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\n')
def vs(stage,state='running'):
    vj('run_status.json',dict(stage=stage,status=state,utc=datetime.now(timezone.utc).isoformat(),environment='google_colab' if IN_COLAB else 'local',goal_complete=False))
def qs(value):return "'"+str(value).replace("'","''")+"'"
assert hashlib.sha256((PROJECT/'docs/SPO2_VENOUS_PRESSURE_PLAN.md').read_bytes()).hexdigest()==VENOUS_PRESSURE_PROTOCOL_SHA256
lock=OUT/'protocol_lock.json'
if lock.exists():assert json.loads(lock.read_text())['sha256']==VENOUS_PRESSURE_PROTOCOL_SHA256
else:vj('protocol_lock.json',dict(sha256=VENOUS_PRESSURE_PROTOCOL_SHA256,utc=datetime.now(timezone.utc).isoformat(),before_new_cvp_extraction=True,previous_project_results_known=True))
try:con.close()
except (NameError,AttributeError):pass
con=duckdb.connect(str(PRIVATE/'spo2_venous_pressure.duckdb'))
con.execute('SET threads=2');con.execute("SET memory_limit='2GB'")
con.execute(f"ATTACH {qs(PRIVATE/'spo2_context.duckdb')} AS prior (READ_ONLY)")
con.execute(f"ATTACH {qs(PRIVATE/'spo2_hemoglobin.duckdb')} AS elig (READ_ONLY)")
for d in ['eicu','mimic']:
    extra=',c.hadm_id,c.admit_time' if d=='mimic' else ''
    con.execute(f'''CREATE OR REPLACE TABLE {d}_cohort AS SELECT e.stay_id,e.exposed,
      cast(c.person_id AS VARCHAR) person_id{extra} FROM elig.{d}_eligible e
      JOIN prior.{d}_cohort c USING(stay_id)''')
    assert con.execute(f'SELECT count(*) FROM {d}_cohort').fetchone()[0]=={'eicu':11452,'mimic':4711}[d]
    con.execute(f'''CREATE OR REPLACE TABLE {d}_first_episode AS
      WITH paired AS (SELECT b.stay_id,b.bin,b.value,b.representative_minute AS b,
        lag(b.value) OVER w AS previous_value,lag(b.representative_minute) OVER w AS a
        FROM prior.{d}_bins b JOIN {d}_cohort c USING(stay_id) WHERE b.concept='spo2'
        WINDOW w AS (PARTITION BY b.stay_id ORDER BY b.bin))
      SELECT stay_id,a,b FROM paired WHERE abs(value-previous_value)>=4 AND b-a>0 AND b-a<=30
      QUALIFY row_number() OVER(PARTITION BY stay_id ORDER BY b)=1''')
    assert con.execute(f'''SELECT count(*) FROM {d}_cohort c LEFT JOIN {d}_first_episode e USING(stay_id)
      WHERE c.exposed<>cast(e.stay_id IS NOT NULL AS INTEGER)''').fetchone()[0]==0
sources={}
def raw(dataset,table):
    p=DRIVE/'Data'/dataset/'Full'/(table+'.csv');s=p.stat()
    sources[dataset+'/'+table]=dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    vj('input_manifest.json',sources)
    return f'read_csv({qs(p)},header=true,all_varchar=true,sample_size=10000,strict_mode=true)'
vs('CVP cohort and first-episode reconstruction complete')
"""

EXTRACT = r"""
vs('eICU CVP raw scan')
source=raw('eICU','vitalPeriodic')
con.execute(f'''CREATE OR REPLACE TABLE eicu_cvp_raw AS
  SELECT c.stay_id,try_cast(v.observationoffset AS DOUBLE) event_minute,
    try_cast(v.cvp AS DOUBLE) numeric_value
  FROM {source} v JOIN eicu_cohort c ON try_cast(v.patientunitstayid AS BIGINT)=c.stay_id
  WHERE try_cast(v.observationoffset AS DOUBLE)>=0 AND try_cast(v.observationoffset AS DOUBLE)<240
    AND v.cvp IS NOT NULL''')
con.execute('''CREATE OR REPLACE TABLE eicu_cvp_times AS SELECT DISTINCT stay_id,event_minute
  FROM eicu_cvp_raw WHERE isfinite(numeric_value)''')
vs('MIMIC CVP dictionary verification')
items=raw('MIMIC','d_items')
item=con.execute(f"SELECT itemid,label,unitname,linksto FROM {items} WHERE itemid='220074'").df()
assert len(item)==1 and item.iloc[0]['label']=='Central Venous Pressure'
assert item.iloc[0]['unitname']=='mmHg' and item.iloc[0]['linksto']=='chartevents'
vj('mimic_dictionary.json',item.to_dict('records'))
vs('MIMIC CVP raw scan')
source=raw('MIMIC','chartevents')
con.execute(f'''CREATE OR REPLACE TABLE mimic_cvp_raw AS
  SELECT c.stay_id,date_diff('second',c.admit_time,try_cast(v.charttime AS TIMESTAMP))/60.0 event_minute,
    date_diff('second',c.admit_time,try_cast(v.storetime AS TIMESTAMP))/60.0 available_minute,
    try_cast(v.valuenum AS DOUBLE) numeric_value,v.valueuom,
    coalesce(try_cast(v.warning AS DOUBLE),0) warning
  FROM {source} v JOIN mimic_cohort c ON try_cast(v.stay_id AS BIGINT)=c.stay_id
    AND try_cast(v.hadm_id AS BIGINT)=c.hadm_id AND v.subject_id=c.person_id
  WHERE try_cast(v.itemid AS BIGINT)=220074
    AND try_cast(v.charttime AS TIMESTAMP)>=c.admit_time
    AND try_cast(v.charttime AS TIMESTAMP)<c.admit_time+INTERVAL '240 minutes' ''')
con.execute('''CREATE OR REPLACE TABLE mimic_cvp_times AS SELECT DISTINCT stay_id,event_minute
  FROM mimic_cvp_raw WHERE warning<>1 AND available_minute<=240
    AND lower(trim(valueuom))='mmhg' AND isfinite(numeric_value) AND numeric_value BETWEEN -5 AND 50''')
for x in sources.values():
    s=Path(x['path']).stat();assert s.st_size==x['bytes'] and s.st_mtime_ns==x['mtime_ns']
vj('source_stability.json',dict(sizes_and_mtimes_unchanged=True,sources=sources))
vs('CVP raw extraction complete')
"""

SUPPORT = r"""
vs('independent CVP timing support validation')
support=[];validation=[];source_audit=[]
for d in ['eicu','mimic']:
    raw_count=con.execute(f'SELECT count(*) FROM {d}_cvp_raw').fetchone()[0]
    times=con.execute(f'SELECT stay_id,event_minute FROM {d}_cvp_times ORDER BY stay_id,event_minute').df()
    assert times.event_minute.ge(0).all() and times.event_minute.lt(240).all()
    source_audit.append(dict(dataset=d,raw_nonnull_source_rows=raw_count,accepted_unique_times=len(times),
      encounters_with_times=int(times.stay_id.nunique()),units='explicit mmHg' if d=='mimic' else 'unresolved native monitor unit; timing upper bound'))
    con.execute(f'''CREATE OR REPLACE TABLE {d}_timing_support AS
      WITH counts AS (SELECT e.stay_id,e.a,e.b,
        count(t.event_minute) FILTER(WHERE t.event_minute>=e.a-60 AND t.event_minute<e.a) n_pre,
        count(t.event_minute) FILTER(WHERE t.event_minute>e.b AND t.event_minute<=e.b+60) n_post,
        min(t.event_minute) FILTER(WHERE t.event_minute>=e.a-60 AND t.event_minute<e.a) pre_first,
        max(t.event_minute) FILTER(WHERE t.event_minute>=e.a-60 AND t.event_minute<e.a) pre_last,
        min(t.event_minute) FILTER(WHERE t.event_minute>e.b AND t.event_minute<=e.b+60) post_first,
        max(t.event_minute) FILTER(WHERE t.event_minute>e.b AND t.event_minute<=e.b+60) post_last
        FROM {d}_first_episode e LEFT JOIN {d}_cvp_times t USING(stay_id) GROUP BY e.stay_id,e.a,e.b)
      SELECT *,a>=60 AND b<180 AS full_window,
        coalesce(a>=60 AND b<180 AND n_pre>=2 AND n_post>=2
        AND pre_last-pre_first>=15 AND post_last-post_first>=15
        AND a-pre_last<=30 AND post_first-b<=30,false) complete_timing FROM counts''')
    sql=con.execute(f'SELECT * FROM {d}_timing_support ORDER BY stay_id').df()
    bins=con.execute(f"SELECT b.stay_id,b.bin,b.value,b.representative_minute FROM prior.{d}_bins b JOIN {d}_cohort c USING(stay_id) WHERE concept='spo2' ORDER BY stay_id,bin").df()
    bins['a']=bins.groupby('stay_id').representative_minute.shift()
    dv=bins.value-bins.groupby('stay_id').value.shift();dt=bins.representative_minute-bins.a
    first=bins.loc[dv.abs().ge(4)&dt.gt(0)&dt.le(30)].groupby('stay_id',sort=False).head(1)
    assert np.array_equal(first.stay_id.to_numpy(),sql.stay_id.to_numpy())
    assert np.allclose(first.a,sql.a,rtol=0,atol=1e-12) and np.allclose(first.representative_minute,sql.b,rtol=0,atol=1e-12)
    bystay={k:v.event_minute.to_numpy() for k,v in times.groupby('stay_id')};python_complete=[]
    for r in first.itertuples():
        a=r.a;b=r.representative_minute;t=bystay.get(r.stay_id,np.array([]))
        before=t[(t>=a-60)&(t<a)];after=t[(t>b)&(t<=b+60)]
        ok=a>=60 and b<180 and len(before)>=2 and len(after)>=2
        if ok:ok=before[-1]-before[0]>=15 and after[-1]-after[0]>=15 and a-before[-1]<=30 and after[0]-b<=30
        if ok:python_complete.append(r.stay_id)
    assert set(python_complete)==set(sql.loc[sql.complete_timing,'stay_id'])
    joined=sql.merge(con.execute(f'SELECT stay_id,person_id FROM {d}_cohort').df(),on='stay_id',validate='one_to_one')
    stages={'first_original_episode':np.ones(len(joined),dtype=bool),'full_windows':joined.full_window.to_numpy()}
    stages['any_both_windows']=stages['full_windows']&joined.n_pre.ge(1)&joined.n_post.ge(1)
    stages['two_both_windows']=stages['any_both_windows']&joined.n_pre.ge(2)&joined.n_post.ge(2)
    stages['span_at_least_15']=stages['two_both_windows']&(joined.pre_last-joined.pre_first).ge(15)&(joined.post_last-joined.post_first).ge(15)
    stages['complete_timing']=joined.complete_timing
    for stage,mask in stages.items():
        subset=joined.loc[mask];support.append(dict(dataset=d,stage=stage,encounters=len(subset),people=int(subset.person_id.nunique())))
    validation.append(dict(dataset=d,first_episode_presence_matches_original_exposure=True,python_first_episode_times_match=True,python_complete_patient_set_matches=True))
    # Private evidence contains timing only, with no pressure trajectory or outcome labels.
    joined.to_csv(PRIVATE/(d+'_venous_pressure_timing.csv'),index=False)
gate=all(x['people']>=100 for x in support if x['stage']=='complete_timing')
manifest=dict(protocol_sha256=VENOUS_PRESSURE_PROTOCOL_SHA256,
    module_sha256=hashlib.sha256((PROJECT/'scripts/venous_pressure_cells.py').read_bytes()).hexdigest(),
    timing_count_floor_passed=gate,comparable_units_established=False,biological_analysis_authorized_by_gates=False,
    pressure_changes_computed=False,clinical_outcomes_inspected=False,goal_complete=False)
vj('timing_support.json',support);vj('validation.json',validation);vj('source_audit.json',source_audit);vj('analysis_manifest.json',manifest)
lines=['# Venous-pressure timing feasibility: actual local results' if not IN_COLAB else '# Venous-pressure timing feasibility: Google Colab results','',
    '| Database | Stage | Encounters | People |','|---|---|---:|---:|']
for r in support:lines.append(f"| {r['dataset']} | {r['stage']} | {r['encounters']} | {r['people']} |")
lines+=['',f'Timing count floor passed: **{gate}**. Comparable pressure units established: **False**.',
    '', 'eICU counts are optimistic timing support from finite native CVP values, without physiological range validation. MIMIC uses explicit mmHg records passing the frozen quality criteria. Neither the size of a timing sample nor its presence proves a biological mechanism.',
    '', 'Independent Python reconstruction agrees with SQL on every first-episode identity/time and every complete-timing encounter. First-episode presence reconciles to the original SpO2 exposure in both databases. No later episode substitution was used.',
    '', 'No CVP change, MAP response, biomarker association or mortality effect was calculated. The gate result governs this specification and must not be represented as absence of venous-congestion biology. No discovery or mortality benefit is established.']
report='\n'.join(lines)+'\n';(PROJECT/'docs/SPO2_VENOUS_PRESSURE_RESULTS.md').write_text(report)
con.close();vs('CVP timing feasibility and independent validation','complete');print(report)
"""
