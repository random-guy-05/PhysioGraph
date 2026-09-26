"""Outcome-blind raw laboratory timing census for original SpO2 episodes."""

SETUP = r"""
import json,hashlib,importlib.util
from pathlib import Path
from datetime import datetime,timezone
import duckdb,numpy as np,pandas as pd
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph';PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905'
OUT=PROJECT/'research/spo2_arterial_raw_recovery';OUT.mkdir(parents=True,exist_ok=True)
def ar_json(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def ar_q(x):return "'"+str(x).replace("'","''")+"'"
def ar_hash(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
assert ar_hash(PROJECT/'docs/SPO2_ARTERIAL_RAW_RECOVERY_PLAN.md')==ARTERIAL_RAW_PROTOCOL_SHA256
lock=OUT/'protocol_lock.json'
if lock.exists():assert json.loads(lock.read_text())['sha256']==ARTERIAL_RAW_PROTOCOL_SHA256
else:ar_json('protocol_lock.json',dict(sha256=ARTERIAL_RAW_PROTOCOL_SHA256,utc=datetime.now(timezone.utc).isoformat(),before_raw_counts=True,earlier_results_known=True))
try:con.close()
except (NameError,AttributeError):pass
con=duckdb.connect(str(PRIVATE/'arterial_raw_recovery.duckdb'))
con.execute('SET threads=2');con.execute("SET memory_limit='2GB'")
con.execute(f'ATTACH {ar_q(PRIVATE/"spo2_context.duckdb")} AS prior (READ_ONLY)')
con.execute(f'ATTACH {ar_q(PRIVATE/"spo2_acid_base.duckdb")} AS gas (READ_ONLY)')
sources={}
for dataset,name in [('eICU','lab'),('MIMIC','labevents'),('MIMIC','admissions'),('MIMIC','d_labitems')]:
    p=DRIVE/'Data'/dataset/'Full'/(name+'.csv');s=p.stat()
    sources[dataset+'/'+name]=dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
for d in ['eicu','mimic']:
    con.execute(f'''CREATE OR REPLACE TABLE {d}_eligible AS SELECT e.*,cast(c.person_id AS VARCHAR) canonical_person
      FROM gas.{d}_eligible e JOIN prior.{d}_cohort c USING(stay_id)''')
    assert con.execute(f'SELECT count(*) FROM {d}_eligible').fetchone()[0]=={'eicu':11452,'mimic':4711}[d]
    con.execute(f'''CREATE OR REPLACE TABLE {d}_first AS WITH p AS (
      SELECT b.stay_id,b.bin,b.value,b.representative_minute b,
       lag(value) OVER w previous_value,lag(representative_minute) OVER w a
      FROM prior.{d}_bins b JOIN {d}_eligible e USING(stay_id) WHERE concept='spo2'
      WINDOW w AS (PARTITION BY b.stay_id ORDER BY bin))
      SELECT stay_id,a,b FROM p WHERE abs(value-previous_value)>=4 AND b-a>0 AND b-a<=30
      QUALIFY row_number() OVER(PARTITION BY stay_id ORDER BY b)=1''')
ar_json('run_status.json',dict(status='running',stage='protocol locked; raw scan pending',environment='google_colab' if IN_COLAB else 'local'))
print('Locked raw timing recovery; no oxygen values or outcomes selected.')
"""

EXTRACT = r"""
def ar_csv(key):return f"read_csv({ar_q(sources[key]['path'])},header=true,all_varchar=true,sample_size=10000,strict_mode=true)"
con.execute(f'''CREATE OR REPLACE TABLE eicu_raw AS
 SELECT e.stay_id,try_cast(l.labid AS BIGINT) lab_id,try_cast(l.labresultoffset AS DOUBLE) event_minute,
 try_cast(l.labresultrevisedoffset AS DOUBLE) available_minute,l.labname raw_name,
 try_cast(l.labtypeid AS INTEGER) lab_type,'explicit' linkage
 FROM {ar_csv('eICU/lab')} l JOIN eicu_eligible e ON try_cast(l.patientunitstayid AS BIGINT)=e.stay_id
 WHERE lower(trim(l.labname)) IN ('pao2','po2') AND try_cast(l.labresultoffset AS DOUBLE) BETWEEN -15 AND 255''')
ar_json('run_status.json',dict(status='running',stage='eICU complete; MIMIC raw scan'))
con.execute(f'''CREATE OR REPLACE TABLE admissions AS SELECT try_cast(subject_id AS BIGINT) person_id,
 try_cast(hadm_id AS BIGINT) hadm_id,try_cast(admittime AS TIMESTAMP) start_time,
 try_cast(dischtime AS TIMESTAMP) end_time FROM {ar_csv('MIMIC/admissions')}''')
con.execute(f'''CREATE OR REPLACE TABLE mimic_candidates AS
 SELECT try_cast(l.labevent_id AS BIGINT) lab_id,try_cast(l.specimen_id AS BIGINT) specimen_id,
 try_cast(l.subject_id AS BIGINT) person_id,try_cast(l.hadm_id AS BIGINT) raw_hadm_id,
 try_cast(l.charttime AS TIMESTAMP) charttime,try_cast(l.storetime AS TIMESTAMP) storetime,
 try_cast(l.itemid AS INTEGER) itemid
 FROM {ar_csv('MIMIC/labevents')} l
 WHERE try_cast(l.itemid AS INTEGER) IN (50821,52042)
 AND EXISTS(SELECT 1 FROM mimic_eligible e WHERE try_cast(l.subject_id AS BIGINT)=e.person_id
  AND date_diff('second',e.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0 BETWEEN -15 AND 255)''')
con.execute('''CREATE OR REPLACE TABLE missing_hospital_candidates AS SELECT l.lab_id,count(a.hadm_id) matches,
 min(a.hadm_id) assigned_hadm_id FROM mimic_candidates l LEFT JOIN admissions a ON l.person_id=a.person_id
 AND l.charttime>=a.start_time AND l.charttime<a.end_time WHERE l.raw_hadm_id IS NULL GROUP BY l.lab_id''')
con.execute('''CREATE OR REPLACE TABLE mimic_raw AS SELECT e.stay_id,l.lab_id,l.specimen_id,l.itemid,
 date_diff('second',e.admit_time,l.charttime)/60.0 event_minute,
 date_diff('second',e.admit_time,l.storetime)/60.0 available_minute,
 CASE WHEN l.raw_hadm_id IS NOT NULL THEN 'explicit' ELSE 'unique_missing_hospital' END linkage
 FROM mimic_candidates l LEFT JOIN missing_hospital_candidates m USING(lab_id)
 JOIN mimic_eligible e ON l.person_id=e.person_id AND e.hadm_id=
 CASE WHEN l.raw_hadm_id IS NOT NULL THEN l.raw_hadm_id WHEN m.matches=1 THEN m.assigned_hadm_id END
 WHERE date_diff('second',e.admit_time,l.charttime)/60.0 BETWEEN -15 AND 255''')
assert con.execute('SELECT count(*)-count(DISTINCT lab_id) FROM mimic_candidates').fetchone()[0]==0
for key,info in sources.items():
    p=Path(info['path']);s=p.stat();assert (s.st_size,s.st_mtime_ns)==(info['bytes'],info['mtime_ns'])
    info['sha256']=ar_hash(p)
ar_json('source_manifest.json',dict(sources=sources,full_raw_lab_scans=True,numeric_oxygen_values_selected=False,outcomes_selected=False))
ar_json('run_status.json',dict(status='running',stage='raw scans complete; independent timing validation'))
print('Both full raw lab scans completed; evaluating timing metadata only.')
"""

SUPPORT = r"""
support=[];validation=[];recovery=[]
for d in ['eicu','mimic']:
    e=con.execute(f'SELECT stay_id,exposed,hospital_start_minute,canonical_person FROM {d}_eligible').df()
    first=con.execute(f'SELECT * FROM {d}_first ORDER BY stay_id').df()
    assert set(first.stay_id)==set(e.loc[e.exposed.eq(1),'stay_id'])
    bins=con.execute(f"SELECT b.stay_id,bin,value,representative_minute FROM prior.{d}_bins b JOIN {d}_eligible e USING(stay_id) WHERE concept='spo2' ORDER BY stay_id,bin").df()
    bins['a']=bins.groupby('stay_id').representative_minute.shift()
    dv=bins.value-bins.groupby('stay_id').value.shift();dt=bins.representative_minute-bins.a
    pyfirst=bins.loc[dv.abs().ge(4)&dt.gt(0)&dt.le(30)].groupby('stay_id',sort=False).head(1)
    assert np.array_equal(first.stay_id,pyfirst.stay_id)
    assert np.allclose(first[['a','b']],pyfirst[['a','representative_minute']],rtol=0,atol=1e-12)
    legacy_filter="raw_name='paO2'" if d=='eicu' else 'itemid=50821'
    old=con.execute(f'SELECT stay_id,lab_id,event_minute FROM gas.{d}_raw WHERE {legacy_filter} AND event_minute BETWEEN -15 AND 240').df()
    fresh=con.execute(f'SELECT * FROM {d}_raw').df()
    check=old.merge(fresh[['stay_id','lab_id','event_minute']],on=['stay_id','lab_id'],how='left',suffixes=('_old','_new'),validate='one_to_one')
    assert check.event_minute_new.notna().all() and np.allclose(check.event_minute_old,check.event_minute_new,rtol=0,atol=1e-12)
    oldkeys=set(zip(old.stay_id,old.lab_id));common=fresh.loc[fresh.event_minute.le(240)]
    extra=common.loc[[key not in oldkeys for key in zip(common.stay_id,common.lab_id)]]
    recovery.append(dict(dataset=d,legacy_common_rows=len(old),fresh_rows=len(fresh),additional_common_rows=len(extra),additional_common_encounters=int(extra.stay_id.nunique())))
    tiers={'canonical_explicit':"raw_name='paO2'" if d=='eicu' else "itemid=50821 AND linkage='explicit'",
           'all_recovered_pao2_labels':'true'}
    for tier,where in tiers.items():
        times=con.execute(f'SELECT DISTINCT stay_id,event_minute FROM {d}_raw WHERE {where}').df()
        con.register('ar_times',times);groups={k:g.event_minute.to_numpy() for k,g in times.groupby('stay_id')}
        for w in [5,10,15]:
            paired=con.execute(f'''SELECT f.stay_id,e.canonical_person,f.a,f.b,
             f.a-{w}>=greatest(-1440,e.hospital_start_minute) AND f.b+{w}<=240 full_window,
             count(t.event_minute) FILTER(WHERE abs(t.event_minute-f.a)<={w} AND t.event_minute<(f.a+f.b)/2)>0 pre,
             count(t.event_minute) FILTER(WHERE abs(t.event_minute-f.b)<={w} AND t.event_minute>=(f.a+f.b)/2)>0 post
             FROM {d}_first f JOIN {d}_eligible e USING(stay_id) LEFT JOIN ar_times t USING(stay_id)
             GROUP BY f.stay_id,e.canonical_person,e.hospital_start_minute,f.a,f.b ORDER BY f.stay_id''').df()
            hs=e.set_index('stay_id').hospital_start_minute.to_dict();flags=[]
            for r in first.itertuples():
                t=groups.get(r.stay_id,np.array([]));mid=(r.a+r.b)/2
                flags.append([r.a-w>=max(-1440,hs[r.stay_id]) and r.b+w<=240,
                    bool(np.any((np.abs(t-r.a)<=w)&(t<mid))),bool(np.any((np.abs(t-r.b)<=w)&(t>=mid)))])
            assert np.array_equal(paired[['full_window','pre','post']].to_numpy(),np.array(flags))
            for stage,mask in [('full_windows',paired.full_window),('pre',paired.full_window&paired.pre),('post',paired.full_window&paired.post),('both',paired.full_window&paired.pre&paired.post)]:
                chosen=paired.loc[mask]
                support.append(dict(dataset=d,tier=tier,tolerance_minutes=w,stage=stage,encounters=len(chosen),people=int(chosen.canonical_person.nunique())))
        con.unregister('ar_times')
    validation.append(dict(dataset=d,first_episode_sql_python_agree=True,all_legacy_rows_reconstruct=True,all_timing_flags_sql_python_agree=True))
missing=con.execute('''SELECT l.lab_id,l.person_id,l.charttime,m.matches,m.assigned_hadm_id FROM mimic_candidates l
 JOIN missing_hospital_candidates m USING(lab_id)''').df()
admissions=con.execute('SELECT * FROM admissions WHERE person_id IN (SELECT person_id FROM mimic_candidates WHERE raw_hadm_id IS NULL)').df()
ag={k:g for k,g in admissions.groupby('person_id')}
for r in missing.itertuples():
    g=ag.get(r.person_id,pd.DataFrame(columns=['start_time','end_time','hadm_id']))
    hits=g.loc[(g.start_time<=r.charttime)&(g.end_time>r.charttime)]
    assert len(hits)==r.matches
    if r.matches:assert int(hits.hadm_id.min())==r.assigned_hadm_id
primary=[r for r in support if r['tier']=='all_recovered_pao2_labels' and r['tolerance_minutes']==5 and r['stage']=='both']
assert len(primary)==2
ar_json('support.json',support);ar_json('recovery.json',recovery)
ar_json('missing_hospital_support.json',con.execute('SELECT matches,count(*) records FROM missing_hospital_candidates GROUP BY matches ORDER BY matches').df().to_dict('records'))
ar_json('validation.json',dict(checks=validation,missing_hospital_assignments_independently_validated=len(missing)))
ar_json('results.json',dict(primary_timing_ceiling=primary,optimistic_ceiling_reaches_prior_gate=all(r['people']>=385 for r in primary),arterial_qualification_performed=False,numeric_oxygen_values_selected=False,outcomes_selected=False,biological_discovery=False))
report=['# Full raw laboratory timing recovery','', 'Actual execution: '+('Google Colab' if IN_COLAB else 'local')+'. Full supplied lab scans; metadata only.','', '| Dataset | Tier | Tolerance | Full-window encounters | Both-side people |','|---|---|---:|---:|---:|']
for d in ['eicu','mimic']:
    for tier in tiers:
        for w in [5,10,15]:
            rows={r['stage']:r for r in support if r['dataset']==d and r['tier']==tier and r['tolerance_minutes']==w}
            report.append(f"| {d} | {tier} | {w} | {rows['full_windows']['encounters']} | {rows['both']['people']} |")
report+=['','Legacy common-window rows reconstruct exactly; independent Python agrees with SQL on original episodes, all timing flags, and missing-hospital admission assignments.',
 'These are permissive laboratory timestamp ceilings. Extra fluid-pO2 labels are not assumed arterial; values and mortality were not selected. This does not census charted gases or all physiological measurements.','']
(PROJECT/'docs/SPO2_ARTERIAL_RAW_RECOVERY_RESULTS.md').write_text('\n'.join(report))
ar_json('run_status.json',dict(status='complete',stage='raw recovery and independent validation',utc=datetime.now(timezone.utc).isoformat(),goal_complete=False))
con.close();print('\n'.join(report));print('Raw row recovery:',recovery)
"""
