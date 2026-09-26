"""Timing-only arterial oxygen support at the original SpO2 pairs."""

ANALYSIS = r"""
import json,hashlib,importlib.util
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import duckdb
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph';PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905'
OUT=PROJECT/'research/spo2_arterial_pairing';OUT.mkdir(parents=True,exist_ok=True)
def ap_json(name,value):(OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\n')
def ap_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def ap_q(value):return "'"+str(value).replace("'","''")+"'"
assert ap_hash(PROJECT/'docs/SPO2_ARTERIAL_PAIRING_PLAN.md')==ARTERIAL_PAIRING_PROTOCOL_SHA256
lock=OUT/'protocol_lock.json'
if lock.exists():assert json.loads(lock.read_text())['sha256']==ARTERIAL_PAIRING_PROTOCOL_SHA256
else:ap_json('protocol_lock.json',dict(sha256=ARTERIAL_PAIRING_PROTOCOL_SHA256,utc=datetime.now(timezone.utc).isoformat(),before_pairing_counts=True,earlier_project_results_known=True))
try:con.close()
except (NameError,AttributeError):pass
con=duckdb.connect(':memory:');con.execute('SET threads=2')
inputs={}
for alias,name in [('prior','spo2_context'),('gas','spo2_acid_base')]:
    path=PRIVATE/(name+'.duckdb');s=path.stat()
    inputs[name]=dict(path=str(path),bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=ap_hash(path))
    con.execute(f'ATTACH {ap_q(path)} AS {alias} (READ_ONLY)')
ap_json('run_status.json',dict(status='running',stage='timing-only pairing',environment='google_colab' if IN_COLAB else 'local',goal_complete=False))
support=[];validation=[];primary=[]
for d in ['eicu','mimic']:
    cohort=con.execute(f'''SELECT e.stay_id,e.exposed,e.hospital_start_minute,
      cast(c.person_id AS VARCHAR) person_id FROM gas.{d}_eligible e
      JOIN prior.{d}_cohort c USING(stay_id)''').df()
    assert len(cohort)=={'eicu':11452,'mimic':4711}[d]
    con.register('ap_cohort',cohort)
    con.execute(f'''CREATE OR REPLACE TEMP TABLE ap_first AS
      WITH paired AS (SELECT b.stay_id,b.bin,b.value,b.representative_minute AS b,
        lag(b.value) OVER w previous_value,lag(b.representative_minute) OVER w a
        FROM prior.{d}_bins b JOIN ap_cohort c USING(stay_id) WHERE b.concept='spo2'
        WINDOW w AS (PARTITION BY b.stay_id ORDER BY b.bin))
      SELECT stay_id,a,b FROM paired WHERE abs(value-previous_value)>=4 AND b-a>0 AND b-a<=30
      QUALIFY row_number() OVER(PARTITION BY stay_id ORDER BY b)=1''')
    first=con.execute('SELECT * FROM ap_first ORDER BY stay_id').df()
    assert set(first.stay_id)==set(cohort.loc[cohort.exposed.eq(1),'stay_id'])
    bins=con.execute(f'''SELECT b.stay_id,b.bin,b.value,b.representative_minute FROM prior.{d}_bins b
      JOIN ap_cohort c USING(stay_id) WHERE concept='spo2' ORDER BY stay_id,bin''').df()
    bins['a']=bins.groupby('stay_id').representative_minute.shift()
    dv=bins.value-bins.groupby('stay_id').value.shift();dt=bins.representative_minute-bins.a
    pyfirst=bins.loc[dv.abs().ge(4)&dt.gt(0)&dt.le(30)].groupby('stay_id',sort=False).head(1)
    assert np.array_equal(first.stay_id,pyfirst.stay_id)
    assert np.allclose(first.a,pyfirst.a,rtol=0,atol=1e-12)
    assert np.allclose(first.b,pyfirst.representative_minute,rtol=0,atol=1e-12)
    selections={
      'qualified_pao2':f"SELECT DISTINCT stay_id,event_minute FROM gas.{d}_components WHERE concept='po2'",
      'raw_pao2_timing_ceiling':f"SELECT DISTINCT stay_id,event_minute FROM gas.{d}_raw WHERE "+("raw_name='paO2'" if d=='eicu' else 'itemid=50821')}
    for source,query in selections.items():
        times=con.execute(query).df();con.register('ap_times',times)
        groups={k:g.event_minute.to_numpy() for k,g in times.groupby('stay_id')}
        for w in [5,10,15]:
            pairs=con.execute(f'''SELECT e.stay_id,c.person_id,e.a,e.b,
              e.a-{w}>=greatest(-1440,c.hospital_start_minute) AND e.b+{w}<=240 full_window,
              count(t.event_minute) FILTER(WHERE abs(t.event_minute-e.a)<={w} AND t.event_minute<(e.a+e.b)/2)>0 pre,
              count(t.event_minute) FILTER(WHERE abs(t.event_minute-e.b)<={w} AND t.event_minute>=(e.a+e.b)/2)>0 post
              FROM ap_first e JOIN ap_cohort c USING(stay_id) LEFT JOIN ap_times t USING(stay_id)
              GROUP BY e.stay_id,c.person_id,c.hospital_start_minute,e.a,e.b ORDER BY e.stay_id''').df()
            checks=[]
            hstart=cohort.set_index('stay_id').hospital_start_minute.to_dict()
            for row in pyfirst.itertuples():
                a=row.a;b=row.representative_minute;t=groups.get(row.stay_id,np.array([]));mid=(a+b)/2
                checks.append((a-w>=max(-1440,hstart[row.stay_id]) and b+w<=240,
                  bool(((abs(t-a)<=w)&(t<mid)).any()),bool(((abs(t-b)<=w)&(t>=mid)).any())))
            assert np.array_equal(pairs[['full_window','pre','post']].to_numpy(),np.array(checks))
            for stage,mask in [('original_episode',np.ones(len(pairs),dtype=bool)),('full_windows',pairs.full_window),
              ('pre_available',pairs.full_window&pairs.pre),('post_available',pairs.full_window&pairs.post),
              ('both_available',pairs.full_window&pairs.pre&pairs.post)]:
                selected=pairs.loc[mask]
                support.append(dict(dataset=d,source=source,tolerance_minutes=w,stage=stage,
                  encounters=len(selected),people=int(selected.person_id.nunique())))
                if source=='qualified_pao2' and w==5 and stage=='both_available':primary.append(int(selected.person_id.nunique()))
            validation.append(dict(dataset=d,source=source,tolerance_minutes=w,exact_sql_python_flags_and_sets=True))
        con.unregister('ap_times')
    con.unregister('ap_cohort')
for info in inputs.values():
    path=Path(info['path']);s=path.stat()
    assert s.st_size==info['bytes'] and s.st_mtime_ns==info['mtime_ns'] and ap_hash(path)==info['sha256']
assert len(primary)==2
gate=all(n>=385 for n in primary)
ap_json('support.json',support)
ap_json('validation.json',dict(checks=validation,original_exposure_sets_match=True,independent_episode_times_match=True,source_hashes_unchanged=True))
ap_json('manifest.json',dict(inputs=inputs,protocol_sha256=ARTERIAL_PAIRING_PROTOCOL_SHA256,
  environment='google_colab' if IN_COLAB else 'local',primary_gate_passed=gate,
  oxygen_values_read=False,mortality_read=False,association_models=0,goal_complete=False))
report=['# Arterial oxygen pairing feasibility','',
  'Actual execution: '+('Google Colab' if IN_COLAB else 'local')+'. Existing caches; no new raw source scan.',
  '', '| Source | Timing tier | Tolerance (min) | Full-window encounters | Both-side encounters | Both-side people |',
  '|---|---|---:|---:|---:|---:|']
for d in ['eicu','mimic']:
    for source in selections:
        for w in [5,10,15]:
            rows={r['stage']:r for r in support if r['dataset']==d and r['source']==source and r['tolerance_minutes']==w}
            report.append(f"| {d} | {source} | {w} | {rows['full_windows']['encounters']} | {rows['both_available']['encounters']} | {rows['both_available']['people']} |")
report+=['', 'Primary requirement: 385 people with paired qualified PaO2 at +/-5 minutes in each database. Gate: '+('PASS' if gate else 'FAIL')+'.',
  '', 'SQL and independent Python reconstruction agree on original episode times, exposed patient sets, and all timing flags. Source SHA-256 hashes were unchanged.',
  '', 'The raw timing ceiling includes unqualified values and nonarterial or ambiguous specimens. It is only a ceiling within the existing hospital-linked extraction. Qualified samples retain prior specimen, unit, range and revision checks; later-available results and samples beyond minute 240 are absent. eICU uses ABG labels without equivalent specimen identifiers. PaO2 does not directly measure SaO2.',
  '', 'No oxygen change, mortality association, mechanism or treatment benefit was estimated. '+('A separate analysis specification is required before values are examined.' if gate else 'This cached-pairing specification is closed; sensitivity windows do not rescue it. Sparse pairing cannot establish that the SpO2 change is artifact or that true arterial oxygen changes are absent.'),'']
(PROJECT/'docs/SPO2_ARTERIAL_PAIRING_RESULTS.md').write_text('\n'.join(report))
ap_json('run_status.json',dict(status='complete',stage='timing-only pairing and independent validation',
  utc=datetime.now(timezone.utc).isoformat(),environment='google_colab' if IN_COLAB else 'local',goal_complete=False))
con.close();print('\n'.join(report))
"""
