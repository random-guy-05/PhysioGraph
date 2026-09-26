"""Outcome-blind oxygen-transition feasibility, embedded in the primary notebook."""

FEASIBILITY = r"""
import json,hashlib,importlib.util
from pathlib import Path
from datetime import datetime,timezone
import duckdb
import pandas as pd
import numpy as np
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph';PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905'
OUT=PROJECT/'research/spo2_oxygen_response';OUT.mkdir(exist_ok=True)
def oj(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def qs(v):return "'"+str(v).replace("'","''")+"'"
lock=OUT/'protocol_lock.json'
if lock.exists():assert json.loads(lock.read_text())['sha256']==OXYGEN_RESPONSE_PROTOCOL_SHA256
else:oj('protocol_lock.json',dict(sha256=OXYGEN_RESPONSE_PROTOCOL_SHA256,
    utc=datetime.now(timezone.utc).isoformat(),outcome_and_response_magnitudes_not_examined=True))
oj('run_status.json',dict(status='running',stage='oxygen-transition timing feasibility',environment='google_colab' if IN_COLAB else 'local'))
fingerprints={}
for name in ['hemoglobin_interaction','spo2_context']:
    p=PRIVATE/(name+'.duckdb');s=p.stat();h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    fingerprints[name]=dict(path=str(p),sha256=h.hexdigest(),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
oj('input_database_fingerprints.json',fingerprints)
con=duckdb.connect(str(PRIVATE/'oxygen_response.duckdb'))
con.execute('SET threads=2');con.execute("SET memory_limit='2GB'")
for alias,name in [('care','hemoglobin_interaction'),('prior','spo2_context')]:
    con.execute(f"ATTACH {qs(PRIVATE/(name+'.duckdb'))} AS {alias} (READ_ONLY)")
support=[];validation=[]
for d in ['eicu','mimic']:
    con.execute(f'''CREATE OR REPLACE TABLE {d}_settings AS
      SELECT stay_id,event_minute,median(fio2_percent) fio2
      FROM care.{d}_fio2 WHERE fio2_percent BETWEEN 21 AND 100
        AND event_minute>=0 AND event_minute<240 GROUP BY stay_id,event_minute''')
    con.execute(f'''CREATE OR REPLACE TABLE {d}_escalations AS
      WITH ordered AS (SELECT *,lag(event_minute) OVER w previous_minute,lag(fio2) OVER w previous_fio2
        FROM {d}_settings WINDOW w AS(PARTITION BY stay_id ORDER BY event_minute))
      SELECT *,row_number() OVER(PARTITION BY stay_id ORDER BY event_minute) escalation_index
      FROM ordered WHERE event_minute>previous_minute AND event_minute-previous_minute<=60
        AND fio2-previous_fio2>=10''')
    con.execute(f'''CREATE OR REPLACE TABLE {d}_timing_support AS
      SELECT e.*,f.spo2_jump_any exposed,p.pre_spo2_minute,q.post_spo2_minute,
        p.pre_spo2_minute IS NOT NULL AND q.post_spo2_minute IS NOT NULL paired_spo2,
        EXISTS(SELECT 1 FROM {d}_settings s WHERE s.stay_id=e.stay_id
          AND s.event_minute>=q.post_spo2_minute AND s.event_minute<=e.event_minute+30
          AND abs(s.fio2-e.fio2)<=5) setting_corroborated
      FROM {d}_escalations e JOIN prior.{d}_features f USING(stay_id)
      LEFT JOIN LATERAL (SELECT representative_minute pre_spo2_minute FROM prior.{d}_bins b
        WHERE b.stay_id=e.stay_id AND b.concept='spo2'
          AND b.representative_minute>=e.event_minute-30 AND b.representative_minute<e.event_minute
        ORDER BY b.representative_minute DESC LIMIT 1) p ON TRUE
      LEFT JOIN LATERAL (SELECT representative_minute post_spo2_minute FROM prior.{d}_bins b
        WHERE b.stay_id=e.stay_id AND b.concept='spo2'
          AND b.representative_minute>=e.event_minute+5 AND b.representative_minute<=e.event_minute+30
        ORDER BY b.representative_minute LIMIT 1) q ON TRUE''')
    t=con.execute(f'SELECT * FROM {d}_timing_support ORDER BY stay_id,event_minute').df()
    settings=con.execute(f'SELECT * FROM {d}_settings ORDER BY stay_id,event_minute').df()
    # Independent pandas/NumPy reconstruction of every escalation and timed pair.
    g=settings.groupby('stay_id',sort=False)
    previous_time=g.event_minute.shift();previous_value=g.fio2.shift()
    candidate=settings.loc[(settings.event_minute-previous_time).between(0,60,inclusive='right')&
        ((settings.fio2-previous_value)>=10)].copy()
    assert list(map(tuple,candidate[['stay_id','event_minute']].to_numpy()))==list(map(tuple,t[['stay_id','event_minute']].to_numpy()))
    sp=con.execute(f"SELECT stay_id,representative_minute FROM prior.{d}_bins WHERE concept='spo2' ORDER BY stay_id,representative_minute").df()
    times={int(k):v.representative_minute.to_numpy() for k,v in sp.groupby('stay_id')}
    settings_by_stay={int(k):v for k,v in settings.groupby('stay_id')}
    reconstructed=[]
    for row in t.itertuples(index=False):
        ts=times[int(row.stay_id)];before=ts[(ts>=row.event_minute-30)&(ts<row.event_minute)]
        after=ts[(ts>=row.event_minute+5)&(ts<=row.event_minute+30)]
        pre=float(before[-1]) if len(before) else np.nan;post=float(after[0]) if len(after) else np.nan
        s=settings_by_stay[int(row.stay_id)]
        corroborated=bool(((s.event_minute>=post)&(s.event_minute<=row.event_minute+30)&
            ((s.fio2-row.fio2).abs()<=5)).any()) if np.isfinite(post) else False
        reconstructed.append((pre,post,np.isfinite(pre) and np.isfinite(post),corroborated))
    if len(t):
        reconstructed=np.asarray(reconstructed,float)
        assert np.allclose(reconstructed[:,:2],t[['pre_spo2_minute','post_spo2_minute']],equal_nan=True)
        assert np.array_equal(reconstructed[:,2:].astype(bool),t[['paired_spo2','setting_corroborated']].to_numpy())
    first=t.loc[t.escalation_index==1]
    full=first.loc[first.paired_spo2&first.setting_corroborated]
    groups={str(x):int((full.exposed==x).sum()) for x in [0,1]}
    support.append(dict(dataset=d,dynamics_eligible=con.execute(f'SELECT count(*) FROM care.{d}_care').fetchone()[0],
        fio2_observed_encounters=int(settings.stay_id.nunique()),fio2_records=len(settings),
        escalation_records=len(t),escalation_encounters=int(t.stay_id.nunique()),
        paired_escalation_records=int(t.paired_spo2.sum()),first_escalation_paired=int(first.paired_spo2.sum()),
        first_escalation_paired_and_corroborated=len(full),full_support_by_original_exposure=groups,
        feasibility_pass=len(full)>=200 and min(groups.values())>=50))
    validation.append(dict(dataset=d,all_escalation_identifiers_match=True,
        all_pre_post_timestamps_match=True,all_corroboration_flags_match=True,records_checked=len(t)))
oj('timing_support.json',support);oj('independent_validation.json',validation)
for name,info in fingerprints.items():
    s=Path(info['path']).stat();assert s.st_size==info['bytes'] and s.st_mtime_ns==info['mtime_ns'],name
con.close()
passed=all(r['feasibility_pass'] for r in support)
oj('analysis_manifest.json',dict(protocol_sha256=OXYGEN_RESPONSE_PROTOCOL_SHA256,
    completed_utc=datetime.now(timezone.utc).isoformat(),environment='google_colab' if IN_COLAB else 'local',
    all_sources_pass=passed,outcomes_examined=False,spo2_response_magnitudes_examined=False,
    biological_discovery=False,goal_complete=False))
lines=['# Oxygen-response timing feasibility','',
    '**Actual execution: '+('Google Colab' if IN_COLAB else 'local')+'. No biological effect or mortality result was estimated.**','',
    '| Database | FiO2 observed | Escalation encounters | First escalation with paired SpO2 | Paired and corroborated | Original exposure 0 / 1 |',
    '|---|---:|---:|---:|---:|---|']
for r in support:
    g=r['full_support_by_original_exposure']
    lines.append(f"| {r['dataset']} | {r['fio2_observed_encounters']} | {r['escalation_encounters']} | {r['first_escalation_paired']} | {r['first_escalation_paired_and_corroborated']} | {g['0']} / {g['1']} |")
lines+=['', 'The frozen requirement was 200 fully supported encounters per source and 50 in each original SpO2 exposure group.', '',
    ('The timing screen passes. Biological validity, confounding and power still require a separate outcome-analysis specification.' if passed else
     'The timing screen fails. Close this specification without widening time windows or relaxing oxygen-step/corroboration rules. These records do not support the proposed cross-database response experiment under the frozen requirements.'),'',
    'Every escalation and paired timestamp was independently reconstructed with pandas/NumPy and matched SQL. No saturation-response magnitude, troponin endpoint or mortality outcome was opened for this candidate. Documentation times do not guarantee intervention times or sustained delivery. These feasibility results do not establish treatment benefit or meet the biological-discovery objective.','']
report='\n'.join(lines);(PROJECT/'docs/SPO2_OXYGEN_RESPONSE_FEASIBILITY_RESULTS.md').write_text(report)
oj('run_status.json',dict(status='complete',stage='oxygen-transition timing feasibility',environment='google_colab' if IN_COLAB else 'local'))
print(report)
"""
