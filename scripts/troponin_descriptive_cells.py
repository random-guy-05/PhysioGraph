"""Explicit descriptive amendment; the failed primary gate is preserved."""

ANALYSIS = r"""
import json,hashlib,importlib.util
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal
import numpy as np
import pandas as pd
import duckdb
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph';PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905'
OUT=PROJECT/'research/spo2_troponin_descriptive';OUT.mkdir(parents=True,exist_ok=True)
def tj(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def ts(stage,state='running'):
    tj('run_status.json',dict(stage=stage,status=state,utc=datetime.now(timezone.utc).isoformat(),environment='google_colab' if IN_COLAB else 'local',goal_complete=False))
def qs(x):return "'"+str(x).replace("'","''")+"'"
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
assert sha(PROJECT/'docs/SPO2_TROPONIN_DESCRIPTIVE_AMENDMENT.md')==TROPONIN_DESCRIPTIVE_PROTOCOL_SHA256
lock=OUT/'protocol_lock.json'
if lock.exists():assert json.loads(lock.read_text())['sha256']==TROPONIN_DESCRIPTIVE_PROTOCOL_SHA256
else:tj('protocol_lock.json',dict(sha256=TROPONIN_DESCRIPTIVE_PROTOCOL_SHA256,utc=datetime.now(timezone.utc).isoformat(),before_directional_values=True,feasibility_counts_and_prior_project_results_known=True,primary_gate_remains_failed=True))
paths={k:PRIVATE/(k+'.duckdb') for k in ['preicu_linkage_amendment','spo2_context']}
inputs={k:dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for k,p in paths.items()}
inputs['preicu_linkage_amendment']['sha256']=sha(paths['preicu_linkage_amendment'])
tj('input_manifest.json',inputs)
try:con.close()
except (NameError,AttributeError):pass
con=duckdb.connect(':memory:');con.execute('SET threads=2')
con.execute(f"ATTACH {qs(paths['preicu_linkage_amendment'])} AS src (READ_ONLY)")
con.execute(f"ATTACH {qs(paths['spo2_context'])} AS prior (READ_ONLY)")
ts('identifier-only selection before concentration joins')
chosen={};support=[]
for d in ['eicu','mimic']:
    meta=con.execute(f'''SELECT s.stay_id,cast(c.person_id AS VARCHAR) person_id
      FROM src.{d}_trajectory_support s JOIN prior.{d}_cohort c USING(stay_id)
      WHERE s.complete_trajectory ORDER BY s.stay_id''').df()
    assert len(meta)=={'eicu':133,'mimic':76}[d] and meta.stay_id.is_unique
    meta['selection_hash']=[hashlib.sha256(f'20260905:{d}:{p}:{s}'.encode()).hexdigest() for p,s in zip(meta.person_id,meta.stay_id)]
    chosen[d]=set(meta.sort_values('selection_hash').drop_duplicates('person_id').stay_id)
    support.append(dict(dataset=d,complete_encounters=len(meta),people=int(meta.person_id.nunique()),all_original_episodes=con.execute(f'SELECT count(*) FROM src.{d}_anchors').fetchone()[0]))
tj('identifier_selection.json',support)
ts('frozen descriptive concentration comparison')
joint=[];marginal=[];timing=[];validation=[]
for d in ['eicu','mimic']:
    con.execute(f'''CREATE OR REPLACE TEMP TABLE {d}_description AS
      SELECT s.stay_id,s.episode_minute,s.concept,s.unit,s.pre_first_time,s.pre_last_time,
        f.value first_pre,l.value last_pre,l.available_minute last_pre_available,
        max(p.value) post_peak,min(p.event_minute) first_delayed_time,max(p.event_minute) last_delayed_time,
        a.hospital_start_minute,a.followup_end_offset_minutes
      FROM src.{d}_trajectory_support s JOIN src.{d}_anchors a USING(stay_id)
      JOIN src.{d}_troponin f ON f.stay_id=s.stay_id AND f.concept=s.concept AND f.unit=s.unit AND f.event_minute=s.pre_first_time
      JOIN src.{d}_troponin l ON l.stay_id=s.stay_id AND l.concept=s.concept AND l.unit=s.unit AND l.event_minute=s.pre_last_time
      JOIN src.{d}_troponin p ON p.stay_id=s.stay_id AND p.concept=s.concept AND p.unit=s.unit
        AND p.event_minute>=s.episode_minute+120 AND p.event_minute<=s.episode_minute+720
      WHERE s.complete_trajectory GROUP BY ALL''')
    frame=con.execute(f'''SELECT *,last_pre/first_pre pre_ratio,post_peak/last_pre post_ratio,
      cast(CAST(last_pre AS DECIMAL(24,10))>=1.5*CAST(first_pre AS DECIMAL(24,10)) AS INTEGER) pre_rise,
      cast(CAST(post_peak AS DECIMAL(24,10))>=1.5*CAST(last_pre AS DECIMAL(24,10)) AS INTEGER) post_rise
      FROM {d}_description ORDER BY stay_id''').df()
    assert len(frame)=={'eicu':133,'mimic':76}[d] and frame.stay_id.is_unique
    assert frame.pre_first_time.ge(frame.episode_minute-720).all()
    assert frame.pre_first_time.ge(frame.hospital_start_minute).all()
    assert frame.pre_last_time.lt(frame.episode_minute).all()
    assert (frame.pre_last_time-frame.pre_first_time).ge(60).all()
    assert frame.first_delayed_time.ge(frame.episode_minute+120).all()
    assert frame.last_delayed_time.le(frame.episode_minute+720).all()
    assert frame.last_delayed_time.le(frame.followup_end_offset_minutes).all()
    labs=con.execute(f'''SELECT t.* FROM src.{d}_troponin t JOIN src.{d}_trajectory_support s USING(stay_id)
      WHERE s.complete_trajectory ORDER BY t.stay_id,t.event_minute''').df()
    grouped={k:v for k,v in labs.groupby('stay_id')}
    for r in frame.itertuples():
        t=grouped[r.stay_id];t=t.loc[t.concept.eq(r.concept)&t.unit.eq(r.unit)]
        pre=t.loc[t.event_minute.ge(max(r.episode_minute-720,r.hospital_start_minute))&t.event_minute.lt(r.episode_minute)]
        post=t.loc[t.event_minute.ge(r.episode_minute+120)&t.event_minute.le(r.episode_minute+720)]
        independent=np.array([pre.iloc[0]['value'],pre.iloc[-1]['value'],post['value'].max()])
        assert np.array_equal(independent,np.array([r.first_pre,r.last_pre,r.post_peak]))
        assert np.max(abs(independent-np.round(independent,10)))<1e-12
        a,b,c=[Decimal(str(round(float(v),10))) for v in independent]
        assert (int(b>=Decimal('1.5')*a),int(c>=Decimal('1.5')*b))==(r.pre_rise,r.post_rise)
    frame['patient_selected']=frame.stay_id.isin(chosen[d])
    frame.to_csv(PRIVATE/(d+'_troponin_descriptive.csv'),index=False)
    for mode,sub in [('all_complete_encounters',frame),('one_per_person',frame.loc[frame.patient_selected])]:
        for pre,post,label in [(0,0,'neither'),(1,0,'pre_only'),(0,1,'post_only'),(1,1,'both')]:
            n=int((sub.pre_rise.eq(pre)&sub.post_rise.eq(post)).sum())
            joint.append(dict(dataset=d,sample=mode,pattern=label,n=n,denominator=len(sub),percent=100*n/len(sub)))
        marginal.append(dict(dataset=d,sample=mode,n=len(sub),pre_rises=int(sub.pre_rise.sum()),post_rises=int(sub.post_rise.sum())))
    spans={'pre_span_minutes':frame.pre_last_time-frame.pre_first_time,'pre_last_to_episode_minutes':frame.episode_minute-frame.pre_last_time,'first_delayed_after_episode_minutes':frame.first_delayed_time-frame.episode_minute}
    timing.append(dict(dataset=d,n=len(frame),last_pre_result_available_by_episode=int(frame.last_pre_available.le(frame.episode_minute).sum()),last_pre_availability_missing=int(frame.last_pre_available.isna().sum()),
        timing_quantiles={k:{str(q):float(v.quantile(q)) for q in [.25,.5,.75]} for k,v in spans.items()}))
    validation.append(dict(dataset=d,all_fixed_windows_pass=True,independent_first_last_peak_values_match=True,independent_decimal_labels_match=True,complete_encounters=len(frame)))
for k,p in paths.items():
    s=p.stat();assert s.st_size==inputs[k]['bytes'] and s.st_mtime_ns==inputs[k]['mtime_ns']
tj('joint_patterns.json',joint);tj('marginal_patterns.json',marginal);tj('timing_context.json',timing);tj('validation.json',validation)
tj('analysis_manifest.json',dict(protocol_sha256=TROPONIN_DESCRIPTIVE_PROTOCOL_SHA256,module_sha256=sha(PROJECT/'scripts/troponin_descriptive_cells.py'),
    inputs_unchanged=True,primary_count_gate_remains_failed=True,descriptive_departure_explicit=True,mortality_joined=False,hypothesis_tests=0,goal_complete=False))
lines=['# Serial troponin: descriptive amendment results','',
    'Actual local execution.' if not IN_COLAB else 'Actual Google Colab execution.',
    'The original primary count gate remains failed. These are selected-case descriptions, without hypothesis tests, confidence intervals or population inference.','',
    '| Database | Sample | Pattern | Count / denominator | Percent |','|---|---|---|---:|---:|']
for r in joint:lines.append(f"| {r['dataset']} | {r['sample']} | {r['pattern']} | {r['n']} / {r['denominator']} | {r['percent']:.1f} |")
lines+=['', 'Pre rise compares last versus first pre-episode concentration; post rise compares the delayed post-episode peak with the last pre-episode concentration. Both use the unchanged 1.5-fold threshold. Windows differ, and the post measure is a maximum rather than a last value.',
    '', 'A pre-episode rise means that the recorded numeric troponin values had already increased in those cases. It does not locate myocardial injury onset, identify a causal mechanism, prove infarction, or establish benefit from oxygen or another treatment. Absence of a pre-rise does not rule out prior injury. Selective repeat testing and assay kinetics limit interpretation.',
    '', 'The episode is the first captured in the original first-four-hour ICU window, not the first possible hypoxic event in the illness. Earlier hypoxemia has not been excluded. These calculations inherit the existing numeric laboratory harmonization; assay-specific censoring and reference-limit adjudication were not newly validated here.',
    '', 'All available complete encounters and the fixed one-per-person sensitivity are reported. SQL and Python independently agree on every first/last/peak concentration and decimal rise label. Mortality labels were not joined. See the explicit amendment for the departure from the original stop rule. No paradigm-shifting discovery is established.']
report='\n'.join(lines)+'\n';(PROJECT/'docs/SPO2_TROPONIN_DESCRIPTIVE_RESULTS.md').write_text(report)
con.close();ts('descriptive comparison and independent validation','complete');print(report)
"""
