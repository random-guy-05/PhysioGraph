"""Outcome-blind capnography timing at original SpO2 transitions."""

SETUP = r"""
import json,hashlib,importlib.util,csv
from pathlib import Path
from datetime import datetime,timezone
import duckdb,numpy as np,pandas as pd
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph';PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905'
OUT=PROJECT/'research/spo2_capnography_feasibility';OUT.mkdir(parents=True,exist_ok=True)
def cf_json(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def cf_q(x):return "'"+str(x).replace("'","''")+"'"
def cf_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def cf_info(path):
    p=Path(path);s=p.stat();return dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
assert cf_hash(PROJECT/'docs/SPO2_CAPNOGRAPHY_FEASIBILITY_PLAN.md')==CAPNOGRAPHY_PROTOCOL_SHA256
lock=OUT/'protocol_lock.json'
if lock.exists():assert json.loads(lock.read_text())['sha256']==CAPNOGRAPHY_PROTOCOL_SHA256
else:cf_json('protocol_lock.json',dict(sha256=CAPNOGRAPHY_PROTOCOL_SHA256,utc=datetime.now(timezone.utc).isoformat(),before_patient_capnography_counts_and_values=True,prior_project_results_known=True))
try:con.close()
except (NameError,AttributeError):pass
con=duckdb.connect(str(PRIVATE/'capnography_feasibility.duckdb'));con.execute('SET threads=2');con.execute("SET memory_limit='2GB'");con.execute('SET enable_progress_bar=false')
inputs={}
for alias,name in [('origin','hemoglobin_oxidation'),('prior','spo2_context')]:
    path=PRIVATE/(name+'.duckdb');info=cf_info(path);info['sha256']=cf_hash(path);inputs[alias]=info
    con.execute(f'ATTACH {cf_q(path)} AS {alias} (READ_ONLY)')
dictionary=DRIVE/'Data/MIMIC/Full/d_items.csv'
with dictionary.open() as f:lookup={int(r['itemid']):r for r in csv.DictReader(f)}
assert lookup[228640]['label']=='EtCO2' and lookup[224687]['label']=='Minute Volume'
assert lookup[224687]['unitname']=='L/min'
for i in [228640,224687]:assert lookup[i]['linksto']=='chartevents' and lookup[i]['param_type']=='Numeric'
dictionary_info=dict(sha256=cf_hash(dictionary),items={str(i):lookup[i] for i in [228640,224687]})
episode_checks=[]
for d in ['eicu','mimic']:
    con.execute(f'CREATE OR REPLACE TABLE {d}_selected AS SELECT * FROM origin.{d}_selected')
    selected=con.execute(f'SELECT stay_id,person_id,exposed FROM {d}_selected ORDER BY stay_id').df()
    assert len(selected)=={'eicu':10167,'mimic':4305}[d] and selected.person_id.nunique()==len(selected)
    con.execute(f'''CREATE OR REPLACE TABLE {d}_episodes AS WITH b AS (
      SELECT b.stay_id,b.bin,b.representative_minute AS b,b.value,
      lag(b.representative_minute) OVER w AS a,lag(b.value) OVER w AS previous_value
      FROM prior.{d}_bins b JOIN {d}_selected s USING(stay_id) WHERE concept='spo2'
      WINDOW w AS (PARTITION BY b.stay_id ORDER BY b.bin))
      SELECT stay_id,a,b,CASE WHEN value>previous_value THEN 'up' ELSE 'down' END direction
      FROM b WHERE abs(value-previous_value)>=4 AND b-a>0 AND b-a<=30
      QUALIFY row_number() OVER(PARTITION BY stay_id ORDER BY bin)=1''')
    episodes=con.execute(f'SELECT * FROM {d}_episodes ORDER BY stay_id').df()
    assert set(episodes.stay_id)==set(selected.loc[selected.exposed.eq(1),'stay_id'])
    assert len(episodes)=={'eicu':3798,'mimic':1276}[d]
    bins=con.execute(f"SELECT b.stay_id,b.bin,b.value,b.representative_minute FROM prior.{d}_bins b JOIN {d}_selected s USING(stay_id) WHERE concept='spo2' ORDER BY stay_id,bin").df()
    bins['a']=bins.groupby('stay_id').representative_minute.shift();delta=bins.value-bins.groupby('stay_id').value.shift();gap=bins.representative_minute-bins.a
    bins['direction']=np.where(delta>0,'up','down')
    expected=bins.loc[delta.abs().ge(4)&gap.gt(0)&gap.le(30)].groupby('stay_id',sort=False).head(1)
    assert np.array_equal(episodes[['stay_id','direction']],expected[['stay_id','direction']])
    assert np.allclose(episodes[['a','b']],expected[['a','representative_minute']],rtol=0,atol=1e-12)
    episode_checks.append(dict(dataset=d,selected_people=len(selected),exposed_people=len(episodes),sql_python_times_signs_and_original_exposure_agree=True))
sources={d:cf_info(DRIVE/'Data'/name/'Full'/filename) for d,name,filename in [('eicu','eICU','vitalPeriodic.csv'),('mimic','MIMIC','chartevents.csv')]}
def cf_csv(d):return f"read_csv({cf_q(sources[d]['path'])},header=true,all_varchar=true,sample_size=10000,quote='\"',escape='\"',strict_mode=true)"
cf_json('episode_validation.json',episode_checks)
cf_json('run_status.json',dict(status='running',stage='protocol locked and original episodes verified',environment='google_colab' if IN_COLAB else 'local'))
print('Capnography metadata protocol locked; original episode timing and direction verified.')
"""

EXTRACT = r"""
con.execute(f'''CREATE OR REPLACE TABLE eicu_raw AS SELECT s.stay_id,try_cast(v.vitalperiodicid AS BIGINT) row_id,
 'etco2' concept,try_cast(v.observationoffset AS DOUBLE) event_minute,NULL::DOUBLE available_minute,
 NULL::VARCHAR unit,NULL::INTEGER warning,nullif(trim(v.respiration),'') IS NOT NULL respiratory_rate_present
 FROM {cf_csv('eicu')} v JOIN eicu_selected s ON try_cast(v.patientunitstayid AS BIGINT)=s.stay_id
 WHERE nullif(trim(v.etco2),'') IS NOT NULL AND try_cast(v.observationoffset AS DOUBLE) BETWEEN 0 AND 240''')
cf_json('run_status.json',dict(status='running',stage='eICU monitor metadata complete; MIMIC raw capnography/minute-volume scan'))
con.execute(f'''CREATE OR REPLACE TABLE mimic_raw AS SELECT s.stay_id,
 CASE WHEN try_cast(v.itemid AS INTEGER)=228640 THEN 'etco2' ELSE 'minute_volume' END concept,
 date_diff('second',s.admit_time,try_cast(v.charttime AS TIMESTAMP))/60.0 event_minute,
 date_diff('second',s.admit_time,try_cast(v.storetime AS TIMESTAMP))/60.0 available_minute,
 v.valueuom unit,try_cast(v.warning AS INTEGER) warning
 FROM {cf_csv('mimic')} v JOIN mimic_selected s ON try_cast(v.stay_id AS BIGINT)=s.stay_id
 AND try_cast(v.subject_id AS BIGINT)=s.subject_id AND try_cast(v.hadm_id AS BIGINT)=s.hadm_id
 WHERE try_cast(v.itemid AS INTEGER) IN (228640,224687)
 AND coalesce(nullif(trim(v.value),''),nullif(trim(v.valuenum),'')) IS NOT NULL
 AND date_diff('second',s.admit_time,try_cast(v.charttime AS TIMESTAMP))/60.0 BETWEEN 0 AND 240''')
print('Full raw metadata extraction tables committed.')
"""

VERIFY = r"""
for d in ['eicu','mimic']:
    columns={r[0] for r in con.execute(f'DESCRIBE {d}_raw').fetchall()}
    assert {'stay_id','concept','event_minute','available_minute','unit','warning'}<=columns
    assert not columns.intersection({'value','valuenum','etco2','respiration','mortality','hospital_mortality'})
    assert con.execute(f'''SELECT count(*) FROM {d}_raw r LEFT JOIN {d}_selected s USING(stay_id)
      WHERE s.stay_id IS NULL OR r.event_minute IS NULL OR r.event_minute<0 OR r.event_minute>240''').fetchone()[0]==0
cf_json('run_status.json',dict(status='running',stage='saved raw tables verified; hashing full native sources'))
for info in sources.values():
    s=Path(info['path']).stat();assert (s.st_size,s.st_mtime_ns)==(info['bytes'],info['mtime_ns'])
    info['sha256']=cf_hash(info['path'])
    s=Path(info['path']).stat();assert (s.st_size,s.st_mtime_ns)==(info['bytes'],info['mtime_ns'])
for info in inputs.values():assert cf_hash(info['path'])==info['sha256']
historical_hashes={'eicu':'09818ad685112afb5e227ec495a1ecadc2d93aad1ae937875b469ac20f321744',
  'mimic':'0d1932df345c2d135ce715df1f6afa603f7124894a5aa8c6ba56aa6ee02160d4'}
for d in sources:assert sources[d]['sha256']==historical_hashes[d]
cf_json('source_manifest.json',dict(raw=sources,inputs=inputs,dictionary=dictionary_info,full_raw_scans=True,explicit_csv_quote_and_escape=True,
  native_source_hashes_match_prior_independent_extractions=True,capnography_values_selected=False,minute_volume_values_selected=False,respiratory_rate_values_selected=False,mortality_selected=False))
cf_json('run_status.json',dict(status='running',stage='both full metadata scans and source hashes complete; pair validation'))
print('Both full raw metadata scans complete. No CO2, ventilation or mortality amplitudes selected.')
"""

SUPPORT = r"""
def cf_nearest(times,target,mid,w,side):
    t=times[(abs(times-target)<=w)&((times<mid) if side=='pre' else (times>=mid))]
    return float(sorted(t,key=lambda z:(abs(z-target),z))[0]) if len(t) else np.nan
def cf_same(a,b):return (pd.isna(a) and pd.isna(b)) or (pd.notna(a) and pd.notna(b) and abs(a-b)<=1e-12)
support=[];checks=[];coverage=[];gates=[]
for d in ['eicu','mimic']:
    raw=con.execute(f'SELECT * FROM {d}_raw').df()
    episodes=con.execute(f'SELECT * FROM {d}_episodes ORDER BY stay_id').df()
    con.execute(f'CREATE OR REPLACE TABLE {d}_times AS SELECT DISTINCT stay_id,concept,event_minute FROM {d}_raw')
    times=con.execute(f'SELECT * FROM {d}_times').df()
    groups={(sid,c):g.event_minute.to_numpy() for (sid,c),g in times.groupby(['stay_id','concept'])}
    rate_flags=raw.groupby(['stay_id','event_minute']).respiratory_rate_present.any().to_dict() if d=='eicu' else {}
    for concept in (['etco2'] if d=='eicu' else ['etco2','minute_volume']):
        g=raw.loc[raw.concept.eq(concept)]
        coverage.append(dict(dataset=d,concept=concept,raw_rows=len(g),distinct_times=len(g[['stay_id','event_minute']].drop_duplicates()),people=int(g.stay_id.nunique())))
    for w in [5,15]:
        con.execute(f'''CREATE OR REPLACE TABLE {d}_pairs_{w} AS SELECT e.*,s.hospital_id,
          e.a-{w}>=0 AND e.b+{w}<=240 full_window,
          (SELECT t.event_minute FROM {d}_times t WHERE t.stay_id=e.stay_id AND t.concept='etco2'
           AND abs(t.event_minute-e.a)<={w} AND t.event_minute<(e.a+e.b)/2 ORDER BY abs(t.event_minute-e.a),t.event_minute LIMIT 1) pre_minute,
          (SELECT t.event_minute FROM {d}_times t WHERE t.stay_id=e.stay_id AND t.concept='etco2'
           AND abs(t.event_minute-e.b)<={w} AND t.event_minute>=(e.a+e.b)/2 ORDER BY abs(t.event_minute-e.b),t.event_minute LIMIT 1) post_minute
          FROM {d}_episodes e JOIN {d}_selected s USING(stay_id)''')
        pairs=con.execute(f'SELECT * FROM {d}_pairs_{w} ORDER BY stay_id').df()
        for r in pairs.itertuples():
            t=groups.get((r.stay_id,'etco2'),np.array([]));mid=(r.a+r.b)/2
            assert bool(r.full_window)==bool(r.a-w>=0 and r.b+w<=240)
            assert cf_same(r.pre_minute,cf_nearest(t,r.a,mid,w,'pre'))
            assert cf_same(r.post_minute,cf_nearest(t,r.b,mid,w,'post'))
            if pd.notna(r.pre_minute) and pd.notna(r.post_minute):assert r.pre_minute<r.post_minute
        context_sql=(f'''EXISTS(SELECT 1 FROM eicu_raw t WHERE t.stay_id=p.stay_id AND t.event_minute=p.pre_minute AND t.respiratory_rate_present)
          AND EXISTS(SELECT 1 FROM eicu_raw t WHERE t.stay_id=p.stay_id AND t.event_minute=p.post_minute AND t.respiratory_rate_present)'''
          if d=='eicu' else f'''EXISTS(SELECT 1 FROM mimic_times t WHERE t.stay_id=p.stay_id AND t.concept='minute_volume'
          AND abs(t.event_minute-p.a)<={w} AND t.event_minute<(p.a+p.b)/2)
          AND EXISTS(SELECT 1 FROM mimic_times t WHERE t.stay_id=p.stay_id AND t.concept='minute_volume'
          AND abs(t.event_minute-p.b)<={w} AND t.event_minute>=(p.a+p.b)/2)''')
        context=con.execute(f'SELECT stay_id,({context_sql}) context_present FROM {d}_pairs_{w} p ORDER BY stay_id').df()
        py_context=[]
        for r in pairs.itertuples():
            if d=='eicu':ok=rate_flags.get((r.stay_id,r.pre_minute),False) and rate_flags.get((r.stay_id,r.post_minute),False)
            else:
                t=groups.get((r.stay_id,'minute_volume'),np.array([]));mid=(r.a+r.b)/2
                ok=pd.notna(cf_nearest(t,r.a,mid,w,'pre')) and pd.notna(cf_nearest(t,r.b,mid,w,'post'))
            py_context.append(bool(ok))
        assert np.array_equal(context.context_present,py_context)
        pairs['context_present']=py_context
        paired=pairs.full_window & pairs.pre_minute.notna() & pairs.post_minute.notna()
        for stage,mask in [('original',np.ones(len(pairs),dtype=bool)),('full_window',pairs.full_window),
          ('pre_available',pairs.full_window & pairs.pre_minute.notna()),('post_available',pairs.full_window & pairs.post_minute.notna()),
          ('paired',paired),('paired_with_context_timestamps',paired & pairs.context_present)]:
            p=pairs.loc[mask];up=int(p.direction.eq('up').sum());down=int(p.direction.eq('down').sum());h=int(p.hospital_id.nunique()) if d=='eicu' else None
            r=dict(dataset=d,tolerance_minutes=w,stage=stage,people=len(p),up=up,down=down,hospitals=h,
              context_definition='respiratory rate field present at both selected monitor timestamps' if d=='eicu' else 'minute volume timestamps available in both side-specific windows')
            if stage=='paired':
                r['support_floor_pass']=len(p)>=50 and up>=20 and down>=20 and (d=='mimic' or h>=10)
                r['pre_timing_error_median_minutes']=float((p.pre_minute-p.a).abs().median()) if len(p) else None
                r['post_timing_error_median_minutes']=float((p.post_minute-p.b).abs().median()) if len(p) else None
                if w==5:gates.append(r['support_floor_pass'])
            support.append(r)
        checks.append(dict(dataset=d,tolerance_minutes=w,all_nearest_timestamps_full_windows_and_forward_order_sql_python_agree=True,context_flags_sql_python_agree=True))
assert len(gates)==2
cf_json('coverage.json',coverage);cf_json('support.json',support)
cf_json('validation.json',dict(checks=checks,one_original_episode_per_exposed_person=True,nearest_pair_ties_choose_earlier=True,no_reused_co2_timestamp=True))
cf_json('results.json',dict(primary_joint_support_pass=all(gates),co2_values_selected=False,ventilation_values_selected=False,mortality_selected=False,association_models=0,biological_discovery=False,
  interpretation='Timing support only. Missing capnography does not establish absence of a ventilation or perfusion disturbance.'))
report=['# Capnography at the original SpO2 events','',
 'Actual execution: '+('Google Colab' if IN_COLAB else 'local')+'. Full eICU vitalPeriodic and MIMIC chartevents metadata scans; no numeric CO2 or ventilation values selected.','',
 '| Database | Tolerance (min) | Full-window people | Paired people | SpO2 up | SpO2 down | eICU hospitals | Context timestamps on both sides | Floor |',
 '|---|---:|---:|---:|---:|---:|---:|---:|---|']
for r in support:
    if r['stage']=='paired':
        context=next(s for s in support if s['dataset']==r['dataset'] and s['tolerance_minutes']==r['tolerance_minutes'] and s['stage']=='paired_with_context_timestamps')
        full=next(s for s in support if s['dataset']==r['dataset'] and s['tolerance_minutes']==r['tolerance_minutes'] and s['stage']=='full_window')
        report.append(f"| {r['dataset']} | {r['tolerance_minutes']} | {full['people']} | {r['people']} | {r['up']} | {r['down']} | {r['hospitals'] if r['hospitals'] is not None else 'NA'} | {context['people']} | {r['support_floor_pass']} |")
report+=['','Primary joint five-minute timing floor: '+str(all(gates))+'. The 15-minute rows are descriptive and do not replace it.',
 'The larger tolerance also requires larger full windows within minutes 0–240 and therefore excludes more events near the boundaries. Its eligible set differs; paired counts need not increase with tolerance.',
 'The original episode signs and timestamps, all nearest choices, forward order, full windows and context flags agree in independent SQL/Python reconstruction. Source hashes were recorded and input database hashes rechecked.',
 'Context means respiratory-rate field presence in eICU and actual minute-volume item timestamps in MIMIC; neither establishes constant minute ventilation. Nonempty chart text, unqualified units and warning flags are still included in this optimistic timing ceiling.',
 'No CO2 change, mortality association, biological mechanism or treatment benefit was estimated. '+('A separate value/contrast specification is required before amplitude analysis.' if all(gates) else 'The primary cross-database event-scale contrast is unsupported by these recordings; this does not reject the underlying biological mechanism.'),'']
(PROJECT/'docs/SPO2_CAPNOGRAPHY_FEASIBILITY_RESULTS.md').write_text('\n'.join(report))
cf_json('run_status.json',dict(status='complete',stage='full raw metadata extraction and independently validated timing support',utc=datetime.now(timezone.utc).isoformat(),goal_complete=False))
con.close();print('\n'.join(report))
"""
