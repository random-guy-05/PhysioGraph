"""Fixed paired ionized-calcium/pH biochemical screen."""

SETUP = r"""
import json,hashlib,importlib.util
from pathlib import Path
from datetime import datetime,timezone
import duckdb,numpy as np,pandas as pd
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph';PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905'
OUT=PROJECT/'research/spo2_ionized_calcium';OUT.mkdir(parents=True,exist_ok=True)
def ca_json(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def ca_q(x):return "'"+str(x).replace("'","''")+"'"
def ca_hash(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
assert ca_hash(PROJECT/'docs/SPO2_IONIZED_CALCIUM_PLAN.md')==CALCIUM_PROTOCOL_SHA256
lock=OUT/'protocol_lock.json'
if lock.exists():assert json.loads(lock.read_text())['sha256']==CALCIUM_PROTOCOL_SHA256
else:ca_json('protocol_lock.json',dict(sha256=CALCIUM_PROTOCOL_SHA256,utc=datetime.now(timezone.utc).isoformat(),before_patient_calcium_counts_and_values=True,prior_negative_results_known=True))
try:con.close()
except (NameError,AttributeError):pass
origin=PRIVATE/'hemoglobin_oxidation.duckdb';origin_hash=ca_hash(origin)
con=duckdb.connect(str(PRIVATE/'ionized_calcium.duckdb'));con.execute('SET threads=2');con.execute("SET memory_limit='2GB'");con.execute('SET enable_progress_bar=false')
con.execute(f'ATTACH {ca_q(origin)} AS origin (READ_ONLY)')
for d in ['eicu','mimic']:
    con.execute(f'CREATE OR REPLACE TABLE {d}_selected AS SELECT * FROM origin.{d}_selected')
    assert con.execute(f'SELECT count(*),count(DISTINCT person_id) FROM {d}_selected').fetchone()==({'eicu':10167,'mimic':4305}[d],)*2
con.execute('CREATE OR REPLACE TABLE admissions AS SELECT * FROM origin.admissions')
old_manifest=json.loads((PROJECT/'research/spo2_hemoglobin_oxidation/source_manifest.json').read_text())
sources={k:dict(v) for k,v in old_manifest['raw'].items()}
for key,info in sources.items():
    s=Path(info['path']).stat();assert (s.st_size,s.st_mtime_ns)==(info['bytes'],info['mtime_ns'])
def ca_csv(key):return f"read_csv({ca_q(sources[key]['path'])},header=true,all_varchar=true,sample_size=10000,strict_mode=true)"
ca_json('run_status.json',dict(status='running',stage='locked; raw biochemical scan pending',environment='google_colab' if IN_COLAB else 'local'))
print('Paired calcium/pH protocol locked; original cohorts retained.')
"""

EXTRACT = r"""
con.execute(f'''CREATE OR REPLACE TABLE eicu_raw AS SELECT e.stay_id,try_cast(l.labid AS BIGINT) lab_id,NULL::BIGINT specimen_id,
 CASE WHEN l.labname='ionized calcium' THEN 'calcium' ELSE 'ph' END assay,
 try_cast(l.labresultoffset AS DOUBLE) event_minute,try_cast(l.labresultrevisedoffset AS DOUBLE) available_minute,
 try_cast(l.labresult AS DOUBLE) raw_value,l.labresulttext raw_text,l.labmeasurenamesystem unit_system,l.labmeasurenameinterface unit_interface
 FROM {ca_csv('eICU/lab')} l JOIN eicu_selected e ON try_cast(l.patientunitstayid AS BIGINT)=e.stay_id
 WHERE ((l.labname='ionized calcium' AND try_cast(l.labtypeid AS INTEGER)=1) OR (l.labname='pH' AND try_cast(l.labtypeid AS INTEGER)=7))
 AND e.followup_end_offset_minutes IS NOT NULL
 AND try_cast(l.labresultoffset AS DOUBLE) BETWEEN greatest(-1440,e.hospital_start_minute) AND least(1680,e.followup_end_offset_minutes)''')
ca_json('run_status.json',dict(status='running',stage='eICU raw extraction complete; MIMIC scan'))
con.execute(f'''CREATE OR REPLACE TABLE mimic_candidates AS SELECT e.stay_id,e.hadm_id expected_hadm_id,
 try_cast(l.labevent_id AS BIGINT) lab_id,try_cast(l.specimen_id AS BIGINT) specimen_id,
 try_cast(l.subject_id AS BIGINT) subject_id,try_cast(l.hadm_id AS BIGINT) raw_hadm_id,try_cast(l.charttime AS TIMESTAMP) charttime,
 CASE WHEN try_cast(l.itemid AS INTEGER)=50808 THEN 'calcium' ELSE 'ph' END assay,
 date_diff('second',e.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0 event_minute,
 date_diff('second',e.admit_time,try_cast(l.storetime AS TIMESTAMP))/60.0 available_minute,
 try_cast(l.valuenum AS DOUBLE) raw_value,l.value raw_text,l.valueuom unit_system,NULL::VARCHAR unit_interface
 FROM {ca_csv('MIMIC/labevents')} l JOIN mimic_selected e ON try_cast(l.subject_id AS BIGINT)=e.subject_id
 WHERE try_cast(l.itemid AS INTEGER) IN (50808,50820) AND e.followup_end_offset_minutes IS NOT NULL
 AND date_diff('second',e.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0 BETWEEN greatest(-1440,e.hospital_start_minute) AND least(1680,e.followup_end_offset_minutes)''')
assert con.execute('SELECT count(*)-count(DISTINCT lab_id) FROM mimic_candidates').fetchone()[0]==0
con.execute('''CREATE OR REPLACE TABLE missing_hospital AS SELECT l.lab_id,count(a.hadm_id) matches,min(a.hadm_id) assigned_hadm_id
 FROM mimic_candidates l LEFT JOIN admissions a ON l.subject_id=a.subject_id AND l.charttime>=a.start_time AND l.charttime<a.end_time
 WHERE l.raw_hadm_id IS NULL GROUP BY l.lab_id''')
con.execute('''CREATE OR REPLACE TABLE mimic_raw AS SELECT l.stay_id,l.lab_id,l.specimen_id,l.assay,l.event_minute,l.available_minute,
 l.raw_value,l.raw_text,l.unit_system,l.unit_interface FROM mimic_candidates l LEFT JOIN missing_hospital m USING(lab_id)
 WHERE l.raw_hadm_id=l.expected_hadm_id OR (l.raw_hadm_id IS NULL AND m.matches=1 AND m.assigned_hadm_id=l.expected_hadm_id)''')
for info in sources.values():
    p=Path(info['path']);s=p.stat();assert (s.st_size,s.st_mtime_ns)==(info['bytes'],info['mtime_ns'])
    assert ca_hash(p)==info['sha256'];s=p.stat();assert (s.st_size,s.st_mtime_ns)==(info['bytes'],info['mtime_ns'])
ca_json('source_manifest.json',dict(raw=sources,original_database=str(origin),original_database_sha256=origin_hash,
 full_raw_lab_scans=True,numeric_rows_copied_privately=True,mortality_selected=False))
ca_json('run_status.json',dict(status='running',stage='raw extraction and source hashes complete; qualification pending'))
print('Raw calcium/pH scans complete, source hashes verified.')
"""

QUALIFY = r"""
support=[];qc=[];checks=[]
def ca_token(x):return '' if pd.isna(x) else ''.join(str(x).casefold().split())
for d in ['eicu','mimic']:
    raw=con.execute(f'SELECT * FROM {d}_raw ORDER BY stay_id,event_minute,assay,lab_id').df()
    us=raw.unit_system.map(ca_token);ui=raw.unit_interface.map(ca_token);isca=raw.assay.eq('calcium')
    raw['value']=raw.raw_value/(4.0078 if d=='eicu' else 1.)
    raw.loc[~isca,'value']=raw.loc[~isca,'raw_value']
    caunits=us.eq('mg/dl')&ui.eq('mg/dl') if d=='eicu' else us.eq('mmol/l')
    phunits=us.isin(['','unit','units','ph','phunits'])&ui.isin(['','unit','units','ph','phunits'])
    raw['unit_ok']=caunits.where(isca,phunits)
    raw['censored']=raw.raw_text.fillna('').str.contains(r'^\s*(?:[<>≤≥]|less\s+than|greater\s+than)',case=False,regex=True)
    raw['valid']=np.isfinite(raw.value)&raw.value.between(.1,3.).where(isca,raw.value.between(6.5,8.))&raw.unit_ok&~raw.censored
    rows=[]
    for (sid,t,assay),g in raw.groupby(['stay_id','event_minute','assay'],sort=True):
        latest=g.loc[g.available_minute.eq(g.available_minute.max())]
        specimen_ok=d=='eicu' or (g.specimen_id.notna().all() and g.specimen_id.nunique()==1)
        ok=bool(g.available_minute.notna().all() and specimen_ok and len(latest)>0 and latest.valid.all() and latest.value.nunique()==1)
        rows.append(dict(stay_id=sid,event_minute=t,assay=assay,qualified=ok,value=float(latest.value.iloc[0]) if ok else np.nan,
          specimen_id=int(g.specimen_id.iloc[0]) if d=='mimic' and specimen_ok else np.nan))
    quality=pd.DataFrame(rows);con.register('ca_quality',quality);con.execute(f'CREATE OR REPLACE TABLE {d}_quality AS SELECT * FROM ca_quality');con.unregister('ca_quality')
    unit_clause="us='mg/dl' AND ui='mg/dl'" if d=='eicu' else "us='mmol/l'"
    div=4.0078 if d=='eicu' else 1.
    sql=con.execute(f'''WITH r AS (SELECT *,lower(regexp_replace(coalesce(unit_system,''),'\\s','','g')) us,
      lower(regexp_replace(coalesce(unit_interface,''),'\\s','','g')) ui,
      CASE WHEN assay='calcium' THEN raw_value/{div} ELSE raw_value END AS value,
      max(available_minute) OVER(PARTITION BY stay_id,event_minute,assay) latest FROM {d}_raw),
      g AS (SELECT stay_id,event_minute,assay,count(*)=count(available_minute) times_ok,
      count(*)=count(specimen_id) AND count(DISTINCT specimen_id)=1 specimen_ok,
      bool_and(coalesce(isfinite(value) AND NOT regexp_matches(coalesce(raw_text,''),'^\\s*([<>≤≥]|less\\s+than|greater\\s+than)','i')
      AND CASE WHEN assay='calcium' THEN ({unit_clause}) AND value BETWEEN .1 AND 3.
      ELSE us IN ('','unit','units','ph','phunits') AND ui IN ('','unit','units','ph','phunits') AND value BETWEEN 6.5 AND 8. END,false)) FILTER(WHERE available_minute=latest) latest_ok,
      count(DISTINCT value) FILTER(WHERE available_minute=latest)=1 one_value,min(value) FILTER(WHERE available_minute=latest) AS value
      FROM r GROUP BY stay_id,event_minute,assay)
      SELECT stay_id,event_minute,assay,coalesce(times_ok AND {'true' if d=='eicu' else 'specimen_ok'} AND latest_ok AND one_value,false) qualified,value
      FROM g ORDER BY stay_id,event_minute,assay''').df()
    assert np.array_equal(quality[['stay_id','event_minute','assay','qualified']],sql[['stay_id','event_minute','assay','qualified']])
    assert np.allclose(quality.loc[quality.qualified,'value'],sql.loc[quality.qualified,'value'],rtol=0,atol=1e-12)
    con.execute(f'''CREATE OR REPLACE TABLE {d}_pairs AS SELECT e.stay_id,e.person_id,e.exposed,e.hospital_id,
      max(r.event_minute) FILTER(WHERE r.event_minute<0) baseline_minute,
      min(r.event_minute) FILTER(WHERE r.event_minute>240) followup_minute FROM {d}_selected e
      LEFT JOIN {d}_raw r ON r.stay_id=e.stay_id AND r.assay='calcium' GROUP BY e.stay_id,e.person_id,e.exposed,e.hospital_id''')
    pairs=con.execute(f'SELECT * FROM {d}_pairs ORDER BY stay_id').df()
    times=raw.loc[isca,['stay_id','event_minute']].drop_duplicates();pre=times.loc[times.event_minute.lt(0)].groupby('stay_id').event_minute.max();post=times.loc[times.event_minute.gt(240)].groupby('stay_id').event_minute.min()
    for r in pairs.itertuples():
        for obs,v in [(r.baseline_minute,pre.get(r.stay_id,np.nan)),(r.followup_minute,post.get(r.stay_id,np.nan))]:assert (pd.isna(obs) and pd.isna(v)) or abs(obs-v)<1e-12
    con.execute(f'''CREATE OR REPLACE TABLE {d}_panels AS SELECT a.stay_id,a.event_minute,a.value calcium,b.value ph FROM {d}_quality a
      JOIN {d}_quality b ON a.stay_id=b.stay_id AND a.event_minute=b.event_minute
      WHERE a.assay='calcium' AND b.assay='ph' AND a.qualified AND b.qualified {'AND a.specimen_id=b.specimen_id' if d=='mimic' else ''}''')
    con.execute(f'''CREATE OR REPLACE TABLE {d}_paired AS SELECT p.*,a.calcium baseline_calcium,b.calcium followup_calcium,a.ph baseline_ph,b.ph followup_ph FROM {d}_pairs p
      JOIN {d}_panels a ON p.stay_id=a.stay_id AND p.baseline_minute=a.event_minute
      JOIN {d}_panels b ON p.stay_id=b.stay_id AND p.followup_minute=b.event_minute''')
    panels=quality.loc[quality.assay.eq('calcium')&quality.qualified].merge(quality.loc[quality.assay.eq('ph')&quality.qualified],on=['stay_id','event_minute'],suffixes=('_ca','_ph'))
    if d=='mimic':panels=panels.loc[panels.specimen_id_ca.eq(panels.specimen_id_ph)]
    panels=panels[['stay_id','event_minute','value_ca','value_ph']]
    py=pairs.merge(panels.rename(columns={'event_minute':'baseline_minute','value_ca':'baseline_calcium','value_ph':'baseline_ph'}),on=['stay_id','baseline_minute']).merge(panels.rename(columns={'event_minute':'followup_minute','value_ca':'followup_calcium','value_ph':'followup_ph'}),on=['stay_id','followup_minute']).sort_values('stay_id')
    paired=con.execute(f'SELECT * FROM {d}_paired ORDER BY stay_id').df();cols=['baseline_calcium','followup_calcium','baseline_ph','followup_ph']
    assert np.array_equal(py.stay_id,paired.stay_id) and np.allclose(py[cols],paired[cols],rtol=0,atol=1e-12)
    for stage,p in [('raw_calcium',pairs.loc[pairs.baseline_minute.notna()&pairs.followup_minute.notna()]),('qualified_joint_panel',paired)]:
        ne=int(p.exposed.sum());nu=len(p)-ne;h=int(p.hospital_id.nunique());gate=len(p)>=50 and ne>=20 and nu>=20 and (d!='eicu' or (h>=10 and p.hospital_id.notna().all()))
        support.append(dict(dataset=d,stage=stage,people=len(p),exposed=ne,unexposed=nu,hospitals=h if d=='eicu' else None,gate=bool(gate)))
    for assay,g in raw.groupby('assay'):qc.append(dict(dataset=d,assay=assay,rows=len(g),unit_rejected_rows=int((~g.unit_ok).sum()),explicit_censor_rows=int(g.censored.sum())))
    checks.append(dict(dataset=d,sql_python_qualification=True,sql_python_fixed_calcium_timestamps=True,sql_python_panel_pairs_and_converted_values=True))
missing=con.execute('SELECT l.lab_id,l.subject_id,l.charttime,m.matches,m.assigned_hadm_id FROM mimic_candidates l JOIN missing_hospital m USING(lab_id)').df()
ad=con.execute('SELECT * FROM admissions WHERE subject_id IN (SELECT subject_id FROM mimic_candidates WHERE raw_hadm_id IS NULL)').df()
ad['py_start']=ad.admit_time.where(~(ad.ed_start.le(ad.admit_time)&(ad.ed_end.isna()|ad.ed_start.le(ad.ed_end))),ad.ed_start)
by={k:g for k,g in ad.groupby('subject_id')}
for r in missing.itertuples():
    a=by.get(r.subject_id,pd.DataFrame(columns=['py_start','end_time','hadm_id']));hit=a.loc[a.py_start.le(r.charttime)&a.end_time.gt(r.charttime)]
    assert len(hit)==r.matches
    if r.matches:assert hit.hadm_id.min()==r.assigned_hadm_id
joint_gate=all(r['gate'] for r in support if r['stage']=='qualified_joint_panel')
ca_json('support.json',support);ca_json('qualification_counts.json',qc);ca_json('qualification_validation.json',dict(checks=checks,independent_missing_hospital_checks=len(missing)))
ca_json('run_status.json',dict(status='running',stage='quality and fixed pairing validated',joint_gate=joint_gate))
print('Fixed paired calcium/pH support:',support)
"""

SCREEN = r"""
def ca_screen():
    if not joint_gate:
        r=dict(qualified_joint_gate=False,changes_computed=False,biological_discovery=False);ca_json('results.json',r)
        (PROJECT/'docs/SPO2_IONIZED_CALCIUM_RESULTS.md').write_text('# Paired ionized calcium / pH\n\nFixed qualified support fails; no exposure contrasts computed. Actual execution: '+('Google Colab' if IN_COLAB else 'local')+'.\n\n'+json.dumps(support,indent=2)+'\n')
        return r
    estimates=[];means=[];checks=[]
    for d in ['eicu','mimic']:
        p=con.execute(f'SELECT *,followup_calcium-baseline_calcium change_calcium,followup_ph-baseline_ph change_ph FROM {d}_paired ORDER BY stay_id').df()
        v=p[['change_calcium','change_ph']].to_numpy();exp=p.exposed.to_numpy(int)
        assert np.allclose(v,p[['followup_calcium','followup_ph']].to_numpy()-p[['baseline_calcium','baseline_ph']].to_numpy(),rtol=0,atol=1e-12)
        codes=pd.Categorical(p.hospital_id,categories=sorted(p.hospital_id.unique())).codes if d=='eicu' else np.arange(len(p));G=int(codes.max())+1
        totals=np.zeros((G,6))
        for arm in [0,1]:
            for j in [0,1]:np.add.at(totals[:,arm*3+j],codes[exp==arm],v[exp==arm,j])
            np.add.at(totals[:,arm*3+2],codes[exp==arm],1)
            means.append(dict(dataset=d,exposed=arm,people=int((exp==arm).sum()),mean_calcium_change=float(v[exp==arm,0].mean()),mean_ph_change=float(v[exp==arm,1].mean())))
        point=v[exp==1].mean(axis=0)-v[exp==0].mean(axis=0)
        rng=np.random.default_rng({'eicu':2026090565,'mimic':2026090566}[d]);draws=[];checked=0;max_diff=0.
        for start in range(0,20000,500):
            w=rng.multinomial(G,np.repeat(1/G,G),size=500);s=w@totals
            with np.errstate(divide='ignore',invalid='ignore'):b=s[:,3:5]/s[:,5:6]-s[:,:2]/s[:,2:3]
            for weights,expected in zip(w[:max(0,100-checked)],b[:max(0,100-checked)]):
                idx=np.repeat(np.arange(len(p)),weights[codes]);a=exp[idx];z=v[idx]
                direct=z[a==1].mean(axis=0)-z[a==0].mean(axis=0) if (a==0).any() and (a==1).any() else np.array([np.nan,np.nan])
                assert np.allclose(direct,expected,rtol=0,atol=1e-10,equal_nan=True)
                if np.isfinite(direct).all():max_diff=max(max_diff,float(np.max(abs(direct-expected))))
                checked+=1
            draws.append(b)
        draws=np.concatenate(draws);valid=bool(np.isfinite(draws).all());assert checked==100
        for j,assay in enumerate(['calcium','ph']):
            sqlpoint=con.execute(f'SELECT avg(followup_{assay}-baseline_{assay}) FILTER(WHERE exposed=1)-avg(followup_{assay}-baseline_{assay}) FILTER(WHERE exposed=0) FROM {d}_paired').fetchone()[0]
            assert abs(point[j]-sqlpoint)<1e-12
            row=dict(dataset=d,analyte=assay,contrast=float(point[j]),bootstrap_valid=valid)
            if valid:
                lo,hi,lc,hc=np.quantile(draws[:,j],[.025,.975,.00625,.99375]);row.update(lower95=float(lo),upper95=float(hi),lower_simultaneous=float(lc),upper_simultaneous=float(hc),advance=bool(hc<0 if assay=='calcium' else lc>0))
            else:row['advance']=False
            estimates.append(row)
        checks.append(dict(dataset=d,sql_python_changes_and_contrasts=True,draws=20000,directly_expanded_draws=checked,max_absolute_direct_difference=max_diff,resampling='hospital' if d=='eicu' else 'person'))
    result=dict(qualified_joint_gate=True,changes_computed=True,estimates=estimates,group_means=means,advancement_gate_pass=all(r['advance'] for r in estimates),biological_discovery=False,mortality_benefit_established=False)
    ca_json('results.json',result);ca_json('bootstrap_validation.json',checks)
    report=['# Paired ionized calcium / pH biochemical screen','','Actual execution: '+('Google Colab' if IN_COLAB else 'local')+'. Original exposure and fixed pre/follow-up samples.','',
      '| Dataset | Analyte | Change contrast | Nominal 95% CI | Simultaneous 98.75% CI |','|---|---|---:|---|---|']
    for r in estimates:
        if r['bootstrap_valid']:report.append(f"| {r['dataset']} | {r['analyte']} | {r['contrast']:.6f} | {r['lower95']:.6f} to {r['upper95']:.6f} | {r['lower_simultaneous']:.6f} to {r['upper_simultaneous']:.6f} |")
        else:report.append(f"| {r['dataset']} | {r['analyte']} | {r['contrast']:.6f} | Invalid bootstrap | Not promoted |")
    report+=['','Calcium units are mmol/L; pH units are dimensionless. Joint directional criterion: '+str(result['advancement_gate_pass'])+'.',
      'Unadjusted selected-patient associations; no claim of a respiratory mechanism, calcium binding/depletion, myocardial intracellular calcium, impaired contractility, causality, novelty, or mortality benefit. eICU exact-time pairing lacks specimen IDs; assay and selection limitations remain.','']
    (PROJECT/'docs/SPO2_IONIZED_CALCIUM_RESULTS.md').write_text('\n'.join(report));print('\n'.join(report))
    return result
result=ca_screen();assert ca_hash(origin)==origin_hash
ca_json('run_status.json',dict(status='complete',utc=datetime.now(timezone.utc).isoformat(),goal_complete=False))
con.close();print('Paired calcium/pH screen:',result)
"""
