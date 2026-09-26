"""Temporal biochemical follow-up of original SpO2 instability."""

SETUP = r"""
import json,hashlib,importlib.util,csv,re
from pathlib import Path
from datetime import datetime,timezone
import duckdb,numpy as np,pandas as pd
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph';PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905'
OUT=PROJECT/'research/spo2_hemoglobin_oxidation';OUT.mkdir(parents=True,exist_ok=True)
def ho_json(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def ho_q(x):return "'"+str(x).replace("'","''")+"'"
def ho_hash(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
assert ho_hash(PROJECT/'docs/SPO2_HEMOGLOBIN_OXIDATION_PLAN.md')==OXIDATION_PROTOCOL_SHA256
lock=OUT/'protocol_lock.json'
if lock.exists():assert json.loads(lock.read_text())['sha256']==OXIDATION_PROTOCOL_SHA256
else:ho_json('protocol_lock.json',dict(sha256=OXIDATION_PROTOCOL_SHA256,utc=datetime.now(timezone.utc).isoformat(),before_patient_assay_counts=True,earlier_project_results_known=True))
try:con.close()
except (NameError,AttributeError):pass
con=duckdb.connect(str(PRIVATE/'hemoglobin_oxidation.duckdb'));con.execute('SET threads=2');con.execute("SET memory_limit='2GB'");con.execute('SET enable_progress_bar=false')
inputs={}
for alias,name in [('prior','spo2_context'),('care','spo2_cardiorenal'),('gas','spo2_acid_base')]:
    p=PRIVATE/(name+'.duckdb');inputs[alias]=dict(path=str(p),sha256=ho_hash(p))
    con.execute(f'ATTACH {ho_q(p)} AS {alias} (READ_ONLY)')
for d in ['eicu','mimic']:
    extra='g.admit_time,g.hadm_id,cast(g.person_id AS BIGINT) subject_id' if d=='mimic' else 'NULL::TIMESTAMP admit_time,NULL::BIGINT hadm_id,NULL::BIGINT subject_id'
    con.execute(f'''CREATE OR REPLACE TABLE {d}_original AS SELECT g.stay_id,g.exposed,g.hospital_start_minute,
      cast(c.person_id AS VARCHAR) person_id,c.followup_end_offset_minutes,p.hospital_id,{extra}
      FROM gas.{d}_eligible g JOIN care.{d}_cohort c USING(stay_id) JOIN prior.{d}_cohort p USING(stay_id)''')
    original=con.execute(f'SELECT * FROM {d}_original ORDER BY stay_id').df()
    assert len(original)=={'eicu':11452,'mimic':4711}[d]
    con.execute(f'''CREATE OR REPLACE TABLE {d}_selected AS SELECT * FROM {d}_original
      QUALIFY row_number() OVER(PARTITION BY person_id ORDER BY stay_id)=1''')
    selected=con.execute(f'SELECT * FROM {d}_selected ORDER BY stay_id').df()
    assert np.array_equal(selected.stay_id,original.groupby('person_id',sort=False).head(1).stay_id)
sources={}
for dataset,name in [('eICU','lab'),('MIMIC','labevents'),('MIMIC','admissions'),('MIMIC','d_labitems')]:
    p=DRIVE/'Data'/dataset/'Full'/(name+'.csv');s=p.stat()
    sources[dataset+'/'+name]=dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
with Path(sources['MIMIC/d_labitems']['path']).open() as f:items={int(r['itemid']):r for r in csv.DictReader(f)}
assert items[50814]['label']=='Methemoglobin' and items[50814]['category']=='Blood Gas' and items[50814]['fluid']=='Blood'
assert items[52144]['label']=='Methemoglobin' and items[52144]['category']=='Hematology'
ho_json('run_status.json',dict(status='running',stage='protocol and original-person selection locked',environment='google_colab' if IN_COLAB else 'local'))
print('Hemoglobin-oxidation protocol locked before new patient assay counts.')
"""

EXTRACT = r"""
def ho_csv(key):return f"read_csv({ho_q(sources[key]['path'])},header=true,all_varchar=true,sample_size=10000,strict_mode=true)"
con.execute(f'''CREATE OR REPLACE TABLE eicu_raw AS SELECT e.stay_id,try_cast(l.labid AS BIGINT) lab_id,
 NULL::BIGINT specimen_id,'Methemoglobin' assay,try_cast(l.labresultoffset AS DOUBLE) event_minute,
 try_cast(l.labresultrevisedoffset AS DOUBLE) available_minute,try_cast(l.labresult AS DOUBLE) raw_value,
 l.labresulttext raw_text,l.labmeasurenamesystem unit_system,l.labmeasurenameinterface unit_interface,
 NULL::VARCHAR raw_comment,'explicit' linkage
 FROM {ho_csv('eICU/lab')} l JOIN eicu_selected e ON try_cast(l.patientunitstayid AS BIGINT)=e.stay_id
 WHERE l.labname='Methemoglobin' AND try_cast(l.labtypeid AS INTEGER)=7 AND e.followup_end_offset_minutes IS NOT NULL
 AND try_cast(l.labresultoffset AS DOUBLE) BETWEEN greatest(-1440,e.hospital_start_minute) AND least(1680,e.followup_end_offset_minutes)''')
ho_json('run_status.json',dict(status='running',stage='eICU extracted; MIMIC raw assay scan'))
con.execute(f'''CREATE OR REPLACE TABLE admissions AS WITH a AS (
 SELECT try_cast(subject_id AS BIGINT) subject_id,try_cast(hadm_id AS BIGINT) hadm_id,
 try_cast(admittime AS TIMESTAMP) admit_time,try_cast(dischtime AS TIMESTAMP) end_time,
 try_cast(edregtime AS TIMESTAMP) ed_start,try_cast(edouttime AS TIMESTAMP) ed_end FROM {ho_csv('MIMIC/admissions')})
 SELECT *,CASE WHEN ed_start<=admit_time AND (ed_end IS NULL OR ed_start<=ed_end) THEN ed_start ELSE admit_time END start_time FROM a''')
assert con.execute('SELECT count(*)-count(DISTINCT hadm_id) FROM admissions').fetchone()[0]==0
con.execute(f'''CREATE OR REPLACE TABLE mimic_candidates AS SELECT e.stay_id,e.hadm_id expected_hadm_id,
 try_cast(l.labevent_id AS BIGINT) lab_id,try_cast(l.specimen_id AS BIGINT) specimen_id,
 try_cast(l.subject_id AS BIGINT) subject_id,try_cast(l.hadm_id AS BIGINT) raw_hadm_id,
 try_cast(l.charttime AS TIMESTAMP) charttime,cast(try_cast(l.itemid AS INTEGER) AS VARCHAR) assay,
 date_diff('second',e.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0 event_minute,
 date_diff('second',e.admit_time,try_cast(l.storetime AS TIMESTAMP))/60.0 available_minute,
 try_cast(l.valuenum AS DOUBLE) raw_value,l.value raw_text,l.valueuom unit_system,
 NULL::VARCHAR unit_interface,l.comments raw_comment
 FROM {ho_csv('MIMIC/labevents')} l JOIN mimic_selected e ON try_cast(l.subject_id AS BIGINT)=e.subject_id
 WHERE try_cast(l.itemid AS INTEGER) IN (50814,52144) AND e.followup_end_offset_minutes IS NOT NULL
 AND date_diff('second',e.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0
 BETWEEN greatest(-1440,e.hospital_start_minute) AND least(1680,e.followup_end_offset_minutes)''')
assert con.execute('SELECT count(*)-count(DISTINCT lab_id) FROM mimic_candidates').fetchone()[0]==0
con.execute('''CREATE OR REPLACE TABLE missing_hospital AS SELECT l.lab_id,count(a.hadm_id) matches,min(a.hadm_id) assigned_hadm_id
 FROM mimic_candidates l LEFT JOIN admissions a ON l.subject_id=a.subject_id AND l.charttime>=a.start_time AND l.charttime<a.end_time
 WHERE l.raw_hadm_id IS NULL GROUP BY l.lab_id''')
con.execute('''CREATE OR REPLACE TABLE mimic_raw AS SELECT l.stay_id,l.lab_id,l.specimen_id,l.assay,l.event_minute,l.available_minute,
 l.raw_value,l.raw_text,l.unit_system,l.unit_interface,l.raw_comment,
 CASE WHEN l.raw_hadm_id IS NOT NULL THEN 'explicit' ELSE 'unique_missing_hospital' END linkage
 FROM mimic_candidates l LEFT JOIN missing_hospital m USING(lab_id)
 WHERE l.raw_hadm_id=l.expected_hadm_id OR (l.raw_hadm_id IS NULL AND m.matches=1 AND m.assigned_hadm_id=l.expected_hadm_id)''')
for key,info in sources.items():
    p=Path(info['path']);s=p.stat();assert (s.st_size,s.st_mtime_ns)==(info['bytes'],info['mtime_ns'])
    info['sha256']=ho_hash(p);s=p.stat();assert (s.st_size,s.st_mtime_ns)==(info['bytes'],info['mtime_ns'])
ho_json('source_manifest.json',dict(raw=sources,inputs=inputs,full_raw_lab_scans=True,numeric_results_copied_privately=True,numeric_distributions_or_changes_queried=False,mortality_selected=False))
ho_json('run_status.json',dict(status='running',stage='raw assays cached privately; timing support pending'))
print('Both raw assay scans completed; no numeric distributions or changes queried.')
"""

TIMING = r"""
timing=[];primary_raw_gates=[];checks=[]
for d in ['eicu','mimic']:
    assays="('Methemoglobin')" if d=='eicu' else "('50814'),('52144')"
    selected=con.execute(f'SELECT stay_id,exposed FROM {d}_selected ORDER BY stay_id').df()
    bins=con.execute(f"SELECT b.stay_id,b.bin,b.value,b.representative_minute FROM prior.{d}_bins b JOIN {d}_selected e USING(stay_id) WHERE concept='spo2' ORDER BY stay_id,bin").df()
    dv=bins.value-bins.groupby('stay_id').value.shift();dt=bins.representative_minute-bins.groupby('stay_id').representative_minute.shift()
    assert set(bins.loc[dv.abs().ge(4)&dt.gt(0)&dt.le(30),'stay_id'])==set(selected.loc[selected.exposed.eq(1),'stay_id'])
    assert con.execute(f'SELECT count(*) FROM {d}_selected e JOIN care.{d}_cohort c USING(stay_id) WHERE e.exposed IS DISTINCT FROM c.exposed').fetchone()[0]==0
    con.execute(f'''CREATE OR REPLACE TABLE {d}_pairs AS SELECT e.stay_id,e.person_id,e.exposed,e.hospital_id,m.assay,
      max(r.event_minute) FILTER(WHERE r.event_minute<0) baseline_minute,
      min(r.event_minute) FILTER(WHERE r.event_minute>240) followup_minute
      FROM {d}_selected e CROSS JOIN (VALUES {assays}) m(assay)
      LEFT JOIN {d}_raw r ON r.stay_id=e.stay_id AND r.assay=m.assay
      GROUP BY e.stay_id,e.person_id,e.exposed,e.hospital_id,m.assay''')
    pairs=con.execute(f'SELECT * FROM {d}_pairs ORDER BY stay_id,assay').df()
    times=con.execute(f'SELECT DISTINCT stay_id,assay,event_minute FROM {d}_raw').df()
    pre=times.loc[times.event_minute.lt(0)].groupby(['stay_id','assay']).event_minute.max()
    post=times.loc[times.event_minute.gt(240)].groupby(['stay_id','assay']).event_minute.min()
    for r in pairs.itertuples():
        for observed,s in [(r.baseline_minute,pre),(r.followup_minute,post)]:
            expected=s.get((r.stay_id,r.assay),np.nan)
            assert (pd.isna(observed) and pd.isna(expected)) or abs(observed-expected)<=1e-12
    for assay,p in pairs.groupby('assay'):
        for stage,mask in [('selected',np.ones(len(p),dtype=bool)),('baseline',p.baseline_minute.notna()),('followup',p.followup_minute.notna()),('paired',p.baseline_minute.notna()&p.followup_minute.notna())]:
            v=p.loc[mask];n=len(v);ne=int(v.exposed.sum());nu=n-ne
            row=dict(dataset=d,assay=assay,stage=stage,people=n,exposed=ne,unexposed=nu)
            if stage=='paired':
                row['gate']=n>=100 and ne>=25 and nu>=25 and (d!='eicu' or (v.hospital_id.notna().all() and v.hospital_id.nunique()>=20))
                if assay=={'eicu':'Methemoglobin','mimic':'50814'}[d]:primary_raw_gates.append(bool(row['gate']))
            timing.append(row)
    checks.append(dict(dataset=d,lowest_original_person_selection_validated=True,pair_times_sql_python_agree=True,original_exposure_reconstructed_from_spo2_bins=True,exposure_matches_independent_care_cache=True))
missing=con.execute('''SELECT l.lab_id,l.subject_id,l.charttime,m.matches,m.assigned_hadm_id FROM mimic_candidates l JOIN missing_hospital m USING(lab_id)''').df()
ad=con.execute('SELECT * FROM admissions WHERE subject_id IN (SELECT subject_id FROM mimic_candidates WHERE raw_hadm_id IS NULL)').df()
ad['python_start']=ad.admit_time.where(~(ad.ed_start.le(ad.admit_time)&(ad.ed_end.isna()|ad.ed_start.le(ad.ed_end))),ad.ed_start)
assert (ad.python_start.eq(ad.start_time)|(ad.python_start.isna()&ad.start_time.isna())).all()
by_person={k:g for k,g in ad.groupby('subject_id')}
for r in missing.itertuples():
    a=by_person.get(r.subject_id,pd.DataFrame(columns=['python_start','end_time','hadm_id']))
    hit=a.loc[a.python_start.le(r.charttime)&a.end_time.gt(r.charttime)]
    assert len(hit)==r.matches
    if r.matches:assert hit.hadm_id.min()==r.assigned_hadm_id
assert len(primary_raw_gates)==2
ho_json('timing_support.json',timing)
ho_json('timing_validation.json',dict(checks=checks,independent_ed_boundary_and_missing_hospital_checks=len(missing)))
ho_json('run_status.json',dict(status='running',stage='timing support established before numeric qualification',raw_joint_gate=all(primary_raw_gates)))
print('Primary raw-timing gate:',all(primary_raw_gates));print('Paired timing support:',[r for r in timing if r['stage']=='paired'])
"""

SCREEN = r"""
def ho_screen():
    if not all(primary_raw_gates):
        result=dict(raw_joint_gate=False,qualified_joint_gate=False,changes_computed=False,biological_discovery=False)
        ho_json('results.json',result)
        (PROJECT/'docs/SPO2_HEMOGLOBIN_OXIDATION_RESULTS.md').write_text('# Hemoglobin oxidation timing support\n\nActual execution: '+('Google Colab' if IN_COLAB else 'local')+'. The primary raw timing floor fails; numeric qualification and exposure-specific changes were not computed.\n')
        return result
    accepted_units={'%','%totalhgb','%thb','%ofhgb','%oftotalhb','%oftotal'}
    def token(x):return '' if pd.isna(x) else ''.join(str(x).casefold().split())
    qualified={};qc_counts=[];validation=[];q_support=[]
    for d in ['eicu','mimic']:
        assay={'eicu':'Methemoglobin','mimic':'50814'}[d]
        raw=con.execute(f'SELECT * FROM {d}_raw WHERE assay={ho_q(assay)}').df()
        raw['unit_ok']=[token(a) in accepted_units and (d=='mimic' or token(b) in accepted_units|{''}) for a,b in zip(raw.unit_system,raw.unit_interface)]
        raw['censored']=raw.raw_text.fillna('').str.contains(r'^\s*(?:[<>≤≥]|less\s+than|greater\s+than)',case=False,regex=True)
        raw['value_ok']=raw.raw_value.notna()&np.isfinite(raw.raw_value)&raw.raw_value.between(0,100)&~raw.censored
        chosen=[]
        for (sid,t),g in raw.groupby(['stay_id','event_minute'],sort=True):
            all_times=g.available_minute.notna().all();latest=g.loc[g.available_minute.eq(g.available_minute.max())]
            specimen_ok=d=='eicu' or g.specimen_id.nunique()<=1
            ok=bool(all_times and specimen_ok and len(latest)>0 and latest.unit_ok.all() and latest.value_ok.all() and latest.raw_value.nunique()==1)
            chosen.append(dict(stay_id=sid,event_minute=t,qualified=ok,value=float(latest.raw_value.iloc[0]) if ok else np.nan))
        quality=pd.DataFrame(chosen)
        con.register('ho_quality',quality);con.execute(f'CREATE OR REPLACE TABLE {d}_quality AS SELECT * FROM ho_quality');con.unregister('ho_quality')
        sql_units="('%','%totalhgb','%thb','%ofhgb','%oftotalhb','%oftotal')"
        unit_clause=f"us IN {sql_units}"+(f" AND (ui='' OR ui IN {sql_units})" if d=='eicu' else '')
        independent=con.execute(f'''WITH r AS (SELECT *,
          lower(regexp_replace(coalesce(unit_system,''),'\\s','','g')) us,
          lower(regexp_replace(coalesce(unit_interface,''),'\\s','','g')) ui,
          max(available_minute) OVER(PARTITION BY stay_id,event_minute) latest FROM {d}_raw WHERE assay={ho_q(assay)}),
          g AS (SELECT stay_id,event_minute,count(*)=count(available_minute) times_ok,
          count(DISTINCT specimen_id)<=1 specimen_ok,
          bool_and(coalesce(({unit_clause}) AND isfinite(raw_value) AND raw_value BETWEEN 0 AND 100
          AND NOT regexp_matches(coalesce(raw_text,''),'^\\s*([<>≤≥]|less\\s+than|greater\\s+than)','i'),false)) FILTER(WHERE available_minute=latest) latest_ok,
          count(DISTINCT raw_value) FILTER(WHERE available_minute=latest)=1 one_value,
           min(raw_value) FILTER(WHERE available_minute=latest) AS value
          FROM r GROUP BY stay_id,event_minute)
          SELECT stay_id,event_minute,coalesce(times_ok AND {'true' if d=='eicu' else 'specimen_ok'} AND latest_ok AND one_value,false) qualified,value
          FROM g ORDER BY stay_id,event_minute''').df()
        assert np.array_equal(quality[['stay_id','event_minute','qualified']],independent[['stay_id','event_minute','qualified']])
        mask=quality.qualified;assert np.allclose(quality.loc[mask,'value'],independent.loc[mask,'value'],rtol=0,atol=1e-12)
        p=con.execute(f'''SELECT p.*,a.value baseline_value,b.value followup_value FROM {d}_pairs p
          JOIN {d}_quality a ON a.stay_id=p.stay_id AND a.event_minute=p.baseline_minute AND a.qualified
          JOIN {d}_quality b ON b.stay_id=p.stay_id AND b.event_minute=p.followup_minute AND b.qualified
          WHERE p.assay={ho_q(assay)} ORDER BY p.stay_id''').df()
        pairs=con.execute(f'SELECT * FROM {d}_pairs WHERE assay={ho_q(assay)}').df()
        py=pairs.merge(quality.rename(columns={'event_minute':'baseline_minute','value':'baseline_value','qualified':'baseline_ok'}),on=['stay_id','baseline_minute'],how='inner').merge(quality.rename(columns={'event_minute':'followup_minute','value':'followup_value','qualified':'followup_ok'}),on=['stay_id','followup_minute'],how='inner')
        py=py.loc[py.baseline_ok&py.followup_ok].sort_values('stay_id')
        assert np.array_equal(p.stay_id,py.stay_id) and np.allclose(p[['baseline_value','followup_value']],py[['baseline_value','followup_value']],rtol=0,atol=1e-12)
        n=len(p);ne=int(p.exposed.sum());nu=n-ne
        gate=n>=100 and ne>=25 and nu>=25 and (d!='eicu' or (p.hospital_id.notna().all() and p.hospital_id.nunique()>=20))
        q_support.append(dict(dataset=d,people=n,exposed=ne,unexposed=nu,hospitals=int(p.hospital_id.nunique()) if d=='eicu' else None,gate=bool(gate)))
        qualified[d]=p
        qc_counts.append(dict(dataset=d,raw_primary_rows=len(raw),measurement_timestamps=len(quality),qualified_timestamps=int(quality.qualified.sum()),unit_rejected_rows=int((~raw.unit_ok).sum()),explicit_censor_rows=int(raw.censored.sum())))
        validation.append(dict(dataset=d,independent_sql_python_unit_entry_value_flags=True,independent_qualified_pair_sets=True))
    ho_json('qualified_support.json',q_support);ho_json('qualification_counts.json',qc_counts);ho_json('qualification_validation.json',validation)
    if not all(r['gate'] for r in q_support):
        result=dict(raw_joint_gate=True,qualified_joint_gate=False,changes_computed=False,biological_discovery=False)
        ho_json('results.json',result)
        (PROJECT/'docs/SPO2_HEMOGLOBIN_OXIDATION_RESULTS.md').write_text('# Hemoglobin oxidation qualification\n\nActual execution: '+('Google Colab' if IN_COLAB else 'local')+'. Qualified paired support fails the frozen joint floor. No exposure-specific changes computed.\n\n'+json.dumps(q_support,indent=2)+'\n')
        return result
    estimates=[];bootstrap_checks=[];means=[]
    for d,p in qualified.items():
        p=p.copy();p['change']=p.followup_value-p.baseline_value
        con.register('ho_paired',p);con.execute(f'CREATE OR REPLACE TABLE {d}_analysis AS SELECT * FROM ho_paired');con.unregister('ho_paired')
        assert np.allclose(con.execute(f'SELECT followup_value-baseline_value AS delta FROM {d}_analysis ORDER BY stay_id').fetchnumpy()['delta'],p.change,rtol=0,atol=1e-12)
        for exp,g in p.groupby('exposed'):means.append(dict(dataset=d,exposed=int(exp),people=len(g),mean_change_pp=float(g.change.mean())))
        point=float(p.loc[p.exposed.eq(1),'change'].mean()-p.loc[p.exposed.eq(0),'change'].mean())
        sql_point=con.execute(f'SELECT avg(followup_value-baseline_value) FILTER(WHERE exposed=1)-avg(followup_value-baseline_value) FILTER(WHERE exposed=0) FROM {d}_analysis').fetchone()[0]
        assert abs(sql_point-point)<1e-12
        if d=='eicu':
            group=pd.Categorical(p.hospital_id,categories=sorted(p.hospital_id.unique()));codes=group.codes;G=len(group.categories)
        else:codes=np.arange(len(p));G=len(p)
        totals=np.zeros((G,4));v=p.change.to_numpy();exp=p.exposed.to_numpy(int)
        for arm in [0,1]:
            np.add.at(totals[:,arm*2],codes[exp==arm],v[exp==arm]);np.add.at(totals[:,arm*2+1],codes[exp==arm],1)
        rng=np.random.default_rng({'eicu':2026090561,'mimic':2026090562}[d]);draws=[];checked=0
        for start in range(0,20000,500):
            w=rng.multinomial(G,np.repeat(1/G,G),size=min(500,20000-start));s=w@totals
            with np.errstate(divide='ignore',invalid='ignore'):b=s[:,2]/s[:,3]-s[:,0]/s[:,1]
            for weights,expected in zip(w[:max(0,100-checked)],b[:max(0,100-checked)]):
                idx=np.repeat(np.arange(len(p)),weights[codes]);a=exp[idx];z=v[idx]
                direct=z[a==1].mean()-z[a==0].mean() if (a==0).any() and (a==1).any() else np.nan
                assert (np.isnan(direct) and np.isnan(expected)) or abs(direct-expected)<1e-10
                checked+=1
            draws.extend(b.tolist())
        draws=np.asarray(draws);valid=bool(np.isfinite(draws).all());assert checked==100
        entry=dict(dataset=d,contrast_mean_change_pp=point,bootstrap_all_denominators_nonzero=valid,finite_fraction=float(np.isfinite(draws).mean()))
        if valid:
            lo,hi=np.quantile(draws,[.025,.975]);lc,hc=np.quantile(draws,[.0125,.9875])
            entry.update(lower95=float(lo),upper95=float(hi),lower_simultaneous=float(lc),upper_simultaneous=float(hc),advance=bool(lc>0))
        else:entry['advance']=False
        estimates.append(entry);bootstrap_checks.append(dict(dataset=d,directly_reconstructed_draws=checked,draws=20000,resampling='hospital' if d=='eicu' else 'person'))
    result=dict(raw_joint_gate=True,qualified_joint_gate=True,changes_computed=True,group_means=means,estimates=estimates,
      advancement_gate_pass=all(r['advance'] for r in estimates),biological_discovery=False,mortality_benefit_established=False)
    ho_json('results.json',result);ho_json('bootstrap_validation.json',bootstrap_checks)
    report=['# Hemoglobin oxidation biochemical screen','', 'Actual execution: '+('Google Colab' if IN_COLAB else 'local')+'. Fixed pre-ICU/follow-up samples, original SpO2 exposure.','',
      '| Source | Exposed minus unexposed mean change, pp | Nominal 95% CI | Simultaneous 97.5% CI |','|---|---:|---|---|']
    for r in estimates:
        if r['bootstrap_all_denominators_nonzero']:report.append(f"| {r['dataset']} | {r['contrast_mean_change_pp']:.6f} | {r['lower95']:.6f} to {r['upper95']:.6f} | {r['lower_simultaneous']:.6f} to {r['upper_simultaneous']:.6f} |")
        else:report.append(f"| {r['dataset']} | {r['contrast_mean_change_pp']:.6f} | Bootstrap validity failed | Not promoted |")
    report+=['','Joint positive-screen criterion: '+str(result['advancement_gate_pass'])+'.',
      'This is an unadjusted biochemical-association screen of recorded numeric fractions. It does not identify ROS/NO flux, resolve drug/transfusion/testing or analyzer effects, establish causation, or show mortality benefit.','']
    (PROJECT/'docs/SPO2_HEMOGLOBIN_OXIDATION_RESULTS.md').write_text('\n'.join(report));print('\n'.join(report))
    return result
result=ho_screen()
for info in inputs.values():assert ho_hash(info['path'])==info['sha256']
ho_json('run_status.json',dict(status='complete',stage='frozen oxidation support/screen and validation',utc=datetime.now(timezone.utc).isoformat(),goal_complete=False))
con.close();print('Hemoglobin-oxidation stage result:',result)
"""
