"""Acute individual-level potassium/glucose metabolic response."""

SETUP = r"""
import json,hashlib,importlib.util,re
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal
import duckdb,numpy as np,pandas as pd
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph';PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905'
OUT=PROJECT/'research/spo2_adrenergic_metabolic';OUT.mkdir(parents=True,exist_ok=True)
def am_json(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def am_q(x):return "'"+str(x).replace("'","''")+"'"
def am_hash(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
assert am_hash(PROJECT/'docs/SPO2_ADRENERGIC_METABOLIC_PLAN.md')==ADRENERGIC_PROTOCOL_SHA256
lock=OUT/'protocol_lock.json'
if lock.exists():assert json.loads(lock.read_text())['sha256']==ADRENERGIC_PROTOCOL_SHA256
else:am_json('protocol_lock.json',dict(sha256=ADRENERGIC_PROTOCOL_SHA256,utc=datetime.now(timezone.utc).isoformat(),before_new_patient_potassium_glucose_counts_and_values=True,prior_project_results_known=True))
try:con.close()
except (NameError,AttributeError):pass
con=duckdb.connect(str(PRIVATE/'adrenergic_metabolic.duckdb'));con.execute('SET threads=2');con.execute("SET memory_limit='2GB'");con.execute('SET enable_progress_bar=false')
origin=PRIVATE/'hemoglobin_oxidation.duckdb';origin_hash=am_hash(origin)
manifest=json.loads((PROJECT/'research/spo2_hemoglobin_oxidation/source_manifest.json').read_text());prior=Path(manifest['inputs']['prior']['path']);prior_hash=am_hash(prior)
assert prior_hash==manifest['inputs']['prior']['sha256']
con.execute(f'ATTACH {am_q(origin)} AS origin (READ_ONLY)');con.execute(f'ATTACH {am_q(prior)} AS prior (READ_ONLY)')
con.execute('CREATE OR REPLACE TABLE admissions AS SELECT * FROM origin.admissions')
anchor_checks=[]
for d in ['eicu','mimic']:
    con.execute(f'''CREATE OR REPLACE TABLE {d}_cases AS WITH b AS (SELECT b.*,
      lag(value) OVER(PARTITION BY stay_id ORDER BY bin) pv,lag(representative_minute) OVER(PARTITION BY stay_id ORDER BY bin) pt
      FROM prior.{d}_bins b JOIN origin.{d}_selected s USING(stay_id) WHERE concept='spo2')
      SELECT stay_id,min_by(representative_minute,bin) anchor_minute FROM b
      WHERE abs(value-pv)>=4 AND representative_minute-pt>0 AND representative_minute-pt<=30 GROUP BY stay_id''')
    cases=con.execute(f'SELECT * FROM {d}_cases ORDER BY stay_id').df();n=len(cases)
    selected=con.execute(f'SELECT * FROM origin.{d}_selected ORDER BY stay_id').df()
    assert set(cases.stay_id)==set(selected.loc[selected.exposed.eq(1),'stay_id'])
    con.execute(f'''CREATE OR REPLACE TABLE {d}_anchors AS WITH a AS (SELECT *,row_number() OVER(ORDER BY stay_id)-1 idx FROM {d}_cases)
      SELECT s.*,CASE WHEN s.exposed=1 THEN actual.anchor_minute ELSE sham.anchor_minute END anchor_minute
      FROM origin.{d}_selected s LEFT JOIN {d}_cases actual USING(stay_id)
      LEFT JOIN a sham ON sham.idx=CAST(('0x'||substr(sha256('PhysioGraph adrenergic sham v1|{d}|'||s.person_id),1,8)) AS UBIGINT)%{n}''')
    anchors=con.execute(f'SELECT * FROM {d}_anchors ORDER BY stay_id').df()
    b=con.execute(f"SELECT b.stay_id,b.bin,b.value,b.representative_minute FROM prior.{d}_bins b JOIN origin.{d}_selected s USING(stay_id) WHERE concept='spo2' ORDER BY stay_id,bin").df()
    dv=b.value-b.groupby('stay_id').value.shift();dt=b.representative_minute-b.groupby('stay_id').representative_minute.shift()
    pycases=b.loc[dv.abs().ge(4)&dt.gt(0)&dt.le(30)].groupby('stay_id',sort=True).first().representative_minute
    assert np.array_equal(cases.stay_id,pycases.index) and np.allclose(cases.anchor_minute,pycases.to_numpy(),rtol=0,atol=1e-12)
    py=[float(pycases.loc[r.stay_id]) if r.exposed else float(pycases.iloc[int(hashlib.sha256(('PhysioGraph adrenergic sham v1|'+d+'|'+r.person_id).encode()).hexdigest()[:8],16)%n]) for r in selected.itertuples()]
    assert np.allclose(anchors.anchor_minute,py,rtol=0,atol=1e-12)
    assert len(anchors)=={'eicu':10167,'mimic':4305}[d] and anchors.person_id.nunique()==len(anchors)
    anchor_checks.append(dict(dataset=d,people=len(anchors),exposed=n,sql_python_original_exposure_and_anchor_times=True,sql_python_control_hash_assignment=True,one_original_stay_per_person=True))
sources={k:dict(v) for k,v in manifest['raw'].items()}
for info in sources.values():
    s=Path(info['path']).stat();assert (s.st_size,s.st_mtime_ns)==(info['bytes'],info['mtime_ns'])
def am_csv(key):return f"read_csv({am_q(sources[key]['path'])},header=true,all_varchar=true,sample_size=10000,strict_mode=true)"
am_json('anchor_validation.json',anchor_checks);am_json('run_status.json',dict(status='running',stage='protocol and original episode/sham anchors verified',environment='google_colab' if IN_COLAB else 'local'))
print('Acute metabolic protocol locked; episode and sham anchors independently verified.')
"""

EXTRACT = r"""
con.execute(f'''CREATE OR REPLACE TABLE eicu_raw AS SELECT a.stay_id,try_cast(l.labid AS BIGINT) lab_id,NULL::BIGINT specimen_id,
 l.labname assay,try_cast(l.labresultoffset AS DOUBLE) event_minute,try_cast(l.labresultrevisedoffset AS DOUBLE) available_minute,
 l.labresult numeric_text,l.labresulttext raw_text,l.labmeasurenamesystem unit_system,l.labmeasurenameinterface unit_interface,NULL::VARCHAR raw_comment
 FROM {am_csv('eICU/lab')} l JOIN eicu_anchors a ON try_cast(l.patientunitstayid AS BIGINT)=a.stay_id
 WHERE l.labname IN ('potassium','glucose') AND try_cast(l.labtypeid AS INTEGER)=1 AND a.followup_end_offset_minutes IS NOT NULL
 AND try_cast(l.labresultoffset AS DOUBLE) BETWEEN greatest(a.hospital_start_minute,a.anchor_minute-180) AND least(a.anchor_minute+120,a.followup_end_offset_minutes)''')
am_json('run_status.json',dict(status='running',stage='eICU raw extraction complete; MIMIC chemistry scan'))
con.execute(f'''CREATE OR REPLACE TABLE mimic_candidates AS SELECT a.stay_id,a.hadm_id expected_hadm_id,try_cast(l.labevent_id AS BIGINT) lab_id,
 try_cast(l.specimen_id AS BIGINT) specimen_id,try_cast(l.subject_id AS BIGINT) subject_id,try_cast(l.hadm_id AS BIGINT) raw_hadm_id,
 try_cast(l.charttime AS TIMESTAMP) charttime,CASE WHEN try_cast(l.itemid AS INTEGER)=50971 THEN 'potassium' ELSE 'glucose' END assay,
 date_diff('second',a.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0 event_minute,
 date_diff('second',a.admit_time,try_cast(l.storetime AS TIMESTAMP))/60.0 available_minute,
 l.valuenum numeric_text,l.value raw_text,l.valueuom unit_system,NULL::VARCHAR unit_interface,l.comments raw_comment
 FROM {am_csv('MIMIC/labevents')} l JOIN mimic_anchors a ON try_cast(l.subject_id AS BIGINT)=a.subject_id
 WHERE try_cast(l.itemid AS INTEGER) IN (50931,50971) AND a.followup_end_offset_minutes IS NOT NULL
 AND date_diff('second',a.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0 BETWEEN greatest(a.hospital_start_minute,a.anchor_minute-180) AND least(a.anchor_minute+120,a.followup_end_offset_minutes)''')
assert con.execute('SELECT count(*)-count(DISTINCT lab_id) FROM mimic_candidates').fetchone()[0]==0
con.execute('''CREATE OR REPLACE TABLE missing_hospital AS SELECT l.lab_id,count(a.hadm_id) matches,min(a.hadm_id) assigned_hadm_id
 FROM mimic_candidates l LEFT JOIN admissions a ON l.subject_id=a.subject_id AND l.charttime>=a.start_time AND l.charttime<a.end_time
 WHERE l.raw_hadm_id IS NULL GROUP BY l.lab_id''')
con.execute('''CREATE OR REPLACE TABLE mimic_raw AS SELECT l.stay_id,l.lab_id,l.specimen_id,l.assay,l.event_minute,l.available_minute,
 l.numeric_text,l.raw_text,l.unit_system,l.unit_interface,l.raw_comment FROM mimic_candidates l LEFT JOIN missing_hospital m USING(lab_id)
 WHERE l.raw_hadm_id=l.expected_hadm_id OR (l.raw_hadm_id IS NULL AND m.matches=1 AND m.assigned_hadm_id=l.expected_hadm_id)''')
for info in sources.values():
    p=Path(info['path']);s=p.stat();assert (s.st_size,s.st_mtime_ns)==(info['bytes'],info['mtime_ns']);assert am_hash(p)==info['sha256']
    s=p.stat();assert (s.st_size,s.st_mtime_ns)==(info['bytes'],info['mtime_ns'])
am_json('source_manifest.json',dict(raw=sources,origin_database=str(origin),origin_sha256=origin_hash,prior_database=str(prior),prior_sha256=prior_hash,full_raw_lab_scans=True,native_numeric_strings_copied_privately=True,mortality_selected=False))
am_json('run_status.json',dict(status='running',stage='raw chemistry and source hashes complete; qualification pending'))
print('Both raw chemistry scans completed; original-source hashes match.')
"""

QUALIFY = r"""
SCALE=100000000
DECIMAL_PATTERN=r'^[+-]?([0-9]+([.][0-9]{0,8}0*)?|[.][0-9]{1,8}0*)$'
def am_token(x):
    z='' if pd.isna(x) else ''.join(str(x).casefold().split())
    return 'mmol/l' if z=='mmoll' else z
def am_scaled(s):
    if pd.isna(s) or re.fullmatch(DECIMAL_PATTERN,str(s).strip()) is None:return None
    v=Decimal(str(s).strip());scaled=v*SCALE
    return int(scaled) if scaled==scaled.to_integral_value() and abs(scaled)<2**63 else None
support=[];checks=[];qc=[]
for d in ['eicu','mimic']:
    raw=con.execute(f'SELECT * FROM {d}_raw ORDER BY stay_id,event_minute,assay,lab_id').df()
    raw['us']=raw.unit_system.map(am_token);raw['ui']=raw.unit_interface.map(am_token)
    raw['scaled']=pd.array([am_scaled(v) for v in raw.numeric_text],dtype='Int64')
    raw['censored']=raw.raw_text.fillna('').str.contains(r'^\s*(?:[<>≤≥]|less\s+than|greater\s+than)',case=False,regex=True)
    raw['hemol']=raw.raw_text.fillna('').str.contains('hemol',case=False,regex=False)|raw.raw_comment.fillna('').str.contains('hemol',case=False,regex=False)
    isk=raw.assay.eq('potassium');kunits=raw.us.isin(['mmol/l','meq/l']);gunits=raw.us.eq('mg/dl')
    if d=='eicu':kunits &= raw.ui.isin(['mmol/l','meq/l']);gunits &= raw.ui.eq('mg/dl')
    raw['unit_ok']=kunits.where(isk,gunits)
    raw['valid']=(raw.scaled.between(SCALE,10*SCALE).where(isk,raw.scaled.between(10*SCALE,1500*SCALE))&raw.unit_ok&~raw.censored&~raw.hemol).fillna(False)
    rows=[]
    for (sid,t,assay),g in raw.groupby(['stay_id','event_minute','assay'],sort=True):
        latest=g.loc[g.available_minute.eq(g.available_minute.max())]
        specimen_ok=d=='eicu' or (g.specimen_id.notna().all() and g.specimen_id.nunique()==1)
        ok=bool(g.available_minute.notna().all() and len(g[['us','ui']].drop_duplicates())==1 and specimen_ok and len(latest)>0 and latest.valid.all() and latest.scaled.nunique()==1)
        rows.append(dict(stay_id=sid,event_minute=t,assay=assay,qualified=ok,scaled=int(latest.scaled.iloc[0]) if ok else None,
          specimen_id=int(g.specimen_id.iloc[0]) if d=='mimic' and specimen_ok else None))
    quality=pd.DataFrame(rows);quality['scaled']=pd.array(quality.scaled,dtype='Int64');quality['specimen_id']=pd.array(quality.specimen_id,dtype='Int64')
    con.register('am_quality',quality);con.execute(f'CREATE OR REPLACE TABLE {d}_quality AS SELECT * FROM am_quality');con.unregister('am_quality')
    kclause="us IN ('mmol/l','meq/l')"+(" AND ui IN ('mmol/l','meq/l')" if d=='eicu' else '')
    gclause="us='mg/dl'"+(" AND ui='mg/dl'" if d=='eicu' else '')
    sql=con.execute(f'''WITH r0 AS (SELECT *,lower(regexp_replace(coalesce(unit_system,''),'\\s','','g')) us0,lower(regexp_replace(coalesce(unit_interface,''),'\\s','','g')) ui0,
      CASE WHEN regexp_full_match(trim(numeric_text),{am_q(DECIMAL_PATTERN)}) THEN try_cast(try_cast(numeric_text AS DECIMAL(30,8))*100000000 AS BIGINT) END scaled,
      max(available_minute) OVER(PARTITION BY stay_id,event_minute,assay) latest FROM {d}_raw),
      r AS (SELECT *,CASE WHEN us0='mmoll' THEN 'mmol/l' ELSE us0 END us,CASE WHEN ui0='mmoll' THEN 'mmol/l' ELSE ui0 END ui FROM r0),
      g AS (SELECT stay_id,event_minute,assay,count(*)=count(available_minute) times_ok,count(DISTINCT (us,ui))=1 unit_unique,
      count(*)=count(specimen_id) AND count(DISTINCT specimen_id)=1 specimen_ok,
      bool_and(coalesce(NOT regexp_matches(coalesce(raw_text,''),'^\\s*([<>≤≥]|less\\s+than|greater\\s+than)','i')
      AND NOT contains(lower(coalesce(raw_text,'')||' '||coalesce(raw_comment,'')),'hemol')
      AND CASE WHEN assay='potassium' THEN ({kclause}) AND scaled BETWEEN 100000000 AND 1000000000
      ELSE ({gclause}) AND scaled BETWEEN 1000000000 AND 150000000000 END,false)) FILTER(WHERE available_minute=latest) latest_ok,
      count(DISTINCT scaled) FILTER(WHERE available_minute=latest)=1 one_value,min(scaled) FILTER(WHERE available_minute=latest) AS scaled
      FROM r GROUP BY stay_id,event_minute,assay)
      SELECT stay_id,event_minute,assay,coalesce(times_ok AND unit_unique AND {'true' if d=='eicu' else 'specimen_ok'} AND latest_ok AND one_value,false) qualified,scaled FROM g ORDER BY stay_id,event_minute,assay''').df()
    assert np.array_equal(quality[['stay_id','event_minute','assay','qualified']],sql[['stay_id','event_minute','assay','qualified']])
    m=quality.qualified;assert np.array_equal(quality.loc[m,'scaled'].to_numpy(dtype=np.int64),sql.loc[m,'scaled'].to_numpy(dtype=np.int64))
    con.execute(f'''CREATE OR REPLACE TABLE {d}_pairs AS SELECT a.stay_id,a.person_id,a.exposed,a.hospital_id,a.anchor_minute,
      max(r.event_minute) FILTER(WHERE r.event_minute<a.anchor_minute) baseline_minute,
      min(r.event_minute) FILTER(WHERE r.event_minute>a.anchor_minute) followup_minute
      FROM {d}_anchors a LEFT JOIN {d}_raw r ON a.stay_id=r.stay_id AND r.assay='potassium'
      GROUP BY a.stay_id,a.person_id,a.exposed,a.hospital_id,a.anchor_minute''')
    pairs=con.execute(f'SELECT * FROM {d}_pairs ORDER BY stay_id').df()
    times=raw.loc[isk,['stay_id','event_minute']].drop_duplicates().merge(pairs[['stay_id','anchor_minute']],on='stay_id')
    pre=times.loc[times.event_minute.lt(times.anchor_minute)].groupby('stay_id').event_minute.max();post=times.loc[times.event_minute.gt(times.anchor_minute)].groupby('stay_id').event_minute.min()
    for r in pairs.itertuples():
        for obs,v in [(r.baseline_minute,pre.get(r.stay_id,np.nan)),(r.followup_minute,post.get(r.stay_id,np.nan))]:assert (pd.isna(obs) and pd.isna(v)) or abs(obs-v)<1e-12
    con.execute(f'''CREATE OR REPLACE TABLE {d}_panels AS SELECT k.stay_id,k.event_minute,k.scaled potassium,g.scaled glucose FROM {d}_quality k
      JOIN {d}_quality g ON k.stay_id=g.stay_id AND k.event_minute=g.event_minute
      WHERE k.assay='potassium' AND g.assay='glucose' AND k.qualified AND g.qualified {'AND k.specimen_id=g.specimen_id' if d=='mimic' else ''}''')
    con.execute(f'''CREATE OR REPLACE TABLE {d}_paired AS SELECT p.*,a.potassium baseline_potassium,b.potassium followup_potassium,a.glucose baseline_glucose,b.glucose followup_glucose
      FROM {d}_pairs p JOIN {d}_panels a ON p.stay_id=a.stay_id AND p.baseline_minute=a.event_minute
      JOIN {d}_panels b ON p.stay_id=b.stay_id AND p.followup_minute=b.event_minute''')
    panel=quality.loc[quality.assay.eq('potassium')&quality.qualified].merge(quality.loc[quality.assay.eq('glucose')&quality.qualified],on=['stay_id','event_minute'],suffixes=('_k','_g'))
    if d=='mimic':panel=panel.loc[panel.specimen_id_k.eq(panel.specimen_id_g)]
    panel=panel[['stay_id','event_minute','scaled_k','scaled_g']]
    py=pairs.merge(panel.rename(columns={'event_minute':'baseline_minute','scaled_k':'baseline_potassium','scaled_g':'baseline_glucose'}),on=['stay_id','baseline_minute']).merge(panel.rename(columns={'event_minute':'followup_minute','scaled_k':'followup_potassium','scaled_g':'followup_glucose'}),on=['stay_id','followup_minute']).sort_values('stay_id')
    paired=con.execute(f'SELECT * FROM {d}_paired ORDER BY stay_id').df();cols=['baseline_potassium','followup_potassium','baseline_glucose','followup_glucose']
    assert np.array_equal(py.stay_id,paired.stay_id) and np.array_equal(py[cols].to_numpy(dtype=np.int64),paired[cols].to_numpy(dtype=np.int64))
    for stage,p in [('raw_potassium',pairs.loc[pairs.baseline_minute.notna()&pairs.followup_minute.notna()]),('qualified_joint_panel',paired)]:
        ne=int(p.exposed.sum());nu=len(p)-ne;h=int(p.hospital_id.nunique());gate=len(p)>=50 and ne>=20 and nu>=20 and (d!='eicu' or (h>=10 and p.hospital_id.notna().all()))
        support.append(dict(dataset=d,stage=stage,people=len(p),exposed=ne,unexposed=nu,hospitals=h if d=='eicu' else None,gate=bool(gate)))
    for assay,g in raw.groupby('assay'):qc.append(dict(dataset=d,assay=assay,rows=len(g),unit_rejected_rows=int((~g.unit_ok).sum()),decimal_rejected_rows=int(g.scaled.isna().sum()),explicit_censor_rows=int(g.censored.sum()),hemol_mentions=int(g.hemol.sum())))
    checks.append(dict(dataset=d,sql_python_entry_unit_decimal_flags_and_values=True,sql_python_fixed_potassium_timestamps=True,sql_python_joint_pairs_and_scaled_values=True))
missing=con.execute('SELECT l.lab_id,l.subject_id,l.charttime,m.matches,m.assigned_hadm_id FROM mimic_candidates l JOIN missing_hospital m USING(lab_id)').df()
ad=con.execute('SELECT * FROM admissions WHERE subject_id IN (SELECT subject_id FROM mimic_candidates WHERE raw_hadm_id IS NULL)').df()
ad['py_start']=ad.admit_time.where(~(ad.ed_start.le(ad.admit_time)&(ad.ed_end.isna()|ad.ed_start.le(ad.ed_end))),ad.ed_start);by={k:g for k,g in ad.groupby('subject_id')}
for r in missing.itertuples():
    a=by.get(r.subject_id,pd.DataFrame(columns=['py_start','end_time','hadm_id']));hit=a.loc[a.py_start.le(r.charttime)&a.end_time.gt(r.charttime)]
    assert len(hit)==r.matches
    if r.matches:assert hit.hadm_id.min()==r.assigned_hadm_id
joint_gate=all(r['gate'] for r in support if r['stage']=='qualified_joint_panel')
am_json('support.json',support);am_json('qualification_counts.json',qc);am_json('qualification_validation.json',dict(checks=checks,independent_missing_hospital_checks=len(missing)))
am_json('run_status.json',dict(status='running',stage='fixed paired support verified before event labels',joint_gate=joint_gate));print('Acute potassium/glucose support:',support)
"""

SCREEN = r"""
def am_screen():
    if not joint_gate:
        r=dict(joint_gate=False,events_computed=False,biological_discovery=False);am_json('results.json',r)
        (PROJECT/'docs/SPO2_ADRENERGIC_METABOLIC_RESULTS.md').write_text('# Acute potassium/glucose support\n\nFixed qualified support failed. No joint events or exposure contrasts computed. Actual execution: '+('Google Colab' if IN_COLAB else 'local')+'.\n\n'+json.dumps(support,indent=2)+'\n')
        return r
    estimates=[];counts=[];categories=[];descriptive=[];checks=[]
    for d in ['eicu','mimic']:
        con.execute(f'''CREATE OR REPLACE TABLE {d}_analysis AS SELECT *,
          baseline_potassium-followup_potassium>=50000000 potassium_fall,
          followup_glucose-baseline_glucose>=2000000000 glucose_absolute_rise,
          followup_glucose*5>=baseline_glucose*6 glucose_relative_rise,
          (baseline_potassium-followup_potassium>=50000000 AND followup_glucose-baseline_glucose>=2000000000 AND followup_glucose*5>=baseline_glucose*6)::INTEGER AS event
          FROM {d}_paired''')
        p=con.execute(f'SELECT * FROM {d}_analysis ORDER BY stay_id').df();exp=p.exposed.to_numpy(int);v=p.event.to_numpy(int)
        k=(p.baseline_potassium-p.followup_potassium).ge(50000000);ga=(p.followup_glucose-p.baseline_glucose).ge(2000000000);gr=(p.followup_glucose*5).ge(p.baseline_glucose*6)
        assert np.array_equal(v,(k&ga&gr).to_numpy(int))
        assert np.array_equal(p[['potassium_fall','glucose_absolute_rise','glucose_relative_rise']],np.column_stack([k,ga,gr]))
        risks={};event_gate=True
        for arm,g in p.groupby('exposed'):
            events=int(g.event.sum());non=len(g)-events;h=int(g.loc[g.event.eq(1),'hospital_id'].nunique());gate=events>=5 and non>=5 and (d!='eicu' or h>=3);event_gate &= gate
            risks[int(arm)]=events/len(g);counts.append(dict(dataset=d,exposed=int(arm),people=len(g),events=events,non_events=non,event_hospitals=h if d=='eicu' else None,risk=events/len(g),event_gate=bool(gate)))
            dk=(g.followup_potassium-g.baseline_potassium)/SCALE;dg=(g.followup_glucose-g.baseline_glucose)/SCALE
            descriptive.append(dict(dataset=d,exposed=int(arm),mean_potassium_change=float(dk.mean()),median_potassium_change=float(dk.median()),mean_glucose_change=float(dg.mean()),median_glucose_change=float(dg.median()),median_sample_spacing_minutes=float((g.followup_minute-g.baseline_minute).median())))
            for kk in [False,True]:
                for aa in [False,True]:
                    for rr in [False,True]:categories.append(dict(dataset=d,exposed=int(arm),potassium_fall=kk,glucose_absolute_rise=aa,glucose_relative_rise=rr,people=int((g.potassium_fall.eq(kk)&g.glucose_absolute_rise.eq(aa)&g.glucose_relative_rise.eq(rr)).sum())))
        point=risks[1]-risks[0];sqlpoint=con.execute(f'SELECT avg(event) FILTER(WHERE exposed=1)-avg(event) FILTER(WHERE exposed=0) FROM {d}_analysis').fetchone()[0];assert abs(point-sqlpoint)<1e-12
        entry=dict(dataset=d,risk_difference=point,risk_ratio=risks[1]/risks[0] if risks[0]>0 else None,event_gate=bool(event_gate),advance=False)
        check=dict(dataset=d,sql_python_exact_joint_labels=True,sql_python_risk_difference=True,bootstrap_run=False)
        if event_gate:
            codes=pd.Categorical(p.hospital_id,categories=sorted(p.hospital_id.unique())).codes if d=='eicu' else np.arange(len(p));G=int(codes.max())+1;totals=np.zeros((G,4))
            for arm in [0,1]:np.add.at(totals[:,2*arm],codes[exp==arm],v[exp==arm]);np.add.at(totals[:,2*arm+1],codes[exp==arm],1)
            rng=np.random.default_rng({'eicu':2026090569,'mimic':2026090570}[d]);draws=[];checked=0;max_diff=0.
            for start in range(0,20000,500):
                w=rng.multinomial(G,np.repeat(1/G,G),size=500);s=w@totals
                with np.errstate(divide='ignore',invalid='ignore'):b=s[:,2]/s[:,3]-s[:,0]/s[:,1]
                for weights,expected in zip(w[:max(0,100-checked)],b[:max(0,100-checked)]):
                    idx=np.repeat(np.arange(len(p)),weights[codes]);a=exp[idx];z=v[idx];direct=z[a==1].mean()-z[a==0].mean() if (a==0).any() and (a==1).any() else np.nan
                    assert np.isclose(direct,expected,rtol=0,atol=1e-12,equal_nan=True)
                    if np.isfinite(direct):max_diff=max(max_diff,abs(direct-expected))
                    checked+=1
                draws.extend(b.tolist())
            draws=np.asarray(draws);valid=bool(np.isfinite(draws).all());assert checked==100;entry['bootstrap_valid']=valid
            if valid:
                lo,hi=np.quantile(draws,[.025,.975]);entry.update(lower95=float(lo),upper95=float(hi),advance=bool(lo>0))
            check.update(bootstrap_run=True,draws=20000,directly_expanded_draws=checked,max_absolute_direct_difference=max_diff,resampling='hospital' if d=='eicu' else 'person')
        estimates.append(entry);checks.append(check)
    result=dict(joint_gate=True,events_computed=True,arm_counts=counts,estimates=estimates,advancement_gate_pass=all(r['advance'] for r in estimates),biological_discovery=False,mortality_benefit_established=False)
    am_json('results.json',result);am_json('joint_categories.json',categories);am_json('descriptive_changes.json',descriptive);am_json('event_bootstrap_validation.json',checks)
    report=['# Acute coupled potassium fall and glucose rise','','Actual execution: '+('Google Colab' if IN_COLAB else 'local')+'. Fixed original episode/sham anchors, first post-episode sample within two hours.','',
      '| Dataset | Arm | People | Joint events | Risk |','|---|---|---:|---:|---:|']
    for r in counts:report.append(f"| {r['dataset']} | {r['exposed']} | {r['people']} | {r['events']} | {100*r['risk']:.3f}% |")
    report+=['','| Dataset | Risk difference, pp | 95% interval, pp |','|---|---:|---|']
    for r in estimates:
        interval=f"{100*r['lower95']:.3f} to {100*r['upper95']:.3f}" if r.get('bootstrap_valid') else 'Event support or bootstrap validity insufficient'
        report.append(f"| {r['dataset']} | {100*r['risk_difference']:.3f} | {interval} |")
    report+=['','Joint cross-database advancement: '+str(result['advancement_gate_pass'])+'.',
      'The event is an investigator-defined coupled recorded change, not a specific assay of endogenous catecholamines or a diagnosis of hypokalemia. Treatment, hemolysis, testing and severity remain alternatives. No causal, novelty or mortality-benefit claim follows from this screen. Marginal bootstrap intervals are approximate and not simultaneous family intervals.','']
    (PROJECT/'docs/SPO2_ADRENERGIC_METABOLIC_RESULTS.md').write_text('\n'.join(report));print('\n'.join(report));return result
result=am_screen();assert am_hash(origin)==origin_hash and am_hash(prior)==prior_hash
am_json('run_status.json',dict(status='complete',utc=datetime.now(timezone.utc).isoformat(),goal_complete=False));con.close();print('Acute metabolic screen:',result)
"""
