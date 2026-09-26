"""Frozen episode-anchored coupled lactate/arterial CO2/pH experiment."""

SETUP = r"""
import json,hashlib,importlib.util,re,csv
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal,getcontext
import duckdb,numpy as np,pandas as pd
getcontext().prec=50
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph';PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905'
OUT=PROJECT/'research/spo2_lactate_co2';OUT.mkdir(parents=True,exist_ok=True)
def lc_json(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def lc_q(x):return "'"+str(x).replace("'","''")+"'"
def lc_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for z in iter(lambda:f.read(8*1024*1024),b''):h.update(z)
    return h.hexdigest()
assert lc_hash(PROJECT/'docs/SPO2_LACTATE_CO2_PLAN.md')==LACTATE_CO2_PROTOCOL_SHA256
lock=OUT/'protocol_lock.json'
if lock.exists():assert json.loads(lock.read_text())['sha256']==LACTATE_CO2_PROTOCOL_SHA256
else:lc_json('protocol_lock.json',dict(sha256=LACTATE_CO2_PROTOCOL_SHA256,utc=datetime.now(timezone.utc).isoformat(),before_new_joint_counts_and_changes=True,prior_project_results_known=True))
try:con.close()
except (NameError,AttributeError):pass
con=duckdb.connect(str(PRIVATE/'lactate_co2.duckdb'));con.execute('SET threads=2');con.execute("SET memory_limit='2GB'");con.execute('SET enable_progress_bar=false')
inputs={}
for alias,name in [('origin','adrenergic_metabolic'),('prior','spo2_context')]:
    p=PRIVATE/(name+'.duckdb');inputs[alias]=dict(path=str(p),sha256=lc_hash(p));con.execute(f'ATTACH {lc_q(p)} AS {alias} (READ_ONLY)')
con.execute('CREATE OR REPLACE TABLE admissions AS SELECT * FROM origin.admissions')
checks=[]
for d in ['eicu','mimic']:
    con.execute(f'CREATE OR REPLACE TABLE {d}_anchors AS SELECT * FROM origin.{d}_anchors')
    a=con.execute(f'SELECT stay_id,person_id,exposed,anchor_minute FROM {d}_anchors ORDER BY stay_id').df()
    assert len(a)=={'eicu':10167,'mimic':4305}[d] and a.person_id.nunique()==len(a)
    b=con.execute(f"SELECT b.stay_id,b.bin,b.value,b.representative_minute FROM prior.{d}_bins b JOIN {d}_anchors a USING(stay_id) WHERE concept='spo2' ORDER BY stay_id,bin").df()
    delta=b.value-b.groupby('stay_id').value.shift();gap=b.representative_minute-b.groupby('stay_id').representative_minute.shift()
    first=b.loc[delta.abs().ge(4)&gap.gt(0)&gap.le(30)].groupby('stay_id',sort=True).first().representative_minute
    assert set(first.index)==set(a.loc[a.exposed.eq(1),'stay_id'])
    expected=[float(first.loc[r.stay_id]) if r.exposed else float(first.iloc[int(hashlib.sha256(('PhysioGraph adrenergic sham v1|'+d+'|'+r.person_id).encode()).hexdigest()[:8],16)%len(first)]) for r in a.itertuples()]
    assert np.allclose(a.anchor_minute,expected,rtol=0,atol=1e-12)
    checks.append(dict(dataset=d,people=len(a),original_exposure_and_episode_times_reproduced=True,prior_control_hash_reproduced=True))
sources=json.loads((PROJECT/'research/spo2_adrenergic_metabolic/source_manifest.json').read_text())['raw']
for info in sources.values():
    s=Path(info['path']).stat();assert (s.st_size,s.st_mtime_ns)==(info['bytes'],info['mtime_ns'])
dictionary=DRIVE/'Data/MIMIC/Full/d_labitems.csv'
with dictionary.open() as f:lookup={int(r['itemid']):r for r in csv.DictReader(f)}
for i,label in [(50813,'Lactate'),(50818,'pCO2'),(50820,'pH'),(52033,'Specimen Type')]:
    assert lookup[i]['label']==label and lookup[i]['fluid']=='Blood' and lookup[i]['category']=='Blood Gas'
def lc_csv(key):return f"read_csv({lc_q(sources[key]['path'])},header=true,all_varchar=true,sample_size=10000,quote='\"',escape='\"',strict_mode=true)"
lc_json('anchor_validation.json',checks)
lc_json('run_status.json',dict(status='running',stage='protocol and original episode/control anchors locked',environment='google_colab' if IN_COLAB else 'local'))
lc_json('input_manifest.json',dict(inputs=inputs,raw_before_scan=sources,dictionary_sha256=lc_hash(dictionary)))
print('Lactate/CO2/pH protocol locked; original episode and control anchors independently reproduced.')
"""

EXTRACT = r"""
con.execute(f'''CREATE OR REPLACE TABLE eicu_raw AS SELECT a.stay_id,try_cast(l.labid AS BIGINT) lab_id,
 NULL::BIGINT specimen_id,CASE l.labname WHEN 'paCO2' THEN 'pco2' WHEN 'pH' THEN 'ph' ELSE 'lactate' END concept,
 try_cast(l.labresultoffset AS DOUBLE) event_minute,try_cast(l.labresultrevisedoffset AS DOUBLE) available_minute,
 l.labresult numeric_text,l.labresulttext raw_text,l.labmeasurenamesystem unit_system,l.labmeasurenameinterface unit_interface
 FROM {lc_csv('eICU/lab')} l JOIN eicu_anchors a ON try_cast(l.patientunitstayid AS BIGINT)=a.stay_id
 WHERE ((l.labname IN ('pH','paCO2') AND try_cast(l.labtypeid AS INTEGER)=7) OR
 (l.labname='lactate' AND try_cast(l.labtypeid AS INTEGER) IN (1,7)))
 AND a.followup_end_offset_minutes IS NOT NULL AND try_cast(l.labresultoffset AS DOUBLE)
 BETWEEN greatest(a.hospital_start_minute,a.anchor_minute-180) AND least(a.anchor_minute+120,a.followup_end_offset_minutes)''')
lc_json('run_status.json',dict(status='running',stage='eICU extraction committed; scanning MIMIC blood-gas source'))
con.execute(f'''CREATE OR REPLACE TABLE mimic_candidates AS SELECT a.stay_id,a.hadm_id expected_hadm_id,
 try_cast(l.labevent_id AS BIGINT) lab_id,try_cast(l.specimen_id AS BIGINT) specimen_id,
 try_cast(l.subject_id AS BIGINT) subject_id,try_cast(l.hadm_id AS BIGINT) raw_hadm_id,try_cast(l.charttime AS TIMESTAMP) charttime,
 CASE try_cast(l.itemid AS INTEGER) WHEN 50813 THEN 'lactate' WHEN 50818 THEN 'pco2' WHEN 50820 THEN 'ph' ELSE 'specimen' END concept,
 date_diff('second',a.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0 event_minute,
 date_diff('second',a.admit_time,try_cast(l.storetime AS TIMESTAMP))/60.0 available_minute,
 l.valuenum numeric_text,l.value raw_text,l.valueuom unit_system,NULL::VARCHAR unit_interface
 FROM {lc_csv('MIMIC/labevents')} l JOIN mimic_anchors a ON try_cast(l.subject_id AS BIGINT)=a.subject_id
 WHERE try_cast(l.itemid AS INTEGER) IN (50813,50818,50820,52033)
 AND a.followup_end_offset_minutes IS NOT NULL AND date_diff('second',a.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0
 BETWEEN greatest(a.hospital_start_minute,a.anchor_minute-180) AND least(a.anchor_minute+120,a.followup_end_offset_minutes)''')
assert con.execute('SELECT count(*)-count(DISTINCT lab_id) FROM mimic_candidates').fetchone()[0]==0
con.execute('''CREATE OR REPLACE TABLE missing_hospital AS SELECT l.lab_id,count(a.hadm_id) matches,min(a.hadm_id) assigned_hadm_id
 FROM mimic_candidates l LEFT JOIN admissions a ON l.subject_id=a.subject_id AND l.charttime>=a.start_time AND l.charttime<a.end_time
 WHERE l.raw_hadm_id IS NULL GROUP BY l.lab_id''')
con.execute('''CREATE OR REPLACE TABLE mimic_raw AS SELECT l.stay_id,l.lab_id,l.specimen_id,l.concept,l.event_minute,l.available_minute,
 l.numeric_text,l.raw_text,l.unit_system,l.unit_interface FROM mimic_candidates l LEFT JOIN missing_hospital m USING(lab_id)
 WHERE l.raw_hadm_id=l.expected_hadm_id OR (l.raw_hadm_id IS NULL AND m.matches=1 AND m.assigned_hadm_id=l.expected_hadm_id)''')
lc_json('run_status.json',dict(status='running',stage='both raw extractions committed; full native source hash verification'))
print('Both raw blood-gas extractions committed. No joint change or outcome analysis yet.')
"""

VERIFY = r"""
for info in sources.values():
    p=Path(info['path']);s=p.stat();assert (s.st_size,s.st_mtime_ns)==(info['bytes'],info['mtime_ns']);assert lc_hash(p)==info['sha256']
    s=p.stat();assert (s.st_size,s.st_mtime_ns)==(info['bytes'],info['mtime_ns'])
for info in inputs.values():assert lc_hash(info['path'])==info['sha256']
missing=con.execute('SELECT l.lab_id,l.subject_id,l.charttime,m.matches,m.assigned_hadm_id FROM mimic_candidates l JOIN missing_hospital m USING(lab_id) ORDER BY l.lab_id').df()
ad=con.execute('SELECT subject_id,hadm_id,start_time,end_time FROM admissions').df()
groups={s:g for s,g in ad.groupby('subject_id')}
for r in missing.itertuples():
    g=groups.get(r.subject_id,ad.iloc[:0]);hit=g.loc[g.start_time.le(r.charttime)&g.end_time.gt(r.charttime)]
    assert len(hit)==r.matches
    if len(hit):assert int(hit.hadm_id.min())==r.assigned_hadm_id
    else:assert pd.isna(r.assigned_hadm_id)
lc_json('source_manifest.json',dict(inputs=inputs,raw=sources,full_raw_lab_scans=True,native_hashes_match_previous_independent_extractions=True,
  dictionary_sha256=lc_hash(dictionary),explicit_csv_quoting=True,mortality_selected=False,missing_hospital_candidates_independently_validated=len(missing)))
lc_json('run_status.json',dict(status='running',stage='raw hashes and hospital linkage verified; fixed panels and numerical qualification'))
print('Full native hashes and all missing-hospital assignments independently verified.')
"""

QUALIFY = r"""
DECIMAL_PATTERN=r'^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)$'
def lc_token(x):
    z='' if pd.isna(x) else ''.join(str(x).casefold().split());return 'mmol/l' if z=='mmoll' else z
def lc_value(r):
    us=lc_token(r.unit_system);ui=lc_token(r.unit_interface)
    if r.concept=='ph':unit_ok=us in ('','ph','units') and ui in ('','ph','units');factor=Decimal(1)
    elif r.concept=='pco2':
        used=[u for u in [us,ui] if u];unit_ok=bool(used) and len(set(used))==1 and used[0] in ('mmhg','kpa')
        factor=Decimal('7.50061683') if unit_ok and used[0]=='kpa' else Decimal(1)
    else:unit_ok=us=='mmol/l' and (r.dataset=='mimic' or ui=='mmol/l');factor=Decimal(1)
    text='' if pd.isna(r.numeric_text) else str(r.numeric_text).strip()
    numeric_ok=re.fullmatch(DECIMAL_PATTERN,text) is not None
    censored=bool(re.search(r'^\s*(?:[<>≤≥]|less\s+than|greater\s+than)','' if pd.isna(r.raw_text) else str(r.raw_text),re.I))
    v=Decimal(text)*factor if numeric_ok else None
    lo,hi={'ph':('6.5','8.5'),'pco2':('5','250'),'lactate':('0','40')}[r.concept]
    range_ok=v is not None and Decimal(lo)<=v<=Decimal(hi)
    return dict(us=us,ui=ui,normalized=str(v) if v is not None else None,unit_ok=unit_ok,numeric_ok=numeric_ok,censored=censored,range_ok=range_ok,valid=unit_ok and numeric_ok and not censored and range_ok)
support=[];quality_checks=[];qc=[]
for d in ['eicu','mimic']:
    raw=con.execute(f'SELECT * FROM {d}_raw ORDER BY stay_id,event_minute,concept,lab_id').df()
    if d=='mimic':
        types=raw.loc[raw.concept.eq('specimen')&raw.specimen_id.notna()].copy();types['label']=types.raw_text.fillna('').str.strip().str.upper()
        accepted={k for k,g in types.groupby(['stay_id','specimen_id']) if g.label.nunique()==1 and g.label.iloc[0] in ('ART.','ART','ARTERIAL')}
        con.execute('''CREATE OR REPLACE TABLE mimic_arterial_specimens AS SELECT stay_id,specimen_id FROM mimic_raw
          WHERE concept='specimen' AND specimen_id IS NOT NULL GROUP BY stay_id,specimen_id
          HAVING count(DISTINCT upper(trim(coalesce(raw_text,''))))=1 AND min(upper(trim(coalesce(raw_text,'')))) IN ('ART.','ART','ARTERIAL')''')
        sqltypes=set(con.execute('SELECT stay_id,specimen_id FROM mimic_arterial_specimens').fetchall());assert accepted==sqltypes
        r=raw.loc[raw.concept.ne('specimen')].copy()
        r=r.loc[[(sid,sp) in accepted for sid,sp in zip(r.stay_id,r.specimen_id)]].copy()
        source_sql='SELECT r.* FROM mimic_raw r JOIN mimic_arterial_specimens a USING(stay_id,specimen_id) WHERE concept!=\'specimen\''
    else:r=raw.copy();source_sql='SELECT * FROM eicu_raw'
    con.execute(f'CREATE OR REPLACE TABLE {d}_source AS '+source_sql)
    con.execute(f'''CREATE OR REPLACE TABLE {d}_fixed AS SELECT a.stay_id,a.person_id,a.exposed,a.hospital_id,
      max(r.event_minute) FILTER(WHERE r.event_minute<a.anchor_minute) pre_minute,
      min(r.event_minute) FILTER(WHERE r.event_minute>a.anchor_minute) post_minute
      FROM {d}_anchors a LEFT JOIN {d}_source r ON r.stay_id=a.stay_id AND r.concept='pco2'
      GROUP BY a.stay_id,a.person_id,a.exposed,a.hospital_id''')
    fixed=con.execute(f'SELECT * FROM {d}_fixed ORDER BY stay_id').df()
    anchors=con.execute(f'SELECT stay_id,anchor_minute FROM {d}_anchors').df().set_index('stay_id').anchor_minute.to_dict()
    tg={s:g.event_minute.to_numpy() for s,g in r.loc[r.concept.eq('pco2')].groupby('stay_id')}
    for z in fixed.itertuples():
        t=tg.get(z.stay_id,np.array([]));a=anchors[z.stay_id]
        pre=t[t<a];post=t[t>a]
        assert (pd.isna(z.pre_minute) and not len(pre)) or (len(pre)>0 and z.pre_minute==pre.max())
        assert (pd.isna(z.post_minute) and not len(post)) or (len(post)>0 and z.post_minute==post.min())
    r['dataset']=d
    details=pd.DataFrame([lc_value(z) for z in r.itertuples()],index=r.index)
    r=pd.concat([r,details],axis=1)
    con.register('lc_normalized',r);con.execute(f'CREATE OR REPLACE TABLE {d}_normalized AS SELECT * FROM lc_normalized');con.unregister('lc_normalized')
    sql=con.execute(f'''WITH b AS (SELECT *,lower(regexp_replace(coalesce(unit_system,''),'\\s','','g')) us0,
      lower(regexp_replace(coalesce(unit_interface,''),'\\s','','g')) ui0 FROM {d}_source), u AS (
      SELECT *,CASE WHEN us0='mmoll' THEN 'mmol/l' ELSE us0 END us,CASE WHEN ui0='mmoll' THEN 'mmol/l' ELSE ui0 END ui FROM b),
      v AS (SELECT *,CASE WHEN regexp_full_match(coalesce(trim(numeric_text),''),{lc_q(DECIMAL_PATTERN)}) THEN try_cast(numeric_text AS DECIMAL(38,16)) END
      *CASE WHEN concept='pco2' AND coalesce(nullif(us,''),ui)='kpa' THEN 7.50061683 ELSE 1 END value
      FROM u)
      SELECT lab_id,CASE WHEN concept='ph' THEN us IN ('','ph','units') AND ui IN ('','ph','units')
      WHEN concept='pco2' THEN (us IN ('mmhg','kpa') OR ui IN ('mmhg','kpa')) AND (us='' OR ui='' OR us=ui)
      ELSE us='mmol/l' AND {'true' if d=='mimic' else "ui='mmol/l'"} END unit_ok,
      regexp_full_match(coalesce(trim(numeric_text),''),{lc_q(DECIMAL_PATTERN)}) numeric_ok,
      regexp_matches(coalesce(raw_text,''),'^\\s*([<>≤≥]|less\\s+than|greater\\s+than)','i') censored,
      coalesce(CASE concept WHEN 'ph' THEN value BETWEEN 6.5 AND 8.5 WHEN 'pco2' THEN value BETWEEN 5 AND 250 ELSE value BETWEEN 0 AND 40 END,false) range_ok
      FROM v ORDER BY lab_id''').df()
    ordered=r.sort_values('lab_id')
    assert np.array_equal(ordered[['lab_id','unit_ok','numeric_ok','censored','range_ok']],sql[['lab_id','unit_ok','numeric_ok','censored','range_ok']])
    comp={};component_rows=[]
    for key,g in r.groupby(['stay_id','event_minute','concept']):
        latest=g.loc[g.available_minute.eq(g.available_minute.max())]
        spok=d=='eicu' or (g.specimen_id.notna().all() and g.specimen_id.nunique()==1)
        values={Decimal(x) for x in latest.normalized.dropna()}
        ok=g.available_minute.notna().all() and len(g[['us','ui']].drop_duplicates())==1 and spok and len(latest)>0 and latest.valid.all() and len(values)==1
        value=next(iter(values)) if ok else None;sp=int(g.specimen_id.iloc[0]) if d=='mimic' and spok else None
        comp[key]=(value,sp)
        component_rows.append(dict(stay_id=key[0],event_minute=key[1],concept=key[2],qualified=bool(ok),normalized=str(value) if value is not None else None,specimen_id=sp))
    cr=pd.DataFrame(component_rows);cr['specimen_id']=pd.array(cr.specimen_id,dtype='Int64')
    con.register('lc_comp',cr);con.execute(f'CREATE OR REPLACE TABLE {d}_components AS SELECT * FROM lc_comp');con.unregister('lc_comp')
    con.register('lc_sql_row_flags',sql)
    independent=con.execute(f'''WITH r AS (SELECT s.*,f.unit_ok AND f.numeric_ok AND NOT f.censored AND f.range_ok valid,
      lower(regexp_replace(coalesce(s.unit_system,''),'\\s','','g')) us,lower(regexp_replace(coalesce(s.unit_interface,''),'\\s','','g')) ui,
      try_cast(s.numeric_text AS DECIMAL(38,16)) native_value,
      max(s.available_minute) OVER(PARTITION BY s.stay_id,s.event_minute,s.concept) latest
      FROM {d}_source s JOIN lc_sql_row_flags f USING(lab_id))
      SELECT stay_id,event_minute,concept,coalesce(count(*)=count(available_minute)
      AND count(DISTINCT(CASE WHEN us='mmoll' THEN 'mmol/l' ELSE us END,CASE WHEN ui='mmoll' THEN 'mmol/l' ELSE ui END))=1
      AND {'true' if d=='eicu' else 'count(*)=count(specimen_id) AND count(DISTINCT specimen_id)=1'}
      AND bool_and(valid) FILTER(WHERE available_minute=latest)
      AND count(DISTINCT native_value) FILTER(WHERE available_minute=latest)=1,false) qualified
      FROM r GROUP BY stay_id,event_minute,concept ORDER BY stay_id,event_minute,concept''').df()
    con.unregister('lc_sql_row_flags')
    assert np.array_equal(cr[['stay_id','event_minute','concept','qualified']],independent[['stay_id','event_minute','concept','qualified']])
    rows=[]
    for z in fixed.itertuples():
        row=dict(stay_id=z.stay_id,person_id=z.person_id,exposed=int(z.exposed),hospital_id=z.hospital_id,
          pre_minute=z.pre_minute,post_minute=z.post_minute,qualified=True)
        for side,t in [('pre',z.pre_minute),('post',z.post_minute)]:
            specs=[]
            for c in ['lactate','pco2','ph']:
                v,sp=comp.get((z.stay_id,t,c),(None,None));row[side+'_'+c]=str(v) if v is not None else None;specs.append(sp)
                row['qualified'] &= v is not None
            if d=='mimic':row['qualified'] &= None not in specs and len(set(specs))==1
        rows.append(row)
    panels=pd.DataFrame(rows);con.register('lc_panels',panels);con.execute(f'CREATE OR REPLACE TABLE {d}_panels AS SELECT * FROM lc_panels');con.unregister('lc_panels')
    # Independent SQL fixed-time joins reconstruct qualification, including specimen equality.
    joins=' '.join(f"LEFT JOIN {d}_components {side}_{c} ON {side}_{c}.stay_id=f.stay_id AND {side}_{c}.event_minute=f.{side}_minute AND {side}_{c}.concept='{c}'" for side in ['pre','post'] for c in ['lactate','pco2','ph'])
    conditions=' AND '.join(f'{side}_{c}.qualified' for side in ['pre','post'] for c in ['lactate','pco2','ph'])
    if d=='mimic':conditions+=' AND '+' AND '.join(f'{side}_lactate.specimen_id={side}_{c}.specimen_id' for side in ['pre','post'] for c in ['ph','pco2'])
    flags=con.execute(f'SELECT f.stay_id,coalesce(({conditions}),false) qualified FROM {d}_fixed f {joins} ORDER BY f.stay_id').df()
    assert np.array_equal(panels[['stay_id','qualified']],flags[['stay_id','qualified']])
    for stage,mask in [('source_eligible_co2_pairs',fixed.pre_minute.notna()&fixed.post_minute.notna()),('qualified_joint_panels',panels.qualified)]:
        p=panels.loc[mask];ne=int(p.exposed.sum());nu=len(p)-ne;h=int(p.hospital_id.nunique()) if d=='eicu' else None
        support.append(dict(dataset=d,stage=stage,people=len(p),exposed=ne,control=nu,hospitals=h,panel_support_pass=len(p)>=50 and ne>=20 and nu>=20 and (d=='mimic' or h>=10)))
    for c,g in r.groupby('concept'):qc.append(dict(dataset=d,concept=c,source_rows=len(g),unit_rejected=int((~g.unit_ok).sum()),nondecimal=int((~g.numeric_ok).sum()),censored=int(g.censored.sum()),range_rejected=int((~g.range_ok).sum())))
    quality_checks.append(dict(dataset=d,specimen_labels_sql_python_agree=True,fixed_sample_times_sql_python_agree=True,row_quality_flags_sql_python_agree=True,latest_component_qualification_sql_python_agree=True,fixed_panel_qualification_sql_python_agree=True))
lc_json('support.json',support);lc_json('quality_validation.json',quality_checks);lc_json('quality_counts.json',qc)
lc_json('run_status.json',dict(status='running',stage='fixed numerical panels independently qualified; joint gate pending'))
print('Fixed three-component panels qualified; no replacement sample times.')
"""

SCREEN = r"""
gates=[r['panel_support_pass'] for r in support if r['stage']=='qualified_joint_panels'];assert len(gates)==2
results=[];checks=[]
if all(gates):
    for d in ['eicu','mimic']:
        p=con.execute(f'SELECT * FROM {d}_panels WHERE qualified ORDER BY stay_id').df()
        for side in ['pre','post']:
            for c in ['lactate','pco2','ph']:p[side+'_'+c]=p[side+'_'+c].map(Decimal)
        p['event']=[int(r.post_lactate-r.pre_lactate>=Decimal('.5') and r.post_pco2-r.pre_pco2>=Decimal('5') and r.post_ph-r.pre_ph<=Decimal('-.03')) for r in p.itertuples()]
        # Exact SQL DECIMAL is adequate only if every qualified numerical string is exactly representable.
        for side in ['pre','post']:
            for c in ['lactate','pco2','ph']:
                for v in p[side+'_'+c]:assert v==v.quantize(Decimal('0.0000000000000001'))
        sql=con.execute(f'''SELECT stay_id,(try_cast(post_lactate AS DECIMAL(38,16))-try_cast(pre_lactate AS DECIMAL(38,16))>=0.5
          AND try_cast(post_pco2 AS DECIMAL(38,16))-try_cast(pre_pco2 AS DECIMAL(38,16))>=5
          AND try_cast(post_ph AS DECIMAL(38,16))-try_cast(pre_ph AS DECIMAL(38,16))<=-0.03)::INTEGER AS event
          FROM {d}_panels WHERE qualified ORDER BY stay_id''').df()
        assert np.array_equal(p[['stay_id','event']],sql[['stay_id','event']])
        con.register('lc_events',p[['stay_id','exposed','hospital_id','event']]);con.execute(f'CREATE OR REPLACE TABLE {d}_events AS SELECT * FROM lc_events');con.unregister('lc_events')
        arms=[]
        for e in [0,1]:
            a=p.loc[p.exposed.eq(e)];ev=int(a.event.sum());he=int(a.loc[a.event.eq(1),'hospital_id'].nunique()) if d=='eicu' else None
            arms.append(dict(exposed=e,people=len(a),events=ev,nonevents=len(a)-ev,event_hospitals=he))
        rd=arms[1]['events']/arms[1]['people']-arms[0]['events']/arms[0]['people']
        event_gate=all(a['events']>=5 and a['nonevents']>=5 and (d=='mimic' or a['event_hospitals']>=3) for a in arms)
        r=dict(dataset=d,arms=arms,risk_difference_pp=100*rd,event_support_pass=event_gate,bootstrap_run=False,ci95_pp=None,advancement_pass=False)
        # Both databases must have event support before either uncertainty calculation.
        results.append(r);checks.append(dict(dataset=d,decimal_python_sql_joint_events_agree=True))
    if all(r['event_support_pass'] for r in results):
        for r in results:
            d=r['dataset'];p=con.execute(f'SELECT * FROM {d}_events ORDER BY stay_id').df();cluster='hospital_id' if d=='eicu' else 'stay_id'
            rows=[];parts=[]
            for _,g in p.groupby(cluster,sort=True):
                rows.append([int(g.exposed.eq(1).sum()),int(g.loc[g.exposed.eq(1),'event'].sum()),int(g.exposed.eq(0).sum()),int(g.loc[g.exposed.eq(0),'event'].sum())]);parts.append(g)
            a=np.asarray(rows);rng=np.random.default_rng(202609061 if d=='eicu' else 202609062);draws=[]
            for k in range(20000):
                idx=rng.integers(0,len(a),len(a));total=a[idx].sum(axis=0);assert total[0]>0 and total[2]>0
                rd=total[1]/total[0]-total[3]/total[2];draws.append(rd)
                if k<100:
                    expanded=pd.concat([parts[i] for i in idx]);direct=expanded.loc[expanded.exposed.eq(1),'event'].mean()-expanded.loc[expanded.exposed.eq(0),'event'].mean()
                    assert abs(rd-direct)<=1e-12
            ci=np.quantile(draws,[.025,.975])*100;r.update(bootstrap_run=True,ci95_pp=ci.tolist(),advancement_pass=bool(ci[0]>0))
            checks.append(dict(dataset=d,bootstrap_draws=20000,direct_expansion_checks=100,all_draws_retain_both_groups=True))
lc_json('event_validation.json',checks)
lc_json('results.json',dict(joint_panel_support_pass=all(gates),results=results,joint_advancement_pass=len(results)==2 and all(r['advancement_pass'] for r in results),biological_discovery=False,mortality_models=0,
  interpretation='Unadjusted coupled biochemical change screen; does not diagnose inadequate physiological compensation, endogenous mechanism or treatment benefit.'))
report=['# Coupled lactate, arterial CO2 and pH response','',
 'Actual execution: '+('Google Colab' if IN_COLAB else 'local')+'. Full raw laboratory scans with original episode/control anchors.','',
 '| Dataset | Qualified people | Exposed | Controls | eICU hospitals | Panel support |','|---|---:|---:|---:|---:|---|']
for r in support:
    if r['stage']=='qualified_joint_panels':report.append(f"| {r['dataset']} | {r['people']} | {r['exposed']} | {r['control']} | {r['hospitals']} | {r['panel_support_pass']} |")
report+=['','Primary event: lactate increase ≥0.5 mmol/L, pCO2 increase ≥5 mmHg AND pH decrease ≥0.03 at the fixed panels.']
if results:
    report+=['','| Dataset | Exposed events | Control events | Risk difference, pp | Event support | 95% interval, pp |','|---|---|---|---:|---|---|']
    for r in results:
        c,e=r['arms'];report.append(f"| {r['dataset']} | {e['events']}/{e['people']} | {c['events']}/{c['people']} | {r['risk_difference_pp']:.6f} | {r['event_support_pass']} | {r['ci95_pp']} |")
else:report+=['','The joint panel-support gate failed; no numerical changes, event labels or effect estimates were calculated.']
report+=['','No mortality or treatment-effect model was fitted. This coupled event is not a validated syndrome and does not locate biochemical onset or distinguish treatment from endogenous physiology. eICU co-timing lacks MIMIC-equivalent specimen linkage and does not prove arterial lactate.',
 'Original/control anchors, full native hashes, missing-hospital assignments, fixed sample times and numerical qualification were independently verified. No replacement times, altered thresholds or component-endpoint rescue was used. The biological-discovery goal remains unfulfilled.','']
(PROJECT/'docs/SPO2_LACTATE_CO2_RESULTS.md').write_text('\n'.join(report))
lc_json('run_status.json',dict(status='complete',stage='fixed coupled biochemical screen and independent validation',utc=datetime.now(timezone.utc).isoformat(),goal_complete=False))
con.close();print('\n'.join(report))
"""
