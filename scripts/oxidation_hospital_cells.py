"""Separate eICU hospital replication of the fixed MetHb change question."""

SETUP = r"""
import json,hashlib,importlib.util
from pathlib import Path
from datetime import datetime,timezone
import duckdb,numpy as np,pandas as pd
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph';PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905'
OUT=PROJECT/'research/spo2_oxidation_hospitals';OUT.mkdir(parents=True,exist_ok=True)
def oh_json(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def oh_hash(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def oh_q(x):return "'"+str(x).replace("'","''")+"'"
assert oh_hash(PROJECT/'docs/SPO2_OXIDATION_HOSPITAL_PLAN.md')==OXIDATION_HOSPITAL_PROTOCOL_SHA256
lock=OUT/'protocol_lock.json'
if lock.exists():assert json.loads(lock.read_text())['sha256']==OXIDATION_HOSPITAL_PROTOCOL_SHA256
else:oh_json('protocol_lock.json',dict(sha256=OXIDATION_HOSPITAL_PROTOCOL_SHA256,utc=datetime.now(timezone.utc).isoformat(),timing_split_counts_known=True,before_numeric_qualification_or_changes=True))
original=json.loads((PROJECT/'research/spo2_hemoglobin_oxidation/results.json').read_text())
assert not original['raw_joint_gate'] and not original['changes_computed']
origin=PRIVATE/'hemoglobin_oxidation.duckdb';origin_hash=oh_hash(origin)
assert oh_hash(PROJECT/'docs/SPO2_HEMOGLOBIN_OXIDATION_PLAN.md')==json.loads((PROJECT/'research/spo2_hemoglobin_oxidation/protocol_lock.json').read_text())['sha256']
try:con.close()
except (NameError,AttributeError):pass
con=duckdb.connect(str(PRIVATE/'oxidation_hospitals.duckdb'));con.execute('SET threads=2');con.execute("SET memory_limit='2GB'")
con.execute(f'ATTACH {oh_q(origin)} AS origin (READ_ONLY)')
con.execute('''CREATE OR REPLACE TABLE selected AS SELECT *,
 (CAST(('0x'||substr(sha256('PhysioGraph oxidation hospital v1|'||cast(hospital_id AS BIGINT)::VARCHAR),1,2)) AS INTEGER)%2)::INTEGER split
 FROM origin.eicu_pairs WHERE assay='Methemoglobin' ''')
selected=con.execute('SELECT * FROM selected ORDER BY stay_id').df()
assert len(selected)==10167 and selected.person_id.nunique()==len(selected)
py=[int(hashlib.sha256(('PhysioGraph oxidation hospital v1|'+str(int(h))).encode()).hexdigest()[:2],16)%2 if pd.notna(h) else np.nan for h in selected.hospital_id]
assert np.array_equal(selected.split.to_numpy(),np.asarray(py),equal_nan=True)
assert selected.groupby('hospital_id').split.nunique().max()==1
oh_json('source_manifest.json',dict(database=str(origin),database_sha256=origin_hash,original_results=original,
 original_source_manifest=json.loads((PROJECT/'research/spo2_hemoglobin_oxidation/source_manifest.json').read_text()),environment='google_colab' if IN_COLAB else 'local'))
oh_json('run_status.json',dict(status='running',stage='hospital split locked before values'))
print('Separate eICU hospital protocol and independent hospital assignment verified.')
"""

QUALIFY = r"""
units={'%','%totalhgb','%thb','%ofhgb','%oftotalhb','%oftotal'}
def oh_token(x):return '' if pd.isna(x) else ''.join(str(x).casefold().split())
raw=con.execute("SELECT * FROM origin.eicu_raw WHERE assay='Methemoglobin'").df()
raw['unit_ok']=[oh_token(a) in units and oh_token(b) in units|{''} for a,b in zip(raw.unit_system,raw.unit_interface)]
raw['censored']=raw.raw_text.fillna('').str.contains(r'^\s*(?:[<>≤≥]|less\s+than|greater\s+than)',case=False,regex=True)
raw['value_ok']=raw.raw_value.notna()&np.isfinite(raw.raw_value)&raw.raw_value.between(0,100)&~raw.censored
rows=[]
for (sid,t),g in raw.groupby(['stay_id','event_minute'],sort=True):
    latest=g.loc[g.available_minute.eq(g.available_minute.max())]
    ok=bool(g.available_minute.notna().all() and len(latest)>0 and latest.unit_ok.all() and latest.value_ok.all() and latest.raw_value.nunique()==1)
    rows.append(dict(stay_id=sid,event_minute=t,qualified=ok,value=float(latest.raw_value.iloc[0]) if ok else np.nan))
quality=pd.DataFrame(rows)
con.register('oh_quality',quality);con.execute('CREATE OR REPLACE TABLE quality AS SELECT * FROM oh_quality');con.unregister('oh_quality')
independent=con.execute(r'''WITH r AS (SELECT *,
 lower(regexp_replace(coalesce(unit_system,''),'\s','','g')) us,
 lower(regexp_replace(coalesce(unit_interface,''),'\s','','g')) ui,
 max(available_minute) OVER(PARTITION BY stay_id,event_minute) latest FROM origin.eicu_raw WHERE assay='Methemoglobin'),
 g AS (SELECT stay_id,event_minute,count(*)=count(available_minute) times_ok,
 bool_and(coalesce(us IN ('%','%totalhgb','%thb','%ofhgb','%oftotalhb','%oftotal')
 AND ui IN ('','%','%totalhgb','%thb','%ofhgb','%oftotalhb','%oftotal')
 AND isfinite(raw_value) AND raw_value BETWEEN 0 AND 100
 AND NOT regexp_matches(coalesce(raw_text,''),'^\s*([<>≤≥]|less\s+than|greater\s+than)','i'),false)) FILTER(WHERE available_minute=latest) latest_ok,
 count(DISTINCT raw_value) FILTER(WHERE available_minute=latest)=1 one_value,
 min(raw_value) FILTER(WHERE available_minute=latest) AS value
 FROM r GROUP BY stay_id,event_minute)
 SELECT stay_id,event_minute,coalesce(times_ok AND latest_ok AND one_value,false) qualified,value FROM g ORDER BY stay_id,event_minute''').df()
assert np.array_equal(quality[['stay_id','event_minute','qualified']],independent[['stay_id','event_minute','qualified']])
assert np.allclose(quality.loc[quality.qualified,'value'],independent.loc[quality.qualified,'value'],rtol=0,atol=1e-12)
con.execute('''CREATE OR REPLACE TABLE paired AS SELECT s.*,a.value baseline_value,b.value followup_value FROM selected s
 JOIN quality a ON a.stay_id=s.stay_id AND a.event_minute=s.baseline_minute AND a.qualified
 JOIN quality b ON b.stay_id=s.stay_id AND b.event_minute=s.followup_minute AND b.qualified''')
paired=con.execute('SELECT * FROM paired ORDER BY stay_id').df()
py=selected.merge(quality.rename(columns={'event_minute':'baseline_minute','value':'baseline_value','qualified':'baseline_ok'}),on=['stay_id','baseline_minute']).merge(quality.rename(columns={'event_minute':'followup_minute','value':'followup_value','qualified':'followup_ok'}),on=['stay_id','followup_minute'])
py=py.loc[py.baseline_ok&py.followup_ok].sort_values('stay_id')
assert np.array_equal(py.stay_id,paired.stay_id) and np.allclose(py[['baseline_value','followup_value']],paired[['baseline_value','followup_value']],rtol=0,atol=1e-12)
support=[]
for stage,frame in [('raw',selected.loc[selected.baseline_minute.notna()&selected.followup_minute.notna()]),('qualified',paired)]:
    for split in [0,1]:
        p=frame.loc[frame.split.eq(split)];ne=int(p.exposed.sum());nu=len(p)-ne;h=p.hospital_id.nunique()
        support.append(dict(stage=stage,split=split,people=len(p),exposed=ne,unexposed=nu,hospitals=int(h),gate=bool(ne>=20 and nu>=20 and h>=10 and p.hospital_id.notna().all())))
joint_gate=all(r['gate'] for r in support if r['stage']=='qualified')
oh_json('support.json',support)
oh_json('qualification_counts.json',dict(raw_rows=len(raw),measurement_timestamps=len(quality),qualified_timestamps=int(quality.qualified.sum()),unit_rejected_rows=int((~raw.unit_ok).sum()),explicit_censor_rows=int(raw.censored.sum())))
oh_json('qualification_validation.json',dict(sql_python_hospital_split=True,people_unique=True,hospitals_disjoint=True,sql_python_quality_flags_and_values=True,sql_python_selected_pairs=True))
oh_json('run_status.json',dict(status='running',stage='qualification complete before exposure contrasts',qualified_joint_gate=joint_gate))
print('Hospital-split paired support:',support)
"""

SCREEN = r"""
def oh_screen():
    if not joint_gate:
        result=dict(qualified_joint_gate=False,changes_computed=False,biological_discovery=False)
        oh_json('results.json',result)
        (PROJECT/'docs/SPO2_OXIDATION_HOSPITAL_RESULTS.md').write_text('# eICU methemoglobin hospital replication\n\nQualified support fails the frozen gate; no exposure-specific changes computed. Actual execution: '+('Google Colab' if IN_COLAB else 'local')+'.\n')
        return result
    con.execute('CREATE OR REPLACE TABLE analysis AS SELECT *,followup_value-baseline_value AS change FROM paired')
    estimates=[];means=[];validation=[]
    for split in [0,1]:
        p=con.execute(f'SELECT * FROM analysis WHERE split={split} ORDER BY stay_id').df()
        assert np.allclose(p.change,p.followup_value-p.baseline_value,rtol=0,atol=1e-12)
        for arm,g in p.groupby('exposed'):
            means.append(dict(split=split,exposed=int(arm),people=len(g),mean_change_pp=float(g.change.mean()),mean_baseline_pct=float(g.baseline_value.mean()),mean_followup_pct=float(g.followup_value.mean())))
        point=float(p.loc[p.exposed.eq(1),'change'].mean()-p.loc[p.exposed.eq(0),'change'].mean())
        sql_point=con.execute(f'SELECT avg(change) FILTER(WHERE exposed=1)-avg(change) FILTER(WHERE exposed=0) FROM analysis WHERE split={split}').fetchone()[0]
        assert abs(point-sql_point)<1e-12
        groups=pd.Categorical(p.hospital_id,categories=sorted(p.hospital_id.unique()));codes=groups.codes;G=len(groups.categories)
        v=p.change.to_numpy();exp=p.exposed.to_numpy(int);totals=np.zeros((G,4))
        for arm in [0,1]:
            np.add.at(totals[:,arm*2],codes[exp==arm],v[exp==arm]);np.add.at(totals[:,arm*2+1],codes[exp==arm],1)
        rng=np.random.default_rng(2026090563+split);draws=[];checked=0;max_diff=0.
        for start in range(0,20000,500):
            w=rng.multinomial(G,np.repeat(1/G,G),size=500);s=w@totals
            with np.errstate(divide='ignore',invalid='ignore'):b=s[:,2]/s[:,3]-s[:,0]/s[:,1]
            for weights,expected in zip(w[:max(0,100-checked)],b[:max(0,100-checked)]):
                idx=np.repeat(np.arange(len(p)),weights[codes]);a=exp[idx];z=v[idx]
                direct=z[a==1].mean()-z[a==0].mean() if (a==0).any() and (a==1).any() else np.nan
                assert (np.isnan(direct) and np.isnan(expected)) or abs(direct-expected)<1e-10
                if np.isfinite(direct):max_diff=max(max_diff,abs(direct-expected))
                checked+=1
            draws.extend(b.tolist())
        draws=np.asarray(draws);valid=bool(np.isfinite(draws).all());assert checked==100
        entry=dict(split=split,people=len(p),hospitals=G,contrast_mean_change_pp=point,bootstrap_all_denominators_nonzero=valid)
        if valid:
            lo,hi,lc,hc=np.quantile(draws,[.025,.975,.0125,.9875]);entry.update(lower95=float(lo),upper95=float(hi),lower_simultaneous=float(lc),upper_simultaneous=float(hc),advance=bool(lc>0))
        else:entry['advance']=False
        estimates.append(entry);validation.append(dict(split=split,sql_python_contrast=True,draws=20000,directly_expanded_draws=checked,max_absolute_direct_difference=max_diff,resampling_unit='hospital'))
    result=dict(qualified_joint_gate=True,changes_computed=True,estimates=estimates,group_means=means,advancement_gate_pass=all(e['advance'] for e in estimates),biological_discovery=False,mortality_benefit_established=False)
    oh_json('results.json',result);oh_json('bootstrap_validation.json',validation)
    report=['# eICU methemoglobin change across disjoint hospitals','','Actual execution: '+('Google Colab' if IN_COLAB else 'local')+'. Separate subsequent protocol; original two-database timing gate remains failed.','',
      '| Hospital split | People | Hospitals | Change contrast, pp | Nominal 95% CI | Simultaneous 97.5% CI |','|---|---:|---:|---:|---|---|']
    for r in estimates:
        if r['bootstrap_all_denominators_nonzero']:report.append(f"| {r['split']} | {r['people']} | {r['hospitals']} | {r['contrast_mean_change_pp']:.6f} | {r['lower95']:.6f} to {r['upper95']:.6f} | {r['lower_simultaneous']:.6f} to {r['upper_simultaneous']:.6f} |")
        else:report.append(f"| {r['split']} | {r['people']} | {r['hospitals']} | {r['contrast_mean_change_pp']:.6f} | Bootstrap validity failed | Not promoted |")
    report+=['','Joint positive-screen criterion: '+str(result['advancement_gate_pass'])+'.',
      'Unadjusted association of recorded methemoglobin changes in selected, repeatedly tested HF patients. No inference of oxidation flux, causal injury, treatment response, novelty, or mortality benefit. Small hospital groups, selection, assay rounding, drugs, and transfusion remain limitations. MIMIC does not supply temporal replication.','']
    (PROJECT/'docs/SPO2_OXIDATION_HOSPITAL_RESULTS.md').write_text('\n'.join(report));print('\n'.join(report))
    return result
result=oh_screen()
assert oh_hash(origin)==origin_hash
oh_json('run_status.json',dict(status='complete',utc=datetime.now(timezone.utc).isoformat(),goal_complete=False))
con.close();print('eICU methemoglobin hospital screen:',result)
"""
