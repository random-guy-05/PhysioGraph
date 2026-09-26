"""First original SpO2 transition, contemporaneous HR direction and mortality."""

SETUP = r'''
import hashlib,importlib.util,json
from pathlib import Path
from datetime import datetime,timezone
import duckdb
import numpy as np
import pandas as pd
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph';PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905'
OUT=PROJECT/'research/spo2_heart_rate_polarity';OUT.mkdir(parents=True,exist_ok=True)
def hj(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def hh(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def hq(x):return "'"+str(x).replace("'","''")+"'"
def hs(stage,status='running'):hj('run_status.json',dict(stage=stage,status=status,utc=datetime.now(timezone.utc).isoformat(),goal_complete=False))
assert hh(PROJECT/'docs/SPO2_HEART_RATE_POLARITY_PLAN.md')==HEART_RATE_POLARITY_PROTOCOL_SHA256
if (OUT/'protocol_lock.json').exists():assert json.loads((OUT/'protocol_lock.json').read_text())['sha256']==HEART_RATE_POLARITY_PROTOCOL_SHA256
else:hj('protocol_lock.json',dict(sha256=HEART_RATE_POLARITY_PROTOCOL_SHA256,utc=datetime.now(timezone.utc).isoformat(),before_joint_phenotype_and_mortality_counts=True,prior_project_results_known=True))
try:con.close()
except (NameError,AttributeError):pass
con=duckdb.connect(str(PRIVATE/'heart_rate_polarity.duckdb'));con.execute('SET threads=2')
refs={}
for alias,name in [('prior','spo2_context'),('care','spo2_cardiorenal')]:
    p=PRIVATE/(name+'.duckdb');refs[name]=dict(path=str(p),sha256=hh(p));con.execute(f'ATTACH {hq(p)} AS {alias} (READ_ONLY)')
manifest=dict(protocol_sha256=HEART_RATE_POLARITY_PROTOCOL_SHA256,references=refs)
if (OUT/'input_manifest.json').exists():assert json.loads((OUT/'input_manifest.json').read_text())==manifest
hj('input_manifest.json',manifest);hs('locked HR-direction experiment')
'''

PHENOTYPE = r'''
hs('first-transition phenotype before mortality')
support=[]
for d in ['eicu','mimic']:
    cohort=con.execute(f'SELECT stay_id,person_id,exposed FROM care.{d}_cohort').df()
    assert len(cohort)=={'eicu':11452,'mimic':4711}[d]
    con.execute(f"""CREATE OR REPLACE TABLE {d}_first AS WITH lagged AS (
      SELECT b.stay_id,b.bin post_bin,b.value post_spo2,b.representative_minute post_minute,
        lag(b.bin) OVER w pre_bin,lag(b.value) OVER w pre_spo2,lag(b.representative_minute) OVER w pre_minute
      FROM prior.{d}_bins b JOIN care.{d}_cohort c USING(stay_id) WHERE b.concept='spo2'
      WINDOW w AS(PARTITION BY b.stay_id ORDER BY b.bin))
      SELECT l.*,c.person_id FROM lagged l JOIN care.{d}_cohort c USING(stay_id)
      WHERE abs(post_spo2-pre_spo2)>=4 AND post_minute-pre_minute>0 AND post_minute-pre_minute<=30
      QUALIFY row_number() OVER(PARTITION BY stay_id ORDER BY post_minute)=1""")
    first=con.execute(f'SELECT * FROM {d}_first ORDER BY stay_id').df()
    assert set(first.stay_id)==set(cohort.loc[cohort.exposed.eq(1),'stay_id'])
    con.execute(f"""CREATE OR REPLACE TABLE {d}_selected AS SELECT * FROM {d}_first
      QUALIFY row_number() OVER(PARTITION BY person_id ORDER BY stay_id)=1""")
    con.execute(f"""CREATE OR REPLACE TABLE {d}_phenotype AS WITH joined AS (
      SELECT s.*,h0.value pre_hr,h1.value post_hr,h0.representative_minute pre_hr_minute,h1.representative_minute post_hr_minute,
      h0.value IS NOT NULL AND h1.value IS NOT NULL AND abs(h0.representative_minute-s.pre_minute)<=5
        AND abs(h1.representative_minute-s.post_minute)<=5 hr_available
      FROM {d}_selected s LEFT JOIN prior.{d}_bins h0 ON s.stay_id=h0.stay_id AND s.pre_bin=h0.bin AND h0.concept='hr'
      LEFT JOIN prior.{d}_bins h1 ON s.stay_id=h1.stay_id AND s.post_bin=h1.bin AND h1.concept='hr')
      SELECT *,CASE WHEN post_spo2<pre_spo2 THEN 'fall' ELSE 'rise' END spo2_direction,
      CASE WHEN NOT hr_available THEN 'unavailable' WHEN post_hr-pre_hr<=-5 THEN 'fall'
        WHEN post_hr-pre_hr>=5 THEN 'rise' ELSE 'flat' END hr_direction FROM joined""")
    bins=con.execute(f"""SELECT b.* FROM prior.{d}_bins b JOIN care.{d}_cohort c USING(stay_id)
      WHERE concept IN ('spo2','hr') ORDER BY stay_id,bin""").df()
    b=bins.loc[bins.concept.eq('spo2')].copy()
    for col in ['bin','value','representative_minute']:b['pre_'+col]=b.groupby('stay_id')[col].shift()
    dt=b.representative_minute-b.pre_representative_minute
    f=b.loc[(b.value-b.pre_value).abs().ge(4)&dt.gt(0)&dt.le(30)].drop_duplicates('stay_id')
    for sqlcol,pycol in [('pre_bin','pre_bin'),('post_bin','bin'),('pre_spo2','pre_value'),('post_spo2','value'),('pre_minute','pre_representative_minute'),('post_minute','representative_minute')]:
        np.testing.assert_allclose(first[sqlcol],f[pycol],rtol=0,atol=1e-12)
    assert np.array_equal(first.stay_id,f.stay_id)
    f=f.merge(cohort[['stay_id','person_id']],on='stay_id',validate='one_to_one').sort_values('stay_id').drop_duplicates('person_id')
    hr=bins.loc[bins.concept.eq('hr'),['stay_id','bin','value','representative_minute']]
    assert not hr.duplicated(['stay_id','bin']).any()
    f=f.merge(hr.rename(columns={'bin':'pre_bin','value':'pre_hr','representative_minute':'pre_hr_minute'}),on=['stay_id','pre_bin'],how='left',validate='one_to_one')
    f=f.merge(hr.rename(columns={'value':'post_hr','representative_minute':'post_hr_minute'}),on=['stay_id','bin'],how='left',validate='one_to_one')
    f['hr_available']=f.pre_hr.notna()&f.post_hr.notna()&(f.pre_hr_minute-f.pre_representative_minute).abs().le(5)&(f.post_hr_minute-f.representative_minute).abs().le(5)
    f['spo2_direction']=np.where(f.value.lt(f.pre_value),'fall','rise')
    delta=f.post_hr-f.pre_hr;f['hr_direction']=np.select([~f.hr_available,delta.le(-5),delta.ge(5)],['unavailable','fall','rise'],default='flat')
    sql=con.execute(f'SELECT * FROM {d}_phenotype ORDER BY stay_id').df();assert sql.person_id.is_unique
    f=f.sort_values('stay_id').reset_index(drop=True)
    cols=['stay_id','person_id','pre_hr','post_hr','pre_hr_minute','post_hr_minute','hr_available','spo2_direction','hr_direction']
    pd.testing.assert_frame_equal(sql[cols],f[cols],check_dtype=False,rtol=0,atol=1e-12)
    support.append(dict(dataset=d,original_encounters=len(cohort),exposed_encounters=len(first),selected_people=len(sql),hr_matched_people=int(sql.hr_available.sum()),
      groups=sql.groupby(['spo2_direction','hr_direction']).size().reset_index(name='people').to_dict('records')))
hj('phenotype_support.json',support);hj('phenotype_validation.json',dict(original_exposed_sets_match=True,all_first_transitions_match=True,all_person_selections_and_hr_directions_match=True,mortality_not_joined=True))
hs('phenotype support completed before mortality','phenotype_complete');print(json.dumps(support,indent=2))
'''

SCREEN = r'''
hs('frozen mortality screen')
frames={};cells=[];gates=[];results=[];bootstrap_validation=[]
categories=[('fall','fall'),('fall','rise'),('rise','fall'),('rise','rise')]
for d in ['eicu','mimic']:
    f=con.execute(f"""SELECT p.*,c.hospital_id,c.hospital_mortality,c.excluded_before_landmark_flag,
      c.followup_end_offset_minutes FROM {d}_phenotype p JOIN prior.{d}_cohort c USING(stay_id)
      WHERE cast(p.person_id AS VARCHAR)=cast(c.person_id AS VARCHAR) ORDER BY stay_id""").df()
    assert len(f)==con.execute(f'SELECT count(*) FROM {d}_phenotype').fetchone()[0]
    assert f.person_id.is_unique and f.excluded_before_landmark_flag.eq(0).all() and f.followup_end_offset_minutes.gt(240).all()
    assert f.hospital_mortality.dropna().isin([0,1]).all()
    for sd in ['fall','rise']:
        for hd in ['fall','flat','rise','unavailable']:
            s=f.loc[f.spo2_direction.eq(sd)&f.hr_direction.eq(hd)];known=s.hospital_mortality.dropna()
            cells.append(dict(dataset=d,spo2_direction=sd,hr_direction=hd,people=len(s),known_outcomes=len(known),deaths=int(known.sum()),survivors=int((known==0).sum()),unknown_outcomes=int(s.hospital_mortality.isna().sum()),risk=float(known.mean()) if len(known) else None))
    p=f.loc[f.hr_direction.isin(['fall','rise'])&f.hospital_mortality.notna()].copy()
    p['group']=[categories.index((s,h)) for s,h in zip(p.spo2_direction,p.hr_direction)]
    p['cell']=p.group*2+p.hospital_mortality.astype(int)
    rows=[r for r in cells if r['dataset']==d and (r['spo2_direction'],r['hr_direction']) in categories]
    missing_hospital=int((p.hospital_id.isna()|p.hospital_id.fillna('').eq('')).sum())
    gate=all(r['known_outcomes']>=25 and min(r['deaths'],r['survivors'])>=5 for r in rows)
    if d=='eicu':gate=gate and p.hospital_id.nunique()>=20 and missing_hospital==0
    gates.append(dict(dataset=d,gate_pass=bool(gate),primary_people=len(p),hospitals=int(p.hospital_id.nunique()),missing_hospital_identifiers=missing_hospital))
    frames[d]=p
    con.register('mortality_frame',f);con.execute(f'CREATE OR REPLACE TABLE {d}_mortality AS SELECT * FROM mortality_frame');con.unregister('mortality_frame')
hj('mortality_cells.json',cells);hj('mortality_support.json',gates)
joint=all(g['gate_pass'] for g in gates)
def statistics(counts):
    c=np.asarray(counts,dtype=float).reshape(-1,4,2);den=c.sum(axis=2)
    with np.errstate(divide='ignore',invalid='ignore'):risk=c[:,:,1]/den
    rd=risk[:,0]-risk[:,1];interaction=rd-(risk[:,2]-risk[:,3])
    return np.column_stack([rd,interaction])
if joint:
    for d,seed in [('eicu',2026090541),('mimic',2026090542)]:
        f=frames[d];obs=np.bincount(f.cell,minlength=8);point=statistics(obs)[0];rng=np.random.Generator(np.random.PCG64(seed))
        if d=='eicu':
            hospitals=sorted(f.hospital_id.unique());index={h:i for i,h in enumerate(hospitals)}
            by=np.zeros((len(hospitals),8),dtype=np.int64)
            for h,g in f.groupby('hospital_id'):by[index[h]]=np.bincount(g.cell,minlength=8)
            weights=rng.multinomial(len(hospitals),np.full(len(hospitals),1/len(hospitals)),size=20000);draws=weights@by
            individual_cells=[f.loc[f.hospital_id.eq(h),'cell'].to_numpy(dtype=int) for h in hospitals]
        else:draws=rng.multinomial(len(f),obs/len(f),size=20000)
        boot=statistics(draws);valid=np.isfinite(boot).all(axis=1);valid_fraction=float(valid.mean())
        for i in range(100):
            if d=='eicu':expanded=np.concatenate([individual_cells[j] for j in np.repeat(np.arange(len(hospitals)),weights[i])])
            else:expanded=np.repeat(np.arange(8),draws[i])
            rates=[]
            for g in range(4):
                subset=expanded[expanded//2==g];rates.append(float(np.mean(subset%2)) if len(subset) else np.nan)
            direct=[rates[0]-rates[1],rates[0]-rates[1]-rates[2]+rates[3]]
            np.testing.assert_allclose(boot[i],direct,rtol=0,atol=1e-12,equal_nan=True)
        bootstrap_validation.append(dict(dataset=d,first_100_direct_expansions_match=True,replicates=20000,finite_fraction=valid_fraction,resampling_unit='hospital' if d=='eicu' else 'person'))
        for j,name in enumerate(['falling_spo2_hr_fall_minus_rise_rd','direction_interaction_rd']):
            ci=np.quantile(boot[valid,j],[.025,.975,.00625,.99375]) if valid_fraction>=.99 else [None]*4
            threshold=.05 if j==0 else 0
            results.append(dict(dataset=d,estimand=name,estimate=float(point[j]),lower95=None if ci[0] is None else float(ci[0]),upper95=None if ci[1] is None else float(ci[1]),
              lower_bonferroni=None if ci[2] is None else float(ci[2]),upper_bonferroni=None if ci[3] is None else float(ci[3]),finite_fraction=valid_fraction,
              advance=bool(ci[2] is not None and ci[2]>threshold)))
advance=len(results)==4 and all(r['advance'] for r in results)
for ref in refs.values():assert hh(Path(ref['path']))==ref['sha256']
hj('results.json',dict(results=results,joint_count_gate_pass=joint,advancement_gate_pass=advance,adjusted_models_fitted=False,biological_discovery_established=False))
hj('validation.json',dict(bootstrap_checks=bootstrap_validation,source_hashes_unchanged=True,unknown_mortality_not_imputed=True,original_person_and_landmark_checks_pass=True))
hj('manifest.json',dict(protocol_sha256=HEART_RATE_POLARITY_PROTOCOL_SHA256,script_sha256=hh(PROJECT/'scripts/heart_rate_polarity_cells.py'),environment='google_colab' if IN_COLAB else 'local',
 source='existing audited first-four-hour bins and original cohort outcomes',goal_complete=False))
report=['# First SpO2 transition, HR direction and mortality','',
 'Execution: '+('Google Colab' if IN_COLAB else 'local incremental notebook cells')+'. Existing audited MIMIC/eICU bins; no fresh raw-source extraction. Crude observational screen.',
 '', '| Source | SpO2 | HR | Deaths / known outcomes | Unknown | Risk |','|---|---|---|---|---:|---:|']
for r in cells:
    risk='unavailable' if r['risk'] is None else f"{100*r['risk']:.2f}%"
    report.append(f"| {r['dataset']} | {r['spo2_direction']} | {r['hr_direction']} | {r['deaths']} / {r['known_outcomes']} | {r['unknown_outcomes']} | {risk} |")
report+=['',f'Joint count gate passed: {joint}.','', '| Source | Estimand | Estimate, pp | Nominal 98.75% bootstrap interval, pp | Advances |','|---|---|---:|---|---|']
for r in results:
    interval='ungraded' if r['lower_bonferroni'] is None else f"{100*r['lower_bonferroni']:.2f} to {100*r['upper_bonferroni']:.2f}"
    report.append(f"| {r['dataset']} | {r['estimand']} | {100*r['estimate']:.2f} | {interval} | {r['advance']} |")
report+=['',f'Both-database advancement: {advance}. No adjusted or treatment-effect models were fitted.',
 '', 'Intervals use hospital resampling in eICU and person resampling in MIMIC with fixed seeds. Percentile bootstrap coverage is approximate. Missing outcomes and unmatched HR remain separate.',
 '', 'These coarse charted states do not measure an acute chemoreflex, cardiac output or confirmed arterial oxygen change. Severity, baseline vital levels, medications, rhythm, pacing, treatment and observation patterns remain uncontrolled. A positive crude association would not establish biological novelty, mechanism or benefit from changing heart rate. A failed screen cannot exclude smaller or confounded effects.',
 '', 'The original exposure, person selections, HR labels, source hashes and available bootstrap arithmetic validate. The discovery goal remains unfulfilled.']
(PROJECT/'docs/SPO2_HEART_RATE_POLARITY_RESULTS.md').write_text('\n'.join(report)+'\n')
con.execute('CHECKPOINT');con.close();hs('HR-direction mortality screen completed','complete');print('\n'.join(report))
'''
