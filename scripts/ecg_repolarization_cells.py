"""Frozen paired-ECG biological discriminator in the original MIMIC cohort."""

SETUP = r'''
import hashlib,importlib.util,json,re,time,urllib.request,urllib.error
from pathlib import Path
from datetime import datetime,timezone
import duckdb
import numpy as np
import pandas as pd
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph';PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905'
META=PRIVATE/'mimic_ecg_metadata_v1';OUT=PROJECT/'research/spo2_ecg_repolarization';OUT.mkdir(parents=True,exist_ok=True)
def rj(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def rh(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def rq(value):return "'"+str(value).replace("'","''")+"'"
def rs(stage,status='running'):rj('run_status.json',dict(stage=stage,status=status,utc=datetime.now(timezone.utc).isoformat(),goal_complete=False))
assert rh(PROJECT/'docs/SPO2_ECG_REPOLARIZATION_PLAN.md')==ECG_REPOLARIZATION_PROTOCOL_SHA256
lock=OUT/'protocol_lock.json'
if lock.exists():assert json.loads(lock.read_text())['sha256']==ECG_REPOLARIZATION_PROTOCOL_SHA256
else:rj('protocol_lock.json',dict(sha256=ECG_REPOLARIZATION_PROTOCOL_SHA256,utc=datetime.now(timezone.utc).isoformat(),before_machine_measurements=True,metadata_support_and_timing_limits_known=True))
assert rh(PROJECT/'docs/SPO2_ECG_TIMESTAMP_AMENDMENT.md')==ECG_TIMESTAMP_AMENDMENT_SHA256
amend=OUT/'timestamp_amendment_lock.json'
if amend.exists():assert json.loads(amend.read_text())['sha256']==ECG_TIMESTAMP_AMENDMENT_SHA256
else:rj('timestamp_amendment_lock.json',dict(sha256=ECG_TIMESTAMP_AMENDMENT_SHA256,utc=datetime.now(timezone.utc).isoformat(),before_quality_counts_and_outcome_changes=True,metadata_timestamp_disagreement_known=True))
try:con.close()
except (NameError,AttributeError):pass
con=duckdb.connect(str(PRIVATE/'ecg_repolarization.duckdb'));con.execute('SET threads=2');con.execute("SET memory_limit='2GB'")
refs={}
for alias,name in [('links','ecg_linkage'),('care','spo2_cardiorenal')]:
    p=PRIVATE/(name+'.duckdb');refs[name]=dict(path=str(p),sha256=rh(p));con.execute(f'ATTACH {rq(p)} AS {alias} (READ_ONLY)')
con.execute("""CREATE OR REPLACE TABLE selected_pairs AS WITH pre AS (
 SELECT *,row_number() OVER(PARTITION BY stay_id ORDER BY recorded_ecg_time DESC,study_id) rn
 FROM links.nominal_icu_links WHERE ecg_minute>=-1440 AND ecg_minute<0), post AS (
 SELECT *,row_number() OVER(PARTITION BY stay_id ORDER BY recorded_ecg_time,study_id) rn
 FROM links.nominal_icu_links WHERE ecg_minute>=240 AND ecg_minute<=1680), pairs AS (
 SELECT c.*,p.study_id pre_study_id,q.study_id post_study_id,p.recorded_ecg_time pre_time,q.recorded_ecg_time post_time,
 p.ecg_minute pre_minute,q.ecg_minute post_minute,
 row_number() OVER(PARTITION BY c.person_id ORDER BY c.stay_id) person_rn
 FROM care.mimic_cohort c JOIN pre p USING(stay_id) JOIN post q USING(stay_id) WHERE p.rn=1 AND q.rn=1)
 SELECT * EXCLUDE(person_rn) FROM pairs WHERE person_rn=1""")
pairs=con.execute('SELECT * FROM selected_pairs ORDER BY stay_id').df()
alllinks=con.execute('SELECT stay_id,person_id,study_id,recorded_ecg_time,ecg_minute FROM links.nominal_icu_links').df()
pre=alllinks.loc[alllinks.ecg_minute.ge(-1440)&alllinks.ecg_minute.lt(0)].sort_values(['stay_id','recorded_ecg_time','study_id'],ascending=[True,False,True]).drop_duplicates('stay_id')
post=alllinks.loc[alllinks.ecg_minute.ge(240)&alllinks.ecg_minute.le(1680)].sort_values(['stay_id','recorded_ecg_time','study_id']).drop_duplicates('stay_id')
py=pre.merge(post,on=['stay_id','person_id'],suffixes=('_pre','_post')).sort_values('stay_id').drop_duplicates('person_id')
pd.testing.assert_frame_equal(pairs[['stay_id','pre_study_id','post_study_id']],py[['stay_id','study_id_pre','study_id_post']].rename(columns={'study_id_pre':'pre_study_id','study_id_post':'post_study_id'}).reset_index(drop=True),check_dtype=False)
prefix='PhysioGraph paired ECG repolarization v1|'
pairs['split']=[hashlib.sha256((prefix+str(int(x))).encode()).digest()[0]%2 for x in pairs.person_id]
con.register('pairs_frame',pairs);con.execute('CREATE OR REPLACE TABLE selected_pairs AS SELECT * FROM pairs_frame');con.unregister('pairs_frame')
assert pairs.person_id.is_unique
manifest=dict(protocol_sha256=ECG_REPOLARIZATION_PROTOCOL_SHA256,references=refs)
if (OUT/'input_manifest.json').exists():assert json.loads((OUT/'input_manifest.json').read_text())==manifest
rj('input_manifest.json',manifest);rj('metadata_support.json',dict(people=len(pairs),exposed=int(pairs.exposed.sum()),split_counts=pairs.groupby(['split','exposed']).size().reset_index(name='n').to_dict('records'),independent_pairs_match=True))
rs('pairing and patient split frozen before measurements')
'''

DOWNLOAD = r'''
name='machine_measurements.csv';path=META/name;receipt_path=META/(name+'.receipt.json');expected=[]
with (META/'SHA256SUMS.txt').open() as f:
    for line in f:
        digest,entry=line.rstrip('\n').split(maxsplit=1)
        if entry.strip().lstrip('*').removeprefix('./')==name:expected.append(digest)
assert len(expected)==1 and re.fullmatch('[0-9a-f]{64}',expected[0])
if not path.exists():
    rs('downloading official machine measurements')
    for attempt in range(3):
        try:
            url='https://physionet.org/files/mimic-iv-ecg/1.0/'+name
            with urllib.request.urlopen(url,timeout=60) as response:
                declared=response.headers.get('Content-Length');count=0;limit=256*1024*1024
                if declared:assert int(declared)<=limit
                receipt=dict(url=url,final_url=response.geturl(),http_status=response.status,etag=response.headers.get('ETag'),last_modified=response.headers.get('Last-Modified'),content_length=declared,retrieved_utc=datetime.now(timezone.utc).isoformat())
                with (META/(name+'.part')).open('wb') as f:
                    for block in iter(lambda:response.read(1024*1024),b''):
                        count+=len(block);assert count<=limit;f.write(block)
                if declared:assert count==int(declared)
            (META/(name+'.part')).replace(path);receipt_path.write_text(json.dumps(receipt,indent=2)+'\n');break
        except (urllib.error.URLError,TimeoutError,ConnectionError):
            if attempt==2:raise
            time.sleep(attempt+1)
assert receipt_path.exists() and rh(path)==expected[0]
rj('download_manifest.json',dict(receipt=json.loads(receipt_path.read_text()),bytes=path.stat().st_size,sha256=expected[0],publisher_sha256_verified=True,
 checksum_list_sha256=rh(META/'SHA256SUMS.txt'),signal_amplitudes_downloaded=False))
rs('machine measurements downloaded and verified','download_complete');print('Publisher-verified machine measurements:',path.stat().st_size,'bytes')
'''

MEASURE = r'''
rs('machine measurement extraction and fixed quality gate')
manifest=json.loads((OUT/'download_manifest.json').read_text());assert rh(META/'machine_measurements.csv')==manifest['sha256']
con.execute(f"""CREATE OR REPLACE TABLE machine_source AS SELECT m.*
 FROM read_csv({rq(META/'machine_measurements.csv')},header=true,all_varchar=true,strict_mode=true) m
 WHERE try_cast(m.study_id AS BIGINT) IN (SELECT pre_study_id FROM selected_pairs UNION SELECT post_study_id FROM selected_pairs)""")
m=con.execute('SELECT * FROM machine_source').df()
for col in ['subject_id','study_id']:m[col]=pd.to_numeric(m[col],errors='raise').astype('int64')
assert m.study_id.is_unique
registry=con.execute('SELECT study_id,subject_id,recorded_ecg_time FROM links.ecg_registry').df()
ident=m[['subject_id','study_id','ecg_time']].merge(registry,on='study_id',suffixes=('_machine','_registry'),validate='one_to_one')
assert len(ident)==len(m) and ident.subject_id_machine.eq(ident.subject_id_registry).all()
ident['timestamp_match']=[datetime.fromisoformat(x)==y.to_pydatetime() for x,y in zip(ident.ecg_time,ident.recorded_ecg_time)]
ts=con.execute("""SELECT try_cast(m.study_id AS BIGINT) study_id,
 try_cast(m.ecg_time AS TIMESTAMP)=r.recorded_ecg_time timestamp_match,
 date_diff('second',r.recorded_ecg_time,try_cast(m.ecg_time AS TIMESTAMP)) difference_seconds
 FROM machine_source m JOIN links.ecg_registry r ON try_cast(m.study_id AS BIGINT)=r.study_id ORDER BY study_id""").df()
pd.testing.assert_frame_equal(ts[['study_id','timestamp_match']],ident[['study_id','timestamp_match']].sort_values('study_id').reset_index(drop=True),check_dtype=False)
rj('timestamp_consistency.json',dict(selected_records=len(ident),exact_matches=int(ident.timestamp_match.sum()),rejected_records=int((~ident.timestamp_match).sum()),
 min_difference_seconds=int(ts.difference_seconds.min()),max_difference_seconds=int(ts.difference_seconds.max()),independent_all_timestamp_comparisons_match=True,
 amendment_sha256=ECG_TIMESTAMP_AMENDMENT_SHA256,original_pair_choices_preserved=True))
con.register('timestamp_frame',ts);con.execute('CREATE OR REPLACE TABLE timestamp_consistency AS SELECT * FROM timestamp_frame');con.unregister('timestamp_frame')
m=m.loc[m.study_id.isin(ident.loc[ident.timestamp_match,'study_id'])].copy()
for col in ['rr_interval','qrs_onset','qrs_end','t_end']:m[col]=pd.to_numeric(m[col],errors='coerce')
reports=[c for c in m if re.fullmatch(r'report_\d+',c)]
assert reports
text=m[reports].fillna('').agg(' '.join,axis=1)
m['excluded_rhythm']=text.str.contains(r'paced|pacing|pacemaker|atrial\s+(?:fibrillation|flutter)',case=False,regex=True)
m['qrs']=m.qrs_end-m.qrs_onset;m['qt']=m.t_end-m.qrs_onset
m['jtcf']=m.qt/np.cbrt(m.rr_interval/1000)-m.qrs
m['valid_duration']=(m.qrs_onset.ge(0)&m.qrs_end.gt(m.qrs_onset)&m.t_end.gt(m.qrs_end)&m.t_end.le(5000)&m.rr_interval.between(300,2000)&m.qrs.between(40,240)&m.qt.between(200,800)&m.jtcf.between(100,650)&np.isfinite(m[['qrs','qt','jtcf','rr_interval']]).all(axis=1))
con.register('measure_frame',m[['study_id','subject_id','qrs','qt','jtcf','rr_interval','excluded_rhythm','valid_duration','cart_id','bandwidth','filtering']]);con.execute('CREATE OR REPLACE TABLE measures AS SELECT * FROM measure_frame');con.unregister('measure_frame')
sql=con.execute("""SELECT try_cast(study_id AS BIGINT) study_id,
 try_cast(qrs_end AS DOUBLE)-try_cast(qrs_onset AS DOUBLE) qrs,
 try_cast(t_end AS DOUBLE)-try_cast(qrs_onset AS DOUBLE) qt,
 (try_cast(t_end AS DOUBLE)-try_cast(qrs_onset AS DOUBLE))/cbrt(try_cast(rr_interval AS DOUBLE)/1000.0)
 -(try_cast(qrs_end AS DOUBLE)-try_cast(qrs_onset AS DOUBLE)) jtcf FROM machine_source
 WHERE try_cast(study_id AS BIGINT) IN (SELECT study_id FROM timestamp_consistency WHERE timestamp_match) ORDER BY study_id""").df()
np.testing.assert_allclose(sql[['qrs','qt','jtcf']],m.sort_values('study_id')[['qrs','qt','jtcf']],rtol=1e-12,atol=1e-10,equal_nan=True)
cols=['study_id','subject_id','qrs','qt','jtcf','rr_interval','excluded_rhythm','valid_duration','cart_id','bandwidth','filtering']
a=pairs.merge(m[cols].add_prefix('pre_'),on='pre_study_id',how='left',validate='one_to_one').merge(m[cols].add_prefix('post_'),on='post_study_id',how='left',validate='one_to_one')
for side in ['pre','post']:
    known=a[side+'_subject_id'].notna();assert a.loc[known,side+'_subject_id'].eq(pd.to_numeric(a.loc[known,'person_id'])).all()
a['both_measured']=a.pre_subject_id.notna()&a.post_subject_id.notna()
a['both_valid']=a.pre_valid_duration.fillna(False).astype(bool)&a.post_valid_duration.fillna(False).astype(bool)
a['rhythm_excluded']=a.pre_excluded_rhythm.fillna(False).astype(bool)|a.post_excluded_rhythm.fillna(False).astype(bool)
a['quality_pass']=a.both_measured&a.both_valid&~a.rhythm_excluded
a['margin_pass']=a.pre_minute.le(-60)&a.post_minute.ge(300)&a.post_minute.le(1620)
a['same_device']=True
for col in ['cart_id','bandwidth','filtering']:
    a['same_device']&=a['pre_'+col].notna()&a['post_'+col].notna()&a['pre_'+col].eq(a['post_'+col])
con.register('analysis_frame',a);con.execute('CREATE OR REPLACE TABLE quality_pairs AS SELECT * FROM analysis_frame');con.unregister('analysis_frame')
support=[]
for split in [0,1]:
    s=a.loc[a.split.eq(split)];ok=s.loc[s.quality_pass]
    counts=[int(ok.exposed.eq(arm).sum()) for arm in [0,1]]
    support.append(dict(split=split,metadata_people=len(s),both_measured=int(s.both_measured.sum()),both_duration_valid=int(s.both_valid.sum()),rhythm_excluded=int(s.rhythm_excluded.sum()),quality_people=len(ok),unexposed=counts[0],exposed=counts[1],gate_pass=len(ok)>=80 and min(counts)>=20))
rj('quality_support.json',dict(splits=support,joint_gate_pass=all(s['gate_pass'] for s in support),matched_machine_rows=len(m),registry_id_and_time_match=True,independent_durations_match=True))
rs('fixed quality support completed','support_complete');print(json.dumps(support,indent=2))
'''

MODELS = r'''
rs('frozen paired ECG models')
from scipy.stats import t as student_t
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests
support=json.loads((OUT/'quality_support.json').read_text());results=[];model_info=[];max_validation_error=0.0
def fit_split(s,split,analysis):
    global max_validation_error
    s=s.copy();n=len(s);counts=[int(s.exposed.eq(v).sum()) for v in [0,1]]
    info=dict(split=split,analysis=analysis,n=n,unexposed=counts[0],exposed=counts[1],gate_failures=[])
    if n<80 or min(counts)<20:
        info['gate_failures'].append('fixed sample support');model_info.append(info);return
    x=pd.DataFrame({'intercept':np.ones(n),'exposed':s.exposed.to_numpy(dtype=float)},index=s.index)
    nuis=s[['pre_jtcf','pre_qrs','pre_rr_interval','age','spo2_mean','hypoxic_fraction','spo2_readings','vent_documented','vasoactive_documented']].copy()
    nuis['spo2_readings']=np.log1p(nuis.spo2_readings)
    nuis['sex_male']=pd.to_numeric(s.is_male,errors='coerce').mask(s.sex_unknown_flag.fillna(1).ne(0))
    nuis['elapsed_hours']=(s.post_time-s.pre_time).dt.total_seconds()/3600
    missing={};removed=[]
    for col in nuis:
        v=pd.to_numeric(nuis[col],errors='coerce').replace([np.inf,-np.inf],np.nan);missing[col]=int(v.isna().sum())
        flag=v.isna().astype(float)
        if flag.nunique()>1:x[col+'_missing']=flag
        med=float(v.median()) if v.notna().any() else 0.0;v=v.fillna(med)
        if v.nunique()<2:removed.append(col);continue
        x[col]=(v-v.mean())/v.std(ddof=0)
    xx=x.to_numpy(dtype=float);k=xx.shape[1];rank=int(np.linalg.matrix_rank(xx));condition=float(np.linalg.cond(xx));df=n-k
    info.update(columns=list(x),missing=missing,removed_constant_nuisance=removed,rank=rank,condition_number=condition,residual_df=df)
    if rank<k or condition>1e8 or df<30:
        info['gate_failures'].append('fixed design rank/condition/residual-df');model_info.append(info);return
    # Outcomes are formed only after both primary support gates and this design gate.
    dj=s.post_jtcf-s.pre_jtcf;dq=s.post_qrs-s.pre_qrs
    outcomes={'jtcf_change':dj,'jtcf_minus_qrs_change':dj-dq,'qrs_change_descriptive':dq}
    bread=np.linalg.inv(xx.T@xx);hat=np.sum((xx@bread)*xx,axis=1);assert np.all(hat<1)
    for endpoint,y in outcomes.items():
        fit=OLS(y.to_numpy(),xx).fit(cov_type='HC3',use_t=True)
        beta=np.linalg.lstsq(xx,y.to_numpy(),rcond=None)[0];resid=y.to_numpy()-xx@beta
        cov=bread@((xx*(resid/(1-hat))[:,None]).T@(xx*(resid/(1-hat))[:,None]))@bread
        np.testing.assert_allclose(beta,fit.params,rtol=1e-8,atol=1e-8)
        np.testing.assert_allclose(cov,fit.cov_params(),rtol=1e-7,atol=1e-7)
        max_validation_error=max(max_validation_error,float(np.max(np.abs(cov-fit.cov_params()))))
        b=float(beta[1]);se=float(np.sqrt(cov[1,1]));critical=float(student_t.ppf(.975,df));p=float(2*student_t.sf(abs(b/se),df))
        results.append(dict(split=split,analysis=analysis,endpoint=endpoint,n=n,estimate_ms=b,se_ms=se,lower_ms=b-critical*se,upper_ms=b+critical*se,
          p=p if analysis=='primary' and endpoint!='qrs_change_descriptive' else None))
    model_info.append(info)
    con.register('modeled_frame',s);con.execute(f'CREATE OR REPLACE TABLE {analysis}_split_{split}_modeled AS SELECT * FROM modeled_frame');con.unregister('modeled_frame')
    con.register('design_frame',x.reset_index(drop=True));con.execute(f'CREATE OR REPLACE TABLE {analysis}_split_{split}_design AS SELECT * FROM design_frame');con.unregister('design_frame')
if support['joint_gate_pass']:
    a=con.execute('SELECT * FROM quality_pairs WHERE quality_pass ORDER BY stay_id').df()
    for split in [0,1]:fit_split(a.loc[a.split.eq(split)],split,'primary')
    for analysis,flag in [('margin','margin_pass'),('same_device','same_device')]:
        for split in [0,1]:fit_split(a.loc[a.split.eq(split)&a[flag]],split,analysis)
primary=[r for r in results if r['analysis']=='primary' and r['p'] is not None]
if len(primary)==4:
    for r,p in zip(primary,multipletests([r['p'] for r in primary],method='holm')[1]):r['holm_p']=float(p)
signal=len(primary)==4 and all(r['lower_ms']>0 and r['holm_p']<.05 and (r['endpoint']!='jtcf_change' or r['estimate_ms']>=20) for r in primary)
for ref in refs.values():assert rh(Path(ref['path']))==ref['sha256']
assert rh(META/'machine_measurements.csv')==json.loads((OUT/'download_manifest.json').read_text())['sha256']
rj('results.json',dict(results=results,primary_prioritization_gate_pass=signal,novel_biological_finding_established=False,mortality_benefit_established=False))
rj('model_info.json',model_info);rj('validation.json',dict(independent_pairs_match=True,registry_identifiers_and_times_match=True,independent_durations_match=True,
 independent_fitted_coefficients_and_hc3_match=bool(results),max_hc3_difference=max_validation_error,source_hashes_unchanged=True,outcomes_computed=bool(results),mortality_accessed=False))
rj('manifest.json',dict(protocol_sha256=ECG_REPOLARIZATION_PROTOCOL_SHA256,timestamp_amendment_sha256=ECG_TIMESTAMP_AMENDMENT_SHA256,script_sha256=rh(PROJECT/'scripts/ecg_repolarization_cells.py'),environment='google_colab' if IN_COLAB else 'local',goal_complete=False))
report=['# Paired ECG repolarization experiment','', 'Execution: '+('Google Colab' if IN_COLAB else 'local incremental notebook cells')+'. MIMIC original cohort only; no external replication or mortality endpoint.',
 '', '| Split | Metadata people | Quality people | Exposed | Unexposed | Support gate |','|---|---:|---:|---:|---:|---|']
for s in support['splits']:report.append(f"| {s['split']} | {s['metadata_people']} | {s['quality_people']} | {s['exposed']} | {s['unexposed']} | {s['gate_pass']} |")
report+=['','| Analysis / split | Outcome | Estimate, ms | 95% CI | Primary Holm p |','|---|---|---:|---|---|']
for r in results:report.append(f"| {r['analysis']} / {r['split']} | {r['endpoint']} | {r['estimate_ms']:.3f} | {r['lower_ms']:.3f} to {r['upper_ms']:.3f} | {r.get('holm_p','not confirmatory')} |")
report+=['',f"Fixed primary prioritization gate passed: {signal}. This is not a novelty, mechanism, or clinical-benefit verdict.",
 '', 'The first execution stopped on timestamp inconsistency. Under the explicit pre-outcome linkage amendment, inconsistent machine records were rejected without replacing the original ECG pairs. See timestamp_consistency.json and SPO2_ECG_TIMESTAMP_AMENDMENT.md.',
 '', 'If the joint support gate failed, no outcome changes or effect models were computed. All denominators and model gate failures are saved. Rhythm exclusions are machine-text flags; absence of a flag is not confirmed sinus rhythm. No ECG waveform-quality validation has been performed.',
 '', 'Clock disagreement, selection for repeated ECGs, medication/electrolyte changes, lead placement and residual clinical confounding remain unresolved. Recorded-time margins do not establish true acquisition ordering. A positive machine-derived result would require the additional validation specified in the protocol; a failed primary cannot be rescued by sensitivities.',
 '', 'The biological discovery goal remains unfulfilled.']
(PROJECT/'docs/SPO2_ECG_REPOLARIZATION_RESULTS.md').write_text('\n'.join(report)+'\n')
con.execute('CHECKPOINT');con.close();rs('paired ECG experiment completed','complete');print('\n'.join(report))
'''
