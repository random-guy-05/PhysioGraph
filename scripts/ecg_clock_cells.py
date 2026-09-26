"""Outcome-blind ECG/procedure temporal corroboration, with no clock correction."""

ANALYSIS = r'''
import hashlib,importlib.util,json
from pathlib import Path
from datetime import datetime,timezone
import duckdb
import numpy as np
import pandas as pd
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph';PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905'
OUT=PROJECT/'research/spo2_ecg_clock';OUT.mkdir(parents=True,exist_ok=True)
def cj(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def ch(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def cq(value):return "'"+str(value).replace("'","''")+"'"
assert ch(PROJECT/'docs/SPO2_ECG_CLOCK_PLAN.md')==ECG_CLOCK_PROTOCOL_SHA256
if (OUT/'protocol_lock.json').exists():assert json.loads((OUT/'protocol_lock.json').read_text())['sha256']==ECG_CLOCK_PROTOCOL_SHA256
else:cj('protocol_lock.json',dict(sha256=ECG_CLOCK_PROTOCOL_SHA256,utc=datetime.now(timezone.utc).isoformat(),before_procedure_timestamps=True))
try:con.close()
except (NameError,AttributeError):pass
con=duckdb.connect(str(PRIVATE/'ecg_clock.duckdb'));con.execute('SET threads=2');con.execute("SET memory_limit='2GB'")
source=DRIVE/'Data/MIMIC/Full/procedureevents.csv';linkdb=PRIVATE/'ecg_linkage.duckdb'
refs={str(p):dict(sha256=ch(p),bytes=p.stat().st_size) for p in [source,linkdb]}
assert json.loads((PROJECT/'research/spo2_ecg_linkage/validation.json').read_text())['publisher_metadata_hashes_verified']
con.execute(f'ATTACH {cq(linkdb)} AS links (READ_ONLY)')
cj('run_status.json',dict(stage='extracting EKG procedure metadata',status='running',environment='local' if not IN_COLAB else 'google_colab'))
con.execute(f"""CREATE OR REPLACE TABLE procedure_source AS
 SELECT try_cast(p.subject_id AS BIGINT) subject_id,try_cast(p.hadm_id AS BIGINT) hadm_id,
   try_cast(p.stay_id AS BIGINT) stay_id,p.starttime,p.endtime,p.storetime,p.orderid,p.linkorderid,p.statusdescription
 FROM read_csv({cq(source)},header=true,all_varchar=true,strict_mode=true) p
 JOIN links.cohort c ON try_cast(p.subject_id AS BIGINT)=c.subject_id
   AND try_cast(p.hadm_id AS BIGINT)=c.hadm_id AND try_cast(p.stay_id AS BIGINT)=c.stay_id
 WHERE try_cast(p.itemid AS BIGINT)=225402""")
raw=con.execute('SELECT * FROM procedure_source').df()
proc=raw.drop_duplicates().sort_values(list(raw.columns),na_position='last').reset_index(drop=True)
duplicate_count=len(raw)-len(proc);proc['procedure_id']=np.arange(len(proc),dtype=np.int64)
for col in ['starttime','endtime','storetime']:proc[col]=pd.to_datetime(proc[col],errors='coerce')
con.register('procedure_frame',proc);con.execute('CREATE OR REPLACE TABLE procedures AS SELECT * FROM procedure_frame');con.unregister('procedure_frame')
con.execute("""CREATE OR REPLACE TABLE candidates AS
 SELECT p.procedure_id,p.subject_id,p.stay_id,p.hadm_id,r.study_id,
 date_diff('second',p.starttime,r.recorded_ecg_time)/60.0 offset_minutes
 FROM procedures p JOIN links.ecg_registry r USING(subject_id)
 WHERE p.starttime IS NOT NULL AND r.path_valid AND r.recorded_ecg_time IS NOT NULL
 AND abs(date_diff('second',p.starttime,r.recorded_ecg_time))<=86400""")
con.execute("""CREATE OR REPLACE TABLE degrees AS SELECT *,
 count(*) OVER(PARTITION BY procedure_id) procedure_degree,
 count(*) OVER(PARTITION BY study_id) ecg_degree FROM candidates""")
con.execute('CREATE OR REPLACE TABLE isolated AS SELECT * FROM degrees WHERE procedure_degree=1 AND ecg_degree=1')
sql=con.execute('SELECT * FROM candidates ORDER BY procedure_id,study_id').df()
registry=con.execute('SELECT study_id,subject_id,recorded_ecg_time FROM links.ecg_registry WHERE path_valid AND recorded_ecg_time IS NOT NULL').df()
py=proc.loc[proc.starttime.notna(),['procedure_id','subject_id','stay_id','hadm_id','starttime']].merge(registry,on='subject_id',validate='many_to_many')
py['offset_minutes']=(py.recorded_ecg_time-py.starttime).dt.total_seconds()/60
py=py.loc[py.offset_minutes.abs().le(1440)].sort_values(['procedure_id','study_id']).reset_index(drop=True)
pd.testing.assert_frame_equal(sql,py[list(sql.columns)],check_dtype=False,rtol=0,atol=1e-12)
py['procedure_degree']=py.groupby('procedure_id').study_id.transform('size')
py['ecg_degree']=py.groupby('study_id').procedure_id.transform('size')
degrees=con.execute('SELECT * FROM degrees ORDER BY procedure_id,study_id').df()
pd.testing.assert_frame_equal(degrees,py[list(degrees.columns)],check_dtype=False,rtol=0,atol=1e-12)
isolated=con.execute('SELECT * FROM isolated ORDER BY procedure_id,study_id').df()
pyisolated=py.loc[py.procedure_degree.eq(1)&py.ecg_degree.eq(1)].reset_index(drop=True)
pd.testing.assert_frame_equal(isolated,pyisolated[list(isolated.columns)],check_dtype=False,rtol=0,atol=1e-12)
paired=isolated.merge(proc[['procedure_id','starttime','endtime','storetime','statusdescription']],on='procedure_id',validate='one_to_one')
def describe(series):
    s=pd.to_numeric(series,errors='coerce').dropna()
    return dict(n=len(s),**{k:(float(s.quantile(q)) if len(s) else None) for k,q in [('p01',.01),('q25',.25),('median',.5),('q75',.75),('p99',.99)]})
def offsets(frame):
    s=frame.offset_minutes
    return dict(signed_minutes=describe(s),absolute_minutes=describe(s.abs()),within_minutes=[dict(minutes=t,n=int(s.abs().le(t).sum()),denominator=len(s),fraction=float(s.abs().le(t).mean()) if len(s) else None) for t in [5,15,30,60,240]])
counts=degrees.groupby('procedure_id').size()
support=dict(original_encounters=con.execute('SELECT count(*) FROM links.cohort').fetchone()[0],raw_procedure_rows=len(raw),exact_duplicate_rows_removed=duplicate_count,
 procedures=len(proc),procedure_people=int(proc.subject_id.nunique()),procedure_encounters=int(proc.stay_id.nunique()),
 invalid_start=int(proc.starttime.isna().sum()),invalid_end=int(proc.endtime.isna().sum()),invalid_store=int(proc.storetime.isna().sum()),
 valid_start_no_candidate=int(len(proc.loc[proc.starttime.notna()])-len(counts)),procedures_multiple_candidates=int(counts.gt(1).sum()),
 procedures_one_candidate=int(counts.eq(1).sum()),candidate_edges=len(sql),isolated_pairs=len(isolated),isolated_people=int(isolated.subject_id.nunique()),
 isolated_encounters=int(isolated.stay_id.nunique()),statuses=proc.statusdescription.fillna('<missing>').value_counts().to_dict())
support['isolated_ecg_outside_nominal_hospital']=con.execute("""SELECT count(*) FROM isolated i JOIN links.ecg_registry r USING(study_id)
 JOIN links.admissions a ON i.hadm_id=a.hadm_id WHERE r.recorded_ecg_time<a.hospital_admit OR r.recorded_ecg_time>a.hospital_discharge""").fetchone()[0]
support['isolated_missing_admission_bounds']=con.execute("""SELECT count(*) FROM isolated i LEFT JOIN links.admissions a ON i.hadm_id=a.hadm_id
 WHERE a.hospital_admit IS NULL OR a.hospital_discharge IS NULL""").fetchone()[0]
summary=dict(all_isolated=offsets(paired),finished_running_isolated=offsets(paired.loc[paired.statusdescription.eq('FinishedRunning')]),
 all_procedure_duration_minutes=describe((proc.endtime-proc.starttime).dt.total_seconds()/60),
 all_procedure_documentation_delay_minutes=describe((proc.storetime-proc.starttime).dt.total_seconds()/60),
 negative_durations=int((proc.endtime-proc.starttime).dt.total_seconds().lt(0).sum()),negative_documentation_delays=int((proc.storetime-proc.starttime).dt.total_seconds().lt(0).sum()))
for name,ref in refs.items():assert ch(Path(name))==ref['sha256']
cj('support.json',support);cj('offset_summary.json',summary)
cj('validation.json',dict(all_candidate_edges_match=True,all_degrees_match=True,all_isolated_pairs_match=True,all_offsets_match=True,source_hashes_unchanged=True,
 mortality_accessed=False,ecg_values_accessed=False,clock_calibration_established=False))
cj('manifest.json',dict(references=refs,protocol_sha256=ECG_CLOCK_PROTOCOL_SHA256,script_sha256=ch(PROJECT/'scripts/ecg_clock_cells.py'),
 environment='local' if not IN_COLAB else 'google_colab',goal_complete=False))
report=['# Diagnostic ECG / charted EKG temporal corroboration','',
 'Execution: '+('Google Colab' if IN_COLAB else 'local incremental notebook cell')+'. Outcome-blind metadata analysis; no ECG values.',
 '',f"{len(proc)} unique charted EKG procedures in {support['procedure_encounters']} original encounters. {len(sql)} candidate edges within ±24 hours; {len(isolated)} isolated pairs in {support['isolated_people']} people.",
 '',f"Valid-start procedures without a candidate: {support['valid_start_no_candidate']}; with multiple candidates: {support['procedures_multiple_candidates']}.",
 '', '| Maximum absolute recorded-time difference | Isolated pairs | Denominator |','|---|---:|---:|']
for r in summary['all_isolated']['within_minutes']:report.append(f"| {r['minutes']} min | {r['n']} | {r['denominator']} |")
report+=['',f"Signed ECG-minus-procedure difference: {json.dumps(summary['all_isolated']['signed_minutes'])} minutes.",
 '', 'These are isolated candidate matches, not validated acquisition links. The matching window truncates possible disagreement; documentation delays and coincidental matches remain possible. No timestamps were corrected, and no acute biological sequence or mortality benefit is established.',
 '', 'Independent SQL/Python candidate edges, degrees, isolated identities and offsets agree. Original failed biological contrasts remain unchanged.']
(PROJECT/'docs/SPO2_ECG_CLOCK_RESULTS.md').write_text('\n'.join(report)+'\n')
con.execute('CHECKPOINT');con.close();cj('run_status.json',dict(stage='completed temporal corroboration',status='complete',goal_complete=False))
print('\n'.join(report))
'''
