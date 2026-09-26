"""Outcome-blind diagnostic-ECG metadata linkage for the original MIMIC cohort."""

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
META=PRIVATE/'mimic_ecg_metadata_v1';META.mkdir(parents=True,exist_ok=True)
OUT=PROJECT/'research/spo2_ecg_linkage';OUT.mkdir(parents=True,exist_ok=True)
BASE='https://physionet.org/files/mimic-iv-ecg/1.0/'
def ej(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def eh(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def eq(value):return "'"+str(value).replace("'","''")+"'"
def es(stage,status='running'):
    ej('run_status.json',dict(stage=stage,status=status,utc=datetime.now(timezone.utc).isoformat(),environment='google_colab' if IN_COLAB else 'local',goal_complete=False))
assert eh(PROJECT/'docs/SPO2_ECG_LINKAGE_PLAN.md')==ECG_LINKAGE_PROTOCOL_SHA256
lock=OUT/'protocol_lock.json'
if lock.exists():assert json.loads(lock.read_text())['sha256']==ECG_LINKAGE_PROTOCOL_SHA256
else:ej('protocol_lock.json',dict(sha256=ECG_LINKAGE_PROTOCOL_SHA256,utc=datetime.now(timezone.utc).isoformat(),before_ecg_index_download_and_overlap_counts=True,
  original_troponin_failures_known=True,clock_unsynchronization_warning_known=True))
try:con.close()
except (NameError,AttributeError):pass
con=duckdb.connect(str(PRIVATE/'ecg_linkage.duckdb'));con.execute('SET threads=2');con.execute("SET memory_limit='2GB'")
refs={}
for alias,name in [('prior','spo2_context'),('care','spo2_cardiorenal')]:
    p=PRIVATE/(name+'.duckdb');refs[name]=dict(path=str(p),sha256=eh(p));con.execute(f'ATTACH {eq(p)} AS {alias} (READ_ONLY)')
for name in ['admissions','d_items']:
    p=DRIVE/'Data/MIMIC/Full'/(name+'.csv');s=p.stat();refs[name]=dict(path=str(p),sha256=eh(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
manifest=dict(references=refs,protocol_sha256=ECG_LINKAGE_PROTOCOL_SHA256)
if (OUT/'input_manifest.json').exists():assert json.loads((OUT/'input_manifest.json').read_text())==manifest
ej('input_manifest.json',manifest);es('ECG metadata protocol locked')
'''

DOWNLOAD = r'''
downloads=[]
def fetch_metadata(name,limit):
    assert name in ['SHA256SUMS.txt','record_list.csv','machine_measurements_data_dictionary.csv','LICENSE.txt']
    path=META/name;receipt_path=META/(name+'.receipt.json')
    if not path.exists():
        es('downloading official ECG '+name)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(BASE+name,timeout=60) as response:
                    declared=response.headers.get('Content-Length');count=0
                    if declared is not None:assert int(declared)<=limit
                    receipt=dict(url=BASE+name,final_url=response.geturl(),http_status=response.status,etag=response.headers.get('ETag'),last_modified=response.headers.get('Last-Modified'),
                      content_length=declared,retrieved_utc=datetime.now(timezone.utc).isoformat())
                    with (META/(name+'.part')).open('wb') as f:
                        for block in iter(lambda:response.read(1024*1024),b''):
                            count+=len(block);assert count<=limit;f.write(block)
                    if declared is not None:assert count==int(declared)
                (META/(name+'.part')).replace(path);receipt_path.write_text(json.dumps(receipt,indent=2)+'\n');break
            except (urllib.error.URLError,TimeoutError,ConnectionError):
                if attempt==2:raise
                time.sleep(attempt+1)
    assert receipt_path.exists() and path.stat().st_size<=limit
    receipt=json.loads(receipt_path.read_text());receipt.update(name=name,bytes=path.stat().st_size,sha256=eh(path))
    downloads.append(receipt);return path
checksums=fetch_metadata('SHA256SUMS.txt',256*1024*1024)
wanted={'record_list.csv','machine_measurements_data_dictionary.csv','LICENSE.txt'};expected={}
with checksums.open() as f:
    for line in f:
        digest,name=line.rstrip('\n').split(maxsplit=1);name=name.strip().lstrip('*').removeprefix('./')
        if name in wanted:
            assert re.fullmatch('[0-9a-f]{64}',digest);expected[name]=digest
assert set(expected)==wanted
for name,limit in [('record_list.csv',96*1024*1024),('machine_measurements_data_dictionary.csv',1024*1024),('LICENSE.txt',1024*1024)]:
    p=fetch_metadata(name,limit);assert eh(p)==expected[name]
ej('download_manifest.json',dict(base_url=BASE,downloads=downloads,publisher_hashes=expected,publisher_hashes_verified=True,
  checksum_list_itself_pinned_by_local_hash_and_https=True,signal_amplitudes_downloaded=False,machine_measurements_downloaded=False,reports_downloaded=False))
es('ECG metadata downloaded and publisher hashes verified','download_complete')
print('Verified ECG metadata files:',len(downloads),'bytes:',sum(d['bytes'] for d in downloads))
'''

LINKAGE = r'''
es('original-cohort ECG metadata linkage')
download_manifest=json.loads((OUT/'download_manifest.json').read_text());assert download_manifest['publisher_hashes_verified']
for item in download_manifest['downloads']:assert eh(META/item['name'])==item['sha256']
registry=pd.read_csv(META/'record_list.csv')
assert {'subject_id','study_id','ecg_time','path'}.issubset(registry.columns)
assert registry.study_id.notna().all() and registry.study_id.is_unique and registry.path.notna().all() and registry.path.is_unique
registry['recorded_ecg_time']=pd.to_datetime(registry.ecg_time,errors='coerce')
parts=registry.path.str.extract(r'^files/p(\d{4})/p(\d{8})/s(\d{8})/(\d{8})$')
registry['path_valid']=(
    parts[1].eq(registry.subject_id.astype(str))&parts[2].eq(registry.study_id.astype(str))&parts[3].eq(parts[2])&parts[0].eq(parts[1].str[:4]))
source_summary=dict(registry_records=len(registry),registry_subjects=int(registry.subject_id.nunique()),unparseable_times=int(registry.recorded_ecg_time.isna().sum()),invalid_paths=int((~registry.path_valid).sum()),columns=list(registry.columns))
cohort=con.execute("""SELECT c.stay_id,cast(c.person_id AS BIGINT) subject_id,c.person_id,c.exposed,p.hadm_id,p.admit_time
  FROM care.mimic_cohort c JOIN prior.mimic_cohort p USING(stay_id)
  WHERE cast(c.person_id AS BIGINT)=cast(p.person_id AS BIGINT) ORDER BY c.stay_id""").df()
assert len(cohort)==con.execute('SELECT count(*) FROM care.mimic_cohort').fetchone()[0] and cohort.stay_id.is_unique
con.register('cohort_frame',cohort);con.execute('CREATE OR REPLACE TABLE cohort AS SELECT * FROM cohort_frame');con.unregister('cohort_frame')
matched=registry.loc[registry.subject_id.isin(cohort.subject_id)].copy()
valid=matched.loc[matched.path_valid&matched.recorded_ecg_time.notna()].copy()
con.register('registry',matched);con.execute('CREATE OR REPLACE TABLE ecg_registry AS SELECT * FROM registry');con.unregister('registry')
ap=Path(refs['admissions']['path'])
admissions=con.execute(f"""SELECT try_cast(subject_id AS BIGINT) subject_id,try_cast(hadm_id AS BIGINT) hadm_id,
  try_cast(admittime AS TIMESTAMP) hospital_admit,try_cast(dischtime AS TIMESTAMP) hospital_discharge
  FROM read_csv({eq(ap)},header=true,all_varchar=true,strict_mode=true)
  WHERE try_cast(subject_id AS BIGINT) IN (SELECT DISTINCT subject_id FROM cohort)""").df()
assert admissions.hadm_id.is_unique
con.register('admissions_frame',admissions);con.execute('CREATE OR REPLACE TABLE admissions AS SELECT * FROM admissions_frame');con.unregister('admissions_frame')
con.execute("""CREATE OR REPLACE TABLE nominal_hospital_links AS
  SELECT r.study_id,r.subject_id,a.hadm_id FROM ecg_registry r JOIN admissions a USING(subject_id)
  WHERE r.path_valid AND r.recorded_ecg_time>=a.hospital_admit AND r.recorded_ecg_time<=a.hospital_discharge""")
python_hospital=valid[['study_id','subject_id','recorded_ecg_time']].merge(admissions,on='subject_id',validate='many_to_many')
python_hospital=python_hospital.loc[python_hospital.recorded_ecg_time.ge(python_hospital.hospital_admit)&python_hospital.recorded_ecg_time.le(python_hospital.hospital_discharge)]
cols=['study_id','subject_id','hadm_id']
sql=con.execute('SELECT * FROM nominal_hospital_links ORDER BY study_id,hadm_id').df()
pd.testing.assert_frame_equal(sql,python_hospital[cols].sort_values(['study_id','hadm_id']).reset_index(drop=True),check_dtype=False)
con.execute("""CREATE OR REPLACE TABLE nominal_icu_links AS
  SELECT c.stay_id,c.subject_id,c.person_id,c.exposed,c.hadm_id,r.study_id,r.path,r.recorded_ecg_time,
    date_diff('second',c.admit_time,r.recorded_ecg_time)/60.0 ecg_minute
  FROM cohort c JOIN nominal_hospital_links h USING(subject_id,hadm_id) JOIN ecg_registry r USING(study_id,subject_id)""")
flags=con.execute("""SELECT c.stay_id,c.person_id,c.exposed,
  EXISTS(SELECT 1 FROM ecg_registry r WHERE r.subject_id=c.subject_id) any_subject_ecg,
  count(l.study_id)>0 nominal_hospital_ecg,
  count(*) FILTER(WHERE l.ecg_minute>=-1440 AND l.ecg_minute<0)>0 pre_icu_ecg,
  count(*) FILTER(WHERE l.ecg_minute>=0 AND l.ecg_minute<240)>0 first4h_ecg,
  count(*) FILTER(WHERE l.ecg_minute>=240 AND l.ecg_minute<=1680)>0 later_ecg
  FROM cohort c LEFT JOIN nominal_icu_links l USING(stay_id) GROUP BY c.stay_id,c.person_id,c.exposed,c.subject_id ORDER BY c.stay_id""").df()
flags['nominal_pre_later_pair']=flags.pre_icu_ecg&flags.later_ecg
py_links=python_hospital.merge(cohort,on=['subject_id','hadm_id'],validate='many_to_many')
py_links['ecg_minute']=(py_links.recorded_ecg_time-py_links.admit_time).dt.total_seconds()/60
py_flags=cohort[['stay_id','person_id','exposed']].copy()
sets={'any_subject_ecg':set(cohort.loc[cohort.subject_id.isin(matched.subject_id),'stay_id']),
 'nominal_hospital_ecg':set(py_links.stay_id),
 'pre_icu_ecg':set(py_links.loc[py_links.ecg_minute.ge(-1440)&py_links.ecg_minute.lt(0),'stay_id']),
 'first4h_ecg':set(py_links.loc[py_links.ecg_minute.ge(0)&py_links.ecg_minute.lt(240),'stay_id']),
 'later_ecg':set(py_links.loc[py_links.ecg_minute.ge(240)&py_links.ecg_minute.le(1680),'stay_id'])}
for key,ids in sets.items():py_flags[key]=py_flags.stay_id.isin(ids)
py_flags['nominal_pre_later_pair']=py_flags.pre_icu_ecg&py_flags.later_ecg
pd.testing.assert_frame_equal(flags,py_flags[flags.columns],check_dtype=False)
stored=con.execute('SELECT stay_id,study_id,ecg_minute FROM nominal_icu_links ORDER BY stay_id,study_id').df()
pd.testing.assert_frame_equal(stored,py_links[['stay_id','study_id','ecg_minute']].sort_values(['stay_id','study_id']).reset_index(drop=True),check_dtype=False,rtol=0,atol=1e-12)
con.register('flags',flags);con.execute('CREATE OR REPLACE TABLE encounter_flags AS SELECT * FROM flags');con.unregister('flags')
ambiguous_hospital=int(sql.groupby('study_id').hadm_id.nunique().gt(1).sum())
in_icu=py_links.loc[py_links.ecg_minute.ge(0)&py_links.ecg_minute.lt(240)]
ambiguous_icu=int(in_icu.groupby('study_id').stay_id.nunique().gt(1).sum())
ambiguity=dict(records_nominally_matching_multiple_hospital_admissions=ambiguous_hospital,records_nominally_matching_multiple_first4h_icu_windows=ambiguous_icu,
  clock_synchronization_verified=False)
summary=[]
for arm in [None,0,1]:
    f=flags if arm is None else flags.loc[flags.exposed.eq(arm)]
    for key in list(sets)+['nominal_pre_later_pair']:
        summary.append(dict(exposed='all' if arm is None else arm,criterion=key,encounters=int(f[key].sum()),people=int(f.loc[f[key],'person_id'].nunique()),eligible_encounters=len(f)))
source_summary.update(eligible_encounters=len(cohort),eligible_people=int(cohort.person_id.nunique()),subject_matched_registry_records=len(matched),subject_matched_registry_people=int(matched.subject_id.nunique()),
  matched_unparseable_times=int(matched.recorded_ecg_time.isna().sum()),matched_invalid_paths=int((~matched.path_valid).sum()))
items=pd.read_csv(refs['d_items']['path'],usecols=['itemid','label','linksto','category'])
ecg_items=items.loc[items.label.fillna('').str.contains(r'\becg\b|\bekg\b|electrocardio',case=False,regex=True)].copy()
item_rows=json.loads(ecg_items.to_json(orient='records'))
for ref in refs.values():assert eh(Path(ref['path']))==ref['sha256']
ej('registry_summary.json',source_summary);ej('nominal_overlap.json',summary);ej('ambiguity.json',ambiguity);ej('ecg_dictionary_items.json',item_rows)
ej('validation.json',dict(independent_hospital_links_match=True,independent_all_encounter_flags_match=True,independent_all_nominal_icu_offsets_match=True,
  source_hashes_unchanged=True,publisher_metadata_hashes_verified=True,mortality_accessed=False,ecg_measurements_accessed=False,clinical_clock_synchronization_validated=False))
ej('manifest.json',dict(protocol_sha256=ECG_LINKAGE_PROTOCOL_SHA256,script_sha256=eh(PROJECT/'scripts/ecg_linkage_cells.py'),environment='google_colab' if IN_COLAB else 'local',
  scope='original MIMIC HF cohort; MIMIC diagnostic ECG metadata only',clock_unsynchronization_warning_preserved=True,biological_discovery=False,goal_complete=False))
report=['# MIMIC diagnostic ECG metadata overlap','', 'Actual execution: '+('Google Colab' if IN_COLAB else 'local incremental notebook cells')+'. No ECG values, signal amplitudes, injury labels or mortality outcomes were examined.',
  '', '| Criterion | Encounters | People | Eligible encounters |','|---|---:|---:|---:|']
for r in summary:
    if r['exposed']=='all':report.append(f"| {r['criterion']} | {r['encounters']} | {r['people']} | {r['eligible_encounters']} |")
report+=['',f"Subject-matched registry records: {len(matched)}. Invalid matched paths: {source_summary['matched_invalid_paths']}; invalid matched times: {source_summary['matched_unparseable_times']}.",
  '',f"Ambiguous nominal hospital assignments: {ambiguous_hospital} records; ambiguous first-four-hour ICU assignments: {ambiguous_icu} records.",
  '', 'These are nominal matches to recorded timestamps. The [source documentation](https://physionet.org/content/mimic-iv-ecg/1.0/) explicitly warns that ECG clocks may be unsynchronized. These counts do not prove ECG acquisition preceded or followed an SpO2 episode, and do not validate an acute injury sequence.',
  '',f"The local dictionary contains {len(item_rows)} ECG/EKG-related entries; their labels and source tables are saved separately for clock-reference assessment. No item has yet been accepted as an independent acquisition timestamp.",
  '', 'Publisher hashes, local source fingerprints, SQL/Python hospital links, all encounter-window flags and nominal offsets validate. The original troponin failures remain unchanged. This creates a possible measurement route for a separately specified electrophysiology study, not a new biological finding or an achieved goal.']
(PROJECT/'docs/SPO2_ECG_LINKAGE_RESULTS.md').write_text('\n'.join(report)+'\n')
con.execute('CHECKPOINT');es('ECG metadata linkage completed','complete');print('\n'.join(report));con.close()
'''
