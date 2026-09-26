"""Public MIMIC waveform metadata overlap with the existing clinical HF cohort."""

OVERLAP = r"""
import sys,json,hashlib,importlib.util,subprocess,re,urllib.request
from pathlib import Path
from datetime import datetime,timezone,timedelta
import duckdb
import pandas as pd
if importlib.util.find_spec('wfdb') is None:
    subprocess.check_call([sys.executable,'-m','pip','install','-q','wfdb==4.3.0'])
import wfdb
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph';PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905'
META=PRIVATE/'mimic_waveform_preview_metadata';META.mkdir(exist_ok=True)
OUT=PROJECT/'research/mimic_waveform_overlap';OUT.mkdir(exist_ok=True)
BASE='https://physionet.org/files/mimic4wdb/0.1.0/'
def wj(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
wj('run_status.json',dict(status='running',stage='public MIMIC waveform metadata overlap',environment='google_colab' if IN_COLAB else 'local'))
downloads=[];expected_hashes={}
def fetch(relative,limit=262144):
    assert not relative.startswith('/') and '..' not in relative.split('/')
    p=META/relative;p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():
        with urllib.request.urlopen(BASE+relative,timeout=30) as response:raw=response.read(limit+1)
        assert len(raw)<=limit,'Unexpected metadata size'
        raw.decode('ascii');p.write_bytes(raw)
    raw=p.read_bytes()
    if relative in expected_hashes:assert hashlib.sha256(raw).hexdigest()==expected_hashes[relative]
    downloads.append(dict(relative_path=relative,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    return p
index=fetch('RECORDS');subject_paths=[s.strip().strip('/') for s in index.read_text().splitlines() if s.strip()]
checksums=fetch('SHA256SUMS.txt',limit=2097152)
for line in checksums.read_text().splitlines():
    digest,name=line.split(maxsplit=1)
    expected_hashes[name.strip().lstrip('*').removeprefix('./')]=digest
assert hashlib.sha256(index.read_bytes()).hexdigest()==expected_hashes['RECORDS']
records=[]
for path in expected_hashes:
    m=re.fullmatch(r'waves/p(\d{3})/p(\d{8})/(\d{8})/\3[.]hea',path)
    if not m:continue
    assert m[2].startswith(m[1])
    assert f'waves/p{m[1]}/p{m[2]}' in subject_paths
    records.append(dict(subject_id=int(m[2]),record_id=m[3],directory=f'waves/p{m[1]}/p{m[2]}/{m[3]}'))
con=duckdb.connect(str(PRIVATE/'spo2_context.duckdb'),read_only=True)
cohort=con.execute('''SELECT c.stay_id,c.person_id subject_id,c.hadm_id,c.admit_time,
  coalesce(f.spo2_bins>=3 AND f.spo2_transitions>=2,false) dynamics_eligible
  FROM mimic_cohort c LEFT JOIN mimic_features f USING(stay_id)''').df()
con.close();assert cohort.stay_id.is_unique
subjects=set(cohort.subject_id.astype(int));selected=[r for r in records if r['subject_id'] in subjects]
rows=[];failures=[]
for r in selected:
    header=fetch(r['directory']+'/'+r['record_id']+'.hea')
    rec=wfdb.rdheader(str(header.with_suffix('')),rd_segments=False)
    comments='\n'.join(rec.comments or [])
    sid=re.search(r'(?im)^\s*subject_id\s+(?:[:=]\s*)?(\d+)',comments)
    hid=re.search(r'(?im)^\s*hadm_id\s+(?:[:=]\s*)?(\d+)',comments)
    if not sid or not hid or rec.base_datetime is None or rec.fs is None:
        failures.append(dict(record_id=r['record_id'],reason='Required header identity/time absent'));continue
    assert int(sid[1])==r['subject_id']
    start=pd.Timestamp(rec.base_datetime);end=start+pd.Timedelta(seconds=rec.sig_len/rec.fs)
    exact=cohort.loc[(cohort.subject_id==int(sid[1]))&(cohort.hadm_id==int(hid[1]))]
    names=[];segments=[]
    if getattr(rec,'seg_name',None):
        offset=0
        for name,length in zip(rec.seg_name,rec.seg_len):
            if name!='~' and length>0:segments.append((offset/rec.fs,(offset+length)/rec.fs))
            offset+=length
        assert offset==rec.sig_len
        layout=next((name for name in rec.seg_name if name!='~'),None)
        if layout:
            assert '/' not in layout and '..' not in layout
            lp=fetch(r['directory']+'/'+layout+'.hea')
            lr=wfdb.rdheader(str(lp.with_suffix('')),rd_segments=False);names=lr.sig_name or []
    else:
        segments=[(0,rec.sig_len/rec.fs)];names=rec.sig_name or []
    for c in exact.itertuples(index=False):
        begin=pd.Timestamp(c.admit_time);finish=begin+pd.Timedelta(hours=4)
        span=max(0,(min(end,finish)-max(start,begin)).total_seconds())
        a=(begin-start).total_seconds();b=(finish-start).total_seconds()
        non_gap=sum(max(0,min(y,b)-max(x,a)) for x,y in segments)
        assert non_gap<=span+1e-6
        rows.append(dict(record_id=r['record_id'],subject_id=r['subject_id'],hadm_id=int(hid[1]),
            stay_id=int(c.stay_id),dynamics_eligible=bool(c.dynamics_eligible),
            record_start=str(start),record_end=str(end),first4h_span_overlap_seconds=span,
            first4h_non_gap_overlap_seconds=non_gap,layout_signal_names=names,
            layout_has_resp='RESP' in {n.upper() for n in names},layout_has_pleth='PLETH' in {n.upper() for n in names},
            layout_has_ecg=bool(set(names)&{'I','II','III','V','aVR','aVL','aVF'})))
(META/'matched_metadata.json').write_text(json.dumps(dict(rows=rows,failures=failures,downloads=downloads),indent=2)+'\n')
positive=[r for r in rows if r['first4h_non_gap_overlap_seconds']>0]
dynamic=[r for r in positive if r['dynamics_eligible']]
support=dict(public_records=len(records),public_subjects=len(set(r['subject_id'] for r in records)),
    current_hf_encounters=len(cohort),subject_matched_records=len(selected),
    subject_matched_subjects=len(set(r['subject_id'] for r in selected)),
    header_identity_or_time_failures=len(failures),exact_hospital_matched_records=len(set(r['record_id'] for r in rows)),
    first4h_non_gap_overlap_records=len(set(r['record_id'] for r in positive)),
    first4h_non_gap_overlap_encounters=len(set(r['stay_id'] for r in positive)),
    original_dynamics_eligible_overlap_encounters=len(set(r['stay_id'] for r in dynamic)),
    dynamics_overlap_with_resp_pleth_ecg_layout=len(set(r['stay_id'] for r in dynamic if r['layout_has_resp'] and r['layout_has_pleth'] and r['layout_has_ecg'])),
    downloaded_metadata_files=len(downloads),downloaded_metadata_bytes=sum(r['bytes'] for r in downloads))
wj('overlap_support.json',support)
wj('analysis_manifest.json',dict(utc=datetime.now(timezone.utc).isoformat(),environment='google_colab' if IN_COLAB else 'local',
    source=BASE,records_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
    private_metadata_sha256=hashlib.sha256((META/'matched_metadata.json').read_bytes()).hexdigest(),
    waveform_amplitudes_downloaded=False,clinical_outcomes_selected=False,
    linkage_requires_subject_admission_and_time=True,biological_discovery=False,goal_complete=False))
report=['# MIMIC waveform-preview overlap','',
    '**Actual execution: '+('Google Colab' if IN_COLAB else 'local')+'. Metadata-only feasibility; no waveform experiment or biological discovery.**','']
report+=['| Quantity | Count |','|---|---:|']+[f'| {k} | {v} |' for k,v in support.items()]
report+=['','The source is the [public MIMIC-IV waveform preview v0.1.0](https://physionet.org/content/mimic4wdb/0.1.0/), joined only to the existing MIMIC HF cohort. No additional non-MIMIC patient dataset was introduced.','',
    'Positive overlap means the metadata contains non-gap segments inside the existing ICU window. Layout channels do not prove simultaneous usable respiratory, ECG and plethysmographic signals; amplitudes, artifact, and physiological validity have not been assessed. No mortality, troponin outcome or SpO2 response was examined.','',
    'These counts determine whether a direct physiological pilot is possible; they do not establish adequate power, a new mechanism, or a mortality-improving intervention. Matched identifiers and downloaded headers remain in private Drive Data.','']
(PROJECT/'docs/MIMIC_WAVEFORM_OVERLAP_RESULTS.md').write_text('\n'.join(report))
wj('run_status.json',dict(status='complete',stage='public MIMIC waveform metadata overlap',environment='google_colab' if IN_COLAB else 'local'))
print('\n'.join(report))
"""

VALIDATION = r"""
# Reconstruct identity, clock origin and durations from plain headers without WFDB.
import numpy as np
from datetime import datetime
assert len(records)==200 and len(set(r['subject_id'] for r in records))==198
assert len(subject_paths)==198 and not failures
checked=[]
for row in rows:
    r=next(r for r in records if r['record_id']==row['record_id'])
    lines=(META/r['directory']/(r['record_id']+'.hea')).read_text().splitlines()
    comments=[line.lstrip('#').strip() for line in lines if line.startswith('#')]
    plain=[line.strip() for line in lines if line.strip() and not line.startswith('#')]
    h=plain[0].split();frequency=float(h[2].split('/')[0]);samples=int(h[3])
    date=datetime.strptime(h[5],'%d/%m/%Y').date()
    start=pd.Timestamp(str(date)+' '+h[4]);end=start+pd.Timedelta(seconds=samples/frequency)
    sid=int(next(c.split()[-1] for c in comments if c.startswith('subject_id ')))
    hid=int(next(c.split()[-1] for c in comments if c.startswith('hadm_id ')))
    assert sid==row['subject_id'] and hid==row['hadm_id']
    c=cohort.loc[cohort.stay_id==row['stay_id']].iloc[0]
    assert int(c.subject_id)==sid and int(c.hadm_id)==hid
    begin=pd.Timestamp(c.admit_time);finish=begin+pd.Timedelta(hours=4)
    expected_span=max(0,(min(end,finish)-max(start,begin)).total_seconds())
    if '/' in h[0]:
        segments=[line.split() for line in plain[1:]]
        lengths=np.asarray([int(s[1]) for s in segments]);assert int(lengths.sum())==samples
        stops=lengths.cumsum()/frequency;starts=np.r_[0,stops[:-1]]
        keep=np.asarray([s[0]!='~' for s in segments])&(lengths>0)
        a=(begin-start).total_seconds();b=(finish-start).total_seconds()
        expected_active=float(np.maximum(0,np.minimum(stops[keep],b)-np.maximum(starts[keep],a)).sum())
    else:expected_active=expected_span
    assert abs(expected_span-row['first4h_span_overlap_seconds'])<.001
    assert abs(expected_active-row['first4h_non_gap_overlap_seconds'])<.001
    assert row['layout_has_resp']==any(s.casefold()=='resp' for s in row['layout_signal_names'])
    assert row['layout_has_pleth']==any(s.casefold()=='pleth' for s in row['layout_signal_names'])
    checked.append(max(abs(expected_span-row['first4h_span_overlap_seconds']),abs(expected_active-row['first4h_non_gap_overlap_seconds'])))
wj('independent_metadata_validation.json',dict(public_index_counts_match_release=True,
    exact_record_encounter_pairs_checked=len(checked),all_plain_header_identities_and_timing_match=True,
    maximum_overlap_difference_seconds=max(checked,default=0),
    metadata_checksums_match_official_manifest=True,waveform_amplitudes_not_read=True))
print('Independent plain-header validation passed for',len(checked),'exact record–encounter pairs.')
"""
