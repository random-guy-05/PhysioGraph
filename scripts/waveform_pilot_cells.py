"""Notebook cells for the frozen, outcome-blind MIMIC respiratory pilot."""

SETUP = r'''
import sys,json,hashlib,importlib.util,subprocess,urllib.request,time,shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import duckdb
if importlib.util.find_spec('wfdb') is None:
    subprocess.check_call([sys.executable,'-m','pip','install','-q','wfdb==4.3.0'])
import wfdb
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph';PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905'
META=PRIVATE/'mimic_waveform_preview_metadata'
PILOT=PRIVATE/'mimic_waveform_pilot';PILOT.mkdir(exist_ok=True)
RAW=PILOT/'raw';RAW.mkdir(exist_ok=True)
OUT=PROJECT/'research/spo2_waveform_pilot';OUT.mkdir(exist_ok=True)
BASE='https://physionet.org/files/mimic4wdb/0.1.0/'
def wj(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def pilot_status(stage,status='running'):
    wj('run_status.json',dict(stage=stage,status=status,utc=datetime.now(timezone.utc).isoformat(),environment='google_colab' if IN_COLAB else 'local',goal_complete=False))
plan=PROJECT/'docs/SPO2_WAVEFORM_PILOT_PLAN.md'
assert hashlib.sha256(plan.read_bytes()).hexdigest()==WAVEFORM_PILOT_PROTOCOL_SHA256
lock=OUT/'protocol_lock.json'
if lock.exists():assert json.loads(lock.read_text())['sha256']==WAVEFORM_PILOT_PROTOCOL_SHA256
else:wj('protocol_lock.json',dict(sha256=WAVEFORM_PILOT_PROTOCOL_SHA256,locked_utc=datetime.now(timezone.utc).isoformat(),before_amplitudes=True))
metadata=json.loads((META/'matched_metadata.json').read_text())
selected=[r for r in metadata['rows'] if r['dynamics_eligible'] and r['first4h_non_gap_overlap_seconds']>0]
assert len(selected)==11 and len({r['stay_id'] for r in selected})==11
con=duckdb.connect(str(PRIVATE/'spo2_context.duckdb'),read_only=True)
ids=','.join(str(r['stay_id']) for r in selected)
clinical=con.execute(f"""SELECT c.stay_id,c.person_id,c.hadm_id,c.admit_time,f.spo2_jump_any exposed
 FROM mimic_cohort c JOIN mimic_features f USING(stay_id) WHERE c.stay_id IN ({ids})""").df()
bins=con.execute(f"""SELECT stay_id,bin,value,representative_minute FROM mimic_bins
 WHERE concept='spo2' AND stay_id IN ({ids}) ORDER BY stay_id,bin""").df()
con.close()
assert int(clinical.exposed.sum())==5 and len(clinical)==11
anchors={}
for sid,g in bins.groupby('stay_id',sort=True):
    delta=g.value.diff().abs();gap=g.representative_minute.diff()
    hit=g.loc[(delta>=4)&(gap>0)&(gap<=30)]
    is_exposed=bool(clinical.loc[clinical.stay_id==sid,'exposed'].iloc[0])
    assert bool(len(hit))==is_exposed
    if len(hit):anchors[int(sid)]=float(hit.iloc[0].representative_minute)
expected_hashes={}
for line in (META/'SHA256SUMS.txt').read_text().splitlines():
    digest,name=line.split(maxsplit=1);expected_hashes[name.strip().lstrip('*').removeprefix('./')]=digest
def sha_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
downloads={}
def fetch(relative):
    assert relative in expected_hashes and not relative.startswith('/') and '..' not in relative.split('/')
    p=RAW/relative;p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists() and (META/relative).exists():shutil.copyfile(META/relative,p)
    if not p.exists():
        partial=p.with_name(p.name+'.part')
        for attempt in range(3):
            try:
                with urllib.request.urlopen(BASE+relative,timeout=60) as response,partial.open('wb') as f:
                    shutil.copyfileobj(response,f,1024*1024)
                assert sha_file(partial)==expected_hashes[relative],'Official checksum mismatch'
                partial.replace(p);break
            except Exception:
                if partial.exists():partial.unlink()
                if attempt==2:raise
                time.sleep(1+attempt)
    assert sha_file(p)==expected_hashes[relative]
    downloads[relative]=dict(bytes=p.stat().st_size,sha256=expected_hashes[relative])
    return p
pilot_status('protocol locked; waveform materialization pending')
print('Locked outcome-blind pilot: 11 encounters; 5 exposed and 6 controls.')
'''

MATERIALIZE = r'''
pilot_status('checksum-verified Resp/Pleth and monitor SpO2 materialization')
materialized=[]
for index,row in enumerate(sorted(selected,key=lambda r:r['stay_id'])):
    sid=row['stay_id'];c=clinical.loc[clinical.stay_id==sid].iloc[0]
    assert int(c.person_id)==row['subject_id'] and int(c.hadm_id)==row['hadm_id']
    rid=row['record_id'];subject=str(row['subject_id'])
    directory=f'waves/p{subject[:3]}/p{subject}/{rid}'
    header=fetch(directory+'/'+rid+'.hea')
    h=wfdb.rdheader(str(header.with_suffix('')),rd_segments=False)
    start=pd.Timestamp(h.base_datetime);admit=pd.Timestamp(c.admit_time)
    offset=(start-admit).total_seconds();fs=float(h.fs)
    assert h.counter_freq and h.counter_freq>0
    a=max(0,int(np.ceil(-offset*fs)));b=min(h.sig_len,int(np.ceil((14400-offset)*fs)))
    assert b>a
    layout_names=[];cursor=0;needed=[]
    # All segment headers are small; only overlapping Resp/Pleth data files are fetched.
    for name,length in zip(h.seg_name,h.seg_len):
        if name!='~':
            hp=fetch(directory+'/'+name+'.hea');sh=wfdb.rdheader(str(hp.with_suffix('')))
            if length==0:layout_names=sh.sig_name
            if length>0 and min(cursor+length,b)>max(cursor,a):
                for label,filename in zip(sh.sig_name,sh.file_name):
                    if label.casefold() in {'resp','pleth'}:
                        assert filename!='~' and '/' not in filename and '..' not in filename
                        needed.append(directory+'/'+filename)
        cursor+=length
    assert cursor==h.sig_len
    names=[next(n for n in layout_names if n.casefold()==wanted) for wanted in ['resp','pleth']]
    for path in sorted(set(needed)):fetch(path)
    r=wfdb.rdrecord(str(header.with_suffix('')),sampfrom=a,sampto=b,channel_names=names,
        physical=True,m2s=True,smooth_frames=True,force_channels=True,return_res=32)
    assert r.sig_len==b-a and abs(r.fs-fs)<1e-12 and r.sig_name==names
    decoded=PILOT/f'{sid}_signals.npz'
    np.savez_compressed(decoded,resp=r.p_signal[:,0],pleth=r.p_signal[:,1],fs=fs,
        first_icu_second=offset+a/fs,frame_start=a,frame_stop=b,
        record_icu_offset_seconds=offset,counter_frequency=float(h.counter_freq))
    numeric=fetch(directory+'/'+rid+'n.csv.gz')
    columns=pd.read_csv(numeric,nrows=0).columns.tolist()
    time_column=next((n for n in columns if n.strip()=='time'),None)
    spo2_column=next((n for n in columns if n.strip()=='SpO2 [%]'),None)
    parts=[]
    if time_column is not None and spo2_column is not None:
        for chunk in pd.read_csv(numeric,usecols=[time_column,spo2_column],chunksize=200000):
            t=pd.to_numeric(chunk[time_column],errors='coerce')/float(h.counter_freq)+offset
            v=pd.to_numeric(chunk[spo2_column],errors='coerce')
            keep=(t>=0)&(t<14400)&v.between(50,100)
            if keep.any():parts.append(pd.DataFrame({'icu_second':t[keep],'spo2':v[keep]}))
    monitor=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame({'icu_second':pd.Series(dtype=float),'spo2':pd.Series(dtype=float)})
    monitor.to_csv(PILOT/f'{sid}_monitor.csv.gz',index=False,compression='gzip')
    materialized.append(dict(stay_id=sid,record_id=rid,exposed=bool(c.exposed),
        anchor_minute=anchors.get(sid),episode_block=int(anchors[sid]//20) if sid in anchors else None,
        fs=fs,counter_frequency=float(h.counter_freq),frames=b-a,
        first_icu_second=offset+a/fs,last_icu_second_exclusive=offset+b/fs,
        numeric_spo2_column_present=spo2_column is not None,valid_monitor_rows=len(monitor),
        decoded_sha256=sha_file(decoded)))
    (PILOT/'materialization.json').write_text(json.dumps(dict(rows=materialized,downloads=downloads),indent=2)+'\n')
    pilot_status(f'materialized {index+1}/11 records')
    print('Materialized record',index+1,'of 11',flush=True)
assert len(materialized)==11
wj('materialization_support.json',dict(encounters=len(materialized),exposed=sum(r['exposed'] for r in materialized),
    records_with_numeric_spo2=sum(r['numeric_spo2_column_present'] for r in materialized),
    downloaded_files=len(downloads),downloaded_bytes=sum(r['bytes'] for r in downloads.values()),
    all_downloads_match_official_checksums=True,clinical_outcomes_read=False))
pilot_status('materialization complete; frozen spectral analysis pending')
'''

ANALYSIS = r'''
from scipy import signal
pilot_status('fixed respiratory modulation and phase-null analysis')
materialization=json.loads((PILOT/'materialization.json').read_text())
materialized=materialization['rows'];assert len(materialized)==11
def fill_short_internal(x,maximum):
    x=np.asarray(x,dtype=float).copy();missing=~np.isfinite(x)
    boundaries=np.diff(np.r_[False,missing,False].astype(int))
    for a,b in zip(np.flatnonzero(boundaries==1),np.flatnonzero(boundaries==-1)):
        if a>0 and b<len(x) and b-a<=maximum:
            x[a:b]=np.linspace(x[a-1],x[b],b-a+2)[1:-1]
    return x
def fft_chunks(x):
    n=len(x)//256
    chunks=np.asarray(x[:n*256]).reshape(n,256)
    chunks=signal.detrend(chunks,axis=1,type='linear')*signal.windows.hann(256,sym=False)
    return np.fft.rfft(chunks,axis=1)
frequency=np.fft.rfftfreq(256,d=1)
band=(frequency>=1/120)&(frequency<=1/40)
assert np.array_equal(np.flatnonzero(band),[3,4,5,6])
def coherence_from_fft(x,y):
    denominator=np.mean(abs(x)**2,axis=-2)*np.mean(abs(y)**2,axis=-2)
    numerator=abs(np.mean(x.conj()*y,axis=-2))**2
    return np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0)
def block_signals(z,monitor,block):
    fs=float(z['fs']);first=float(z['first_icu_second']);last=first+len(z['resp'])/fs
    begin=block*1200;end=begin+1200
    info=dict(block=block,status='pending',waveform_extent=False)
    # At most one frame's quantization error is allowed at an interval boundary.
    if first>begin+1/fs or last<end-1/fs:
        info['status']='incomplete_waveform_extent';return info,None
    info['waveform_extent']=True
    a=max(0,int(np.ceil((begin-first)*fs-1e-8)))
    b=min(len(z['resp']),int(np.ceil((end-first)*fs-1e-8)))
    values=[]
    for name in ['resp','pleth']:
        x=np.asarray(z[name][a:b],dtype=float)
        info[name+'_finite_fraction']=float(np.isfinite(x).mean())
        if info[name+'_finite_fraction']<.95:
            info['status']=name+'_finite_below_95pct';return info,None
        x=fill_short_internal(x,int(np.floor(.5*fs)))
        if not np.isfinite(x).all():
            info['status']=name+'_gap_exceeds_limit';return info,None
        if np.ptp(x)==0:
            info['status']=name+'_constant';return info,None
        values.append(x)
    m=monitor.loc[(monitor.icu_second>=begin)&(monitor.icu_second<end)].copy()
    m['second']=np.floor(m.icu_second-begin).astype(int)
    spo2=m.groupby('second').spo2.median().reindex(range(1200)).to_numpy()
    info['spo2_observed_fraction']=float(np.isfinite(spo2).mean())
    if info['spo2_observed_fraction']<.90:
        info['status']='spo2_observed_below_90pct';return info,None
    spo2=fill_short_internal(spo2,10)
    if not np.isfinite(spo2).all():
        info['status']='spo2_gap_exceeds_limit';return info,None
    info['spo2_range']=float(np.ptp(spo2))
    if info['spo2_range']<1:
        info['status']='spo2_range_uninformative';return info,None
    resp,pleth=values
    filt=signal.sosfiltfilt(signal.butter(4,[.1,.7],btype='bandpass',fs=fs,output='sos'),resp)
    width=int(round(5*fs));square=np.r_[0,np.cumsum(filt**2)]
    envelope=np.full(len(filt),np.nan)
    envelope[width-1:]=np.sqrt(np.maximum(0,(square[width:]-square[:-width])/width))
    sec=np.floor(first+(a+np.arange(b-a))/fs-begin).astype(int)
    env=pd.Series(envelope).groupby(sec).mean().reindex(range(1200)).to_numpy()[10:-10]
    if not np.isfinite(env).all() or np.ptp(env)==0:
        info['status']='resp_envelope_uninformative';return info,None
    pf,pp=signal.welch(pleth,fs=fs,window='hann',nperseg=int(round(30*fs)),detrend='constant')
    denominator=pp[(pf>=.05)&(pf<=8)].sum()
    ratio=float(pp[(pf>=.5)&(pf<=4)].sum()/denominator) if denominator>0 else 0.
    info.update(status='usable',pleth_pulsatility_ratio=ratio,pleth_screen_pass=ratio>=.5)
    return info,(env,spo2[10:-10])
blocks=[];episode_results=[];rng=np.random.default_rng(20260905);spectral_validation=[]
episode_number=0
for row in sorted(materialized,key=lambda r:r['stay_id']):
    sid=row['stay_id'];z=np.load(PILOT/f'{sid}_signals.npz')
    assert sha_file(PILOT/f'{sid}_signals.npz')==row['decoded_sha256']
    monitor=pd.read_csv(PILOT/f'{sid}_monitor.csv.gz')
    for block in range(12):
        info,pair=block_signals(z,monitor,block)
        private_info=dict(stay_id=sid,exposed=row['exposed'],**info);blocks.append(private_info)
        if not row['exposed'] or block!=row['episode_block']:continue
        episode_number+=1
        result=dict(episode=f'E{episode_number}',status=info['status'],usable=pair is not None,
            coherence=None,peak_period_seconds=None,pleth_pulsatility_ratio=info.get('pleth_pulsatility_ratio'),
            pleth_screen_pass=info.get('pleth_screen_pass',False),raw_p=1.,holm_p=1.,positive_flag=False)
        if pair is not None:
            resp,spo2=pair;x=fft_chunks(resp);y=fft_chunks(spo2);assert x.shape[0]==4
            observed=coherence_from_fft(x,y)[band]
            if not np.isfinite(observed).all():
                result.update(status='spectral_denominator_uninformative',usable=False)
            else:
                maximum=float(observed.max());peak=float(frequency[band][np.argmax(observed)])
                phases=rng.uniform(-np.pi,np.pi,size=(499,)+y[:,band].shape)
                surrogate=abs(y[:,band])[None,:,:]*np.exp(1j*phases)
                null=coherence_from_fft(x[:,band][None,:,:],surrogate).max(axis=1)
                p=float((1+np.sum(null>=maximum))/500)
                result.update(coherence=maximum,peak_period_seconds=1/peak,raw_p=p)
                # Independent SciPy implementation on the actual clinical waveforms.
                vf,vc=signal.coherence(resp,spo2,fs=1,window='hann',nperseg=256,noverlap=0,detrend='linear')
                error=float(np.max(abs(vc[band]-observed)))
                assert np.array_equal(vf,frequency) and error<1e-10
                loop=np.asarray([max(abs(np.mean(x[:,band].conj()*ys,axis=0))**2/
                    (np.mean(abs(x[:,band])**2,axis=0)*np.mean(abs(ys)**2,axis=0))) for ys in surrogate])
                assert np.max(abs(loop-null))<1e-12
                spectral_validation.append(dict(episode=result['episode'],scipy_max_absolute_difference=error,
                    surrogate_vector_loop_max_difference=float(np.max(abs(loop-null))),chunks=4,frequency_bins=4))
                np.savez_compressed(PILOT/f'{sid}_episode_spectra.npz',resp_envelope=resp,spo2=spo2,
                    frequency=frequency,coherence=coherence_from_fft(x,y),surrogate_maxima=null)
        episode_results.append(result)
    z.close()
assert len(blocks)==132 and len(episode_results)==5
p=np.asarray([r['raw_p'] for r in episode_results]);order=np.argsort(p,kind='stable')
adjusted=np.minimum(1,np.maximum.accumulate(p[order]*(5-np.arange(5))))
for index,value in zip(order,adjusted):
    r=episode_results[index];r['holm_p']=float(value)
    r['positive_flag']=bool(r['usable'] and value<.05 and r['coherence']>=.5 and r['pleth_screen_pass'])
block_frame=pd.DataFrame(blocks)
coverage=[]
for exposed,g in block_frame.groupby('exposed'):
    coverage.append(dict(exposed=bool(exposed),encounters=int(g.stay_id.nunique()),blocks=len(g),
        usable_blocks=int((g.status=='usable').sum()),
        encounters_with_usable_blocks=int(g.loc[g.status=='usable','stay_id'].nunique()),
        statuses={k:int(v) for k,v in g.status.value_counts().items()}))
usable=sum(r['usable'] for r in episode_results);positive=sum(r['positive_flag'] for r in episode_results)
advance=usable>=4 and positive>=3
(PILOT/'block_quality.json').write_text(json.dumps(blocks,indent=2,allow_nan=False,default=str)+'\n')
wj('episode_results.json',episode_results);wj('block_coverage.json',coverage)
wj('spectral_validation.json',dict(usable_episode_implementations_compared=len(spectral_validation),rows=spectral_validation,
    clinical_outcomes_read=False,signal_null_not_synthetic_patients=True))
wj('analysis_manifest.json',dict(protocol_sha256=WAVEFORM_PILOT_PROTOCOL_SHA256,
    completed_utc=datetime.now(timezone.utc).isoformat(),environment='google_colab' if IN_COLAB else 'local',
    selected_encounters=11,exposed_encounters=5,usable_episode_blocks=usable,positive_pilot_flags=positive,
    advancement_gate_pass=advance,biological_discovery=False,mortality_benefit_demonstrated=False,goal_complete=False,
    private_materialization_sha256=sha_file(PILOT/'materialization.json'),
    phase_null_replicates_per_estimable_episode=499,phase_null_episodes_computed=len(spectral_validation),
    phase_null_realizations_computed=499*len(spectral_validation),multiple_testing_slots=5,seed=20260905))
report=['# MIMIC respiratory modulation pilot','',
    '**Actual execution: '+('Google Colab' if IN_COLAB else 'local')+'. No mortality or troponin outcome analysis.**','',
    f'The fixed pilot included 11 existing MIMIC HF encounters: five exposed to the original charted SpO2 instability and six controls. {usable}/5 first-episode blocks were usable; {positive}/5 met the corrected phase-coherence and Pleth screen. Advancement gate: **'+('PASS' if advance else 'FAIL')+'**.','',
    '| Episode | Status | Coherence | Pleth ratio | Raw p | Holm p | Positive |','|---|---|---:|---:|---:|---:|---|']
def display_number(v):return '—' if v is None else f'{v:.4f}'
for r in episode_results:
    report.append(f"| {r['episode']} | {r['status']} | {display_number(r['coherence'])} | {display_number(r['pleth_pulsatility_ratio'])} | {r['raw_p']:.4f} | {r['holm_p']:.4f} | {r['positive_flag']} |")
report+=['','Missing or uninformative episodes occupy p=1 placeholders in five-slot Holm correction. Episode labels identify analysis slots without patient identifiers. The gate requires at least four usable exposed episodes and at least three positive exposed patients.','',
    '| Group | Encounters | Usable blocks / 20-minute blocks | Encounters with usable blocks |','|---|---:|---:|---:|']
for r in coverage:report.append(f"| {'Exposed' if r['exposed'] else 'Control'} | {r['encounters']} | {r['usable_blocks']} / {r['blocks']} | {r['encounters_with_usable_blocks']} |")
report+=['',f'The frequency band, first-episode selection, quality checks, planned 499 phase-null replicates per estimable episode and gate were locked before amplitudes were inspected. Actual episode coherence tests computed: {len(spectral_validation)}. A failed completeness screen is insufficient evidence, not evidence that respiratory coupling is absent. Respiratory impedance amplitude is not calibrated ventilation; the Pleth screen does not exclude shared motion. Phase coherence does not establish causality, central apnea, ischemia, or mortality benefit. HF periodic breathing and oxygen saturation oscillations have substantial prior literature. This pilot does not establish a novel biological discovery.','',
    'All downloaded source files passed the official MIMIC waveform release checksums. Numeric time uses header counter frequency; waveform time uses frame sampling frequency. Patient arrays and detailed source membership remain in private Drive Data.','']
(PROJECT/'docs/SPO2_WAVEFORM_PILOT_RESULTS.md').write_text('\n'.join(report))
pilot_status('frozen respiratory modulation pilot','complete')
print('\n'.join(report))
'''

VALIDATION = r'''
# Independent plain-text/CSV clocks and direct FLAC calibration, without WFDB decoding.
import csv,gzip,re
import soundfile as sf
materialized=json.loads((PILOT/'materialization.json').read_text())['rows']
pilot_status('independent numeric-clock and full overlapping FLAC calibration validation')
validation=[]
for row in materialized:
    sid=row['stay_id'];rid=row['record_id']
    source=next(r for r in selected if r['stay_id']==sid);subject=str(source['subject_id'])
    directory=RAW/f'waves/p{subject[:3]}/p{subject}/{rid}'
    plain=[s.strip() for s in (directory/(rid+'.hea')).read_text().splitlines() if s.strip() and not s.startswith('#')]
    tokens=plain[0].split();fs=float(tokens[2].split('/')[0])
    counter=float(tokens[2].split('/')[1].split('(')[0])
    date=datetime.strptime(tokens[5],'%d/%m/%Y').date()
    origin=pd.Timestamp(str(date)+' '+tokens[4]);admit=pd.Timestamp(clinical.loc[clinical.stay_id==sid,'admit_time'].iloc[0])
    offset=(origin-admit).total_seconds()
    assert fs==row['fs'] and counter==row['counter_frequency']
    extracted=pd.read_csv(PILOT/f'{sid}_monitor.csv.gz');times=[];values=[]
    with gzip.open(directory/(rid+'n.csv.gz'),'rt') as f:
        for raw in csv.DictReader(f):
            try:t=float(raw['time'])/counter+offset;v=float(raw.get('SpO2 [%]',''))
            except (ValueError,TypeError):continue
            if 0<=t<14400 and 50<=v<=100:times.append(t);values.append(v)
    assert len(times)==len(extracted)
    error=float(np.max(abs(np.asarray(times)-extracted.icu_second.to_numpy()))) if times else 0.
    assert error<1e-8 and np.allclose(values,extracted.spo2,rtol=0,atol=1e-10)
    z=np.load(PILOT/f'{sid}_signals.npz');frame_start=int(z['frame_start']);frame_stop=int(z['frame_stop'])
    assert abs(float(z['first_icu_second'])-(offset+frame_start/fs))<1e-8
    cursor=0;checked={};decoded_channels={label:z[label] for label in ['resp','pleth']}
    for segment in plain[1:]:
        name,length=segment.split();length=int(length)
        if name!='~' and length>0 and min(cursor+length,frame_stop)>max(cursor,frame_start):
            lines=[s.split() for s in (directory/(name+'.hea')).read_text().splitlines() if s.strip() and not s.startswith('#')][1:]
            for fields in lines:
                label=fields[-1].casefold()
                if label not in {'resp','pleth'}:continue
                fmt=re.fullmatch(r'516(?:x(\d+))?',fields[1]);assert fmt,'Unvalidated FLAC format'
                spf=int(fmt[1] or 1)
                gainmatch=re.fullmatch(r'([\d.eE+\-]+)\((-?\d+)\)/.+',fields[2]);assert gainmatch
                gain=float(gainmatch[1]);baseline=int(gainmatch[2])
                start_frame=max(cursor,frame_start);stop_frame=min(cursor+length,frame_stop)
                begin=(start_frame-cursor)*spf;count=(stop_frame-start_frame)*spf
                file_channels=[v for v in lines if v[0]==fields[0]]
                channel_index=file_channels.index(fields)
                with sf.SoundFile(directory/fields[0]) as audio:
                    audio.seek(begin);digital=audio.read(count,dtype='int16',always_2d=True)[:,channel_index]
                assert len(digital)==count
                frames=digital.reshape(-1,spf)
                sentinel=(frames==-32768)
                assert not np.any(sentinel.any(axis=1)&~sentinel.all(axis=1)),'Partial missing subframe needs investigation'
                # WFDB smooth_frames=True averages digital counts into an integer array before DAC.
                averaged=np.trunc(frames.mean(axis=1))
                reference=(averaged-baseline)/gain;reference[averaged==-32768]=np.nan
                got=decoded_channels[label][start_frame-frame_start:stop_frame-frame_start]
                valid=np.isfinite(reference)&np.isfinite(got)
                assert np.array_equal(np.isfinite(reference),np.isfinite(got))
                if valid.any():
                    maximum=float(np.max(abs(reference[valid]-got[valid])))
                    assert np.allclose(reference[valid],got[valid],rtol=2e-6,atol=1e-5)
                    previous=checked.get(label,dict(frames=0,maximum_absolute_difference=0.))
                    checked[label]=dict(frames=previous['frames']+int(valid.sum()),
                        maximum_absolute_difference=max(previous['maximum_absolute_difference'],maximum),samples_per_frame=spf)
        cursor+=length
    assert set(checked)=={'resp','pleth'}
    validation.append(dict(record_number=len(validation)+1,valid_numeric_rows=len(times),
        maximum_numeric_clock_difference_seconds=error,channels=checked))
    z.close()
assert len(validation)==11
wj('independent_input_validation.json',dict(records_checked=11,plain_header_and_csv_clocks_match=True,
    direct_flac_calibration_matches=True,all_overlapping_available_channel_frames_checked=True,
    partial_missing_subframes_absent=True,rows=validation,
    checker_correction_sha256=sha_file(PROJECT/'docs/SPO2_WAVEFORM_CALIBRATION_VALIDATION_CORRECTION.md')))
manifest=json.loads((OUT/'analysis_manifest.json').read_text())
manifest['independent_input_validation_pass']=True
manifest['validation_completed_utc']=datetime.now(timezone.utc).isoformat()
wj('analysis_manifest.json',manifest)
pilot_status('frozen respiratory pilot and independent validation','complete')
print('All 11 records passed independent numeric-clock and direct FLAC calibration checks.')
'''
