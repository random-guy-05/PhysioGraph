"""Author and optionally execute the standalone observation-policy experiment."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import tempfile
import textwrap

import nbformat

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "PhysioGraph_Observation_Experiment.ipynb"
PLAN = ROOT / "docs/OBSERVATION_RESEARCH_PLAN.md"
cells = []


def md(source):
    cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(source).strip()))


def code(source):
    cells.append(nbformat.v4.new_code_cell(textwrap.dedent(source).strip()))


md("""
# Does SpO2 instability survive a change in recording schedule?

**Exploratory measurement research.** Run all cells in order in a fresh Colab
runtime. No Google Drive mount, private dataset, local package or credentials
are needed. Downloads only public BIDMC numerics (~0.3 MB plus checksum list).
All tables and figures are rebuilt. The outputs delivered with this notebook
were executed **locally**, not in Colab. First-time dependency setup needs internet.

This tests a specific normalization assumption in PhysioGraph; it does not
demonstrate a new disease mechanism, HF prognosis, causality or clinical utility.
The 53 short records originate in MIMIC-II and are not independent external
clinical validation of the MIMIC/eICU study. Phases are repeated measurements.
""")
md(PLAN.read_text())
code("""
import importlib.util, subprocess, sys
requirements = {'numpy': 'numpy==2.2.6', 'pandas': 'pandas==2.2.3',
                'scipy': 'scipy==1.15.3', 'matplotlib': 'matplotlib==3.10.3',
                'wfdb': 'wfdb==4.3.0'}
missing = [spec for name, spec in requirements.items() if importlib.util.find_spec(name) is None]
if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *missing])
import os, json, hashlib, platform, time, warnings, urllib.request
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.stats import spearmanr, rankdata
import wfdb
from IPython.display import display, Markdown

SEED = 20260905
OUT = Path('research/observation_audit')
DATA = OUT / 'data'
DATA.mkdir(parents=True, exist_ok=True)
START = datetime.now(timezone.utc).isoformat()
ENVIRONMENT = 'google_colab' if 'google.colab' in sys.modules else 'local'
CONFIG = dict(seed=SEED, cadences_seconds=[1,5,15,30,60], lag_seconds=60,
              min_pairs=3, min_coverage=0.90, bootstrap=2000, artificial_labels=500,
              simulation_replicates=1000, protocol='observation_policy_v1')
def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\\n')
save_json('run_status.json', dict(status='running', started_utc=START, environment=ENVIRONMENT))
print('Execution:', ENVIRONMENT, '| Seed:', SEED)
""")
code(f"PROTOCOL_SHA256 = {hashlib.sha256(PLAN.read_bytes()).hexdigest()!r}\nsave_json('frozen_protocol.json', dict(config=CONFIG, protocol_sha256=PROTOCOL_SHA256, frozen_utc=START))")
md("""
## Experiment 1: an analytic counterexample and simulation controls

For stationary OU variance $V$ and recovery time $\\tau$, the mean squared
increment at lag $h$ is $2V(1-e^{-h/\\tau})$. Independent measurement noise of
variance $s^2$ adds $2s^2$. PhysioGraph's normalization multiplies by $10/h$
when time is in minutes. Its expectation is constant in $h$ for Brownian
increments, but generally not for OU or measurement noise. We report the
square root of the pooled mean squared statistic, not the mean of roots.
Brownian trajectories are mathematical controls, not plausible saturations.
""")
code("""
def normalized_rmssd(times, values, max_gap=np.inf, factor=600.0):
    t, x = np.asarray(times), np.asarray(values)
    dt, dx = np.diff(t), np.diff(x)
    ok = np.isfinite(x[:-1]) & np.isfinite(x[1:]) & (dt > 0) & (dt <= max_gap)
    if ok.sum() < CONFIG['min_pairs']:
        return np.nan
    return float(np.sqrt(factor * np.sum(dx[ok]**2) / np.sum(dt[ok])))

def fixed_lag_rmssd(times, values, lag=60):
    t, x = np.asarray(times, dtype=int), np.asarray(values, dtype=float)
    lookup = dict(zip(t, x))
    diffs = [lookup[int(ti + lag)] - xi for ti, xi in zip(t, x)
             if int(ti + lag) in lookup and np.isfinite(xi) and np.isfinite(lookup[int(ti + lag)])]
    return float(np.sqrt(np.mean(np.square(diffs)))) if len(diffs) >= CONFIG['min_pairs'] else np.nan

def measures(times, values):
    t, x = np.asarray(times), np.asarray(values)
    valid = np.isfinite(x)
    ok = valid[:-1] & valid[1:]
    d = np.diff(x)[ok]
    return dict(normalized=normalized_rmssd(t,x), fixed60=fixed_lag_rmssd(t,x),
                raw_rmssd=float(np.sqrt(np.mean(d*d))) if len(d)>=3 else np.nan,
                sd=float(np.std(x[valid],ddof=1)) if valid.sum()>1 else np.nan,
                mean=float(np.mean(x[valid])) if valid.any() else np.nan,
                any_jump4=float(np.any(np.abs(d)>=4)) if len(d)>=3 else np.nan,
                pairs=int(ok.sum()))

# Meaningful controls: exact units, no bridging missing values, support failure.
assert np.isclose(normalized_rmssd([0,60,120,180], [90,91,92,93]), np.sqrt(10))
assert np.isnan(normalized_rmssd([0,1,2,3,4], [90,np.nan,98,np.nan,90]))
assert np.isnan(fixed_lag_rmssd([0,70,140,210,280], [90,91,92,93,94]))
assert np.isclose(fixed_lag_rmssd([0,60,120,180], [90,91,92,93]), 1)
assert np.isnan(normalized_rmssd([0,60,120,180], [90,91,92,93], max_gap=30))

rng = np.random.default_rng(SEED)
sim_rows=[]
for model, tau, noise in [('brownian',0,0), ('ou5',5,0), ('ou30',30,0),
                         ('ou120',120,0), ('ou30_noise',30,0.25)]:
    n = CONFIG['simulation_replicates']
    x = np.empty((n,241))
    x[:,0] = 0 if model=='brownian' else rng.normal(size=n)
    a = np.exp(-1/tau) if tau else 1.0
    scale = np.sqrt(1-a*a) if tau else 1.0
    for j in range(1,241):
        x[:,j] = a*x[:,j-1] + scale*rng.normal(size=n)
    if noise:
        x += rng.normal(scale=np.sqrt(noise),size=x.shape)
    for h in [1,5,15,60]:
        y=x[:,::h]
        estimate=np.sqrt(np.mean(np.diff(y,axis=1)**2)*10/h)
        expected=np.sqrt(10 if model=='brownian' else (2*(1-np.exp(-h/tau))+2*noise)*10/h)
        stride=60//h
        fixed=np.sqrt(np.mean((y[:,stride:]-y[:,:-stride])**2))
        sim_rows.append(dict(model=model,cadence_min=h,normalized_rms=estimate,
                             theoretical_rms=expected,fixed60min_rms=fixed))
simulation=pd.DataFrame(sim_rows)
assert np.max(np.abs(simulation.normalized_rms/simulation.theoretical_rms-1)) < 0.08
simulation.to_json(OUT/'simulation.json',orient='records',indent=2)
display(simulation.round(4))
""")
md("""
## Experiment 2: intervene on the recording schedule, keeping the signal fixed

The [BIDMC dataset](https://physionet.org/content/bidmc/1.0.0/) contains 53
eight-minute ICU recordings with 1 Hz monitor numerics. Cite Pimentel et al.,
[doi:10.1109/TBME.2016.2613124](https://doi.org/10.1109/TBME.2016.2613124),
dataset [doi:10.13026/C2208R](https://doi.org/10.13026/C2208R), and the
PhysioNet citation listed on the dataset page. Files are distributed under
the Open Data Commons Attribution License v1.0; the license is saved below.

These are processed monitor readings. Device averaging, quantization and
artifacts remain present. This experiment estimates the effect of additional
recording/thinning, not the error relative to true arterial oxygen saturation.
""")
code("""
BASE='https://physionet.org/files/bidmc/1.0.0/'
def fetch_bytes(name):
    req=urllib.request.Request(BASE+name,headers={'User-Agent':'PhysioGraph measurement research'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req,timeout=30) as response:
                return response.read()
        except Exception:
            if attempt==2: raise
            time.sleep(attempt+1)

checksum_payload=fetch_bytes('SHA256SUMS.txt')
published={}
for line in checksum_payload.decode().splitlines():
    digest,name=line.split(maxsplit=1)
    published[name.lstrip('*').removeprefix('./')]=digest
(DATA/'SHA256SUMS.txt').write_bytes(checksum_payload)
(DATA/'LICENSE').write_bytes(fetch_bytes('LICENSE'))
names=[f'bidmc{i:02d}n.{suffix}' for i in range(1,54) for suffix in ('hea','dat')]
def verified_download(name):
    path=DATA/name
    payload=path.read_bytes() if path.exists() else fetch_bytes(name)
    digest=hashlib.sha256(payload).hexdigest()
    if digest != published[name]:
        raise ValueError('Publisher checksum mismatch: '+name)
    if not path.exists(): path.write_bytes(payload)
    return dict(file=name,url=BASE+name,sha256=digest,bytes=len(payload))
with ThreadPoolExecutor(max_workers=6) as pool:
    provenance=list(pool.map(verified_download,names))
save_json('data_provenance.json', dict(source=BASE,files=provenance,
    checksum_manifest_sha256=hashlib.sha256(checksum_payload).hexdigest(),
    license='Open Data Commons Attribution License v1.0'))
signals={}
coverage=[]
for i in range(1,54):
    name=f'bidmc{i:02d}n'
    record=wfdb.rdrecord(str(DATA/name))
    assert record.fs==1 and record.sig_len>=480
    channels=[j for j,s in enumerate(record.sig_name) if 'spo2' in s.lower()]
    assert len(channels)==1
    x=record.p_signal[:480,channels[0]].copy()
    x[(x<50)|(x>100)|~np.isfinite(x)]=np.nan
    fraction=float(np.isfinite(x).mean())
    include=fraction>=CONFIG['min_coverage']
    coverage.append(dict(record=name,coverage=fraction,included=include,valid_samples=int(np.isfinite(x).sum())))
    if include: signals[name]=x
coverage=pd.DataFrame(coverage)
coverage.to_json(OUT/'coverage.json',orient='records',indent=2)
assert len(coverage)==53 and len(signals)>10
print('All 106 source files verified. Included:',len(signals),'of 53 recordings.')
display(coverage.loc[~coverage.included])
""")
code("""
rows=[]
reference={}
for name,x in signals.items():
    reference[name]=measures(np.arange(480),x)
    for cadence in CONFIG['cadences_seconds']:
        for phase in range(cadence):
            t=np.arange(phase,480,cadence)
            rows.append(dict(record=name,cadence=cadence,phase=phase,**measures(t,x[t])))
phase_table=pd.DataFrame(rows)
phase_table.to_json(OUT/'record_phase_metrics.json',orient='records',indent=2)
ref=pd.DataFrame.from_dict(reference,orient='index')
ref.index.name='record'
ref.reset_index().to_json(OUT/'native_reference.json',orient='records',indent=2)
metric_names=['normalized','fixed60','raw_rmssd','sd','mean']
summary=[]
for cadence in CONFIG['cadences_seconds']:
    subset=phase_table[phase_table.cadence==cadence]
    for metric in metric_names:
        baseline=subset.record.map(ref[metric])
        ok=(baseline>1e-8)&subset[metric].notna()
        z=subset.loc[ok,['record','phase']].copy()
        z['absolute_relative_drift']=np.abs(subset.loc[ok,metric]/baseline[ok]-1)
        per_record=z.groupby('record').absolute_relative_drift.mean()
        summary.append(dict(cadence=cadence,metric=metric,n_records=len(per_record),
            mean_absolute_relative_drift=float(per_record.mean()),
            median_absolute_relative_drift=float(per_record.median()),
            unavailable_phase_fraction=float(subset[metric].isna().mean()),
            native_zero_records=int((ref[metric]<=1e-8).sum())))
drift=pd.DataFrame(summary)
drift.to_json(OUT/'drift_summary.json',orient='records',indent=2)
display(drift.round(4))
""")
md("""
## Primary contrast and descriptive sensitivity checks

Each record is a bootstrap unit. The primary estimand averages absolute
relative drift across every phase **within each record first**; it does not
pool phases to obtain artificially precise estimates. Native near-zero
references are excluded explicitly on a common sample. The fixed-lag metric
uses a different, explicitly specified physical estimand; lower drift would
not establish superior clinical information. Confidence intervals are
pointwise; the primary contrast is the sole inferential test in this audit.
""")
code("""
common=ref.index[(ref.normalized>1e-8)&(ref.fixed60>1e-8)]
sub=phase_table[(phase_table.cadence==60)&phase_table.record.isin(common)].copy()
for metric in ['normalized','fixed60']:
    sub[metric+'_drift']=np.abs(sub[metric]/sub.record.map(ref[metric])-1)
record_drift=sub.groupby('record')[['normalized_drift','fixed60_drift']].mean().dropna()
delta=(record_drift.normalized_drift-record_drift.fixed60_drift).to_numpy()
assert len(delta)>=10
rng=np.random.default_rng(SEED+1)
draws=rng.integers(0,len(delta),size=(CONFIG['bootstrap'],len(delta)))
bootstrap=delta[draws].mean(axis=1)
primary=dict(n_records=len(delta),normalized_mean_drift=float(record_drift.normalized_drift.mean()),
    fixed60_mean_drift=float(record_drift.fixed60_drift.mean()),
    paired_mean_difference=float(delta.mean()),ci95=np.quantile(bootstrap,[.025,.975]).tolist(),
    bootstrap_repetitions=CONFIG['bootstrap'],unit='record',
    excluded_nonpositive_native_reference=int(len(ref)-len(common)),
    excluded_no_paired_phase_support=int(len(common)-len(record_drift)))
save_json('primary_result.json',primary)
record_drift.reset_index().to_json(OUT/'primary_record_drift.json',orient='records',indent=2)
display(pd.DataFrame([primary]))

rank_rows=[]
for cadence in CONFIG['cadences_seconds']:
    for metric in metric_names:
        correlations=[]
        for phase in range(cadence):
            sample=phase_table[(phase_table.cadence==cadence)&(phase_table.phase==phase)].set_index('record')
            aligned=sample[[metric]].join(ref[[metric]],rsuffix='_ref').dropna()
            if len(aligned)>3 and aligned[metric].nunique()>1 and aligned[metric+'_ref'].nunique()>1:
                correlations.append(float(spearmanr(aligned[metric],aligned[metric+'_ref']).statistic))
        rank_rows.append(dict(cadence=cadence,metric=metric,phases=len(correlations),
            median_rho=float(np.median(correlations)) if correlations else None,
            min_rho=float(np.min(correlations)) if correlations else None,
            max_rho=float(np.max(correlations)) if correlations else None))
ranks=pd.DataFrame(rank_rows)
ranks.to_json(OUT/'rank_stability.json',orient='records',indent=2)
jump_rows=[]
for cadence in CONFIG['cadences_seconds']:
    sample=phase_table[phase_table.cadence==cadence].copy()
    valid=sample.any_jump4.notna()
    sample=sample[valid]
    sample['switch']=(sample.any_jump4!=sample.record.map(ref.any_jump4)).astype(float)
    sample['created']=(sample.any_jump4.eq(1)&sample.record.map(ref.any_jump4).eq(0)).astype(float)
    jump_rows.append(dict(cadence=cadence,
        native_positive_records=int(ref.any_jump4.sum()),
        mean_record_switch_fraction=float(sample.groupby('record')['switch'].mean().mean()),
        mean_record_created_fraction=float(sample.groupby('record').created.mean().mean()),
        phase_unavailable=int((~valid).sum())))
jumps=pd.DataFrame(jump_rows)
jumps.to_json(OUT/'jump_relabeling.json',orient='records',indent=2)
display(ranks[ranks.cadence==60].round(3))
display(jumps.round(4))
""")
md("""
## Experiment 3: artificial labels and recording policy

Random labels contain no biological information by construction. We impose a
label-dependent schedule solely as a negative control. Each assignment uses
all eligible records, with one random phase per record in the sparse arm.
Native and sparse policies share labels. Distributions across assignments are
randomization summaries, **not clinical AUROCs or confidence intervals**.
""")
code("""
def auc(y,score):
    y=np.asarray(y); score=np.asarray(score)
    ok=np.isfinite(score); y=y[ok]; score=score[ok]
    n1=int(y.sum()); n0=len(y)-n1
    return float((rankdata(score)[y==1].sum()-n1*(n1+1)/2)/(n1*n0)) if n1 and n0 else np.nan

assert np.isclose(auc([0,1,0,1],[0,1,0,1]),1)
assert np.isclose(auc([0,1,0,1],[1,1,1,1]),.5)
names=list(ref.index)
sparse={m:phase_table[phase_table.cadence==60].pivot(index='record',columns='phase',values=m).loc[names].to_numpy()
        for m in ['normalized','fixed60']}
rng=np.random.default_rng(SEED+2)
control_rows=[]
for repetition in range(CONFIG['artificial_labels']):
    labels=np.zeros(len(names),dtype=int)
    labels[rng.choice(len(names),len(names)//2,replace=False)]=1
    phase=rng.integers(0,60,size=len(names))
    for metric in ['normalized','fixed60']:
        native=ref.loc[names,metric].to_numpy()
        policy=np.where(labels==1,sparse[metric][np.arange(len(names)),phase],native)
        control_rows.append(dict(repetition=repetition,metric=metric,
            native_auc=auc(labels,native),policy_auc=auc(labels,policy)))
controls=pd.DataFrame(control_rows)
controls.to_json(OUT/'artificial_label_replicates.json',orient='records',indent=2)
control_summary=controls.groupby('metric')[['native_auc','policy_auc']].agg(['mean','std'])
display(control_summary.round(4))
""")
md("""
## Figures and auditable output manifest

Intervals on the primary contrast quantify between-record sampling uncertainty.
Phase ranges on rank correlations quantify recording-clock sensitivity. Neither
is uncertainty about unobserved HF outcomes. All source numerics and resulting
tables are retained, including exclusions and controls.
""")
code("""
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors={'normalized':'#c65331','fixed60':'#167f83','raw_rmssd':'#9a7d0a','sd':'#75529b','mean':'#526878'}
fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
for model,g in simulation.groupby('model',sort=False):
    axes[0].plot(g.cadence_min,g.normalized_rms/g.normalized_rms.iloc[0],marker='o',label=model)
axes[0].set(xscale='log',xlabel='Simulation cadence (minutes)',ylabel='Normalized RMS / 1-minute value',title='A. Normalization depends on dynamics')
axes[0].legend(fontsize=8)
for metric in ['normalized','fixed60','sd','mean']:
    g=drift[drift.metric==metric]
    axes[1].plot(g.cadence,100*g.mean_absolute_relative_drift,marker='o',label=metric,color=colors[metric])
axes[1].set(xscale='log',xlabel='BIDMC recording cadence (seconds)',ylabel='Mean absolute relative drift (%)',title='B. Same recordings, different schedules')
axes[1].legend(fontsize=8)
for metric in ['normalized','fixed60','sd']:
    g=ranks[ranks.metric==metric]
    axes[2].plot(g.cadence,g.median_rho,marker='o',label=metric,color=colors[metric])
    axes[2].fill_between(g.cadence,g.min_rho,g.max_rho,color=colors[metric],alpha=.12)
axes[2].set(xscale='log',xlabel='BIDMC recording cadence (seconds)',ylabel='Spearman rank correlation',title='C. Record ranking; shading = phase range')
axes[2].legend(fontsize=8)
fig.suptitle('SpO2 observation-policy audit | real monitor numerics + labeled simulations',fontsize=13)
fig.savefig(OUT/'observation_audit.png',dpi=180)
fig.savefig(OUT/'observation_audit.svg')
plt.show()

fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,metric in zip(axes,['normalized','fixed60']):
    g=controls[controls.metric==metric]
    ax.hist(g.native_auc,bins=np.linspace(0,1,31),alpha=.55,label='Common 1-second policy',color='#526878')
    ax.hist(g.policy_auc,bins=np.linspace(0,1,31),alpha=.55,label='Artificial label-dependent policy',color=colors[metric])
    ax.axvline(.5,color='black',linestyle='--',lw=1)
    ax.set(xlabel='AUROC for artificial random labels',ylabel='Assignments',title=metric,xlim=(0,1))
    ax.legend(fontsize=8)
fig.suptitle('Negative control: no clinical labels or clinical prediction claim')
fig.savefig(OUT/'artificial_label_control.png',dpi=180)
plt.show()
""")
code("""
software={name:importlib.import_module(name).__version__ for name in ['numpy','pandas','scipy','matplotlib','wfdb']}
outputs={p.name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
         for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in ['manifest.json','run_status.json']}
save_json('manifest.json',dict(environment=ENVIRONMENT,started_utc=START,
    finished_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,
    platform=platform.platform(),software=software,config=CONFIG,
    protocol_sha256=PROTOCOL_SHA256,source='public BIDMC v1.0.0 monitor numerics',
    outputs=outputs,clinical_endpoints_tested=0,novelty_confirmed=False))
save_json('run_status.json',dict(status='complete',environment=ENVIRONMENT,started_utc=START,
    finished_utc=datetime.now(timezone.utc).isoformat(),included_records=len(signals)))
ci=primary['ci95']
print(f"Primary paired mean drift difference: {100*primary['paired_mean_difference']:.1f} percentage points; "
      f"record bootstrap 95% CI {100*ci[0]:.1f} to {100*ci[1]:.1f}; n={primary['n_records']}.")
print('Completed', ENVIRONMENT, '| output directory:', OUT.resolve())
print('A measurement-method finding is not a new biological mechanism or validated clinical biomarker.')
""")
md("""
## Research decision

Interpret the primary contrast and controls above before changing any clinical
claim. If drift is reduced at a common physical lag, prioritize measurement
invariance and support diagnostics in the next clinical protocol. If not,
retain the negative result and reject this comparator for the tested setting.
The experiment does not establish the cause of MIMIC/eICU transport failure.

The next clinical study needs long HF monitor recordings and actual endpoints.
Use fixed physical lag support, independently estimated recording schedules,
one frozen primary endpoint and patient/site-held-out evaluation. Recovery,
sensor noise and clinician actions require separate measurements or explicit
identifiability assumptions. Existing literature already establishes sampling
effects and structure-function methods. Any novel contribution remains a
candidate until a focused literature comparison and independent replication.
""")


md((ROOT / 'docs/OBSERVATION_REPLICATION_PLAN.md').read_text())
code(f"REPLICATION_PROTOCOL_SHA256 = {hashlib.sha256((ROOT / 'docs/OBSERVATION_REPLICATION_PLAN.md').read_bytes()).hexdigest()!r}")
code("""
# Independent replication. Decisions above precede OSV signal downloads.
save_json('run_status.json', dict(status='running', stage='independent_replication',
    environment=ENVIRONMENT, started_utc=START))
save_json('replication_protocol.json', dict(sha256=REPLICATION_PROTOCOL_SHA256,
    frozen_utc=datetime.now(timezone.utc).isoformat(), bootstrap_seed=20260915))
OSV_BASE='https://physionet.org/files/osv/1.0.0/'
OSV_DATA=DATA/'osv'
OSV_DATA.mkdir(exist_ok=True)
def osv_fetch(name):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(OSV_BASE+name,timeout=30) as response:
                return response.read()
        except Exception:
            if attempt==2: raise
            time.sleep(attempt+1)
osv_checksum=osv_fetch('SHA256SUMS.txt')
osv_published={}
for line in osv_checksum.decode().splitlines():
    digest,name=line.split(maxsplit=1)
    osv_published[name.lstrip('*').removeprefix('./')]=digest
(OSV_DATA/'SHA256SUMS.txt').write_bytes(osv_checksum)
def osv_download(name):
    path=OSV_DATA/name
    payload=path.read_bytes() if path.exists() else osv_fetch(name)
    digest=hashlib.sha256(payload).hexdigest()
    if digest != osv_published[name]: raise ValueError('OSV checksum mismatch: '+name)
    if not path.exists(): path.write_bytes(payload)
    return dict(file=name,url=OSV_BASE+name,sha256=digest,bytes=len(payload))
osv_provenance=[osv_download('RECORDS')]
osv_names=(OSV_DATA/'RECORDS').read_text().splitlines()
assert len(osv_names)==36 and len(set(osv_names))==36
with ThreadPoolExecutor(max_workers=6) as pool:
    osv_provenance+=list(pool.map(osv_download,[n+'.'+s for n in osv_names for s in ['hea','dat']]))
save_json('osv_data_provenance.json',dict(source=OSV_BASE,files=osv_provenance,
    checksum_manifest_sha256=hashlib.sha256(osv_checksum).hexdigest(),
    license='Open Data Commons Attribution License v1.0',
    license_source='https://physionet.org/content/osv/1.0.0/'))
osv_signals={}; osv_coverage=[]
for name in osv_names:
    record=wfdb.rdrecord(str(OSV_DATA/name))
    assert record.fs==1 and record.n_sig==1
    x=record.p_signal[:3600,0].copy()
    x[(x<50)|(x>100)|~np.isfinite(x)]=np.nan
    fraction=float(np.isfinite(x).mean())
    include=len(x)>=3500 and fraction>=CONFIG['min_coverage']
    osv_coverage.append(dict(record=name,samples=len(x),coverage=fraction,included=include))
    if include: osv_signals[name]=x
pd.DataFrame(osv_coverage).to_json(OUT/'osv_coverage.json',orient='records',indent=2)
assert len(osv_signals)>=10
osv_rows=[]; osv_reference={}
for name,x in osv_signals.items():
    osv_reference[name]=measures(np.arange(len(x)),x)
    for cadence in CONFIG['cadences_seconds']:
        for phase in range(cadence):
            t=np.arange(phase,len(x),cadence)
            osv_rows.append(dict(record=name,cadence=cadence,phase=phase,**measures(t,x[t])))
osv_phase=pd.DataFrame(osv_rows)
osv_ref=pd.DataFrame.from_dict(osv_reference,orient='index')
osv_ref.index.name='record'
osv_phase.to_json(OUT/'osv_record_phase_metrics.json',orient='records',indent=2)
osv_ref.reset_index().to_json(OUT/'osv_native_reference.json',orient='records',indent=2)
osv_common=osv_ref.index[(osv_ref.normalized>1e-8)&(osv_ref.fixed60>1e-8)]
osv_sub=osv_phase[(osv_phase.cadence==60)&osv_phase.record.isin(osv_common)].copy()
for metric in ['normalized','fixed60']:
    osv_sub[metric+'_drift']=np.abs(osv_sub[metric]/osv_sub.record.map(osv_ref[metric])-1)
osv_record_drift=osv_sub.groupby('record')[['normalized_drift','fixed60_drift']].mean().dropna()
osv_delta=(osv_record_drift.normalized_drift-osv_record_drift.fixed60_drift).to_numpy()
rng=np.random.default_rng(20260915)
osv_boot=osv_delta[rng.integers(0,len(osv_delta),size=(2000,len(osv_delta)))].mean(axis=1)
osv_primary=dict(n_records=len(osv_delta),
    normalized_mean_drift=float(osv_record_drift.normalized_drift.mean()),
    fixed60_mean_drift=float(osv_record_drift.fixed60_drift.mean()),
    paired_mean_difference=float(osv_delta.mean()),ci95=np.quantile(osv_boot,[.025,.975]).tolist(),
    excluded_nonpositive_native_reference=len(osv_ref)-len(osv_common),
    excluded_no_paired_phase_support=len(osv_common)-len(osv_record_drift),
    bootstrap_repetitions=2000,unit='record',clinical_validation=False)
save_json('osv_primary_result.json',osv_primary)
osv_record_drift.reset_index().to_json(OUT/'osv_primary_record_drift.json',orient='records',indent=2)
display(pd.DataFrame([dict(dataset='BIDMC',**primary),dict(dataset='OSV',**osv_primary)]))
""")
code("""
identity_rows=[]; replication_summary=[]; osv_controls=[]
for dataset,collection,ptable,rtable in [('BIDMC',signals,phase_table,ref),('OSV',osv_signals,osv_phase,osv_ref)]:
    for name,x in collection.items():
        sparse_row=ptable[(ptable.record==name)&(ptable.cadence==60)]
        supported=sparse_row[['normalized','fixed60']].dropna()
        scale_error=float(np.max(np.abs(supported.normalized-np.sqrt(10)*supported.fixed60)))
        assert scale_error < 1e-10
        complete=bool(np.isfinite(x).all())
        partition_error=None
        if complete:
            pooled=float(np.average(sparse_row.fixed60**2,weights=sparse_row.pairs))
            partition_error=abs(pooled-rtable.loc[name,'fixed60']**2)
            assert partition_error < 1e-10
        identity_rows.append(dict(dataset=dataset,record=name,complete=complete,
            scale_identity_error=scale_error,pair_partition_error=partition_error))
    for cadence in CONFIG['cadences_seconds']:
        sample=ptable[ptable.cadence==cadence]
        for metric in ['normalized','fixed60','sd','mean']:
            base=sample.record.map(rtable[metric])
            valid=(base>1e-8)&sample[metric].notna()
            rel=np.abs(sample.loc[valid,metric]/base[valid]-1)
            per_record=rel.groupby(sample.loc[valid,'record']).mean()
            rho=[]
            for phase in range(cadence):
                aligned=sample[sample.phase==phase].set_index('record')[[metric]].join(rtable[[metric]],rsuffix='_native').dropna()
                if len(aligned)>3 and aligned[metric].nunique()>1:
                    rho.append(float(spearmanr(aligned[metric],aligned[metric+'_native']).statistic))
            replication_summary.append(dict(dataset=dataset,cadence=cadence,metric=metric,
                n_records=len(per_record),mean_absolute_relative_drift=float(per_record.mean()),
                median_rank_rho=float(np.median(rho)),min_rank_rho=float(np.min(rho)),max_rank_rho=float(np.max(rho)),
                unavailable_phase_fraction=float(sample[metric].isna().mean()),
                native_zero_records=int((rtable[metric]<=1e-8).sum())))
names=list(osv_ref.index)
osv_sparse={m:osv_phase[osv_phase.cadence==60].pivot(index='record',columns='phase',values=m).loc[names].to_numpy()
            for m in ['normalized','fixed60']}
rng=np.random.default_rng(20260916)
for repetition in range(500):
    labels=np.zeros(len(names),dtype=int)
    labels[rng.choice(len(names),len(names)//2,replace=False)]=1
    phase=rng.integers(0,60,size=len(names))
    for metric in ['normalized','fixed60']:
        native=osv_ref.loc[names,metric].to_numpy()
        policy=np.where(labels==1,osv_sparse[metric][np.arange(len(names)),phase],native)
        osv_controls.append(dict(repetition=repetition,metric=metric,native_auc=auc(labels,native),policy_auc=auc(labels,policy)))
pd.DataFrame(identity_rows).to_json(OUT/'estimator_identity_checks.json',orient='records',indent=2)
replication_summary=pd.DataFrame(replication_summary)
replication_summary.to_json(OUT/'replication_summary.json',orient='records',indent=2)
osv_controls=pd.DataFrame(osv_controls)
osv_controls.to_json(OUT/'osv_artificial_label_replicates.json',orient='records',indent=2)
display(replication_summary[replication_summary.cadence==60].round(4))
display(osv_controls.groupby('metric')[['native_auc','policy_auc']].mean().round(4))
fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
for ax,dataset in zip(axes,['BIDMC','OSV']):
    for metric in ['normalized','fixed60','sd','mean']:
        g=replication_summary[(replication_summary.dataset==dataset)&(replication_summary.metric==metric)]
        ax.plot(g.cadence,100*g.mean_absolute_relative_drift,marker='o',label=metric,color=colors[metric])
    ax.set(xscale='log',xlabel='Recording cadence (seconds)',ylabel='Mean absolute relative drift (%)',title=dataset)
    ax.legend(fontsize=8)
fig.suptitle('Independent measurement replication | no clinical endpoints')
fig.savefig(OUT/'independent_replication.png',dpi=180)
plt.show()
manifest=json.loads((OUT/'manifest.json').read_text())
manifest.update(replication_protocol_sha256=REPLICATION_PROTOCOL_SHA256,
    replication_source=OSV_BASE,replication_records=len(osv_signals),
    finished_utc=datetime.now(timezone.utc).isoformat())
manifest['outputs']={p.name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
    for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in ['manifest.json','run_status.json']}
save_json('manifest.json',manifest)
save_json('run_status.json',dict(status='complete',environment=ENVIRONMENT,started_utc=START,
    finished_utc=datetime.now(timezone.utc).isoformat(),included_records=len(signals),replication_records=len(osv_signals)))
print('Both datasets executed:',ENVIRONMENT,'| Estimator identity checks passed.')
print('Paradigm-shifting discovery: NOT established. Clinical endpoints tested: 0.')
""")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    nb=nbformat.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':platform_version()},
        'colab':{'name':TARGET.name,'provenance':[]},
        'physiograph':{'protocol_sha256':hashlib.sha256(PLAN.read_bytes()).hexdigest()}})
    for i,cell in enumerate(nb.cells):
        cell.id=f'observation-{i:02d}'
    nbformat.validate(nb)
    if TARGET.exists():
        previous=nbformat.read(TARGET,as_version=4)
        if len(previous.cells)>len(nb.cells):
            raise RuntimeError('Refusing to remove pre-existing notebook cells.')
        for old,new in zip(previous.cells,nb.cells):
            if old.source!=new.source:
                old.source=new.source
                if old.cell_type=='code': old.outputs=[]; old.execution_count=None
        previous.cells.extend(nb.cells[len(previous.cells):])
        nb=previous
    nbformat.write(nb,TARGET)
    if args.execute:
        from nbclient import NotebookClient
        from jupyter_client import KernelManager
        from jupyter_client.kernelspec import KernelSpecManager
        with tempfile.TemporaryDirectory(prefix='physiograph-kernel-') as directory:
            folder=Path(directory)/'audit'
            folder.mkdir()
            (folder/'kernel.json').write_text(json.dumps(dict(
                argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
                display_name='Audit runtime',language='python')))
            manager=KernelManager(kernel_name='audit',kernel_spec_manager=KernelSpecManager(kernel_dirs=[directory]))
            client=NotebookClient(nb,km=manager,timeout=600,startup_timeout=180,
                resources={'metadata':{'path':str(ROOT)}})
            try:
                client.execute()
            except Exception as exc:
                output=ROOT/'research/observation_audit'
                output.mkdir(parents=True,exist_ok=True)
                (output/'run_status.json').write_text(json.dumps(dict(
                    status='failed',environment='local',error_type=type(exc).__name__)))
                raise
            finally:
                nbformat.write(nb,TARGET)
        print('Executed notebook:',TARGET)
    else:
        print('Authored notebook:',TARGET)


def platform_version():
    return '.'.join(map(str,sys.version_info[:3]))


if __name__=='__main__':
    main()
