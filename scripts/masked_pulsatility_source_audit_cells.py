"""Outcome-blind source forensic audit cells for masked pulsatility."""

SETUP = r'''
import hashlib,importlib.util,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
requirements={'duckdb':'duckdb==1.4.3','pandas':'pandas==2.2.3','tabulate':'tabulate==0.9.0'}
missing=[package for module,package in requirements.items() if importlib.util.find_spec(module) is None]
if missing:subprocess.check_call([sys.executable,'-m','pip','install','-q',*missing])
import duckdb,pandas as pd
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph';PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905';EICU=DRIVE/'Data/eICU/Full';MIMIC=DRIVE/'Data/MIMIC/Full'
OUT=PROJECT/'research/masked_pulsatility';OUT.mkdir(parents=True,exist_ok=True)
def audit_json(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def aq(value):return "'"+str(value).replace("'","''")+"'"
plan=PROJECT/'docs/MASKED_PULSATILITY_SOURCE_AUDIT_PLAN.md';plan_sha=hashlib.sha256(plan.read_bytes()).hexdigest();assert plan_sha==MASKED_PULSATILITY_SOURCE_AUDIT_SHA256
availability={'eicu_vital_periodic':(EICU/'vitalPeriodic.csv').is_file(),'eicu_vital_aperiodic':(EICU/'vitalAperiodic.csv').is_file(),'eicu_nurse_charting':(EICU/'nurseCharting.csv').is_file(),'mimic_chartevents':(MIMIC/'chartevents.csv').is_file()}
audit_json('source_forensic_protocol.json',dict(sha256=plan_sha,frozen_utc=datetime.now(timezone.utc).isoformat(),outcome_blind=True,frozen_analysis_unchanged=True,source_availability=availability))
phase_db=PRIVATE/'masked_pulsatility_phase_a.duckdb';assert phase_db.is_file()
try:audit_con.close()
except (NameError,AttributeError):pass
audit_db=PRIVATE/'masked_pulsatility_source_audit.duckdb';audit_con=duckdb.connect(str(audit_db));audit_con.execute('SET threads=2');audit_con.execute("SET memory_limit='3GB'");audit_con.execute('SET enable_progress_bar=false')
audit_con.execute(f"ATTACH {aq(phase_db)} AS phase (READ_ONLY)")
def audit_exists(name):return bool(audit_con.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema='main' AND table_name=?",[name]).fetchone()[0])
print(json.dumps(availability,indent=2),flush=True)
'''

EXTRACT = r'''
if not audit_exists('eicu_bp_rows'):
 audit_con.execute(f"""
 CREATE TABLE eicu_bp_rows AS
 SELECT 'eicu' dataset,c.stay_id,try_cast(v.observationoffset AS DOUBLE) event_minute,'invasive_arterial' AS "source",
  try_cast(v.systemicsystolic AS DOUBLE) sbp,try_cast(v.systemicdiastolic AS DOUBLE) dbp,try_cast(v.systemicmean AS DOUBLE) AS "map"
 FROM read_csv({aq(EICU/'vitalPeriodic.csv')},header=true,all_varchar=true,parallel=true) v
 JOIN phase.eicu_cohort c ON try_cast(v.patientunitstayid AS BIGINT)=c.stay_id
 WHERE try_cast(v.observationoffset AS DOUBLE)>=0 AND try_cast(v.observationoffset AS DOUBLE)<240
  AND (try_cast(v.systemicsystolic AS DOUBLE) IS NOT NULL OR try_cast(v.systemicdiastolic AS DOUBLE) IS NOT NULL)
 """)
if availability['eicu_vital_aperiodic'] and not audit_exists('eicu_aperiodic_bp_rows'):
 audit_con.execute(f"""
 CREATE TABLE eicu_aperiodic_bp_rows AS
 SELECT 'eicu' dataset,c.stay_id,try_cast(v.observationoffset AS DOUBLE) event_minute,'noninvasive_cuff' AS "source",
  try_cast(v.noninvasivesystolic AS DOUBLE) sbp,try_cast(v.noninvasivediastolic AS DOUBLE) dbp,try_cast(v.noninvasivemean AS DOUBLE) AS "map"
 FROM read_csv({aq(EICU/'vitalAperiodic.csv')},header=true,all_varchar=true,parallel=true) v
 JOIN phase.eicu_cohort c ON try_cast(v.patientunitstayid AS BIGINT)=c.stay_id
 WHERE try_cast(v.observationoffset AS DOUBLE)>=0 AND try_cast(v.observationoffset AS DOUBLE)<240
 """)
if not audit_exists('mimic_bp_rows'):
 audit_con.execute(f"""
 CREATE TABLE mimic_bp_rows AS
 WITH selected AS (
  SELECT c.stay_id,date_diff('second',c.admit_time,try_cast(v.charttime AS TIMESTAMP))/60.0 event_minute,
   CASE WHEN try_cast(v.itemid AS BIGINT) IN (220050,220051,220052) THEN 'invasive_arterial' ELSE 'noninvasive_cuff' END AS "source",
   try_cast(v.itemid AS BIGINT) itemid,try_cast(v.valuenum AS DOUBLE) val
  FROM read_csv({aq(MIMIC/'chartevents.csv')},header=true,all_varchar=true,parallel=true) v JOIN phase.mimic_cohort c ON try_cast(v.stay_id AS BIGINT)=c.stay_id
  WHERE try_cast(v.itemid AS BIGINT) IN (220050,220051,220052,220179,220180,220181)
   AND coalesce(try_cast(v.warning AS BIGINT),0)<>1
   AND date_diff('second',c.admit_time,try_cast(v.charttime AS TIMESTAMP))/60.0>=0
   AND date_diff('second',c.admit_time,try_cast(v.charttime AS TIMESTAMP))/60.0<240
 )
 SELECT 'mimic' dataset,stay_id,event_minute,"source",
  median(val) FILTER(WHERE itemid IN (220050,220179)) sbp,
  median(val) FILTER(WHERE itemid IN (220051,220180)) dbp,
  median(val) FILTER(WHERE itemid IN (220052,220181)) AS "map"
 FROM selected GROUP BY stay_id,event_minute,"source"
 """)
if not audit_exists('valid_bp_pairs'):
 parts=["SELECT * FROM eicu_bp_rows","SELECT * FROM mimic_bp_rows"]
 if availability['eicu_vital_aperiodic']:parts.append("SELECT * FROM eicu_aperiodic_bp_rows")
 audit_con.execute("CREATE TABLE IF NOT EXISTS all_bp_rows AS "+" UNION ALL ".join(parts))
 audit_con.execute("""
 CREATE TABLE valid_bp_pairs AS SELECT *,floor(event_minute/60)::INTEGER bp_hour
 FROM all_bp_rows WHERE sbp BETWEEN 50 AND 300 AND dbp BETWEEN 20 AND 200 AND sbp>dbp
 """)
 audit_con.execute("""
 CREATE TABLE hourly_bp_pairs AS
 SELECT dataset,stay_id,"source",bp_hour,max(event_minute) anchor_minute,median(sbp) sbp,median(dbp) dbp,
  median("map") AS "map",(median(sbp)-median(dbp))/median(sbp) prpp,count(*) measurement_count
 FROM valid_bp_pairs GROUP BY dataset,stay_id,"source",bp_hour
 """)
if not audit_exists('crossing_anchors'):
 audit_con.execute("""
 CREATE TABLE crossing_anchors AS
 SELECT dataset,stay_id,"source",candidate_id,anchor_minute,intervention_free,baseline_lactate
 FROM phase.candidate_screen WHERE candidate_type='crossing'
 """)
if not audit_exists('mimic_post_pressor'):
 audit_con.execute(f"""
 CREATE TABLE mimic_post_pressor AS
 SELECT DISTINCT a.candidate_id,date_diff('second',c.admit_time,try_cast(i.starttime AS TIMESTAMP))/60.0 event_minute
 FROM crossing_anchors a JOIN phase.mimic_cohort c USING(stay_id)
 JOIN read_csv({aq(MIMIC/'inputevents.csv')},header=true,all_varchar=true,parallel=true) i ON try_cast(i.stay_id AS BIGINT)=a.stay_id
 WHERE a.dataset='mimic' AND a.intervention_free
  AND try_cast(i.itemid AS BIGINT) IN (221906,221289,221662,221653,221749,222315,221986)
  AND coalesce(try_cast(i.rate AS DOUBLE),try_cast(i.amount AS DOUBLE),0)>0
  AND date_diff('second',c.admit_time,try_cast(i.starttime AS TIMESTAMP))/60.0>a.anchor_minute
  AND date_diff('second',c.admit_time,try_cast(i.starttime AS TIMESTAMP))/60.0<=a.anchor_minute+720
 """)
if not audit_exists('eicu_post_pressor'):
 audit_con.execute(f"""
 CREATE TABLE eicu_post_pressor AS
 SELECT DISTINCT a.candidate_id,try_cast(i.infusionoffset AS DOUBLE) event_minute
 FROM crossing_anchors a
 JOIN read_csv({aq(EICU/'infusionDrug.csv')},header=true,all_varchar=true,parallel=true) i ON try_cast(i.patientunitstayid AS BIGINT)=a.stay_id
 WHERE a.dataset='eicu' AND a.intervention_free
  AND regexp_matches(lower(coalesce(i.drugname,'')),'norepinephrine|levophed|epinephrine|adrenalin|phenylephrine|neosynephrine|vasopressin|dopamine|dobutamine|milrinone')
  AND coalesce(try_cast(i.drugrate AS DOUBLE),try_cast(i.infusionrate AS DOUBLE),try_cast(regexp_extract(coalesce(i.drugrate,i.infusionrate,''),'[-+]?[0-9]*\\.?[0-9]+',0) AS DOUBLE),0)>0
  AND try_cast(i.infusionoffset AS DOUBLE)>a.anchor_minute AND try_cast(i.infusionoffset AS DOUBLE)<=a.anchor_minute+720
 """)
print('forensic extraction complete',flush=True)
'''

REPORT = r'''
def scalar(sql):return audit_con.execute(sql).fetchone()[0]
def source_row(dataset,source,available,reason=None):
 if not available:return dict(dataset=dataset,source=source,source_available=False,unavailable_reason=reason,any_sbp_or_dbp_stays=None,paired_sbp_dbp_stays=None,ge3_hourly_pair_bins=None,initially_low=None,incident_pair_crossings=None)
 where=f"dataset={aq(dataset)} AND \"source\"={aq(source)}"
 any_stays=scalar(f"SELECT count(DISTINCT stay_id) FROM all_bp_rows WHERE {where} AND (sbp IS NOT NULL OR dbp IS NOT NULL)")
 paired=scalar(f"SELECT count(DISTINCT stay_id) FROM valid_bp_pairs WHERE {where}")
 stats=audit_con.execute(f"""WITH s AS (SELECT stay_id,count(*) bins,arg_min(prpp,bp_hour) first_prpp,min(bp_hour) FILTER(WHERE prpp<0.25) first_low,min(bp_hour) first_hour FROM hourly_bp_pairs WHERE {where} GROUP BY stay_id) SELECT count(*) FILTER(WHERE bins>=3),count(*) FILTER(WHERE first_prpp<0.25),count(*) FILTER(WHERE first_prpp>=0.25 AND first_low>first_hour) FROM s""").fetchone()
 return dict(dataset=dataset,source=source,source_available=True,unavailable_reason=None,any_sbp_or_dbp_stays=int(any_stays),paired_sbp_dbp_stays=int(paired),ge3_hourly_pair_bins=int(stats[0]),initially_low=int(stats[1]),incident_pair_crossings=int(stats[2]))
sources=[
 source_row('eicu','invasive_arterial',True),
 source_row('eicu','noninvasive_cuff_vitalAperiodic',availability['eicu_vital_aperiodic'],'vitalAperiodic.csv absent from local eICU export'),
 source_row('eicu','noninvasive_nurseCharting',availability['eicu_nurse_charting'],'nurseCharting.csv absent from local eICU export'),
 source_row('mimic','invasive_arterial',True),source_row('mimic','noninvasive_cuff',True)]
def union_row(dataset):
 any_stays=scalar(f"SELECT count(DISTINCT stay_id) FROM all_bp_rows WHERE dataset={aq(dataset)} AND (sbp IS NOT NULL OR dbp IS NOT NULL)")
 paired=scalar(f"SELECT count(DISTINCT stay_id) FROM valid_bp_pairs WHERE dataset={aq(dataset)}")
 stats=audit_con.execute(f"""WITH n AS (SELECT stay_id,"source",count(*) bins FROM hourly_bp_pairs WHERE dataset={aq(dataset)} GROUP BY 1,2),chosen AS (SELECT *,row_number() OVER(PARTITION BY stay_id ORDER BY bins DESC,CASE "source" WHEN 'invasive_arterial' THEN 0 WHEN 'noninvasive_cuff' THEN 1 ELSE 2 END) rn FROM n),s AS (SELECT h.stay_id,count(*) bins,arg_min(h.prpp,h.bp_hour) first_prpp,min(h.bp_hour) FILTER(WHERE h.prpp<0.25) first_low,min(h.bp_hour) first_hour FROM hourly_bp_pairs h JOIN chosen c ON h.stay_id=c.stay_id AND h."source"=c."source" AND c.rn=1 WHERE h.dataset={aq(dataset)} GROUP BY h.stay_id) SELECT count(*) FILTER(WHERE bins>=3),count(*) FILTER(WHERE first_prpp<0.25),count(*) FILTER(WHERE first_prpp>=0.25 AND first_low>first_hour) FROM s""").fetchone()
 return dict(dataset=dataset,source='source_consistent_union',source_available=True,unavailable_reason=None,any_sbp_or_dbp_stays=int(any_stays),paired_sbp_dbp_stays=int(paired),ge3_hourly_pair_bins=int(stats[0]),initially_low=int(stats[1]),incident_pair_crossings=int(stats[2]))
sources.extend([union_row('eicu'),union_row('mimic')])
pd.DataFrame(sources).to_csv(OUT/'source_forensic_attrition.csv',index=False)
frozen=[]
for dataset in ['eicu','mimic']:
 cohort=int(scalar(f'SELECT count(*) FROM phase.{dataset}_cohort'))
 h=audit_con.execute(f"""WITH s AS (SELECT stay_id,count(*) bins,arg_min(prpp,hour) first_prpp,min(hour) FILTER(WHERE prpp<0.25) first_low,min(hour) first_hour FROM phase.hourly_bp WHERE dataset={aq(dataset)} GROUP BY stay_id) SELECT count(*),count(*) FILTER(WHERE bins>=3),count(*) FILTER(WHERE first_prpp>=0.25 AND first_low>first_hour) FROM s""").fetchone()
 c=audit_con.execute(f"""SELECT count(DISTINCT stay_id),count(DISTINCT stay_id) FILTER(WHERE baseline_lactate IS NULL),count(DISTINCT stay_id) FILTER(WHERE baseline_lactate<2),count(DISTINCT stay_id) FILTER(WHERE baseline_lactate>=2),count(DISTINCT stay_id) FILTER(WHERE intervention_free),count(DISTINCT stay_id) FILTER(WHERE baseline_lactate<2 AND intervention_free) FROM phase.candidate_screen WHERE dataset={aq(dataset)} AND candidate_type='crossing'""").fetchone()
 post=audit_con.execute(f"""SELECT count(DISTINCT p.candidate_id) FILTER(WHERE p.event_minute<=a.anchor_minute+360),count(DISTINCT p.candidate_id) FROM {dataset}_post_pressor p JOIN crossing_anchors a USING(candidate_id)""").fetchone()
 frozen.append(dict(dataset=dataset,hf_cohort=cohort,valid_source_selected_triplet_stays=int(h[0]),ge3_hourly_triplet_bins=int(h[1]),incident_crossings_before_compensation=int(h[2]),compensated_crossings=int(c[0]),crossings_no_preanchor_lactate=int(c[1]),crossings_lactate_lt2=int(c[2]),crossings_lactate_ge2=int(c[3]),crossings_without_prior_pressor_mcs=int(c[4]),crossings_lactate_lt2_without_prior_support=int(c[5]),subsequent_pressor_6h=int(post[0]),subsequent_pressor_12h=int(post[1])))
pd.DataFrame(frozen).to_csv(OUT/'source_forensic_frozen_attrition.csv',index=False)
mapping=[
 dict(dataset='eicu',source='vitalPeriodic.systemic*',file_present=True,configured=True,sbp_mapped=True,dbp_mapped=True,included_in_prior_notebook=True),
 dict(dataset='eicu',source='vitalAperiodic.noninvasive*',file_present=availability['eicu_vital_aperiodic'],configured=True,sbp_mapped=True,dbp_mapped=False,included_in_prior_notebook=False),
 dict(dataset='eicu',source='nurseCharting NIBP',file_present=availability['eicu_nurse_charting'],configured=False,sbp_mapped=False,dbp_mapped=False,included_in_prior_notebook=False),
 dict(dataset='mimic',source='chartevents invasive',file_present=True,configured=True,sbp_mapped=True,dbp_mapped=True,included_in_prior_notebook=True),
 dict(dataset='mimic',source='chartevents cuff',file_present=True,configured=True,sbp_mapped=True,dbp_mapped=True,included_in_prior_notebook=True)]
pd.DataFrame(mapping).to_csv(OUT/'source_forensic_mapping.csv',index=False)
concentration=[];lactate_coverage=[]
for dataset in ['eicu','mimic']:
 table=f'phase.{dataset}_bp_triplets'
 for source in [r[0] for r in audit_con.execute(f'SELECT DISTINCT "source" FROM {table} ORDER BY 1').fetchall()]:
  trip=audit_con.execute(f"""WITH n AS (SELECT stay_id,count(*) measurements FROM {table} WHERE "source"={aq(source)} GROUP BY stay_id) SELECT sum(measurements),count(*),median(measurements),max(measurements) FROM n""").fetchone()
  concentration.append(dict(dataset=dataset,source=source,complete_triplets=int(trip[0]),instrumented_stays=int(trip[1]),median_triplets_per_instrumented_stay=float(trip[2]),maximum_triplets_per_stay=int(trip[3])))
 earliest=audit_con.execute(f"""WITH x AS (SELECT *,row_number() OVER(PARTITION BY stay_id ORDER BY anchor_minute,candidate_id) rn FROM phase.candidate_screen WHERE dataset={aq(dataset)}) SELECT count(*),count(*) FILTER(WHERE baseline_lactate IS NULL),count(*) FILTER(WHERE baseline_lactate<2),count(*) FILTER(WHERE baseline_lactate>=2) FROM x WHERE rn=1""").fetchone()
 any_anchor=audit_con.execute(f"""SELECT count(DISTINCT stay_id) FILTER(WHERE baseline_lactate IS NULL),count(DISTINCT stay_id) FILTER(WHERE baseline_lactate<2),count(DISTINCT stay_id) FILTER(WHERE baseline_lactate>=2) FROM phase.candidate_screen WHERE dataset={aq(dataset)}""").fetchone()
 lactate_coverage.append(dict(dataset=dataset,compensated_stays=int(earliest[0]),earliest_candidate_no_lactate=int(earliest[1]),earliest_candidate_lactate_lt2=int(earliest[2]),earliest_candidate_lactate_ge2=int(earliest[3]),any_candidate_anchor_no_lactate=int(any_anchor[0]),any_candidate_anchor_lactate_lt2=int(any_anchor[1]),any_candidate_anchor_lactate_ge2=int(any_anchor[2]),any_anchor_counts_are_nonexclusive=True))
summary=dict(audit_status='completed',classification='invalid_for_intended_routine_bp_hypothesis_missing_sources',frozen_analysis_changed=False,clinical_outcomes_inspected=False,post_anchor_lactate_values_inspected=False,source_rows=sources,frozen_attrition=frozen,triplet_concentration=concentration,candidate_lactate_coverage=lactate_coverage,mapping=mapping,interpretation='The prior eICU gate evaluated an invasive arterial-line subset. Routine eICU cuff BP was not available to or included by the notebook, and the package vitalAperiodic mapping omits DBP. The biological hypothesis is not falsified.',recommended_action='obtain_complete_eicu_vitalAperiodic_and_nurseCharting_then_rerun_source_audit')
audit_json('source_forensic_audit.json',summary)
lines=['# Masked pulsatility source forensic audit','',f"Classification: `{summary['classification']}`",'',f"Protocol: `{plan_sha}`",'','The frozen analysis was not changed. No clinical outcomes or post-anchor lactate values were inspected.','','## Source attrition','',pd.DataFrame(sources).to_markdown(index=False),'','## Frozen-analysis attrition','',pd.DataFrame(frozen).to_markdown(index=False),'','## Triplet concentration','',pd.DataFrame(concentration).to_markdown(index=False),'','## Lactate availability','',pd.DataFrame(lactate_coverage).to_markdown(index=False),'','Earliest-candidate categories are mutually exclusive. Any-candidate-anchor counts reproduce the prior coverage metric but are nonexclusive because a stay can contribute several candidate anchors.','','## Mapping audit','',pd.DataFrame(mapping).to_markdown(index=False),'','The 39,625 eICU complete triplets were repeated invasive measurements concentrated in the arterial-line subset; they did not represent routine cuff coverage.']
(OUT/'source_forensic_audit.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2,allow_nan=False),flush=True)
'''

__all__ = ['EXTRACT', 'REPORT', 'SETUP']
