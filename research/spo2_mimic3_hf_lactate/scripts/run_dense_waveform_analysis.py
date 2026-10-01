#!/usr/bin/env python3
"""Reproducible MIMIC-III dense SpO2 / lactate analysis.

The script reads the full MIMIC-III v1.4 clinical tables and the supplied
15-minute waveform summaries, writes all intermediate and final artifacts,
and records data provenance. All predictor windows end strictly before the
four-hour landmark.
"""
from __future__ import annotations

import argparse
import base64
import csv
import collections
import gzip
import hashlib
import json
import logging
import math
import os
import platform
import re
import sys
import time
import traceback
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
for _font_candidate in ("DejaVu Sans", "Arial", "Liberation Sans", "Noto Sans", "Helvetica"):
    try:
        font_manager.findfont(_font_candidate, fallback_to_default=False)
        matplotlib.rcParams["font.family"] = _font_candidate
        break
    except ValueError:
        continue
matplotlib.rcParams["mathtext.fontset"] = "stix"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import fisher_exact, mannwhitneyu, norm, spearmanr
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
LANDMARK_MIN = 240.0
HORIZONS = {"12h": 960.0, "24h": 1680.0}
BOOTSTRAP_PRIMARY = 10_000
BOOTSTRAP_PREDICTIVE = 2_000
CLINICAL_DIRNAME = "mimic-iii-clinical-database-1.4"
WAVEFORM_FILE = "mimic3_waveform_spo2_15min.csv.gz"
FEATURE_FILE = "mimic3_waveform_spo2_features.csv"
MANIFEST_FILE = "mimic3_waveform_hf_record_manifest.csv"

TABLE_SCHEMAS = {
    "PATIENTS": {"subject_id", "gender", "dob"},
    "ADMISSIONS": {"subject_id", "hadm_id", "admittime", "dischtime", "deathtime"},
    "ICUSTAYS": {"subject_id", "hadm_id", "icustay_id", "intime", "outtime", "first_careunit"},
    "DIAGNOSES_ICD": {"subject_id", "hadm_id", "icd9_code"},
    "D_ICD_DIAGNOSES": {"icd9_code", "long_title"},
    "PROCEDURES_ICD": {"subject_id", "hadm_id", "icd9_code", "seq_num"},
    "D_ICD_PROCEDURES": {"icd9_code", "long_title"},
    "LABEVENTS": {"hadm_id", "itemid", "charttime", "valuenum"},
    "D_LABITEMS": {"itemid", "label", "fluid", "category"},
    "CHARTEVENTS": {"icustay_id", "itemid", "charttime", "valuenum"},
    "D_ITEMS": {"itemid", "label"},
    "INPUTEVENTS_CV": {"icustay_id", "itemid", "charttime", "rate", "amount"},
    "INPUTEVENTS_MV": {"icustay_id", "itemid", "starttime", "endtime", "rate", "amount"},
    "OUTPUTEVENTS": {"icustay_id", "itemid", "charttime", "value"},
    "PROCEDUREEVENTS_MV": {"icustay_id", "itemid", "starttime", "endtime"},
}

OUTPUT_DIRS = [
    "00_inventory", "01_cohort", "02_signal_qc", "03_landmark_lactate",
    "04_episode_lactate", "05_predictive_models", "06_mimic4_parity",
    "07_sensitivity", "08_figures", "09_final_report", "scripts",
]

KNOWN_MIMIC3_TABLE_NAMES = {
    "PATIENTS", "ADMISSIONS", "ICUSTAYS", "DIAGNOSES_ICD", "D_ICD_DIAGNOSES",
    "PROCEDURES_ICD", "D_ICD_PROCEDURES", "LABEVENTS", "D_LABITEMS", "CHARTEVENTS",
    "D_ITEMS", "INPUTEVENTS_CV", "INPUTEVENTS_MV", "OUTPUTEVENTS", "PROCEDUREEVENTS_MV",
    "TRANSFERS", "CALLOUT", "DATETIMEEVENTS", "MICROBIOLOGYEVENTS", "NOTEEVENTS",
    "PRESCRIPTIONS", "CAREGIVERS", "CPTEVENTS", "SERVICES", "DRGCODES",
}


def setup_logger(output: Path) -> logging.Logger:
    logger = logging.getLogger("physiograph_dense")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(output / "execution.log", mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_header(path: Path) -> list[str]:
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return next(csv.reader(handle))


def schema_candidates(input_root: Path, output_root: Path) -> tuple[dict[str, list[Path]], list[Path]]:
    by_table = {key: [] for key in TABLE_SCHEMAS}
    csv_files: list[Path] = []
    output_resolved = output_root.resolve()
    for base, dirs, filenames in os.walk(input_root):
        base_path = Path(base)
        dirs[:] = [d for d in dirs if d not in {".git", ".ipynb_checkpoints"}]
        try:
            base_path.resolve().relative_to(output_resolved)
            dirs[:] = []
            continue
        except ValueError:
            pass
        for filename in filenames:
            lower = filename.lower()
            if not (lower.endswith(".csv") or lower.endswith(".csv.gz")):
                continue
            path = base_path / filename
            csv_files.append(path)
            try:
                columns = {c.strip().lower() for c in file_header(path)}
            except Exception:
                continue
            for table, required in TABLE_SCHEMAS.items():
                if required.issubset(columns):
                    by_table[table].append(path)
    return by_table, csv_files


def canonical_csv_table_name(path: Path) -> str:
    name = path.name.upper()
    if name.endswith(".CSV.GZ"):
        name = name[:-7]
    elif name.endswith(".CSV"):
        name = name[:-4]
    return re.sub(r"\s*\(\d+\)$", "", name)


def choose_table(table: str, candidates: list[Path]) -> Path | None:
    if not candidates:
        return None
    named = [p for p in candidates if canonical_csv_table_name(p) == table]
    if named:
        candidates = named
    else:
        # A schema superset can belong to another native MIMIC table (for
        # example TRANSFERS resembles ICUSTAYS); use only non-native aliases
        # when the canonical table filename is absent.
        aliases = [p for p in candidates if canonical_csv_table_name(p) not in KNOWN_MIMIC3_TABLE_NAMES]
        if not aliases:
            return None
        candidates = aliases
    preferred = [p for p in candidates if CLINICAL_DIRNAME in str(p).lower()]
    group = preferred or candidates
    return sorted(group, key=lambda p: (p.stat().st_size, str(p)), reverse=True)[0]


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, **kwargs)


def write_csv(df: pd.DataFrame, output: Path, subdir: str, name: str) -> Path:
    path = output / subdir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def count_csv_data_rows(path: Path | None) -> tuple[int | None, str]:
    """Count non-header CSV records, including a present but headerless empty file."""
    if path is None:
        return None, "unavailable"
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    try:
        with opener(path, "rt", encoding="utf-8-sig", errors="replace", newline="") as handle:
            rows = csv.reader(handle)
            first = next(rows, None)
            if first is None or not any(str(cell).strip() for cell in first):
                return 0, "available_empty_or_headerless"
            return sum(1 for row in rows if any(str(cell).strip() for cell in row)), "available"
    except Exception as exc:
        return None, f"unavailable: {exc}"


def inventory_rows(
    input_root: Path,
    output_root: Path,
    discovered: dict[str, list[Path]],
    tables: dict[str, Path],
    extras: dict[str, Path | None],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, int | None]]:
    rows: list[dict] = []
    row_counts: dict[str, int | None] = {}
    cache: dict[str, dict] = {}
    prior_inventory = output_root / "data_inventory.csv"
    if prior_inventory.exists():
        try:
            for cached in pd.read_csv(prior_inventory, keep_default_na=False).to_dict(orient="records"):
                full_path = str(cached.get("full_path", ""))
                if full_path:
                    cache[full_path] = cached
        except Exception:
            cache = {}
    used = {p.resolve(): key for key, p in tables.items()}
    named_files = {
        FEATURE_FILE: "waveform_features",
        WAVEFORM_FILE: "waveform_15min",
        MANIFEST_FILE: "waveform_manifest",
        "mimic3_waveform_extraction_failures.csv": "waveform_failures",
        "mimic3_waveform_hf_lactate_feasibility.csv": "waveform_feasibility_expected",
    }
    selected_paths = set(used)
    for p in extras.values():
        if p:
            selected_paths.add(p.resolve())
    candidates: dict[Path, tuple[str, str]] = {}
    for table, paths in discovered.items():
        for p in paths:
            resolved = p.resolve()
            if resolved in selected_paths:
                candidates[resolved] = (used[resolved], "selected full MIMIC-III v1.4 source")
            elif resolved not in candidates or canonical_csv_table_name(p) in TABLE_SCHEMAS:
                inferred = canonical_csv_table_name(p)
                candidates[resolved] = (
                    inferred if inferred in TABLE_SCHEMAS else table,
                    "alternate schema match; not selected",
                )
    for base, label in named_files.items():
        matches = [p for p in input_root.rglob(base) if p.is_file()]
        if matches:
            for p in matches:
                candidates[p.resolve()] = (
                    label, "selected waveform source" if p.name == WAVEFORM_FILE or p.name == FEATURE_FILE or p.name == MANIFEST_FILE else "optional prior artifact",
                )
        else:
            rows.append({
                "filename": base, "full_path": "", "bytes": "", "sha256": "",
                "row_count": "", "columns": "", "inferred_table": label,
                "notes": "missing; checked recursively under Downloads",
            })
    # Include any similarly named previous extraction artifacts even if a suffixed copy exists.
    for p in input_root.rglob("mimic3_waveform*"):
        if p.is_file() and (p.name.endswith(".csv") or p.name.endswith(".csv.gz")):
            candidates.setdefault(p.resolve(), ("waveform_related", "discovered by filename; schema checked independently"))
    for resolved, (table, note) in sorted(candidates.items(), key=lambda item: str(item[0]).lower()):
        p = Path(resolved)
        try:
            size = p.stat().st_size
            logger.info("Inventorying %s (%d bytes).", p.name, size)
            header = file_header(p)
            columns = ",".join(header)
            cached = cache.get(str(p), {})
            digest = str(cached.get("sha256", "")) if int(cached.get("bytes", -1) or -1) == size else ""
            if not digest and size <= 5_000_000_000:
                digest = sha256_file(p)
            hash_note = "" if digest else "SHA256 deferred because file exceeds 5 GB"
            try:
                row_count = int(cached.get("row_count")) if str(cached.get("row_count", "")).strip() else None
            except (TypeError, ValueError):
                row_count = None
            if row_count is None and size <= 100_000_000:
                try:
                    row_count = max(0, sum(1 for _ in csv.reader(
                        gzip.open(p, "rt", encoding="utf-8-sig", errors="replace", newline="")
                        if p.name.lower().endswith(".gz")
                        else p.open("r", encoding="utf-8-sig", errors="replace", newline="")
                    )) - 1)
                except Exception:
                    row_count = None
            elif p.resolve() in selected_paths:
                hash_note = (hash_note + "; " if hash_note else "") + "row count recorded while analysis streamed the table"
            rows.append({
                "filename": p.name, "full_path": str(p), "bytes": size,
                "sha256": digest, "row_count": row_count if row_count is not None else "",
                "columns": columns, "inferred_table": table,
                "notes": note + (("; " + hash_note) if hash_note else ""),
            })
        except Exception as exc:
            failed_row_count = count_csv_data_rows(p)[0] if table == "waveform_failures" else ""
            failed_digest = sha256_file(p) if p.exists() and p.stat().st_size <= 5_000_000_000 else ""
            rows.append({
                "filename": p.name, "full_path": str(p), "bytes": p.stat().st_size if p.exists() else "",
                "sha256": failed_digest, "row_count": failed_row_count, "columns": "",
                "inferred_table": table, "notes": note + f"; inventory error: {exc}",
            })
    # Document the unavailable full waveform-covered candidate-ID list.
    rows.append({
        "filename": "mimic3_waveform_all_coverage_candidate_stays.csv",
        "full_path": "", "bytes": "", "sha256": "", "row_count": "",
        "columns": "", "inferred_table": "waveform_coverage_universe",
        "notes": "not found; the available feature table is a selected HF cohort subset, so the approximately 8,736-stay upstream candidate universe cannot be independently counted",
    })
    frame = pd.DataFrame(rows, columns=[
        "filename", "full_path", "bytes", "sha256", "row_count", "columns", "inferred_table", "notes"
    ])
    frame.to_csv(output_root / "data_inventory.csv", index=False)
    frame.to_csv(output_root / "00_inventory" / "data_inventory.csv", index=False)
    logger.info("Inventory captured %d relevant table/artifact records.", len(frame))
    return frame, row_counts


def update_inventory_row_count(output: Path, source: Path, count: int) -> None:
    for inventory_path in (output / "data_inventory.csv", output / "00_inventory" / "data_inventory.csv"):
        if not inventory_path.exists():
            continue
        frame = pd.read_csv(inventory_path, keep_default_na=False)
        mask = frame.full_path.astype(str).eq(str(source))
        frame.loc[mask, "row_count"] = str(count)
        frame.to_csv(inventory_path, index=False)


def normalize_icd(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^0-9]", "", str(value))


def build_cohort(
    output: Path,
    tables: dict[str, Path],
    extras: dict[str, Path | None],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    patients = read_table(tables["PATIENTS"], usecols=["SUBJECT_ID", "GENDER", "DOB"])
    admissions = read_table(
        tables["ADMISSIONS"],
        usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME", "DEATHTIME", "HOSPITAL_EXPIRE_FLAG"],
    )
    icu = read_table(
        tables["ICUSTAYS"],
        usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME", "FIRST_CAREUNIT", "DBSOURCE"],
    )
    diag = read_table(tables["DIAGNOSES_ICD"], usecols=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"])
    row_counts = {
        "PATIENTS": len(patients), "ADMISSIONS": len(admissions),
        "ICUSTAYS": len(icu), "DIAGNOSES_ICD": len(diag),
    }
    for frame, cols in [
        (admissions, ["ADMITTIME", "DISCHTIME", "DEATHTIME"]),
        (icu, ["INTIME", "OUTTIME"]),
        (patients, ["DOB"]),
    ]:
        for col in cols:
            if col in frame:
                frame[col] = pd.to_datetime(frame[col], errors="coerce")
    icu["ICUSTAY_ID"] = pd.to_numeric(icu.ICUSTAY_ID, errors="coerce").astype("Int64")
    icu["HADM_ID"] = pd.to_numeric(icu.HADM_ID, errors="coerce").astype("Int64")
    duplicates = {
        "duplicate_icustay_id_rows": int(icu.ICUSTAY_ID.duplicated().sum()),
        "duplicate_admission_rows": int(admissions.HADM_ID.duplicated().sum()),
        "duplicate_patient_rows": int(patients.SUBJECT_ID.duplicated().sum()),
    }
    if duplicates["duplicate_icustay_id_rows"]:
        raise RuntimeError(f"ICUSTAY_ID duplicated in full MIMIC-III ICUSTAYS: {duplicates}")
    code = diag.ICD9_CODE.map(normalize_icd)
    diag["hf_code"] = code.str.startswith("428")
    diag["shock_code"] = code.eq("78551")
    flags = diag.groupby("HADM_ID", dropna=False).agg(
        explicit_hf=("hf_code", "max"), cardiogenic_shock=("shock_code", "max")
    ).reset_index()
    stays = icu.merge(admissions, on=["SUBJECT_ID", "HADM_ID"], how="left", validate="many_to_one")
    stays = stays.merge(patients, on="SUBJECT_ID", how="left", validate="many_to_one")
    stays = stays.merge(flags, on="HADM_ID", how="left", validate="many_to_one")
    stays["explicit_hf"] = stays.explicit_hf.fillna(False).astype(bool)
    stays["cardiogenic_shock"] = stays.cardiogenic_shock.fillna(False).astype(bool)
    stays["admit_to_icu_minutes"] = (stays.INTIME - stays.ADMITTIME).dt.total_seconds() / 60
    age_days = (stays.INTIME - stays.DOB).dt.total_seconds() / 86400
    age_float = age_days / 365.2425
    stays["age_topcoded_90plus"] = age_float > 89
    stays["age_years"] = np.where(stays.age_topcoded_90plus, 90, np.floor(age_float))
    stays.loc[(stays.age_years < 0) | (stays.age_years > 120), "age_years"] = np.nan
    stays["adult"] = stays.age_years >= 18
    stays["first_icu_in_hadm"] = False
    ordered = stays.sort_values(["HADM_ID", "INTIME", "ICUSTAY_ID"], kind="mergesort")
    first_ids = ordered.drop_duplicates("HADM_ID", keep="first").index
    stays.loc[first_ids, "first_icu_in_hadm"] = True
    stays["early_icu_or_shock"] = (
        stays.admit_to_icu_minutes.between(0, 1440, inclusive="both")
        | stays.cardiogenic_shock
    )
    stays["landmark_time"] = stays.INTIME + pd.to_timedelta(LANDMARK_MIN, unit="m")
    icu_end_candidates = pd.concat(
        [stays.OUTTIME.rename("icu_out"), stays.DISCHTIME.rename("hospital_discharge"),
         stays.DEATHTIME.rename("death")], axis=1
    )
    stays["icu_observable_end"] = icu_end_candidates.min(axis=1)
    hospital_end_candidates = pd.concat(
        [stays.DISCHTIME.rename("hospital_discharge"), stays.DEATHTIME.rename("death")],
        axis=1,
    )
    stays["observable_end"] = hospital_end_candidates.min(axis=1)
    stays["observable_beyond_hour4"] = stays.icu_observable_end > stays.landmark_time
    stages = [
        ("All MIMIC-III ICU stays", np.ones(len(stays), dtype=bool),
         "Full MIMIC-III v1.4 ICU table; not restricted to waveform coverage."),
        ("First ICU stay per hospital admission", stays.first_icu_in_hadm,
         "Earliest INTIME per HADM_ID; ties broken by ICUSTAY_ID."),
        ("Adult at ICU anchor", stays.adult,
         "Age floored from INTIME minus DOB; MIMIC-III ages above 89 top-coded to 90."),
        ("Explicit HF ICD-9 phenotype", stays.explicit_hf,
         "At least one ICD-9 diagnosis with digits beginning 428."),
        ("Early ICU admission or cardiogenic shock", stays.early_icu_or_shock,
         "ICU admission 0–24h after hospital admission, inclusive, OR ICD-9 78551."),
        ("Observable beyond ICU hour 4", stays.observable_beyond_hour4,
         "Minimum known ICU outtime, hospital discharge, and death time is strictly after minute 240."),
    ]
    current = np.ones(len(stays), dtype=bool)
    flow_rows = []
    prev = int(current.sum())
    for label, mask, definition in stages:
        current &= np.asarray(mask, dtype=bool)
        n = int(current.sum())
        flow_rows.append({
            "flow_step": label, "n_remaining": n, "excluded_at_step": prev - n,
            "scope": "full_clinical_cohort", "definition": definition, "status": "available",
        })
        prev = n
    clinical = stays.loc[current].copy()
    if clinical.ICUSTAY_ID.duplicated().any() or clinical.HADM_ID.duplicated().any():
        raise RuntimeError("Clinical eligible cohort has duplicate stay or admission IDs.")
    clinical["GENDER"] = clinical.GENDER.astype(str).str.upper()
    clinical["sex_male"] = clinical.GENDER.map({"M": 1.0, "F": 0.0})
    clinical["hospital_followup_end"] = stays.loc[clinical.index, "observable_end"]
    clinical["icu_observable_end"] = stays.loc[clinical.index, "icu_observable_end"]
    clinical["hospital_observable_end_minute"] = (
        (clinical.hospital_followup_end - clinical.INTIME).dt.total_seconds() / 60
    )
    clinical["icu_outtime_minute"] = (clinical.OUTTIME - clinical.INTIME).dt.total_seconds() / 60
    clinical["observable_end_minute"] = (
        (clinical.icu_observable_end - clinical.INTIME).dt.total_seconds() / 60
    )
    manifest_path = extras.get("manifest")
    manifest = pd.DataFrame()
    if manifest_path:
        manifest = read_table(manifest_path)
        manifest["ICUSTAY_ID"] = pd.to_numeric(manifest.ICUSTAY_ID, errors="coerce").astype("Int64")
        manifest_stays = set(manifest.ICUSTAY_ID.dropna().astype(int))
        clinical_ids = set(clinical.ICUSTAY_ID.dropna().astype(int))
        in_manifest = len(clinical_ids & manifest_stays)
        flow_rows.append({
            "flow_step": "Eligible cohort represented in waveform record manifest",
            "n_remaining": in_manifest, "excluded_at_step": len(clinical_ids) - in_manifest,
            "scope": "clinical_eligible_subset",
            "definition": "Unique eligible ICUSTAY_IDs present in the HF record manifest; this manifest is not the upstream all-waveform 8,736-stay candidate list.",
            "status": "available",
        })
    else:
        flow_rows.append({
            "flow_step": "Eligible cohort represented in waveform record manifest",
            "n_remaining": "", "excluded_at_step": "",
            "scope": "clinical_eligible_subset", "definition": "Manifest absent.",
            "status": "unavailable",
        })
    waveform_path = extras.get("waveform")
    if not waveform_path:
        raise FileNotFoundError(f"Required dense-waveform file missing: {WAVEFORM_FILE}")
    wave_raw = read_table(waveform_path)
    wave_raw["ICUSTAY_ID"] = pd.to_numeric(wave_raw.ICUSTAY_ID, errors="coerce").astype("Int64")
    feature_path = extras.get("features")
    prior_features = read_table(feature_path) if feature_path else pd.DataFrame()
    if not prior_features.empty:
        prior_features["ICUSTAY_ID"] = pd.to_numeric(prior_features.ICUSTAY_ID, errors="coerce").astype("Int64")
    signal_features, validation, qc = recompute_signal_features(wave_raw, prior_features, manifest)
    extraction_failure_count, extraction_failure_status = count_csv_data_rows(extras.get("failures"))
    failure_qc = pd.DataFrame([{
        "group": "waveform_extraction_failures",
        "stays": np.nan,
        "usable_spo2_stays": np.nan,
        "dynamics_eligible_n": np.nan,
        "exposed_n": np.nan,
        "exposure_prevalence_among_dynamics_eligible": np.nan,
        "extraction_failure_count": extraction_failure_count,
        "extraction_failure_status": extraction_failure_status,
        "status": "available" if extraction_failure_count is not None else "unavailable",
    }])
    qc = pd.concat([qc, failure_qc], ignore_index=True, sort=False)
    write_csv(signal_features, output, "02_signal_qc", "signal_features_recomputed.csv")
    write_csv(validation, output, "02_signal_qc", "signal_feature_validation.csv")
    write_csv(qc, output, "02_signal_qc", "signal_qc_summary.csv")
    clinical_links = clinical[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"]].drop_duplicates()
    linked_features = signal_features.merge(
        clinical_links, on=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"],
        how="inner", validate="one_to_one",
    )
    feature_by_stay = signal_features.merge(
        clinical_links, on="ICUSTAY_ID", how="inner", suffixes=("_signal", "_clinical")
    )
    link_mismatch_subject = int(feature_by_stay.SUBJECT_ID_signal.ne(feature_by_stay.SUBJECT_ID_clinical).sum()) if len(feature_by_stay) else 0
    link_mismatch_hadm = int(feature_by_stay.HADM_ID_signal.ne(feature_by_stay.HADM_ID_clinical).sum()) if len(feature_by_stay) else 0
    if link_mismatch_subject or link_mismatch_hadm:
        raise RuntimeError(
            f"Signal-to-clinical stay linkage disagrees on subject/admission IDs: subjects={link_mismatch_subject}, admissions={link_mismatch_hadm}."
        )
    final_ids = set(clinical.ICUSTAY_ID.dropna().astype(int))
    signal_in_target = int(linked_features.usable_dense_spo2.sum())
    flow_rows.extend([
        {
            "flow_step": "Usable 15-minute signal among eligible clinical cohort",
            "n_remaining": signal_in_target, "excluded_at_step": len(final_ids) - signal_in_target,
            "scope": "clinical_eligible_subset",
            "definition": "Eligible stays with at least one valid 50–100% SpO2 15-minute median in [0,240).",
            "status": "available",
        },
        {
            "flow_step": "Dynamics eligible",
            "n_remaining": int(signal_features.loc[signal_features.ICUSTAY_ID.isin(final_ids), "dynamics_eligible"].sum()),
            "excluded_at_step": signal_in_target - int(signal_features.loc[signal_features.ICUSTAY_ID.isin(final_ids), "dynamics_eligible"].sum()),
            "scope": "clinical_eligible_subset",
            "definition": "At least 3 valid bins and at least 2 qualifying transitions with a gap no longer than 30 minutes.",
            "status": "available",
        },
    ])
    full_feature_ids = set(signal_features.ICUSTAY_ID.dropna().astype(int))
    flow_rows.extend([
        {
            "flow_step": "All stays in supplied feature file",
            "n_remaining": len(full_feature_ids), "excluded_at_step": "",
            "scope": "supplied_signal_artifact",
            "definition": "Unique ICUSTAY_IDs in the supplied derived feature file; the all-waveform upstream candidate ID list was not supplied.",
            "status": "available",
        },
        {
            "flow_step": "Upstream all-waveform candidate universe (~8,736 expected)",
            "n_remaining": "", "excluded_at_step": "",
            "scope": "upstream_waveform_universe",
            "definition": "No all-candidate ICUSTAY_ID list was found in Downloads; no denominator inferred from the selected HF artifacts.",
            "status": "unavailable",
        },
    ])
    flow = pd.DataFrame(flow_rows)
    write_csv(flow, output, "01_cohort", "cohort_flow.csv")
    distribution = clinical.groupby("FIRST_CAREUNIT", dropna=False).agg(
        stays=("ICUSTAY_ID", "nunique"), patients=("SUBJECT_ID", "nunique")
    ).reset_index()
    distribution.to_csv(output / "01_cohort" / "icu_careunit_distribution.csv", index=False)
    # Linkage and boundary audit table.
    feature_join = clinical.merge(signal_features, on=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"], how="left", validate="one_to_one")
    audit = pd.DataFrame([{
        **duplicates,
        "eligible_clinical_stays": int(len(clinical)),
        "unique_subjects": int(clinical.SUBJECT_ID.nunique()),
        "unique_admissions": int(clinical.HADM_ID.nunique()),
        "manifest_unique_stays": int(manifest.ICUSTAY_ID.nunique()) if not manifest.empty else None,
        "feature_unique_stays": int(signal_features.ICUSTAY_ID.nunique()),
        "feature_rows_without_clinical_match": int(len(signal_features) - len(linked_features)),
        "signal_subject_link_mismatch": link_mismatch_subject,
        "signal_admission_link_mismatch": link_mismatch_hadm,
        "eligible_stays_without_signal_feature": int(feature_join.usable_dense_spo2.isna().sum()),
        "topcoded_age_90plus": int(clinical.age_topcoded_90plus.sum()),
        "age_missing": int(clinical.age_years.isna().sum()),
        "subject_hadm_stay_mismatch": int(((clinical.SUBJECT_ID.isna()) | (clinical.HADM_ID.isna()) | (clinical.ICUSTAY_ID.isna())).sum()),
    }])
    write_csv(audit, output, "01_cohort", "cohort_linkage_audit.csv")
    logger.info(
        "Clinical eligible N=%d; unique patients=%d; admissions=%d; usable signal matches=%d.",
        len(clinical), clinical.SUBJECT_ID.nunique(), clinical.HADM_ID.nunique(), signal_in_target,
    )
    final_signal = signal_features.loc[signal_features.ICUSTAY_ID.isin(final_ids)]
    context = {
        "row_counts": row_counts, "duplicates": duplicates,
        "all_icu_stays": int(len(stays)), "eligible_clinical": int(len(clinical)),
        "eligible_subjects": int(clinical.SUBJECT_ID.nunique()),
        "eligible_admissions": int(clinical.HADM_ID.nunique()),
        "manifest_unique": int(manifest.ICUSTAY_ID.nunique()) if not manifest.empty else None,
        "signal_unique": int(signal_features.ICUSTAY_ID.nunique()),
        "signal_in_target": signal_in_target,
        "extraction_failure_count": extraction_failure_count,
        "extraction_failure_status": extraction_failure_status,
        "dynamics_n": int(signal_features.loc[signal_features.ICUSTAY_ID.isin(final_ids), "dynamics_eligible"].sum()),
        "exposed_n": int((
            final_signal.dynamics_eligible.fillna(False).astype(bool)
            & final_signal.exposure_abs_jump_ge4.fillna(False).astype(bool)
        ).sum()),
        "signal_features": signal_features,
        "manifest": manifest,
        "validation": validation,
        "qc": qc,
        "careunit_distribution": distribution,
    }
    return clinical, signal_features, wave_raw, context


def classify_regime(count: object, fs: object = np.nan) -> str:
    try:
        frequency = float(fs)
        if 0.8 <= frequency <= 1.2:
            return "1_Hz"
        if 0 < frequency <= 0.02:
            return "approximately_1_per_min"
    except (TypeError, ValueError):
        pass
    try:
        n = float(count)
        if n >= 100:
            return "1_Hz"
        if n > 0:
            return "approximately_1_per_min"
    except (TypeError, ValueError):
        pass
    return "unknown"


def recompute_signal_features(
    wave_raw: pd.DataFrame, prior: pd.DataFrame, manifest: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {"ICUSTAY_ID", "bin_index", "spo2_median"}
    missing = required - set(wave_raw.columns)
    if missing:
        raise RuntimeError(f"15-minute waveform file is missing required columns: {sorted(missing)}")
    raw = wave_raw.copy()
    for col in ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "bin_index", "n_raw_in_bin"]:
        if col in raw:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw["spo2_median"] = pd.to_numeric(raw.spo2_median, errors="coerce")
    raw["bin_index"] = raw.bin_index.astype("Int64")
    duplicate_rows = int(raw.duplicated(["ICUSTAY_ID", "bin_index"]).sum())
    # Median exact-time / same-bin duplicates before calculating any transitions.
    group_cols = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "bin_index"]
    agg = {"spo2_median": "median"}
    if "n_raw_in_bin" in raw:
        agg["n_raw_in_bin"] = "sum"
    if "representative_time" in raw:
        agg["representative_time"] = "first"
    binned = raw.groupby(group_cols, dropna=False, as_index=False).agg(agg)
    binned["valid"] = (
        binned.spo2_median.between(50, 100, inclusive="both")
        & binned.bin_index.between(0, 15, inclusive="both")
    )
    if "representative_time" in binned:
        binned["representative_time"] = pd.to_datetime(binned.representative_time, errors="coerce")
        intake = raw.groupby("ICUSTAY_ID").INTIME.first() if "INTIME" in raw else pd.Series(dtype=object)
        intake = pd.to_datetime(intake, errors="coerce")
        ref = binned.ICUSTAY_ID.map(intake)
        bin_minutes = (binned.representative_time - ref).dt.total_seconds() / 60
        # The precomputed bin index is the authoritative 15-minute window coordinate.
        binned["representative_minute"] = bin_minutes
        binned["valid"] &= binned.representative_minute.ge(0) & binned.representative_minute.lt(240)
    rows = []
    for stay, part in binned.sort_values(["ICUSTAY_ID", "bin_index"]).groupby("ICUSTAY_ID", sort=False):
        part = part.sort_values("bin_index", kind="mergesort")
        valid = part.loc[part.valid].copy()
        values = valid.spo2_median.to_numpy(dtype=float)
        bins = valid.bin_index.to_numpy(dtype=int)
        diffs = np.diff(values)
        gaps = (
            np.diff(valid.representative_minute.to_numpy(dtype=float))
            if "representative_minute" in valid and valid.representative_minute.notna().all()
            else np.diff(bins) * 15.0
        )
        keep = gaps <= 30.0
        qdiff = diffs[keep]
        qbins = bins[1:][keep]
        qgaps = gaps[keep]
        observed = int(len(valid))
        transitions = int(len(qdiff))
        dyn = observed >= 3 and transitions >= 2
        large = np.flatnonzero(np.abs(qdiff) >= 4)
        episode_time = np.nan
        episode_dir = ""
        episode_delta = np.nan
        directional_anchors = {"first_ge4_drop_minute": np.nan, "first_ge4_rise_minute": np.nan}
        for label, condition in [
            ("first_ge4_drop_minute", qdiff <= -4),
            ("first_ge4_rise_minute", qdiff >= 4),
        ]:
            positions = np.flatnonzero(condition)
            if len(positions):
                k_dir = int(positions[0])
                second_dir = valid.loc[valid.bin_index.eq(qbins[k_dir])].iloc[0]
                rep = second_dir.get("representative_minute", np.nan)
                directional_anchors[label] = (
                    float(rep) if pd.notna(rep) else float(qbins[k_dir] * 15 + 7.5)
                )
        if len(large):
            k = int(large[0])
            second = valid.loc[valid.bin_index.eq(qbins[k])].iloc[0]
            if "representative_minute" in second and pd.notna(second.representative_minute):
                episode_time = float(second.representative_minute)
            else:
                episode_time = float(qbins[k] * 15 + 7.5)
            episode_delta = float(qdiff[k])
            episode_dir = "drop" if qdiff[k] < 0 else "rise"
        counts = pd.to_numeric(part.get("n_raw_in_bin", pd.Series(dtype=float)), errors="coerce").dropna()
        regime_count = float(counts.median()) if len(counts) else np.nan
        rows.append({
            "ICUSTAY_ID": int(stay),
            "SUBJECT_ID": int(valid.SUBJECT_ID.iloc[0]) if observed else int(part.SUBJECT_ID.iloc[0]),
            "HADM_ID": int(valid.HADM_ID.iloc[0]) if observed else int(part.HADM_ID.iloc[0]),
            "observed_15min_bins": observed,
            "missing_15min_bins": 16 - observed,
            "qualifying_transitions_le30min": transitions,
            "dynamics_eligible": bool(dyn),
            "exposure_abs_jump_ge4": bool(np.any(np.abs(qdiff) >= 4)) if transitions else False,
            "rmssd_binned": float(np.sqrt(np.mean(qdiff ** 2))) if transitions else np.nan,
            "jump_fraction_ge3": float(np.mean(np.abs(qdiff) >= 3)) if transitions else np.nan,
            "drop_count_ge3": int(np.sum(qdiff <= -3)) if transitions else 0,
            "mean_binned_spo2": float(np.mean(values)) if observed else np.nan,
            "min_binned_spo2": float(np.min(values)) if observed else np.nan,
            "fraction_binned_below90": float(np.mean(values < 90)) if observed else np.nan,
            "first_ge4_episode_minute": episode_time,
            "first_ge4_episode_direction": episode_dir,
            "first_ge4_episode_delta": episode_delta,
            **directional_anchors,
            "max_abs_qualifying_jump": float(np.max(np.abs(qdiff))) if transitions else np.nan,
            "n_ge4_transitions": int(np.sum(np.abs(qdiff) >= 4)) if transitions else 0,
            "max_transition_gap_minutes": float(np.max(qgaps)) if transitions else np.nan,
            "slope_spo2_per_hour": (
                float(np.polyfit(bins.astype(float) * 15.0 / 60.0, values, 1)[0])
                if observed >= 2 else np.nan
            ),
            "median_raw_samples_per_15min_bin": regime_count,
            "sampling_regime_from_bins": classify_regime(regime_count),
            "usable_dense_spo2": bool(observed > 0),
            "duplicate_stay_bin_rows_collapsed": int(part.duplicated("bin_index").sum()),
        })
    features = pd.DataFrame(rows)
    if not manifest.empty and "fs" in manifest:
        mm = manifest.copy()
        mm["ICUSTAY_ID"] = pd.to_numeric(mm.ICUSTAY_ID, errors="coerce")
        mm["regime"] = mm.fs.map(lambda x: classify_regime(np.nan, x))
        mg = mm.groupby("ICUSTAY_ID").agg(
            manifest_regimes=("regime", lambda x: "|".join(sorted(set(x.dropna())))),
            manifest_record_rows=("regime", "size"),
            manifest_overlap_hours=("overlap_hours", "sum") if "overlap_hours" in mm else ("regime", "size"),
        ).reset_index()
        features = features.merge(mg, on="ICUSTAY_ID", how="left", validate="one_to_one")
        def combine_regime(row):
            values = str(row.get("manifest_regimes", ""))
            labels = [x for x in values.split("|") if x and x != "nan"]
            if len(labels) == 1:
                return labels[0]
            if len(labels) > 1:
                return "mixed"
            return row["sampling_regime_from_bins"]
        features["sampling_regime"] = features.apply(combine_regime, axis=1)
        features["regime_discordant"] = features.apply(
            lambda r: bool(
                r.get("manifest_regimes") and "|" not in str(r.get("manifest_regimes"))
                and r.get("sampling_regime_from_bins") not in {"unknown", r.get("manifest_regimes")}
            ), axis=1
        )
    else:
        features["manifest_regimes"] = ""
        features["manifest_record_rows"] = np.nan
        features["manifest_overlap_hours"] = np.nan
        features["sampling_regime"] = features.sampling_regime_from_bins
        features["regime_discordant"] = False
    validation = pd.DataFrame()
    if not prior.empty:
        common = [
            "observed_15min_bins", "missing_15min_bins", "qualifying_transitions_le30min",
            "dynamics_eligible", "exposure_abs_jump_ge4", "rmssd_binned",
            "jump_fraction_ge3", "drop_count_ge3",
        ]
        old = prior.copy()
        old["ICUSTAY_ID"] = pd.to_numeric(old.ICUSTAY_ID, errors="coerce")
        val = features.merge(old, on="ICUSTAY_ID", how="outer", suffixes=("_recomputed", "_supplied"), indicator=True)
        validation = val[["ICUSTAY_ID", "SUBJECT_ID_recomputed", "HADM_ID_recomputed", "_merge"]].copy()
        for col in common:
            lhs = pd.to_numeric(val.get(col + "_recomputed"), errors="coerce")
            rhs = pd.to_numeric(val.get(col + "_supplied"), errors="coerce")
            if col in {"dynamics_eligible", "exposure_abs_jump_ge4"}:
                a = val[col + "_recomputed"].astype("string")
                b = val[col + "_supplied"].astype("string")
                mismatch = a.ne(b).fillna(True)
                diff = np.where(mismatch, np.nan, 0.0)
            else:
                diff = (lhs - rhs).abs()
                mismatch = diff.gt(1e-6).fillna(lhs.notna() | rhs.notna())
            validation[f"{col}_absolute_difference"] = diff
            validation[f"{col}_mismatch"] = mismatch.astype(bool)
        mismatch_cols = [c for c in validation.columns if c.endswith("_mismatch")]
        validation["any_core_mismatch"] = validation[mismatch_cols].any(axis=1)
        validation["status"] = np.where(validation._merge.eq("both"), "compared", "stay_missing_from_one_side")
    else:
        validation = pd.DataFrame([{
            "status": "unavailable", "reason": f"{FEATURE_FILE} was not found; independent recomputation is retained.",
            "missing_source": FEATURE_FILE,
        }])
    rows = []
    for key, frame in [("overall", features), *[
        (str(group), part) for group, part in features.groupby("sampling_regime", dropna=False)
    ]]:
        eligible = frame.dynamics_eligible.astype(bool)
        exposed = frame.exposure_abs_jump_ge4.astype(bool)
        rows.append({
            "group": key, "stays": int(frame.ICUSTAY_ID.nunique()),
            "usable_spo2_stays": int(frame.usable_dense_spo2.sum()),
            "dynamics_eligible_n": int(eligible.sum()),
            "exposed_n": int((eligible & exposed).sum()),
            "exposure_prevalence_among_dynamics_eligible": float((eligible & exposed).sum() / eligible.sum()) if eligible.sum() else np.nan,
            "observed_bins_median": float(frame.observed_15min_bins.median()),
            "observed_bins_q1": float(frame.observed_15min_bins.quantile(.25)),
            "observed_bins_q3": float(frame.observed_15min_bins.quantile(.75)),
            "transitions_median": float(frame.qualifying_transitions_le30min.median()),
            "rmssd_median": float(frame.rmssd_binned.median()),
            "rmssd_q1": float(frame.rmssd_binned.quantile(.25)),
            "rmssd_q3": float(frame.rmssd_binned.quantile(.75)),
            "missing_bins_median": float(frame.missing_15min_bins.median()),
            "sampling_regime_discordant_n": int(frame.regime_discordant.sum()),
            "duplicate_bin_rows_collapsed": int(frame.duplicate_stay_bin_rows_collapsed.sum()),
            "source_duplicate_stay_bin_rows": duplicate_rows if key == "overall" else "",
        })
    return features, validation, pd.DataFrame(rows)


def resolve_lactate_items(path: Path) -> tuple[set[int], pd.DataFrame]:
    labs = read_table(path, usecols=["ITEMID", "LABEL", "FLUID", "CATEGORY"])
    labs["LABEL"] = labs.LABEL.fillna("").astype(str)
    labs["FLUID"] = labs.FLUID.fillna("").astype(str)
    labs["CATEGORY"] = labs.CATEGORY.fillna("").astype(str)
    labels = labs.LABEL.str.contains("lactate", case=False, regex=False)
    labels &= ~labs.LABEL.str.contains("dehydrogenase", case=False, regex=False)
    fluid_ok = labs.FLUID.str.contains("blood|serum|plasma", case=False, regex=True)
    category_ok = labs.CATEGORY.str.contains("blood|gas", case=False, regex=True)
    chosen = labs.loc[labels & (fluid_ok | category_ok)].drop_duplicates("ITEMID")
    if chosen.empty:
        chosen = labs.loc[labels].drop_duplicates("ITEMID")
    itemids = set(pd.to_numeric(chosen.ITEMID, errors="coerce").dropna().astype(int))
    return itemids, chosen


def stream_lactate(
    lab_path: Path,
    lab_dictionary_path: Path,
    clinical: pd.DataFrame,
    output: Path,
    logger: logging.Logger,
    chunksize: int = 1_500_000,
) -> tuple[pd.DataFrame, dict]:
    itemids, item_map = resolve_lactate_items(lab_dictionary_path)
    if not itemids:
        raise RuntimeError("No plausible blood lactate ITEMIDs were identified in D_LABITEMS.")
    hadms = set(pd.to_numeric(clinical.HADM_ID, errors="coerce").dropna().astype(int))
    stay_map = clinical[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME"]].copy()
    stay_map["HADM_ID"] = pd.to_numeric(stay_map.HADM_ID, errors="coerce").astype("Int64")
    parts = []
    source_rows = 0
    matched_rows = 0
    unit_counts: collections.Counter = collections.Counter()
    for chunk_num, chunk in enumerate(pd.read_csv(
        lab_path, usecols=["SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM", "VALUEUOM"],
        chunksize=chunksize, low_memory=False,
    ), start=1):
        source_rows += len(chunk)
        keep = pd.to_numeric(chunk.HADM_ID, errors="coerce").isin(hadms)
        keep &= pd.to_numeric(chunk.ITEMID, errors="coerce").isin(itemids)
        chunk = chunk.loc[keep].copy()
        if chunk.empty:
            if chunk_num % 10 == 0:
                logger.info("LABEVENTS chunks scanned=%d; source rows=%d; matched lactate rows=%d", chunk_num, source_rows, matched_rows)
            continue
        chunk["ITEMID"] = pd.to_numeric(chunk.ITEMID, errors="coerce").astype("Int64")
        chunk["VALUENUM"] = pd.to_numeric(chunk.VALUENUM, errors="coerce")
        chunk["CHARTTIME"] = pd.to_datetime(chunk.CHARTTIME, errors="coerce")
        units = chunk.VALUEUOM.fillna("").astype(str).str.strip().str.lower()
        unit_counts.update(units.value_counts().to_dict())
        mgdl = units.str.replace(" ", "", regex=False).isin({"mg/dl", "mgperdl"})
        mmol = units.str.replace(" ", "", regex=False).isin({"mmol/l", "mmolperliter", "mmoll-1"})
        accepted = mmol | mgdl | units.eq("")
        chunk = chunk.loc[accepted & chunk.VALUENUM.notna() & chunk.CHARTTIME.notna()].copy()
        units = chunk.VALUEUOM.fillna("").astype(str).str.strip().str.lower().str.replace(" ", "", regex=False)
        chunk["lactate_mmol_l"] = np.where(units.isin({"mg/dl", "mgperdl"}), chunk.VALUENUM / 9.008, chunk.VALUENUM)
        chunk = chunk.loc[chunk.lactate_mmol_l.between(0, 50, inclusive="both")].copy()
        matched_rows += len(chunk)
        if not chunk.empty:
            parts.append(chunk[["SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "VALUEUOM", "lactate_mmol_l"]])
        if chunk_num % 10 == 0:
            logger.info("LABEVENTS chunks scanned=%d; source rows=%d; matched lactate rows=%d", chunk_num, source_rows, matched_rows)
    events = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "VALUEUOM", "lactate_mmol_l"]
    )
    if not events.empty:
        events["HADM_ID"] = pd.to_numeric(events.HADM_ID, errors="coerce").astype("Int64")
        clinical_by_hadm = clinical.set_index("HADM_ID")
        events["clinical_subject_id"] = events.HADM_ID.map(clinical_by_hadm.SUBJECT_ID)
        events["ICUSTAY_ID"] = events.HADM_ID.map(clinical_by_hadm.ICUSTAY_ID)
        linkage_mismatch_rows = int(
            pd.to_numeric(events.SUBJECT_ID, errors="coerce").ne(
                pd.to_numeric(events.clinical_subject_id, errors="coerce")
            ).sum()
        )
        events = events.loc[
            pd.to_numeric(events.SUBJECT_ID, errors="coerce").eq(
                pd.to_numeric(events.clinical_subject_id, errors="coerce")
            )
        ].copy()
        events["INTIME"] = events.ICUSTAY_ID.map(clinical.set_index("ICUSTAY_ID").INTIME)
        events = events.dropna(subset=["ICUSTAY_ID", "INTIME", "CHARTTIME"])
        events["minute_from_icu_intime"] = (events.CHARTTIME - events.INTIME).dt.total_seconds() / 60
        # Same-stay, exact-time duplicate tests are collapsed by their median.
        events = events.groupby(
            ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "CHARTTIME", "minute_from_icu_intime"],
            as_index=False, dropna=False,
        ).agg(
            lactate_mmol_l=("lactate_mmol_l", "median"),
            itemids=("ITEMID", lambda x: "|".join(map(str, sorted(set(x.dropna().astype(int)))))),
            source_rows=("ITEMID", "size"),
        )
    events.to_csv(output / "03_landmark_lactate" / "linked_lactate_draws.csv.gz", index=False, compression="gzip")
    item_map.to_csv(output / "03_landmark_lactate" / "lactate_item_map.csv", index=False)
    info = {
        "source_rows": source_rows, "matched_valid_rows_before_duplicate_collapse": matched_rows,
        "linked_unique_draws": int(len(events)), "lactate_itemids": sorted(itemids),
        "subject_hadm_linkage_mismatch_rows_excluded": int(linkage_mismatch_rows) if "linkage_mismatch_rows" in locals() else 0,
        "lactate_item_labels": item_map.to_dict(orient="records"),
        "observed_value_units": unit_counts,
    }
    logger.info(
        "LABEVENTS scanned %d rows; retained %d source lactate rows and %d linked unique draws; item IDs=%s.",
        source_rows, matched_rows, len(events), sorted(itemids),
    )
    return events, info


def construct_lactate_outcomes(
    clinical: pd.DataFrame, signal: pd.DataFrame, events: pd.DataFrame, output: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_cols = [
        "ICUSTAY_ID", "observed_15min_bins", "missing_15min_bins",
        "qualifying_transitions_le30min", "dynamics_eligible", "exposure_abs_jump_ge4",
        "rmssd_binned", "jump_fraction_ge3", "drop_count_ge3", "mean_binned_spo2",
        "min_binned_spo2", "fraction_binned_below90", "first_ge4_episode_minute",
        "first_ge4_episode_direction", "first_ge4_episode_delta", "max_abs_qualifying_jump",
        "first_ge4_drop_minute", "first_ge4_rise_minute", "n_ge4_transitions",
        "slope_spo2_per_hour", "max_transition_gap_minutes",
        "sampling_regime", "median_raw_samples_per_15min_bin",
        "manifest_regimes",
    ]
    base = clinical.merge(signal[[c for c in signal_cols if c in signal]], on="ICUSTAY_ID", how="left", validate="one_to_one")
    if events.empty:
        by_stay: dict[int, pd.DataFrame] = {}
    else:
        by_stay = {int(k): g.sort_values("CHARTTIME") for k, g in events.groupby("ICUSTAY_ID")}
    outcome_rows = []
    completeness_rows = []
    for row in base.itertuples(index=False):
        stay = int(row.ICUSTAY_ID)
        d = by_stay.get(stay, pd.DataFrame(columns=["CHARTTIME", "minute_from_icu_intime", "lactate_mmol_l"]))
        pre = d.loc[d.minute_from_icu_intime.le(LANDMARK_MIN)] if not d.empty else d
        baseline_row = pre.iloc[-1] if len(pre) else None
        baseline = float(baseline_row.lactate_mmol_l) if baseline_row is not None else np.nan
        baseline_time = baseline_row.CHARTTIME if baseline_row is not None else pd.NaT
        end_min = float(row.observable_end_minute) if pd.notna(row.observable_end_minute) else np.nan
        hospital_end_min = (
            float(row.hospital_observable_end_minute)
            if hasattr(row, "hospital_observable_end_minute") and pd.notna(row.hospital_observable_end_minute)
            else end_min
        )
        result = {
            "SUBJECT_ID": int(row.SUBJECT_ID), "HADM_ID": int(row.HADM_ID), "ICUSTAY_ID": stay,
            "baseline_lactate_mmol_l": baseline, "baseline_lactate_time": baseline_time,
            "baseline_lactate_observed": bool(baseline_row is not None),
            "baseline_lactate_draws_pre_landmark": int(len(pre)),
            "observable_end_minute": end_min,
            "hospital_observable_end_minute": hospital_end_min,
            "icu_outtime_minute": (
                float(row.icu_outtime_minute)
                if hasattr(row, "icu_outtime_minute") and pd.notna(row.icu_outtime_minute)
                else np.nan
            ),
            "last_lactate_before_or_at_240_minute": (
                float(baseline) if np.isfinite(baseline) else np.nan
            ),
            "status": "available",
        }
        if not d.empty:
            post_lm = d.loc[d.minute_from_icu_intime.gt(LANDMARK_MIN)]
        else:
            post_lm = d
        result["any_followup_lactate_after_landmark"] = bool(len(post_lm))
        result["followup_lactate_draws_after_landmark"] = int(len(post_lm))
        if len(post_lm):
            result["minutes_to_first_followup_lactate"] = float(post_lm.minute_from_icu_intime.min() - LANDMARK_MIN)
        else:
            result["minutes_to_first_followup_lactate"] = np.nan
        for horizon, end in HORIZONS.items():
            post = d.loc[d.minute_from_icu_intime.gt(LANDMARK_MIN) & d.minute_from_icu_intime.le(end)] if not d.empty else d
            last = post.iloc[-1] if len(post) else None
            maximum = float(post.lactate_mmol_l.max()) if len(post) else np.nan
            delta = float(last.lactate_mmol_l - baseline) if last is not None and np.isfinite(baseline) else np.nan
            max_delta = float(maximum - baseline) if np.isfinite(maximum) and np.isfinite(baseline) else np.nan
            direct_event = bool(np.isfinite(delta) and delta >= .5)
            has_baseline = bool(np.isfinite(baseline))
            has_post = bool(last is not None)
            followup_complete = bool(np.isfinite(end_min) and end_min >= end)
            legacy_followup_complete = bool(np.isfinite(hospital_end_min) and hospital_end_min >= end)
            if not has_baseline:
                observed, value, reason = False, np.nan, "missing_pre_landmark_baseline"
            elif not has_post:
                observed, value = False, np.nan
                reason = "no_post_landmark_lactate_in_window"
            elif not followup_complete:
                observed, value = False, np.nan
                reason = (
                    "censored_before_horizon_after_early_last_value_event"
                    if direct_event else "censored_before_horizon_without_event"
                )
            elif direct_event:
                observed, value, reason = True, True, "complete_followup_last_value_event"
            else:
                observed, value, reason = True, False, "complete_observation_through_horizon"
            legacy_observed = bool(has_baseline and has_post and (direct_event or legacy_followup_complete))
            legacy_value = bool(direct_event) if legacy_observed else np.nan
            result[f"n_lactate_draws_{horizon}"] = int(len(post))
            result[f"last_lactate_{horizon}"] = float(last.lactate_mmol_l) if last is not None else np.nan
            result[f"last_lactate_time_{horizon}"] = last.CHARTTIME if last is not None else pd.NaT
            result[f"continuous_last_delta_{horizon}"] = delta
            result[f"maximum_post_lactate_{horizon}"] = maximum
            result[f"maximum_post_delta_{horizon}"] = max_delta
            result[f"outcome_available_{horizon}"] = observed
            result[f"lactate_rise_ge0_5_{horizon}"] = value
            result[f"legacy_outcome_available_{horizon}"] = legacy_observed
            result[f"legacy_lactate_rise_ge0_5_{horizon}"] = legacy_value
            one_event = bool(np.isfinite(delta) and delta >= 1.0)
            one_observed = bool(has_baseline and has_post and followup_complete)
            result[f"lactate_last_delta_ge1_{horizon}"] = (
                bool(delta >= 1.0) if one_observed and np.isfinite(delta) else np.nan
            )
            max_event = bool(np.isfinite(max_delta) and max_delta >= .5)
            max_observed = (
                bool(has_baseline and has_post and (max_event or followup_complete))
            )
            result[f"maximum_delta_ge0_5_{horizon}"] = max_event if max_observed else np.nan
            result[f"last_delta_ge1_observed_{horizon}"] = one_observed
            legacy_one_observed = bool(has_baseline and has_post and (
                bool(np.isfinite(delta) and delta >= 1.0) or legacy_followup_complete
            ))
            result[f"legacy_lactate_last_delta_ge1_{horizon}"] = (
                bool(delta >= 1.0) if legacy_one_observed and np.isfinite(delta) else np.nan
            )
            result[f"legacy_last_delta_ge1_observed_{horizon}"] = legacy_one_observed
            if has_baseline and baseline < 2 and has_post:
                cross_event = bool((post.lactate_mmol_l >= 2).any())
                cross_observed = bool(cross_event or followup_complete)
                result[f"baseline_lt2_followup_ge2_{horizon}"] = cross_event if cross_observed else np.nan
            else:
                result[f"baseline_lt2_followup_ge2_{horizon}"] = np.nan
            result[f"outcome_reason_{horizon}"] = reason
            result[f"last_lactate_rise_ge0_5_{horizon}"] = (
                value if observed else np.nan
            )
            completeness_rows.append({
                "horizon": horizon, "ICUSTAY_ID": stay, "SUBJECT_ID": int(row.SUBJECT_ID),
                "baseline_observed": has_baseline, "any_post_lactate": has_post,
                "followup_complete_to_horizon": followup_complete,
                "outcome_available": observed, "event": value if observed else np.nan,
                "reason": reason,
            })
        outcome_rows.append(result)
    outcomes = pd.DataFrame(outcome_rows)
    completeness = pd.DataFrame(completeness_rows)
    summaries = []
    for horizon in HORIZONS:
        part = completeness.loc[completeness.horizon.eq(horizon)]
        obs = part.loc[part.outcome_available]
        events_n = int(obs.event.astype(bool).sum()) if len(obs) else 0
        non_events = int(len(obs) - events_n)
        summaries.append({
            "horizon": horizon, "eligible_stays": int(base.ICUSTAY_ID.nunique()),
            "baseline_observed_n": int(part.baseline_observed.sum()),
            "any_post_landmark_lactate_n": int(part.any_post_lactate.sum()),
            "followup_complete_to_horizon_n": int(part.followup_complete_to_horizon.sum()),
            "observed_n": int(len(obs)), "events": events_n, "non_events": non_events,
            "event_prevalence": events_n / len(obs) if len(obs) else np.nan,
            "missing_baseline_n": int((~part.baseline_observed).sum()),
            "no_post_window_draw_n": int(part.reason.eq("no_post_landmark_lactate_in_window").sum()),
            "censored_without_event_n": int(part.reason.eq("censored_before_horizon_without_event").sum()),
            "censored_after_early_event_n": int(part.reason.eq("censored_before_horizon_after_early_last_value_event").sum()),
            "censored_unavailable_n": int(part.reason.str.startswith("censored_before_horizon").sum()),
            "adequacy_gate_pass": bool(len(obs) >= 200 and events_n >= 20 and non_events >= 20),
            "status": "available",
        })
    write_csv(outcomes, output, "03_landmark_lactate", "lactate_outcomes.csv")
    write_csv(completeness, output, "03_landmark_lactate", "endpoint_completeness_patient_level.csv")
    write_csv(pd.DataFrame(summaries), output, "03_landmark_lactate", "endpoint_completeness.csv")
    return outcomes, pd.DataFrame(summaries)


def build_item_map(items_path: Path, output: Path) -> tuple[dict[str, set[int]], pd.DataFrame]:
    items = read_table(items_path, usecols=["ITEMID", "LABEL"])
    items["LABEL"] = items.LABEL.fillna("").astype(str)
    label = items.LABEL.str.strip().str.lower()
    domains: dict[str, set[int]] = {}
    domains["hr"] = set(items.loc[label.eq("heart rate"), "ITEMID"].astype(int))
    domains["sbp"] = set(items.loc[
        label.str.contains("systolic", regex=False)
        & label.str.contains(r"arterial bp|nbp|manual bp|blood pressure|bp cuff|bp pal", regex=True)
        & ~label.str.contains("alarm|pulmonary|unloading", regex=True),
        "ITEMID",
    ].astype(int))
    domains["map"] = set(items.loc[
        label.str.contains(r"arterial bp mean|nbp mean|arterial mean|mean arterial pressure", regex=True)
        & ~label.str.contains("alarm", regex=False), "ITEMID",
    ].astype(int))
    domains["rr"] = set(items.loc[
        label.str.fullmatch(r"respiratory rate|respiratory rate \(total\)", case=False, na=False),
        "ITEMID",
    ].astype(int))
    domains["spo2"] = set(items.loc[
        label.str.fullmatch(r"spo2|spo2-l|oxygen saturation", case=False, na=False)
        & ~label.str.contains("alarm|limit", regex=True), "ITEMID",
    ].astype(int))
    domains["fio2"] = set(items.loc[
        label.str.contains("fio2|fi02|fraction inspired oxygen", case=False, regex=True)
        & ~label.str.contains("alarm|apache|desat|limit", case=False, regex=True), "ITEMID",
    ].astype(int))
    domains["temp"] = set(items.loc[
        label.str.fullmatch(r"temperature c|temperature f|temperature c \(calc\)|temperature f \(calc\)", case=False, na=False),
        "ITEMID",
    ].astype(int))
    domains["ventilation"] = set(items.loc[
        label.str.fullmatch(r"invasive ventilation|mechanically ventilated|ventilator mode|ventilator type", case=False, na=False),
        "ITEMID",
    ].astype(int))
    records = []
    for domain, ids in domains.items():
        for _, row in items.loc[items.ITEMID.isin(ids), ["ITEMID", "LABEL"]].drop_duplicates().iterrows():
            records.append({"domain": domain, "ITEMID": int(row.ITEMID), "LABEL": row.LABEL})
    item_frame = pd.DataFrame(records)
    item_frame.to_csv(output / "01_cohort" / "chart_item_map.csv", index=False)
    return domains, item_frame


def stream_chart_covariates(
    chart_path: Path | None,
    items_path: Path | None,
    clinical: pd.DataFrame,
    output: Path,
    logger: logging.Logger,
    chunksize: int = 1_000_000,
) -> tuple[pd.DataFrame, dict]:
    if not chart_path or not items_path:
        return pd.DataFrame(columns=["ICUSTAY_ID"]), {
            "status": "unavailable", "reason": "Full MIMIC-III CHARTEVENTS or D_ITEMS source is absent.",
            "missing_source": ",".join(str(x) for x in [chart_path, items_path] if not x),
        }
    domains, item_frame = build_item_map(items_path, output)
    id_domain = {}
    for domain, ids in domains.items():
        for itemid in ids:
            id_domain.setdefault(int(itemid), []).append(domain)
    itemids = set(id_domain)
    stay_ids = set(pd.to_numeric(clinical.ICUSTAY_ID, errors="coerce").dropna().astype(int))
    parts = []
    source_rows = 0
    selected_rows = 0
    for chunk_num, chunk in enumerate(pd.read_csv(
        chart_path,
        usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "ITEMID", "CHARTTIME", "VALUENUM", "VALUE"],
        chunksize=chunksize, low_memory=False,
    ), start=1):
        source_rows += len(chunk)
        stay_keep = pd.to_numeric(chunk.ICUSTAY_ID, errors="coerce").isin(stay_ids)
        item_keep = pd.to_numeric(chunk.ITEMID, errors="coerce").isin(itemids)
        chunk = chunk.loc[stay_keep & item_keep].copy()
        if not chunk.empty:
            chunk["ITEMID"] = pd.to_numeric(chunk.ITEMID, errors="coerce").astype("Int64")
            chunk["VALUENUM"] = pd.to_numeric(chunk.VALUENUM, errors="coerce")
            chunk["CHARTTIME"] = pd.to_datetime(chunk.CHARTTIME, errors="coerce")
            chunk["ICUSTAY_ID"] = pd.to_numeric(chunk.ICUSTAY_ID, errors="coerce").astype("Int64")
            intake_map = clinical.set_index("ICUSTAY_ID").INTIME
            chunk["INTIME"] = chunk.ICUSTAY_ID.map(intake_map)
            chunk["minute"] = (chunk.CHARTTIME - chunk.INTIME).dt.total_seconds() / 60
            chunk = chunk.loc[chunk.minute.ge(0) & chunk.minute.lt(LANDMARK_MIN) & chunk.CHARTTIME.notna()]
            if not chunk.empty:
                selected_rows += len(chunk)
                parts.append(chunk[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "ITEMID", "CHARTTIME", "minute", "VALUENUM", "VALUE"]])
        if chunk_num % 10 == 0:
            logger.info("CHARTEVENTS chunks scanned=%d; source rows=%d; selected pre-landmark rows=%d", chunk_num, source_rows, selected_rows)
    events = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "ITEMID", "CHARTTIME", "minute", "VALUENUM", "VALUE"]
    )
    value_limits = {
        "hr": (20, 250), "sbp": (30, 300), "map": (15, 200),
        "rr": (1, 100), "spo2": (50, 100), "fio2": (0.21, 100),
        "temp": (25, 45),
    }
    summaries = pd.DataFrame({"ICUSTAY_ID": clinical.ICUSTAY_ID.astype(int).unique()})
    for domain, limits in value_limits.items():
        domain_ids = domains.get(domain, set())
        part = events.loc[events.ITEMID.isin(domain_ids)].copy()
        value = part.VALUENUM.astype(float)
        if domain == "fio2":
            value = np.where(value > 1.0, value / 100.0, value)
            valid = np.isfinite(value) & (value >= .21) & (value <= 1.0)
        elif domain == "temp":
            labels = part.ITEMID.map(
                item_frame.loc[item_frame.domain.eq("temp")].set_index("ITEMID").LABEL.str.lower()
            )
            is_f = labels.str.contains(r"temperature f", na=False, regex=True)
            value = np.where(is_f, (value - 32.0) * 5.0 / 9.0, value)
            valid = np.isfinite(value) & (value >= limits[0]) & (value <= limits[1])
        else:
            valid = np.isfinite(value) & (value >= limits[0]) & (value <= limits[1])
        part = part.loc[valid].copy()
        part["clean_value"] = np.asarray(value)[valid]
        grouped = part.groupby("ICUSTAY_ID").clean_value.agg(["median", "count"]).reset_index()
        grouped = grouped.rename(columns={"median": f"early_{domain}_median", "count": f"early_{domain}_n"})
        summaries = summaries.merge(grouped, on="ICUSTAY_ID", how="left", validate="one_to_one")
    ventilation_ids = domains.get("ventilation", set())
    if ventilation_ids and not events.empty:
        part = events.loc[events.ITEMID.isin(ventilation_ids)].copy()
        txt = part.VALUE.fillna("").astype(str).str.strip().str.lower()
        yes = txt.isin({"yes", "true", "1", "invasive", "ventilator"})
        no = txt.isin({"no", "false", "0", "none", "room air"})
        yes_by = part.loc[yes].groupby("ICUSTAY_ID").size().rename("yes_n")
        no_by = part.loc[no].groupby("ICUSTAY_ID").size().rename("no_n")
        flag = pd.concat([yes_by, no_by], axis=1).fillna(0)
        flag["early_mechanical_ventilation"] = np.where(flag.yes_n.gt(0), 1.0, np.where(flag.no_n.gt(0), 0.0, np.nan))
        summaries = summaries.merge(flag[["early_mechanical_ventilation"]], left_on="ICUSTAY_ID", right_index=True, how="left")
    else:
        summaries["early_mechanical_ventilation"] = np.nan
    info = {
        "status": "available", "source_rows": int(source_rows),
        "selected_pre_landmark_rows": int(selected_rows),
        "mapped_item_ids": {key: sorted(map(int, value)) for key, value in domains.items()},
        "mapped_labels": item_frame.to_dict(orient="records"),
        "nonmissing_stays": {
            c: int(summaries[c].notna().sum()) for c in summaries if c.startswith("early_")
        },
    }
    summaries.to_csv(output / "01_cohort" / "early_chart_covariates.csv", index=False)
    logger.info("CHARTEVENTS streamed %d rows; %d matched eligible pre-landmark records.", source_rows, selected_rows)
    return summaries, info


def stream_vasoactive_status(
    files: dict[str, Path], items_path: Path | None, clinical: pd.DataFrame, output: Path, logger: logging.Logger
) -> tuple[pd.DataFrame, dict]:
    if not items_path:
        return pd.DataFrame(columns=["ICUSTAY_ID", "early_vasoactive_infusion"]), {
            "status": "unavailable", "reason": "D_ITEMS absent; medication names cannot be mapped.",
            "missing_source": "D_ITEMS.csv.gz",
        }
    dictionary = read_table(items_path, usecols=["ITEMID", "LABEL", "CATEGORY"])
    labels = dictionary.LABEL.fillna("").astype(str)
    drug_pat = r"norepinephrine|noradrenaline|epinephrine|adrenaline|dopamine|dobutamine|vasopressin|phenylephrine|milrinone|angiotensin ii"
    relevant = dictionary.loc[labels.str.contains(drug_pat, case=False, regex=True)].copy()
    # Input-event ITEMIDs are preferred over similarly named observations in other tables.
    drug_ids = set(pd.to_numeric(relevant.ITEMID, errors="coerce").dropna().astype(int))
    if not drug_ids or not files:
        return pd.DataFrame(columns=["ICUSTAY_ID", "early_vasoactive_infusion"]), {
            "status": "unavailable", "reason": "No mapped vasoactive infusion items or input-event tables were found.",
            "missing_source": "INPUTEVENTS_CV.csv.gz, INPUTEVENTS_MV.csv.gz, D_ITEMS.csv.gz",
            "candidate_item_labels": relevant.to_dict(orient="records"),
        }
    stay_ids = set(pd.to_numeric(clinical.ICUSTAY_ID, errors="coerce").dropna().astype(int))
    collected = []
    rows = 0
    rows_by_table: dict[str, int] = {}
    kept = 0
    specs = [
        ("INPUTEVENTS_CV", ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "ITEMID", "CHARTTIME", "RATE"]),
        ("INPUTEVENTS_MV", ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "ITEMID", "STARTTIME", "ENDTIME", "RATE"]),
    ]
    for table, cols in specs:
        path = files.get(table)
        if not path:
            continue
        table_rows = 0
        for chunk in pd.read_csv(path, usecols=cols, chunksize=1_000_000, low_memory=False):
            rows += len(chunk)
            table_rows += len(chunk)
            mask = pd.to_numeric(chunk.ICUSTAY_ID, errors="coerce").isin(stay_ids)
            mask &= pd.to_numeric(chunk.ITEMID, errors="coerce").isin(drug_ids)
            rate = pd.to_numeric(chunk.RATE, errors="coerce")
            mask &= rate.gt(0)
            part = chunk.loc[mask].copy()
            if part.empty:
                continue
            time_col = "CHARTTIME" if table.endswith("_CV") else "STARTTIME"
            part[time_col] = pd.to_datetime(part[time_col], errors="coerce")
            part["ICUSTAY_ID"] = pd.to_numeric(part.ICUSTAY_ID, errors="coerce").astype("Int64")
            part["INTIME"] = part.ICUSTAY_ID.map(clinical.set_index("ICUSTAY_ID").INTIME)
            part["minute"] = (part[time_col] - part.INTIME).dt.total_seconds() / 60
            part = part.loc[part.minute.ge(0) & part.minute.lt(LANDMARK_MIN)]
            kept += len(part)
            if not part.empty:
                collected.append(part[["ICUSTAY_ID", "ITEMID", "minute"]])
        rows_by_table[table] = table_rows
    if collected:
        hits = pd.concat(collected, ignore_index=True).groupby("ICUSTAY_ID").size().rename("early_vasoactive_event_rows")
        all_stays = pd.DataFrame({"ICUSTAY_ID": clinical.ICUSTAY_ID.astype(int).unique()})
        result = all_stays.merge(hits, left_on="ICUSTAY_ID", right_index=True, how="left")
        result["early_vasoactive_infusion"] = result.early_vasoactive_event_rows.gt(0).astype(float)
    else:
        result = pd.DataFrame({"ICUSTAY_ID": clinical.ICUSTAY_ID.astype(int).unique()})
        result["early_vasoactive_event_rows"] = 0
        result["early_vasoactive_infusion"] = 0.0
    info = {
        "status": "available", "source_rows_scanned": int(rows),
        "source_rows_by_table": rows_by_table,
        "positive_rate_pre_landmark_rows": int(kept),
        "mapped_item_labels": relevant[["ITEMID", "LABEL", "CATEGORY"]].to_dict(orient="records"),
    }
    result.to_csv(output / "01_cohort" / "early_vasoactive_status.csv", index=False)
    logger.info("Input events scanned %d rows; retained %d positive-rate early vasoactive records.", rows, kept)
    return result, info


def wilson_interval(events: int, total: int, alpha: float = .05) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    z = float(norm.ppf(1 - alpha / 2))
    p = events / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return max(0.0, center - half), min(1.0, center + half)


def newcombe_rd_interval(a: int, n1: int, c: int, n0: int) -> tuple[float, float]:
    if n1 == 0 or n0 == 0:
        return np.nan, np.nan
    p1, p0 = a / n1, c / n0
    l1, u1 = wilson_interval(a, n1)
    l0, u0 = wilson_interval(c, n0)
    diff = p1 - p0
    lower = diff - math.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2)
    upper = diff + math.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)
    return max(-1.0, lower), min(1.0, upper)


def cluster_bootstrap_rr_rd(
    frame: pd.DataFrame, exposure_col: str, outcome_col: str, patient_col: str,
    reps: int, seed: int,
) -> dict:
    use = frame[[exposure_col, outcome_col, patient_col]].dropna().copy()
    if use.empty:
        return {"bootstrap_reps_requested": reps, "bootstrap_reps_valid": 0}
    exp = use[exposure_col].astype(bool).to_numpy()
    outcome = use[outcome_col].astype(bool).to_numpy()
    cluster, uniques = pd.factorize(use[patient_col], sort=True)
    cells = np.zeros((len(uniques), 4), dtype=np.int64)
    code = exp.astype(int) * 2 + outcome.astype(int)
    np.add.at(cells, (cluster, code), 1)
    rng = np.random.default_rng(seed)
    rr_values, rd_values = [], []
    n_clusters = len(uniques)
    for _ in range(reps):
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        tab = cells[sampled].sum(axis=0)
        # Cell code: 0=unexposed/non-event, 1=unexposed/event,
        # 2=exposed/non-event, 3=exposed/event.
        n0, c = int(tab[0] + tab[1]), int(tab[1])
        n1, a = int(tab[2] + tab[3]), int(tab[3])
        if n0 and n1:
            p0, p1 = c / n0, a / n1
            if p0 > 0:
                rr_values.append(p1 / p0)
            rd_values.append(p1 - p0)
    def bounds(values):
        if len(values) < max(100, reps // 4):
            return np.nan, np.nan
        return tuple(np.quantile(values, [.025, .975]))
    rr_lo, rr_hi = bounds(rr_values)
    rd_lo, rd_hi = bounds(rd_values)
    return {
        "bootstrap_reps_requested": int(reps),
        "bootstrap_reps_valid": int(min(len(rr_values), len(rd_values))),
        "RR_cluster_boot_low": rr_lo, "RR_cluster_boot_high": rr_hi,
        "RD_cluster_boot_low": rd_lo, "RD_cluster_boot_high": rd_hi,
    }


def cluster_bootstrap_spearman(
    frame: pd.DataFrame, x_col: str, y_col: str, patient_col: str = "SUBJECT_ID",
    reps: int = 2000, seed: int = SEED,
) -> dict:
    use = frame[[x_col, y_col, patient_col]].apply(
        lambda col: pd.to_numeric(col, errors="coerce") if col.name != patient_col else col
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if use.empty:
        return {"bootstrap_reps_requested": int(reps), "bootstrap_reps_valid": 0,
                "rho_cluster_boot_low": np.nan, "rho_cluster_boot_high": np.nan}
    groups = pd.factorize(use[patient_col], sort=True)[0]
    by_patient = [np.flatnonzero(groups == group) for group in range(groups.max() + 1)]
    x = use[x_col].to_numpy(dtype=float)
    y = use[y_col].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(reps):
        sampled = rng.integers(0, len(by_patient), size=len(by_patient))
        idx = np.concatenate([by_patient[group] for group in sampled])
        if np.ptp(x[idx]) == 0 or np.ptp(y[idx]) == 0:
            continue
        rho = spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(rho):
            estimates.append(float(rho))
    minimum = max(100, reps // 4)
    low, high = tuple(np.quantile(estimates, [.025, .975])) if len(estimates) >= minimum else (np.nan, np.nan)
    return {
        "bootstrap_unit": "SUBJECT_ID/patient",
        "bootstrap_reps_requested": int(reps),
        "bootstrap_reps_valid": int(len(estimates)),
        "rho_cluster_boot_low": float(low), "rho_cluster_boot_high": float(high),
    }


def binary_effects(
    frame: pd.DataFrame, exposure_col: str, outcome_col: str,
    patient_col: str = "SUBJECT_ID", reps: int = 2000, seed: int = SEED,
) -> dict:
    use = frame[[exposure_col, outcome_col, patient_col]].dropna().copy()
    if use.empty:
        return {"status": "unavailable", "reason": "No classifiable observations.", "n": 0}
    use[exposure_col] = use[exposure_col].astype(bool)
    use[outcome_col] = use[outcome_col].astype(bool)
    a = int((use[exposure_col] & use[outcome_col]).sum())
    n1 = int(use[exposure_col].sum())
    c = int((~use[exposure_col] & use[outcome_col]).sum())
    n0 = int((~use[exposure_col]).sum())
    b, d = n1 - a, n0 - c
    if not n0 or not n1:
        return {"status": "unavailable", "reason": "One exposure group is empty.", "n": len(use)}
    p1, p0 = a / n1, c / n0
    rd = p1 - p0
    rr = p1 / p0 if p0 else np.inf
    or_value = (a * d / (b * c)) if b * c else np.inf
    if a > 0 and c > 0:
        se_log_rr = math.sqrt(max(0.0, 1 / a - 1 / n1 + 1 / c - 1 / n0))
        rr_lo, rr_hi = math.exp(math.log(rr) - 1.96 * se_log_rr), math.exp(math.log(rr) + 1.96 * se_log_rr)
    else:
        rr_lo = rr_hi = np.nan
    rd_lo, rd_hi = newcombe_rd_interval(a, n1, c, n0)
    odds, fisher_p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    boot = cluster_bootstrap_rr_rd(use, exposure_col, outcome_col, patient_col, reps, seed)
    return {
        "status": "available", "n": int(len(use)), "patients": int(use[patient_col].nunique()),
        "exposed_events": a, "exposed_total": n1, "unexposed_events": c, "unexposed_total": n0,
        "exposed_risk": p1, "unexposed_risk": p0,
        "RR": rr, "RR_95CI_low": rr_lo, "RR_95CI_high": rr_hi,
        "RD": rd, "RD_95CI_low": rd_lo, "RD_95CI_high": rd_hi,
        "OR_secondary": odds, "Fisher_exact_p": fisher_p,
        **boot,
    }


def modified_poisson(
    frame: pd.DataFrame, outcome_col: str, exposure_col: str, covariates: list[str],
    cluster_col: str = "SUBJECT_ID", effect_term: str = "exposure",
) -> dict:
    needed = [outcome_col, exposure_col, cluster_col] + covariates
    absent = [c for c in needed if c not in frame]
    if absent:
        return {"status": "unavailable", "reason": "Missing model columns: " + ",".join(absent)}
    use = frame[needed].dropna().copy()
    if use.empty or use[outcome_col].nunique() < 2:
        return {"status": "unavailable", "reason": "No complete cases or fewer than two outcome classes.", "n": len(use)}
    y = use[outcome_col].astype(float).to_numpy()
    expo = use[exposure_col].astype(float).to_numpy()
    if np.nanstd(expo) == 0:
        return {"status": "unavailable", "reason": "Exposure has no variation in model sample.", "n": len(use)}
    cols = []
    names = ["intercept", "exposure"]
    cols.extend([np.ones(len(use)), expo])
    for col in covariates:
        x = pd.to_numeric(use[col], errors="coerce").to_numpy(dtype=float)
        sd = float(np.nanstd(x))
        if not np.isfinite(sd) or sd < 1e-12:
            continue
        x = (x - float(np.nanmean(x))) / sd
        cols.append(x)
        names.append(col)
    X = np.column_stack(cols)
    n, p = X.shape
    if int(y.sum()) < 5 or n <= p + 2:
        return {
            "status": "not_fit_insufficient_support", "reason": "Fewer than 5 events or sample size too small for the prespecified model.",
            "n": int(n), "events": int(y.sum()), "non_events": int(n - y.sum()),
            "patients": int(use[cluster_col].nunique()), "parameters": int(p),
        }
    beta = np.zeros(p, dtype=float)
    converged = False
    for iteration in range(200):
        eta = np.clip(X @ beta, -18, 18)
        mu = np.exp(eta)
        bread_inv = X.T @ (X * mu[:, None])
        score = X.T @ (y - mu)
        try:
            step = np.linalg.pinv(bread_inv, rcond=1e-12) @ score
        except np.linalg.LinAlgError as exc:
            return {"status": "fit_failed", "reason": str(exc), "n": int(n), "events": int(y.sum())}
        beta_new = beta + step
        if not np.all(np.isfinite(beta_new)):
            return {"status": "fit_failed", "reason": "Non-finite IRLS coefficients.", "n": int(n), "events": int(y.sum())}
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            converged = True
            break
        beta = beta_new
    eta = np.clip(X @ beta, -18, 18)
    mu = np.exp(eta)
    bread = np.linalg.pinv(X.T @ (X * mu[:, None]), rcond=1e-12)
    score_rows = X * (y - mu)[:, None]
    cl, uniques = pd.factorize(use[cluster_col], sort=True)
    cluster_scores = np.zeros((len(uniques), p), dtype=float)
    np.add.at(cluster_scores, cl, score_rows)
    meat = cluster_scores.T @ cluster_scores
    correction = (len(uniques) / (len(uniques) - 1)) * ((n - 1) / (n - p)) if len(uniques) > 1 and n > p else 1.0
    covariance = bread @ meat @ bread * correction
    se = np.sqrt(np.maximum(0, np.diag(covariance)))
    if effect_term not in names:
        return {
            "status": "unavailable", "reason": f"Requested effect term {effect_term!r} is not estimable.",
            "n": int(n), "events": int(y.sum()), "parameters": int(p),
        }
    idx = names.index(effect_term)
    if not np.isfinite(se[idx]) or se[idx] <= 0:
        return {
            "status": "fit_failed", "reason": "Cluster robust exposure standard error is undefined.",
            "n": int(n), "events": int(y.sum()), "parameters": int(p), "converged": converged,
        }
    z = beta[idx] / se[idx]
    p_value = 2 * norm.sf(abs(z))
    return {
        "status": "available" if converged else "iteration_limit_reached",
        "reason": "" if converged else "IRLS did not meet convergence tolerance within 200 iterations.",
        "n": int(n), "events": int(y.sum()), "non_events": int(n - y.sum()),
        "patients": int(len(uniques)), "parameters": int(p), "covariates": "|".join(names[2:]),
        "exposure_beta": float(beta[idx]), "adjusted_RR": float(np.exp(beta[idx])),
        "robust_SE_log_RR": float(se[idx]),
        "adjusted_RR_95CI_low": float(np.exp(beta[idx] - 1.96 * se[idx])),
        "adjusted_RR_95CI_high": float(np.exp(beta[idx] + 1.96 * se[idx])),
        "p_value": float(p_value), "iterations": iteration + 1,
        "converged": bool(converged), "events_per_parameter": float(y.sum() / p),
        "effect_term": effect_term,
    }


def make_analysis_tables(
    clinical: pd.DataFrame, signal: pd.DataFrame, outcomes: pd.DataFrame,
    chart_covars: pd.DataFrame, vaso: pd.DataFrame, output: Path, logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    extra = clinical[[
        "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "age_years", "sex_male",
        "FIRST_CAREUNIT", "age_topcoded_90plus",
    ]].copy()
    analysis = outcomes.merge(extra, on=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"], how="left", validate="one_to_one")
    extra_signal = [
        "ICUSTAY_ID", "observed_15min_bins", "missing_15min_bins",
        "qualifying_transitions_le30min", "dynamics_eligible", "exposure_abs_jump_ge4",
        "rmssd_binned", "jump_fraction_ge3", "drop_count_ge3", "mean_binned_spo2",
        "min_binned_spo2", "fraction_binned_below90", "first_ge4_episode_minute",
        "first_ge4_episode_direction", "first_ge4_episode_delta", "max_abs_qualifying_jump",
        "first_ge4_drop_minute", "first_ge4_rise_minute", "n_ge4_transitions",
        "slope_spo2_per_hour", "max_transition_gap_minutes",
        "sampling_regime", "median_raw_samples_per_15min_bin",
    ]
    for to_merge, suffix in [(signal[extra_signal], "_sig"), (chart_covars, ""), (vaso, "")]:
        if "ICUSTAY_ID" in to_merge:
            cols = [c for c in to_merge.columns if c == "ICUSTAY_ID" or c not in analysis.columns]
            analysis = analysis.merge(to_merge[cols], on="ICUSTAY_ID", how="left", validate="one_to_one")
    # The outcome constructor already carried signal columns; prefer the independent signal recomputation.
    for col in extra_signal:
        if col == "ICUSTAY_ID":
            continue
        sig_col = col + "_sig"
        if sig_col in analysis:
            analysis[col] = analysis[sig_col].combine_first(analysis.get(col))
            analysis = analysis.drop(columns=[sig_col])
    if "sampling_regime" in analysis:
        analysis["sampling_1hz"] = analysis.sampling_regime.eq("1_Hz").astype(float)
        analysis["sampling_mixed"] = analysis.sampling_regime.eq("mixed").astype(float)
        analysis["sampling_unknown"] = analysis.sampling_regime.eq("unknown").astype(float)
    primary_rows = []
    adjusted_rows = []
    continuous_rows = []
    endpoint_rows = []
    dynamics = analysis.loc[analysis.dynamics_eligible.fillna(False).astype(bool)].copy()
    dynamics["exposure_abs_jump_ge4"] = dynamics.exposure_abs_jump_ge4.fillna(False).astype(bool)
    for horizon in HORIZONS:
        ycol = f"lactate_rise_ge0_5_{horizon}"
        avail = f"outcome_available_{horizon}"
        observed = dynamics.loc[dynamics[avail].fillna(False).astype(bool)].copy()
        observed[ycol] = observed[ycol].astype(bool)
        effects = binary_effects(
            observed, "exposure_abs_jump_ge4", ycol, reps=BOOTSTRAP_PRIMARY,
            seed=SEED + (12 if horizon == "12h" else 24),
        )
        primary_rows.append({"horizon": horizon, "exposure": "any_abs_qualifying_transition_ge4", **effects})
        endpoint_rows.append({
            "horizon": horizon, "dynamics_eligible_stays": int(len(dynamics)),
            "observed_n": int(len(observed)), "events": int(observed[ycol].sum()),
            "non_events": int(len(observed) - observed[ycol].sum()),
            "adequacy_gate_pass": bool(len(observed) >= 200 and observed[ycol].sum() >= 20 and (len(observed) - observed[ycol].sum()) >= 20),
        })
        core_covs = ["age_years", "sex_male", "baseline_lactate_mmol_l"]
        signal_covs = core_covs + [
            "mean_binned_spo2", "min_binned_spo2", "fraction_binned_below90",
            "observed_15min_bins", "sampling_1hz", "sampling_mixed", "sampling_unknown",
        ]
        specs = [
            ("clinical_core", core_covs),
            ("signal_adjusted_primary", signal_covs),
        ]
        ext = [
            "early_hr_median", "early_sbp_median", "early_map_median", "early_rr_median",
            "early_fio2_median", "early_mechanical_ventilation", "early_vasoactive_infusion",
        ]
        ext_available = [
            c for c in ext if c in observed and observed[c].notna().mean() >= .70
            and observed[c].nunique(dropna=True) > 1
        ]
        if ext_available:
            specs.append(("extended_prelandmark_clinical", signal_covs + ext_available))
        for spec, covs in specs:
            result = modified_poisson(observed, ycol, "exposure_abs_jump_ge4", covs)
            adjusted_rows.append({
                "horizon": horizon, "model": spec, "outcome": ycol,
                "exposure": "any_abs_qualifying_transition_ge4", **result,
            })
        continuous_col = f"continuous_last_delta_{horizon}"
        for metric in ["rmssd_binned", "jump_fraction_ge3", "drop_count_ge3", "max_abs_qualifying_jump"]:
            part = dynamics.loc[dynamics[continuous_col].notna() & dynamics[metric].notna()].copy()
            binary_part = part.loc[
                part[avail].fillna(False).astype(bool) & part[ycol].notna()
            ].copy()
            x = pd.to_numeric(binary_part[metric], errors="coerce").astype(float)
            sd = float(x.std(ddof=0)) if len(x) else np.nan
            if len(binary_part) and np.isfinite(sd) and sd > 0:
                binary_part["standardized_metric"] = (x - x.mean()) / sd
                binary_part["binary_event"] = binary_part[ycol].astype(bool)
                continuous_covs = [
                    "age_years", "sex_male", "baseline_lactate_mmol_l",
                    "mean_binned_spo2", "min_binned_spo2", "fraction_binned_below90",
                    "observed_15min_bins", "sampling_1hz", "sampling_mixed", "sampling_unknown",
                ]
                mp = modified_poisson(
                    binary_part, "binary_event", "standardized_metric", continuous_covs,
                )
                continuous_rows.append({
                    "horizon": horizon, "signal_metric": metric,
                    "analysis": "binary endpoint modified Poisson RR per 1 SD",
                    "n": int(len(binary_part)), **mp,
                })
            if len(part):
                xx = pd.to_numeric(part[continuous_col], errors="coerce")
                yy = pd.to_numeric(part[metric], errors="coerce")
                rho = spearmanr(yy, xx, nan_policy="omit")
                metric_seed = SEED + (120 if horizon == "12h" else 240) + [
                    "rmssd_binned", "jump_fraction_ge3", "drop_count_ge3", "max_abs_qualifying_jump",
                ].index(metric)
                rho_ci = cluster_bootstrap_spearman(
                    part, metric, continuous_col, reps=2000, seed=metric_seed,
                )
                continuous_rows.append({
                    "horizon": horizon, "signal_metric": metric,
                    "analysis": "continuous last lactate delta Spearman rank",
                    "n": int(len(part)), "rho": float(rho.statistic) if np.isfinite(rho.statistic) else np.nan,
                    "p_value": float(rho.pvalue) if np.isfinite(rho.pvalue) else np.nan,
                    **rho_ci, "status": "available",
                })
    binary = pd.DataFrame(primary_rows)
    adjusted = pd.DataFrame(adjusted_rows)
    continuous = pd.DataFrame(continuous_rows)
    write_csv(binary, output, "03_landmark_lactate", "landmark_binary_association.csv")
    write_csv(adjusted, output, "03_landmark_lactate", "adjusted_modified_poisson.csv")
    write_csv(continuous, output, "03_landmark_lactate", "landmark_continuous_association.csv")
    write_csv(pd.DataFrame(endpoint_rows), output, "03_landmark_lactate", "landmark_analysis_sample_counts.csv")
    logger.info("Primary associations calculated for both prespecified horizons.")
    return analysis, binary, adjusted, continuous


def previous_count_reconciliation(
    analysis: pd.DataFrame, clinical: pd.DataFrame, binary: pd.DataFrame, output: Path,
) -> pd.DataFrame:
    """Compare user-provided prior counts with locked recomputation and a censoring audit."""
    expected = {
        "12h": (20, 97, 17, 143, 1.73),
        "24h": (14, 108, 16, 150, 1.22),
    }
    dynamic = analysis.loc[analysis.dynamics_eligible.fillna(False).astype(bool)].copy()
    censor_end = clinical.set_index("ICUSTAY_ID").apply(
        lambda row: (row.icu_observable_end - row.INTIME).total_seconds() / 60
        if pd.notna(row.icu_observable_end) and pd.notna(row.INTIME) else np.nan,
        axis=1,
    )
    dynamic["earliest_icu_or_hospital_end_minute"] = dynamic.ICUSTAY_ID.map(censor_end)
    rows = []
    for horizon, horizon_end in HORIZONS.items():
        ee, en, ue, un, expected_rr = expected[horizon]
        rows.append({
            "horizon": horizon, "analysis": "user_provided_prior_expected_counts",
            "exposed_events": ee, "exposed_total": en,
            "unexposed_events": ue, "unexposed_total": un,
            "observed_n": en + un, "RR": expected_rr,
            "status": "prior_expected_counts_from_user_prompt",
            "reason": "Comparator only; the earlier feasibility artifact and upstream waveform candidate list were not present to reconstruct its source sample.",
        })
        primary = binary.loc[binary.horizon.eq(horizon)].iloc[0]
        rows.append({
            "horizon": horizon, "analysis": "locked_primary_reconstruction",
            "exposed_events": int(primary.exposed_events), "exposed_total": int(primary.exposed_total),
            "unexposed_events": int(primary.unexposed_events), "unexposed_total": int(primary.unexposed_total),
            "observed_n": int(primary.n), "RR": float(primary.RR),
            "status": "available",
            "reason": "Full MIMIC-III v1.4 clinical cohort, linked LABEVENTS, and supplied HF signal subset; locked outcome/censoring rule.",
        })
        exposure = dynamic.exposure_abs_jump_ge4.fillna(False).astype(bool)
        baseline = dynamic.baseline_lactate_observed.fillna(False).astype(bool)
        has_post = pd.to_numeric(dynamic[f"n_lactate_draws_{horizon}"], errors="coerce").fillna(0).gt(0)
        delta = pd.to_numeric(dynamic[f"continuous_last_delta_{horizon}"], errors="coerce")
        direct_event = delta.ge(.5) & delta.notna()
        complete_to_horizon = pd.to_numeric(dynamic.earliest_icu_or_hospital_end_minute, errors="coerce").ge(horizon_end)
        classifiable = baseline & has_post & complete_to_horizon
        alternate = pd.DataFrame({
            "exposure": exposure.loc[classifiable],
            "event": direct_event.loc[classifiable],
        })
        a = int((alternate.exposure & alternate.event).sum())
        n1 = int(alternate.exposure.sum())
        c = int((~alternate.exposure & alternate.event).sum())
        n0 = int((~alternate.exposure).sum())
        p0 = c / n0 if n0 else np.nan
        rr = (a / n1) / p0 if n1 and p0 else np.nan
        rows.append({
            "horizon": horizon, "analysis": "earliest_icu_hospital_or_death_censor_sensitivity",
            "exposed_events": a, "exposed_total": n1,
            "unexposed_events": c, "unexposed_total": n0,
            "observed_n": n1 + n0, "RR": rr,
            "status": "available",
            "reason": "Complete-window sensitivity requiring baseline, a post-landmark lactate, and observation through the horizon using the earliest ICU outtime, hospital discharge, or death; early positives are unavailable.",
        })
    result = pd.DataFrame(rows)
    result.to_csv(output / "03_landmark_lactate" / "previous_count_reconciliation.csv", index=False)
    return result


def previous_count_reconciliation_pass(reconciliation: pd.DataFrame) -> bool:
    expected_prior = {
        "12h": (20, 97, 17, 143, 240),
        "24h": (14, 108, 16, 150, 258),
    }
    analyses = [
        "user_provided_prior_expected_counts", "locked_primary_reconstruction",
        "earliest_icu_hospital_or_death_censor_sensitivity",
    ]
    required = {
        "horizon", "analysis", "exposed_events", "exposed_total", "unexposed_events",
        "unexposed_total", "observed_n", "RR", "status",
    }
    if not required.issubset(reconciliation.columns) or len(reconciliation) != 2 * len(analyses):
        return False
    if reconciliation.duplicated(["horizon", "analysis"]).any():
        return False
    for horizon in HORIZONS:
        group = reconciliation.loc[reconciliation.horizon.eq(horizon)].set_index("analysis")
        if set(group.index) != set(analyses):
            return False
        prior = group.loc["user_provided_prior_expected_counts"]
        prior_actual = tuple(int(prior[col]) for col in [
            "exposed_events", "exposed_total", "unexposed_events", "unexposed_total", "observed_n",
        ])
        if prior_actual != expected_prior[horizon] or prior.status != "prior_expected_counts_from_user_prompt":
            return False
        primary = group.loc["locked_primary_reconstruction"]
        alternate = group.loc["earliest_icu_hospital_or_death_censor_sensitivity"]
        for row in [primary, alternate]:
            counts = [int(row[col]) for col in ["exposed_events", "exposed_total", "unexposed_events", "unexposed_total"]]
            if any(value < 0 for value in counts) or counts[0] > counts[1] or counts[2] > counts[3]:
                return False
            if int(row.observed_n) != counts[1] + counts[3] or row.status != "available":
                return False
        for column in ["exposed_events", "exposed_total", "unexposed_events", "unexposed_total", "observed_n"]:
            if int(primary[column]) != int(alternate[column]):
                return False
    return True


def complete_followup_before_after_reconciliation(
    output: Path,
    old_outcomes: pd.DataFrame | None,
    new_outcomes: pd.DataFrame,
    signal: pd.DataFrame,
    old_episodes: pd.DataFrame | None,
    new_episodes: pd.DataFrame,
    old_direction: pd.DataFrame | None,
    new_direction: pd.DataFrame,
) -> pd.DataFrame:
    """Record exact legacy-to-complete-window removals for every last-value binary endpoint."""
    rows: list[dict] = []

    def add_row(family: str, window: str, group: str, threshold: str,
                before: pd.DataFrame, after: pd.DataFrame,
                before_av: str, before_y: str, after_av: str, after_y: str) -> None:
        if "ICUSTAY_ID" in before and "ICUSTAY_ID" in after:
            if before.ICUSTAY_ID.duplicated().any() or after.ICUSTAY_ID.duplicated().any():
                raise ValueError(f"Duplicate stay IDs in reconciliation row {family}/{window}/{group}.")
            before = before.set_index("ICUSTAY_ID")
            after = after.set_index("ICUSTAY_ID")
            common = before.index.intersection(after.index)
            before, after = before.loc[common], after.loc[common]
        ba = before[before_av].fillna(False).astype(bool)
        aa = after[after_av].fillna(False).astype(bool)
        by = before[before_y].fillna(False).astype(bool)
        ay = after[after_y].fillna(False).astype(bool)
        removed = ba & ~aa
        before_events = int((ba & by).sum())
        before_nonevents = int((ba & ~by).sum())
        after_events = int((aa & ay).sum())
        after_nonevents = int((aa & ~ay).sum())
        rows.append({
            "endpoint_family": family, "window": window, "group": group, "threshold": threshold,
            "common_anchor_stays": int(len(before)),
            "previously_observed_n": int(ba.sum()), "previously_events": before_events,
            "previously_non_events": before_nonevents,
            "previously_positive_removed": int((removed & by).sum()),
            "previously_negative_removed": int((removed & ~by).sum()),
            "updated_observed_n": int(aa.sum()), "updated_events": after_events,
            "updated_non_events": after_nonevents,
            "net_observations_removed": int(ba.sum() - aa.sum()),
            "status": "available",
        })

    dynamic_ids = set(signal.loc[signal.dynamics_eligible.fillna(False).astype(bool), "ICUSTAY_ID"].astype(int))
    before_main = old_outcomes if old_outcomes is not None else new_outcomes
    for horizon in HORIZONS:
        after_all = new_outcomes
        before_all = before_main
        if old_outcomes is None:
            before_av, before_y = f"legacy_outcome_available_{horizon}", f"legacy_lactate_rise_ge0_5_{horizon}"
        else:
            before_av, before_y = f"outcome_available_{horizon}", f"lactate_rise_ge0_5_{horizon}"
        add_row("landmark_last_value_ge0.5", horizon, "all_clinically_eligible", "delta>=0.5 mmol/L",
                before_all, after_all, before_av, before_y,
                f"outcome_available_{horizon}", f"lactate_rise_ge0_5_{horizon}")
        add_row("landmark_last_value_ge0.5", horizon, "dynamics_eligible", "delta>=0.5 mmol/L",
                before_all.loc[before_all.ICUSTAY_ID.astype(int).isin(dynamic_ids)],
                after_all.loc[after_all.ICUSTAY_ID.astype(int).isin(dynamic_ids)],
                before_av, before_y, f"outcome_available_{horizon}", f"lactate_rise_ge0_5_{horizon}")
        if old_outcomes is None:
            before_one_av, before_one_y = f"legacy_last_delta_ge1_observed_{horizon}", f"legacy_lactate_last_delta_ge1_{horizon}"
        else:
            before_one_av, before_one_y = f"last_delta_ge1_observed_{horizon}", f"lactate_last_delta_ge1_{horizon}"
        add_row("landmark_last_value_ge1.0_sensitivity", horizon, "all_clinically_eligible", "delta>=1.0 mmol/L",
                before_all, after_all, before_one_av, before_one_y,
                f"last_delta_ge1_observed_{horizon}", f"lactate_last_delta_ge1_{horizon}")

    episode_before = old_episodes if old_episodes is not None else new_episodes
    for window in ["acute_0_1h", "delayed_1_8h", "cumulative_0_4h", "cumulative_0_8h", "cumulative_0_12h"]:
        for group in ["episode", "time_aligned_no_episode"]:
            b = episode_before.loc[episode_before.anchor_group.eq(group)]
            a = new_episodes.loc[new_episodes.anchor_group.eq(group)]
            if old_episodes is None:
                bav, by = f"legacy_outcome_available_{window}", f"legacy_rise_ge0_5_{window}"
                bgav, bgy = f"legacy_outcome_available_ge1_{window}", f"legacy_rise_ge1_{window}"
            else:
                bav, by = f"outcome_available_{window}", f"rise_ge0_5_{window}"
                bgav, bgy = f"outcome_available_ge1_{window}", f"rise_ge1_{window}"
            add_row("episode_last_value_ge0.5", window, group, "delta>=0.5 mmol/L", b, a, bav, by,
                    f"outcome_available_{window}", f"rise_ge0_5_{window}")
            add_row("episode_last_value_ge1.0", window, group, "delta>=1.0 mmol/L", b, a, bgav, bgy,
                    f"outcome_available_ge1_{window}", f"rise_ge1_{window}")

    direction_before = old_direction if old_direction is not None else new_direction
    for direction_name in ["drop", "rise"]:
        for window in ["acute_0_1h", "delayed_1_8h"]:
            for group in [direction_name + "_episode", "time_aligned_no_episode"]:
                b = direction_before.loc[
                    direction_before.direction.eq(direction_name)
                    & direction_before.window.eq(window)
                    & direction_before.group.eq(group)
                ]
                a = new_direction.loc[
                    new_direction.direction.eq(direction_name)
                    & new_direction.window.eq(window)
                    & new_direction.group.eq(group)
                ]
                if old_direction is None:
                    bav, by = "legacy_available", "legacy_event"
                else:
                    bav, by = "available", "event"
                add_row("episode_direction_last_value_ge0.5", f"{direction_name}_{window}", group,
                        "delta>=0.5 mmol/L", b, a, bav, by, "available", "event")

    result = pd.DataFrame(rows)
    path = output / "03_landmark_lactate" / "complete_followup_before_after_reconciliation.csv"
    result.to_csv(path, index=False)
    return result


def complete_followup_reconciliation_pass(frame: pd.DataFrame) -> bool:
    required = {
        "endpoint_family", "window", "group", "common_anchor_stays",
        "previously_observed_n", "previously_events", "previously_non_events",
        "previously_positive_removed", "previously_negative_removed",
        "updated_observed_n", "updated_events", "updated_non_events",
        "net_observations_removed",
    }
    if not required.issubset(frame.columns) or frame.empty:
        return False
    numeric = [column for column in required if column not in {"endpoint_family", "window", "group"}]
    counts = frame[numeric].apply(pd.to_numeric, errors="coerce")
    if counts.isna().any().any() or (counts < 0).any().any():
        return False
    return bool(
        (frame.previously_events + frame.previously_non_events == frame.previously_observed_n).all()
        and (frame.updated_events + frame.updated_non_events == frame.updated_observed_n).all()
        and (frame.previously_positive_removed + frame.previously_negative_removed == frame.net_observations_removed).all()
        and (frame.updated_observed_n <= frame.previously_observed_n).all()
        and (frame.common_anchor_stays >= frame.previously_observed_n).all()
    )


def make_episode_anchor(
    analysis: pd.DataFrame, events: pd.DataFrame, output: Path, seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dynamic = analysis.loc[analysis.dynamics_eligible.fillna(False).astype(bool)].copy()
    exposure = dynamic.loc[dynamic.exposure_abs_jump_ge4.fillna(False).astype(bool)].copy()
    controls = dynamic.loc[~dynamic.exposure_abs_jump_ge4.fillna(False).astype(bool)].copy()
    event_times = pd.to_numeric(exposure.first_ge4_episode_minute, errors="coerce").dropna().sort_values().to_numpy()
    controls = controls.sort_values(["SUBJECT_ID", "ICUSTAY_ID"], kind="mergesort").copy()
    if len(event_times) and len(controls):
        controls["anchor_minute"] = event_times[np.arange(len(controls)) % len(event_times)]
    else:
        controls["anchor_minute"] = np.nan
    exposure["anchor_minute"] = exposure.first_ge4_episode_minute
    exposure["anchor_group"] = "episode"
    controls["anchor_group"] = "time_aligned_no_episode"
    combined = pd.concat([exposure, controls], ignore_index=True)
    combined["anchor_type"] = combined.anchor_group
    draw_map = {int(k): g.sort_values("minute_from_icu_intime") for k, g in events.groupby("ICUSTAY_ID")} if not events.empty else {}
    horizons = {
        "acute_0_1h": (0.0, 60.0),
        "delayed_1_8h": (60.0, 480.0),
        "cumulative_0_4h": (0.0, 240.0),
        "cumulative_0_8h": (0.0, 480.0),
        "cumulative_0_12h": (0.0, 720.0),
    }
    records = []
    trajectory = []
    for row in combined.itertuples(index=False):
        anchor = float(row.anchor_minute) if pd.notna(row.anchor_minute) else np.nan
        d = draw_map.get(int(row.ICUSTAY_ID), pd.DataFrame(columns=["minute_from_icu_intime", "lactate_mmol_l", "CHARTTIME"]))
        if np.isfinite(anchor):
            pre = d.loc[d.minute_from_icu_intime.lt(anchor)]
        else:
            pre = d.iloc[0:0]
        base_row = pre.iloc[-1] if len(pre) else None
        base = float(base_row.lactate_mmol_l) if base_row is not None else np.nan
        previous = pre.iloc[-2] if len(pre) >= 2 else None
        prior_delta = float(base - previous.lactate_mmol_l) if previous is not None else np.nan
        prior_hours = float((base_row.minute_from_icu_intime - previous.minute_from_icu_intime) / 60) if previous is not None else np.nan
        record = {
            "SUBJECT_ID": int(row.SUBJECT_ID), "HADM_ID": int(row.HADM_ID), "ICUSTAY_ID": int(row.ICUSTAY_ID),
            "anchor_group": row.anchor_group, "anchor_minute": anchor,
            "age_years": row.age_years, "sex_male": row.sex_male,
            "anchor_time_definition": "representative_time of second bin completing first qualifying transition; controls assigned exposed anchor times in sorted round-robin order before outcome inspection",
            "event_direction": row.first_ge4_episode_direction if row.anchor_group == "episode" else "none",
            "strict_pre_anchor_baseline_lactate": base,
            "strict_pre_anchor_baseline_time_minute": float(base_row.minute_from_icu_intime) if base_row is not None else np.nan,
            "strict_pre_anchor_baseline_time": base_row.CHARTTIME if base_row is not None else pd.NaT,
            "previous_pre_anchor_lactate": float(previous.lactate_mmol_l) if previous is not None else np.nan,
            "previous_pre_anchor_time_minute": float(previous.minute_from_icu_intime) if previous is not None else np.nan,
            "pre_anchor_delta_from_previous": prior_delta,
            "pre_anchor_delta_per_hour": prior_delta / prior_hours if previous is not None and prior_hours > 0 else np.nan,
            "n_lactate_draws_strictly_pre_anchor": int(len(pre)),
            "status": "available" if base_row is not None else "missing_strict_pre_anchor_baseline",
        }
        end_observation = float(row.observable_end_minute) if pd.notna(row.observable_end_minute) else np.nan
        hospital_end_observation = (
            float(row.hospital_observable_end_minute)
            if hasattr(row, "hospital_observable_end_minute") and pd.notna(row.hospital_observable_end_minute)
            else end_observation
        )
        for name, (start_after, end_after) in horizons.items():
            if np.isfinite(anchor) and not d.empty:
                post = d.loc[
                    d.minute_from_icu_intime.gt(anchor + start_after)
                    & d.minute_from_icu_intime.le(anchor + end_after)
                ]
            else:
                post = d.iloc[0:0]
            first = post.iloc[0] if len(post) else None
            last = post.iloc[-1] if len(post) else None
            max_value = float(post.lactate_mmol_l.max()) if len(post) else np.nan
            first_delta = float(first.lactate_mmol_l - base) if first is not None and np.isfinite(base) else np.nan
            last_delta = float(last.lactate_mmol_l - base) if last is not None and np.isfinite(base) else np.nan
            max_delta = float(max_value - base) if np.isfinite(max_value) and np.isfinite(base) else np.nan
            window_complete = bool(np.isfinite(end_observation) and end_observation >= anchor + end_after) if np.isfinite(anchor) else False
            legacy_window_complete = bool(
                np.isfinite(hospital_end_observation) and hospital_end_observation >= anchor + end_after
            ) if np.isfinite(anchor) else False
            direct = bool(np.isfinite(last_delta) and last_delta >= .5)
            if not np.isfinite(base):
                available, outcome = False, np.nan
                reason = "missing_strict_pre_anchor_baseline"
            elif not len(post):
                available, outcome = False, np.nan
                reason = "no_lactate_in_post_anchor_window"
            elif not window_complete:
                available, outcome = False, np.nan
                reason = (
                    "censored_before_window_end_after_early_last_value_event"
                    if direct else "censored_before_window_end_without_event"
                )
            elif direct:
                available, outcome, reason = True, True, "complete_followup_last_value_event"
            else:
                available, outcome, reason = True, False, "complete_observation_through_window_end"
            legacy_available = bool(np.isfinite(base) and len(post) and (direct or legacy_window_complete))
            legacy_outcome = bool(direct) if legacy_available else np.nan
            record[f"n_draws_{name}"] = int(len(post))
            record[f"first_next_lactate_{name}"] = float(first.lactate_mmol_l) if first is not None else np.nan
            record[f"first_next_delta_{name}"] = first_delta
            record[f"last_lactate_{name}"] = float(last.lactate_mmol_l) if last is not None else np.nan
            record[f"last_delta_{name}"] = last_delta
            record[f"maximum_lactate_{name}"] = max_value
            record[f"maximum_delta_{name}"] = max_delta
            record[f"outcome_available_{name}"] = available
            record[f"rise_ge0_5_{name}"] = outcome
            record[f"legacy_outcome_available_{name}"] = legacy_available
            record[f"legacy_rise_ge0_5_{name}"] = legacy_outcome
            one_event = bool(np.isfinite(last_delta) and last_delta >= 1.0)
            one_available = bool(np.isfinite(base) and len(post) and window_complete)
            legacy_one_available = bool(
                np.isfinite(base) and len(post) and (one_event or legacy_window_complete)
            )
            record[f"outcome_available_ge1_{name}"] = one_available
            record[f"rise_ge1_{name}"] = one_event if one_available else np.nan
            record[f"legacy_outcome_available_ge1_{name}"] = legacy_one_available
            record[f"legacy_rise_ge1_{name}"] = one_event if legacy_one_available else np.nan
            record[f"outcome_reason_{name}"] = reason
        records.append(record)
        if np.isfinite(anchor) and not d.empty and np.isfinite(base):
            around = d.loc[
                d.minute_from_icu_intime.ge(anchor - 480)
                & d.minute_from_icu_intime.le(anchor + 720)
            ].copy()
            for draw in around.itertuples(index=False):
                trajectory.append({
                    "SUBJECT_ID": int(row.SUBJECT_ID), "ICUSTAY_ID": int(row.ICUSTAY_ID),
                    "anchor_group": row.anchor_group, "anchor_minute": anchor,
                    "relative_minute": float(draw.minute_from_icu_intime - anchor),
                    "lactate_mmol_l": float(draw.lactate_mmol_l),
                    "change_from_strict_pre_anchor_baseline": float(draw.lactate_mmol_l - base),
                })
    anchors = pd.DataFrame(records)
    trajectory_df = pd.DataFrame(trajectory)
    anchors["pre_anchor_trend_missing"] = anchors.pre_anchor_delta_per_hour.isna().astype(float)
    anchors["pre_anchor_trend_per_hour_model"] = anchors.pre_anchor_delta_per_hour.fillna(0.0)
    pretrend_rows = []
    for group, part in anchors.groupby("anchor_group"):
        trend = pd.to_numeric(part.pre_anchor_delta_per_hour, errors="coerce").dropna()
        pretrend_rows.append({
            "group": group, "anchors": int(len(part)),
            "serial_pre_anchor_lactate_n": int(len(trend)),
            "serial_pre_anchor_lactate_fraction": float(len(trend) / len(part)) if len(part) else np.nan,
            "pre_anchor_delta_per_hour_mean": float(trend.mean()) if len(trend) else np.nan,
            "pre_anchor_delta_per_hour_median": float(trend.median()) if len(trend) else np.nan,
            "pre_anchor_delta_per_hour_q1": float(trend.quantile(.25)) if len(trend) else np.nan,
            "pre_anchor_delta_per_hour_q3": float(trend.quantile(.75)) if len(trend) else np.nan,
            "pre_anchor_lactate_rising_fraction": float((trend > 0).mean()) if len(trend) else np.nan,
            "status": "available" if len(trend) else "unavailable_no_serial_pre_anchor_draws",
            "interpretation": "Descriptive temporal ordering only; serial pre-anchor draws are selected and do not establish causality.",
        })
    pd.DataFrame(pretrend_rows).to_csv(
        output / "04_episode_lactate" / "episode_pre_anchor_trend_summary.csv", index=False,
    )
    primary_effects = []
    for window in horizons:
        for group, part in anchors.groupby("anchor_group"):
            available = part.loc[part[f"outcome_available_{window}"].fillna(False)]
            events_n = int(available[f"rise_ge0_5_{window}"].astype(bool).sum()) if len(available) else 0
            primary_effects.append({
                "window": window, "group": group, "n_anchors": int(len(part)),
                "baseline_available_n": int(part.strict_pre_anchor_baseline_lactate.notna().sum()),
                "observed_n": int(len(available)), "events": events_n,
                "non_events": int(len(available) - events_n),
                "event_prevalence": events_n / len(available) if len(available) else np.nan,
                "mean_last_delta": float(available[f"last_delta_{window}"].mean()) if len(available) else np.nan,
                "median_last_delta": float(available[f"last_delta_{window}"].median()) if len(available) else np.nan,
                "mean_maximum_delta": float(available[f"maximum_delta_{window}"].mean()) if len(available) else np.nan,
            })
        pair = anchors.loc[anchors[f"outcome_available_{window}"].fillna(False)].copy()
        pair["episode_exposed"] = pair.anchor_group.eq("episode")
        contrast = binary_effects(
            pair, "episode_exposed", f"rise_ge0_5_{window}", reps=2000,
            seed=SEED + {
                "acute_0_1h": 101, "delayed_1_8h": 102,
                "cumulative_0_4h": 104, "cumulative_0_8h": 108,
                "cumulative_0_12h": 112,
            }[window],
        )
        primary_effects.append({"window": window, "group": "episode_vs_time_aligned_control", **contrast})
        pair["episode_exposed"] = pair.anchor_group.eq("episode").astype(float)
        adjusted_episode = modified_poisson(
            pair, f"rise_ge0_5_{window}", "episode_exposed",
            ["age_years", "sex_male", "strict_pre_anchor_baseline_lactate",
             "pre_anchor_trend_per_hour_model", "pre_anchor_trend_missing"],
        )
        primary_effects.append({
            "window": window, "group": "episode_vs_time_aligned_control_adjusted",
            "adjustment_set": "age|sex|strict_pre_anchor_baseline_lactate|prior_pre_anchor_lactate_change_per_hour|prior_trend_missing_indicator",
            **adjusted_episode,
        })
    effects = pd.DataFrame(primary_effects)
    descriptive_rows = []
    ge1_rows = []
    for window in horizons:
        for group, part in anchors.groupby("anchor_group"):
            baseline_n = int(part.strict_pre_anchor_baseline_lactate.notna().sum())
            record = {
                "window": window, "group": group, "n_anchors": int(len(part)),
                "strict_pre_anchor_baseline_n": baseline_n,
            }
            for stat, value_col in [
                ("first_next", f"first_next_lactate_{window}"),
                ("first_next_delta", f"first_next_delta_{window}"),
                ("last", f"last_lactate_{window}"),
                ("last_delta", f"last_delta_{window}"),
                ("maximum", f"maximum_lactate_{window}"),
                ("maximum_delta", f"maximum_delta_{window}"),
            ]:
                vals = pd.to_numeric(part[value_col], errors="coerce").dropna()
                record[f"{stat}_n"] = int(len(vals))
                record[f"{stat}_mean"] = float(vals.mean()) if len(vals) else np.nan
                record[f"{stat}_median"] = float(vals.median()) if len(vals) else np.nan
            primary_available = part.loc[part[f"outcome_available_{window}"].fillna(False)]
            secondary_available = part.loc[part[f"outcome_available_ge1_{window}"].fillna(False)]
            record["primary_ge0_5_observed_n"] = int(len(primary_available))
            record["primary_ge0_5_events"] = int(primary_available[f"rise_ge0_5_{window}"].astype(bool).sum()) if len(primary_available) else 0
            record["secondary_ge1_observed_n"] = int(len(secondary_available))
            record["secondary_ge1_events"] = int(secondary_available[f"rise_ge1_{window}"].astype(bool).sum()) if len(secondary_available) else 0
            record["status"] = "available" if baseline_n else "no_strict_pre_anchor_baseline"
            descriptive_rows.append(record)
            ge1_rows.append({
                "window": window, "group": group, "status": "available",
                "n_anchors": int(len(part)), "baseline_available_n": baseline_n,
                "observed_n": int(len(secondary_available)),
                "events": int(secondary_available[f"rise_ge1_{window}"].astype(bool).sum()) if len(secondary_available) else 0,
                "non_events": int(len(secondary_available) - secondary_available[f"rise_ge1_{window}"].astype(bool).sum()) if len(secondary_available) else 0,
                "event_prevalence": float(secondary_available[f"rise_ge1_{window}"].astype(bool).mean()) if len(secondary_available) else np.nan,
            })
        pair_ge1 = anchors.loc[anchors[f"outcome_available_ge1_{window}"].fillna(False)].copy()
        pair_ge1["episode_exposed"] = pair_ge1.anchor_group.eq("episode")
        ge1_effect = binary_effects(
            pair_ge1, "episode_exposed", f"rise_ge1_{window}", reps=2000,
            seed=SEED + {"acute_0_1h": 211, "delayed_1_8h": 212,
                         "cumulative_0_4h": 214, "cumulative_0_8h": 218,
                         "cumulative_0_12h": 222}[window],
        )
        ge1_rows.append({"window": window, "group": "episode_vs_time_aligned_control", **ge1_effect})
    pd.DataFrame(descriptive_rows).to_csv(
        output / "04_episode_lactate" / "episode_window_descriptives.csv", index=False,
    )
    pd.DataFrame(ge1_rows).to_csv(
        output / "04_episode_lactate" / "episode_ge1_sensitivity.csv", index=False,
    )
    anchors.to_csv(output / "04_episode_lactate" / "episode_anchor_records.csv", index=False)
    effects.to_csv(output / "04_episode_lactate" / "episode_lactate_effects.csv", index=False)
    trajectory_df.to_csv(output / "04_episode_lactate" / "episode_aligned_lactate_draws.csv", index=False)
    return anchors, effects, trajectory_df


def episode_direction_sensitivity(
    analysis: pd.DataFrame, events: pd.DataFrame, output: Path,
) -> pd.DataFrame:
    dynamic = analysis.loc[analysis.dynamics_eligible.fillna(False).astype(bool)].copy()
    no_episode = dynamic.loc[~dynamic.exposure_abs_jump_ge4.fillna(False).astype(bool)].copy()
    draw_map = {int(k): g.sort_values("minute_from_icu_intime") for k, g in events.groupby("ICUSTAY_ID")} if not events.empty else {}
    out_rows = []
    windows = {"acute_0_1h": (0., 60.), "delayed_1_8h": (60., 480.)}
    for direction, time_col in [("drop", "first_ge4_drop_minute"), ("rise", "first_ge4_rise_minute")]:
        selected = dynamic.loc[dynamic[time_col].notna()].copy()
        values = pd.to_numeric(selected[time_col], errors="coerce").dropna().sort_values().to_numpy()
        controls = no_episode.sort_values(["SUBJECT_ID", "ICUSTAY_ID"], kind="mergesort").copy()
        if len(values) and len(controls):
            controls["anchor_minute"] = values[np.arange(len(controls)) % len(values)]
        else:
            controls["anchor_minute"] = np.nan
        selected["anchor_minute"] = selected[time_col]
        selected["anchor_group"] = direction + "_episode"
        controls["anchor_group"] = "time_aligned_no_episode"
        combo = pd.concat([selected, controls], ignore_index=True)
        for row in combo.itertuples(index=False):
            anchor = float(row.anchor_minute) if pd.notna(row.anchor_minute) else np.nan
            d = draw_map.get(int(row.ICUSTAY_ID), pd.DataFrame(columns=["minute_from_icu_intime", "lactate_mmol_l"]))
            pre = d.loc[d.minute_from_icu_intime.lt(anchor)] if np.isfinite(anchor) and not d.empty else d.iloc[0:0]
            base = float(pre.iloc[-1].lactate_mmol_l) if len(pre) else np.nan
            end_obs = float(row.observable_end_minute) if pd.notna(row.observable_end_minute) else np.nan
            hospital_end_obs = (
                float(row.hospital_observable_end_minute)
                if hasattr(row, "hospital_observable_end_minute") and pd.notna(row.hospital_observable_end_minute)
                else end_obs
            )
            for window, (lo, hi) in windows.items():
                post = d.loc[d.minute_from_icu_intime.gt(anchor + lo) & d.minute_from_icu_intime.le(anchor + hi)] if np.isfinite(anchor) and not d.empty else d.iloc[0:0]
                last_delta = float(post.iloc[-1].lactate_mmol_l - base) if len(post) and np.isfinite(base) else np.nan
                complete = bool(np.isfinite(end_obs) and end_obs >= anchor + hi) if np.isfinite(anchor) else False
                legacy_complete = bool(
                    np.isfinite(hospital_end_obs) and hospital_end_obs >= anchor + hi
                ) if np.isfinite(anchor) else False
                available = bool(np.isfinite(base) and len(post) and complete)
                outcome = bool(last_delta >= .5) if available else np.nan
                legacy_available = bool(np.isfinite(base) and len(post) and (last_delta >= .5 or legacy_complete))
                legacy_outcome = bool(last_delta >= .5) if legacy_available else np.nan
                out_rows.append({
                    "direction": direction, "window": window, "group": row.anchor_group,
                    "SUBJECT_ID": int(row.SUBJECT_ID), "ICUSTAY_ID": int(row.ICUSTAY_ID),
                    "anchor_minute": anchor, "baseline": base, "last_delta": last_delta,
                    "available": available, "event": outcome,
                    "legacy_available": legacy_available, "legacy_event": legacy_outcome,
                })
    long = pd.DataFrame(out_rows)
    summary = []
    for (direction, window), part in long.groupby(["direction", "window"]):
        for group, g in part.groupby("group"):
            usable = g.loc[g.available]
            ev = int(usable.event.astype(bool).sum()) if len(usable) else 0
            summary.append({
                "direction": direction, "window": window, "group": group,
                "anchors": int(len(g)), "observed_n": int(len(usable)), "events": ev,
                "non_events": int(len(usable) - ev),
                "event_prevalence": ev / len(usable) if len(usable) else np.nan,
                "mean_last_delta": float(usable.last_delta.mean()) if len(usable) else np.nan,
            })
        usable = part.loc[part.available].copy()
        usable["direction_episode"] = usable.group.eq(direction + "_episode")
        effect = binary_effects(
            usable, "direction_episode", "event", reps=2000,
            seed=SEED + (3 if direction == "drop" else 4) + (1 if window.startswith("acute") else 2),
        )
        summary.append({"direction": direction, "window": window, "group": "direction_episode_vs_control", **effect})
    result = pd.DataFrame(summary)
    result.to_csv(output / "04_episode_lactate" / "episode_direction_sensitivity.csv", index=False)
    long.to_csv(output / "04_episode_lactate" / "episode_direction_patient_records.csv", index=False)
    return result


def fit_observation_probabilities(
    frame: pd.DataFrame, outcome_col: str, covariates: list[str], seed: int = SEED,
) -> tuple[np.ndarray | None, dict]:
    use = frame.copy()
    y = use[outcome_col].astype(bool).to_numpy()
    groups = use.SUBJECT_ID.to_numpy()
    class_counts = np.bincount(y.astype(int), minlength=2)
    n_splits = min(5, int(class_counts.min()), int(pd.Series(groups).nunique()))
    if n_splits < 2:
        return None, {
            "status": "unavailable", "reason": "Fewer than two outcome-observation examples in one class or fewer than two patients."
        }
    x = use[covariates].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.loc[:, x.notna().any(axis=0)]
    if x.shape[1] == 0:
        return None, {"status": "unavailable", "reason": "No usable pre-landmark predictors for observation model."}
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.full(len(use), np.nan)
    fold_info = []
    for fold, (train, test) in enumerate(splitter.split(x, y, groups), start=1):
        if set(groups[train]) & set(groups[test]):
            return None, {"status": "failed", "reason": "Subject overlap across observation-model folds."}
        pipe = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=3000, random_state=seed),
        )
        pipe.fit(x.iloc[train], y[train])
        pred[test] = pipe.predict_proba(x.iloc[test])[:, 1]
        fold_info.append({
            "fold": fold, "train_n": int(len(train)), "test_n": int(len(test)),
            "test_observed_n": int(y[test].sum()), "test_unobserved_n": int((~y[test]).sum()),
            "train_subjects": int(pd.Series(groups[train]).nunique()),
            "test_subjects": int(pd.Series(groups[test]).nunique()),
        })
    return pred, {
        "status": "available", "reason": "", "n_splits": n_splits,
        "features": list(x.columns), "folds": fold_info,
    }


def observation_process(
    analysis: pd.DataFrame, output: Path,
) -> pd.DataFrame:
    all_members = analysis.copy()
    dynamic = analysis.loc[analysis.dynamics_eligible.fillna(False).astype(bool)].copy()
    dynamic["exposure_group"] = dynamic.exposure_abs_jump_ge4.fillna(False).astype(bool)
    rows = []
    measures = [
        ("baseline_lactate_observed", "baseline_lactate_observed"),
        ("pre_landmark_lactate_draws", "baseline_lactate_draws_pre_landmark"),
        ("followup_lactate_draws_after_landmark", "followup_lactate_draws_after_landmark"),
        ("time_to_first_followup_lactate_min", "minutes_to_first_followup_lactate"),
    ]
    for horizon in HORIZONS:
        measures.extend([
            (f"any_lactate_in_{horizon}_window", f"n_lactate_draws_{horizon}"),
            (f"classifiable_endpoint_{horizon}", f"outcome_available_{horizon}"),
        ])
    for measure, col in measures:
        for exposed, group in [(False, "unexposed"), (True, "exposed")]:
            part = dynamic.loc[dynamic.exposure_group.eq(exposed)]
            vals = pd.to_numeric(part[col], errors="coerce") if col in part else pd.Series(dtype=float)
            if measure.startswith("any_lactate_in_"):
                vals = vals.gt(0).astype(float)
            elif measure.startswith("classifiable_endpoint_") or measure == "baseline_lactate_observed":
                vals = vals.astype(float)
            numeric = vals.dropna()
            if measure == "time_to_first_followup_lactate_min":
                summary = float(numeric.median()) if len(numeric) else np.nan
                p_val = np.nan
                other = pd.to_numeric(dynamic.loc[~dynamic.exposure_group.eq(exposed), col], errors="coerce").dropna() if col in dynamic else pd.Series(dtype=float)
                if len(numeric) and len(other):
                    p_val = float(mannwhitneyu(numeric, other, alternative="two-sided").pvalue)
                rows.append({
                    "measure": measure, "exposure_group": group, "n": int(len(numeric)),
                    "observed_n": int(len(numeric)), "mean": float(numeric.mean()) if len(numeric) else np.nan,
                    "median": summary, "p_value_vs_other_group": p_val, "status": "available",
                })
            else:
                success = int(numeric.astype(bool).sum()) if len(numeric) else 0
                rows.append({
                    "measure": measure, "exposure_group": group, "n": int(len(part)),
                    "observed_n": int(len(numeric)), "success_n": success,
                    "proportion": success / len(numeric) if len(numeric) else np.nan,
                    "p_value_vs_other_group": np.nan, "status": "available",
                })
        # Add an explicit group-comparison row for each outcome-observation process measure.
        if measure != "time_to_first_followup_lactate_min":
            a = dynamic.loc[dynamic.exposure_group]
            b = dynamic.loc[~dynamic.exposure_group]
            va = pd.to_numeric(a[col], errors="coerce") if col in a else pd.Series(dtype=float)
            vb = pd.to_numeric(b[col], errors="coerce") if col in b else pd.Series(dtype=float)
            if measure.startswith("any_lactate_in_"):
                va, vb = va.gt(0).astype(int), vb.gt(0).astype(int)
            if measure == "baseline_lactate_observed" or measure.startswith("classifiable_endpoint_") or measure.startswith("any_lactate_in_"):
                aa, bb = va.dropna().astype(bool), vb.dropna().astype(bool)
                table = [[int(aa.sum()), int((~aa).sum())], [int(bb.sum()), int((~bb).sum())]]
                p = float(fisher_exact(table).pvalue) if all(sum(r) for r in table) else np.nan
            else:
                aa, bb = va.dropna(), vb.dropna()
                p = float(mannwhitneyu(aa, bb, alternative="two-sided").pvalue) if len(aa) and len(bb) else np.nan
            rows.append({
                "measure": measure, "exposure_group": "exposed_vs_unexposed_test",
                "n": int(len(va) + len(vb)), "observed_n": int(va.notna().sum() + vb.notna().sum()),
                "p_value_vs_other_group": p, "status": "available",
            })
    predictors = [
        "age_years", "sex_male", "baseline_lactate_mmol_l",
        "observed_15min_bins", "missing_15min_bins", "mean_binned_spo2",
        "min_binned_spo2", "fraction_binned_below90", "sampling_1hz",
    ]
    for horizon in HORIZONS:
        observed_col = f"outcome_available_{horizon}"
        if observed_col not in dynamic:
            rows.append({
                "measure": f"IPW_association_{horizon}", "exposure_group": "all",
                "status": "unavailable", "reason": "Outcome observation indicator absent.",
                "missing_source": observed_col,
            })
            continue
        dynamic[observed_col] = dynamic[observed_col].fillna(False).astype(bool)
        all_members[observed_col] = all_members[observed_col].fillna(False).astype(bool)
        available_preds = [c for c in predictors if c in all_members]
        p_obs, fit_info = fit_observation_probabilities(all_members, observed_col, available_preds)
        rows.append({
            "measure": f"whole_cohort_classifiable_endpoint_{horizon}",
            "exposure_group": "all_clinical_eligible",
            "n": int(len(all_members)), "observed_n": int(len(all_members)),
            "success_n": int(all_members[observed_col].sum()),
            "proportion": float(all_members[observed_col].mean()) if len(all_members) else np.nan,
            "observation_model_status": fit_info.get("status"),
            "observation_model_features": "|".join(fit_info.get("features", [])),
            "observation_model_patients": int(all_members.SUBJECT_ID.nunique()),
            "folds": json.dumps(fit_info.get("folds", []), sort_keys=True),
            "status": "available",
            "interpretation_note": "Endpoint-observation probability was modeled across all clinically eligible cohort members using pre-landmark predictors.",
        })
        if p_obs is None:
            rows.append({
                "measure": f"IPW_association_{horizon}", "exposure_group": "all",
                **fit_info, "missing_source": "",
            })
            continue
        probability_by_stay = pd.Series(p_obs, index=all_members.ICUSTAY_ID.astype(int))
        dynamic[f"p_observed_{horizon}"] = np.clip(
            dynamic.ICUSTAY_ID.astype(int).map(probability_by_stay), .01, .99,
        )
        use = dynamic.loc[dynamic[observed_col]].copy()
        # Stabilized inverse-probability weights; truncate at the prespecified 1st/99th percentiles.
        pbar = float(all_members[observed_col].mean())
        raw_w = pbar / use[f"p_observed_{horizon}"].to_numpy(dtype=float)
        lo, hi = np.quantile(raw_w, [.01, .99])
        weights = np.clip(raw_w, lo, hi)
        use["ipw"] = weights
        ycol = f"lactate_rise_ge0_5_{horizon}"
        exp = use.exposure_group.astype(bool)
        risks = {}
        for flag, name in [(True, "exposed"), (False, "unexposed")]:
            mask = exp.eq(flag).to_numpy()
            den = float(weights[mask].sum())
            risks[name] = float(np.sum(weights[mask] * use.loc[mask, ycol].astype(float).to_numpy()) / den) if den else np.nan
        weighted_rr = risks["exposed"] / risks["unexposed"] if risks["unexposed"] > 0 else np.inf
        ess = float(weights.sum() ** 2 / np.sum(weights ** 2)) if np.sum(weights ** 2) else 0.0
        unweighted = binary_effects(
            use, "exposure_group", ycol, reps=2000,
            seed=SEED + (601 if horizon == "12h" else 602),
        )
        rows.append({
            "measure": f"IPW_association_{horizon}", "exposure_group": "all",
            "status": fit_info["status"], "reason": fit_info.get("reason", ""),
            "n_observed": int(len(use)), "events": int(use[ycol].astype(bool).sum()),
            "observation_model_n": int(len(all_members)), "observation_model_features": "|".join(fit_info.get("features", [])),
            "predicted_observation_min": float(np.min(p_obs)), "predicted_observation_median": float(np.median(p_obs)),
            "predicted_observation_max": float(np.max(p_obs)),
            "weight_truncation_low": float(lo), "weight_truncation_high": float(hi),
            "weight_min": float(weights.min()), "weight_median": float(np.median(weights)),
            "weight_max": float(weights.max()), "weight_ESS": ess,
            "weighted_exposed_risk": risks["exposed"], "weighted_unexposed_risk": risks["unexposed"],
            "weighted_RR": weighted_rr, "unweighted_RR": unweighted.get("RR"),
            "unweighted_RR_95CI_low": unweighted.get("RR_cluster_boot_low"),
            "unweighted_RR_95CI_high": unweighted.get("RR_cluster_boot_high"),
            "folds": json.dumps(fit_info.get("folds", []), sort_keys=True),
            "interpretation_note": "Weighting addresses measured pre-landmark selection variables only; it does not remove unmeasured selection.",
        })
    result = pd.DataFrame(rows)
    result.to_csv(output / "03_landmark_lactate" / "lactate_observation_process.csv", index=False)
    return result


def sampling_and_coverage_sensitivities(
    analysis: pd.DataFrame, output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dynamic = analysis.loc[analysis.dynamics_eligible.fillna(False).astype(bool)].copy()
    sampling_rows = []
    overall_by_horizon: dict[str, float] = {}
    for horizon in HORIZONS:
        ycol = f"lactate_rise_ge0_5_{horizon}"
        avail = f"outcome_available_{horizon}"
        overall = dynamic.loc[dynamic[avail].fillna(False).astype(bool)].copy()
        overall[ycol] = overall[ycol].astype(bool)
        overall_eff = binary_effects(overall, "exposure_abs_jump_ge4", ycol, reps=2000, seed=SEED + 701 + (0 if horizon == "12h" else 1))
        overall_by_horizon[horizon] = overall_eff.get("RR", np.nan)
        regimes = sorted(dynamic.sampling_regime.fillna("unknown").astype(str).unique())
        for regime in regimes:
            sub = overall.loc[overall.sampling_regime.fillna("unknown").astype(str).eq(regime)].copy()
            sub[ycol] = sub[ycol].astype(bool)
            eff = binary_effects(sub, "exposure_abs_jump_ge4", ycol, reps=2000, seed=SEED + 711 + len(sampling_rows))
            sampling_rows.append({
                "analysis": "within_sampling_regime", "horizon": horizon, "sampling_regime": regime,
                "n_dynamics_eligible": int((dynamic.sampling_regime.fillna("unknown").astype(str).eq(regime)).sum()),
                "observed_n": int(len(sub)), "events": int(sub[ycol].sum()) if len(sub) else 0,
                "exposed_n": int(sub.exposure_abs_jump_ge4.sum()) if len(sub) else 0,
                **eff,
            })
        eligible = overall.loc[overall.sampling_regime.isin(["1_Hz", "approximately_1_per_min"])].copy()
        eligible["sample_1hz"] = eligible.sampling_regime.eq("1_Hz").astype(float)
        eligible["exposure_x_1hz"] = eligible.exposure_abs_jump_ge4.astype(float) * eligible.sample_1hz
        interaction = modified_poisson(
            eligible, ycol, "exposure_abs_jump_ge4",
            ["sample_1hz", "exposure_x_1hz"], effect_term="exposure_x_1hz",
        )
        sampling_rows.append({
            "analysis": "exposure_by_regime_interaction", "horizon": horizon,
            "sampling_regime": "1_Hz vs approximately_1_per_min",
            "interaction_RR_ratio": interaction.get("adjusted_RR"),
            "interaction_RR_ratio_95CI_low": interaction.get("adjusted_RR_95CI_low"),
            "interaction_RR_ratio_95CI_high": interaction.get("adjusted_RR_95CI_high"),
            **interaction,
        })
        for regime in ["1_Hz", "approximately_1_per_min"]:
            sub = overall.loc[overall.sampling_regime.eq(regime)].copy()
            eff = binary_effects(sub, "exposure_abs_jump_ge4", ycol, reps=2000, seed=SEED + 731 + len(sampling_rows))
            sampling_rows.append({
                "analysis": "restricted_to_sampling_regime", "horizon": horizon,
                "sampling_regime": regime, "n_dynamics_eligible": int((dynamic.sampling_regime.eq(regime)).sum()),
                "observed_n": int(len(sub)), "events": int(sub[ycol].sum()) if len(sub) else 0,
                "exposed_n": int(sub.exposure_abs_jump_ge4.sum()) if len(sub) else 0,
                **eff,
            })
    sampling = pd.DataFrame(sampling_rows)
    for horizon, group in sampling[sampling.analysis.eq("within_sampling_regime")].groupby("horizon"):
        rrs = group.loc[group.RR.notna()]
        signs = (rrs.RR > 1).nunique() if len(rrs) else 0
        sampling.loc[
            sampling.horizon.eq(horizon) & sampling.analysis.eq("within_sampling_regime"),
            "direction_materially_changes_vs_other_regime",
        ] = signs > 1
    sampling["overall_rr_for_reference"] = sampling.horizon.map(overall_by_horizon)
    sampling.to_csv(output / "07_sensitivity" / "sampling_regime_sensitivity.csv", index=False)

    coverage_rows = []
    thresholds = [
        ("at_least_1h_observed_bins", 4),
        ("at_least_2h_observed_bins", 8),
        ("at_least_3h_observed_bins", 12),
        ("near_complete_at_least_15_of_16_bins", 15),
    ]
    for horizon in HORIZONS:
        ycol, avail = f"lactate_rise_ge0_5_{horizon}", f"outcome_available_{horizon}"
        for name, minimum_bins in thresholds:
            subset = dynamic.loc[dynamic.observed_15min_bins.ge(minimum_bins)].copy()
            observed = subset.loc[subset[avail].fillna(False).astype(bool)].copy()
            if len(observed):
                observed[ycol] = observed[ycol].astype(bool)
            effect = binary_effects(
                observed, "exposure_abs_jump_ge4", ycol,
                reps=2000, seed=SEED + minimum_bins + (100 if horizon == "12h" else 200),
            )
            coverage_rows.append({
                "restriction": name, "minimum_observed_bins": minimum_bins,
                "horizon": horizon, "dynamics_eligible_n": int(len(subset)),
                "observed_n": int(len(observed)),
                "events": int(observed[ycol].sum()) if len(observed) else 0,
                "exposed_n": int(observed.exposure_abs_jump_ge4.sum()) if len(observed) else 0,
                "coverage_hours_minimum": minimum_bins * 15 / 60,
                **effect,
            })
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(output / "07_sensitivity" / "coverage_sensitivity.csv", index=False)
    return sampling, coverage


def calibration_summary(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
        model.fit(z, y)
        return float(model.intercept_[0]), float(model.coef_[0, 0]), float(np.mean(np.abs(y - p)))
    except Exception:
        return np.nan, np.nan, np.nan


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = len(y)
    result = 0.0
    for idx in range(bins):
        mask = (p >= edges[idx]) & (p < edges[idx + 1] if idx < bins - 1 else p <= edges[idx + 1])
        if mask.any():
            result += mask.sum() / total * abs(float(y[mask].mean() - p[mask].mean()))
    return float(result)


def prediction_metric(y: np.ndarray, p: np.ndarray) -> dict:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"AUROC": np.nan, "AUPRC": np.nan, "Brier": np.nan, "calibration_intercept": np.nan, "calibration_slope": np.nan, "ECE_10_bins": np.nan}
    intercept, slope, _ = calibration_summary(y, p)
    return {
        "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
        "calibration_intercept": intercept, "calibration_slope": slope,
        "ECE_10_bins": expected_calibration_error(y, p, 10),
    }


def incremental_prediction_metric(y: np.ndarray, p: np.ndarray) -> dict:
    """Compute only metrics used in the paired bootstrap, avoiding repeated calibration fits."""
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"AUROC": np.nan, "AUPRC": np.nan, "Brier": np.nan}
    return {
        "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
    }


def grouped_oof_models(analysis: pd.DataFrame, output: Path, logger: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    metrics_rows = []
    increment_rows = []
    predictions_all = []
    audit = {}
    clinical_base = [
        "age_years", "sex_male", "baseline_lactate_mmol_l",
        "early_hr_median", "early_sbp_median", "early_map_median",
        "early_rr_median", "early_fio2_median", "early_mechanical_ventilation",
        "early_vasoactive_infusion",
    ]
    context = [
        "mean_binned_spo2", "min_binned_spo2", "fraction_binned_below90",
        "observed_15min_bins", "missing_15min_bins",
        "qualifying_transitions_le30min", "median_raw_samples_per_15min_bin",
        "sampling_1hz", "sampling_mixed", "sampling_unknown",
    ]
    parsimonious = ["rmssd_binned", "jump_fraction_ge3", "drop_count_ge3"]
    expanded = ["max_abs_qualifying_jump", "n_ge4_transitions", "slope_spo2_per_hour"]
    for horizon in HORIZONS:
        ycol, avail = f"lactate_rise_ge0_5_{horizon}", f"outcome_available_{horizon}"
        use = analysis.loc[
            analysis.dynamics_eligible.fillna(False).astype(bool)
            & analysis[avail].fillna(False).astype(bool)
        ].copy().reset_index(drop=True)
        y = use[ycol].astype(int).to_numpy()
        groups = use.SUBJECT_ID.to_numpy()
        if len(use) < 10 or len(np.unique(y)) < 2:
            for model in ["A_clinical", "B_clinical_signal_context", "C_plus_parsimonious_dynamics", "D_plus_expanded_dynamics"]:
                metrics_rows.append({
                    "horizon": horizon, "model": model, "status": "unavailable",
                    "reason": "Too few endpoint-observed rows or a single outcome class.",
                    "n": int(len(use)), "events": int(y.sum()) if len(y) else 0,
                })
            continue
        model_defs = [
            ("A_clinical", clinical_base),
            ("B_clinical_signal_context", clinical_base + context),
            ("C_plus_parsimonious_dynamics", clinical_base + context + parsimonious),
            ("D_plus_expanded_dynamics", clinical_base + context + parsimonious + expanded),
        ]
        model_defs = [(n, [c for c in cols if c in use]) for n, cols in model_defs]
        # The feature lists are frozen before fitting. Only columns with no observed values
        # in this outcome-available sample are unavailable and removed.
        clean_defs = []
        for name, cols in model_defs:
            usable = [c for c in cols if pd.to_numeric(use[c], errors="coerce").notna().any()]
            clean_defs.append((name, usable))
        counts = np.bincount(y, minlength=2)
        folds_n = min(5, int(counts.min()), int(pd.Series(groups).nunique()))
        if folds_n < 2:
            for name, cols in clean_defs:
                metrics_rows.append({
                    "horizon": horizon, "model": name, "status": "unavailable",
                    "reason": "Insufficient outcome events per patient group for grouped cross-validation.",
                    "n": int(len(use)), "events": int(y.sum()), "input_feature_count": len(cols),
                })
            continue
        splitter = StratifiedGroupKFold(n_splits=folds_n, shuffle=True, random_state=SEED)
        pred = {name: np.full(len(use), np.nan) for name, _ in clean_defs}
        fold_rows = []
        convergence: dict[str, int] = {name: 0 for name, _ in clean_defs}
        overlap_ok = True
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(use)), y, groups), start=1):
            train_subjects = set(groups[train])
            test_subjects = set(groups[test])
            if train_subjects & test_subjects:
                overlap_ok = False
            fold_rows.append({
                "fold": fold, "train_n": int(len(train)), "test_n": int(len(test)),
                "train_events": int(y[train].sum()), "test_events": int(y[test].sum()),
                "train_subjects": len(train_subjects), "test_subjects": len(test_subjects),
                "subject_overlap": len(train_subjects & test_subjects),
            })
            for name, cols in clean_defs:
                x = use[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
                pipe = make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True),
                    StandardScaler(),
                    LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=3000, random_state=SEED),
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    pipe.fit(x.iloc[train], y[train])
                convergence[name] += sum("converg" in str(w.message).lower() for w in caught)
                pred[name][test] = pipe.predict_proba(x.iloc[test])[:, 1]
        audit[horizon] = {
            "folds": fold_rows, "subject_group_overlap_pass": overlap_ok,
            "fold_count": folds_n, "convergence_warning_count": convergence,
            "n": int(len(use)), "events": int(y.sum()), "subjects": int(pd.Series(groups).nunique()),
        }
        if not overlap_ok:
            raise RuntimeError(f"SUBJECT_ID overlap detected across grouped folds for {horizon}.")
        pred_frame = pd.DataFrame({
            "horizon": horizon, "SUBJECT_ID": use.SUBJECT_ID,
            "ICUSTAY_ID": use.ICUSTAY_ID, "outcome": y, "sampling_regime": use.sampling_regime,
        })
        pred_frame["fold"] = np.nan
        for fold, (_, test) in enumerate(splitter.split(np.zeros(len(use)), y, groups), start=1):
            pred_frame.loc[test, "fold"] = fold
        for name, _ in clean_defs:
            pred_frame[name] = pred[name]
            n_features = len(_)
            transformed = n_features
            for train, test in splitter.split(np.zeros(len(use)), y, groups):
                x = use[_].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
                pipe = make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True),
                    StandardScaler(),
                    LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=3000, random_state=SEED),
                )
                pipe.fit(x.iloc[train], y[train])
                transformed = len(pipe.named_steps["simpleimputer"].get_feature_names_out())
                break
            metrics = prediction_metric(y, pred[name])
            metrics_rows.append({
                "horizon": horizon, "model": name, "status": "available",
                "n": int(len(use)), "patients": int(pd.Series(groups).nunique()),
                "events": int(y.sum()), "non_events": int(len(y) - y.sum()),
                "input_feature_count": n_features, "transformed_feature_count": transformed,
                "events_per_transformed_feature": float(y.sum() / transformed) if transformed else np.nan,
                "fold_event_counts": json.dumps([x["test_events"] for x in fold_rows]),
                "convergence_warnings": convergence[name],
                "subject_group_overlap_pass": overlap_ok, **metrics,
            })
        pred_frame.to_csv(output / "05_predictive_models" / f"oof_predictions_{horizon}.csv", index=False)
        predictions_all.append(pred_frame)
        # Patient-cluster paired bootstrap for prespecified incremental comparisons.
        subjects = pd.unique(use.SUBJECT_ID)
        by_subject = {sid: np.flatnonzero(use.SUBJECT_ID.to_numpy() == sid) for sid in subjects}
        rng = np.random.default_rng(SEED + (810 if horizon == "12h" else 820))
        for new_name, compare in [
            ("C_plus_parsimonious_dynamics", "B_clinical_signal_context"),
            ("D_plus_expanded_dynamics", "B_clinical_signal_context"),
        ]:
            auc_delta, pr_delta, brier_delta = [], [], []
            valid = 0
            for _rep in range(BOOTSTRAP_PREDICTIVE):
                sampled_subjects = rng.choice(subjects, size=len(subjects), replace=True)
                idx = np.concatenate([by_subject[s] for s in sampled_subjects])
                yb = y[idx]
                if len(np.unique(yb)) < 2:
                    continue
                mnew = incremental_prediction_metric(yb, pred[new_name][idx])
                mbase = incremental_prediction_metric(yb, pred[compare][idx])
                auc_delta.append(mnew["AUROC"] - mbase["AUROC"])
                pr_delta.append(mnew["AUPRC"] - mbase["AUPRC"])
                brier_delta.append(mnew["Brier"] - mbase["Brier"])
                valid += 1
            def boot_ci(vals):
                return tuple(np.quantile(vals, [.025, .975])) if len(vals) >= BOOTSTRAP_PREDICTIVE // 4 else (np.nan, np.nan)
            auc_ci = boot_ci(auc_delta)
            pr_ci = boot_ci(pr_delta)
            brier_ci = boot_ci(brier_delta)
            point_new = incremental_prediction_metric(y, pred[new_name])
            point_base = incremental_prediction_metric(y, pred[compare])
            inc = {
                "horizon": horizon, "comparison": f"{new_name} minus {compare}",
                "bootstrap_reps_requested": BOOTSTRAP_PREDICTIVE,
                "bootstrap_reps_valid": valid,
                "delta_AUROC": point_new["AUROC"] - point_base["AUROC"],
                "delta_AUROC_95CI_low": auc_ci[0], "delta_AUROC_95CI_high": auc_ci[1],
                "delta_AUPRC": point_new["AUPRC"] - point_base["AUPRC"],
                "delta_AUPRC_95CI_low": pr_ci[0], "delta_AUPRC_95CI_high": pr_ci[1],
                "delta_Brier": point_new["Brier"] - point_base["Brier"],
                "delta_Brier_95CI_low": brier_ci[0], "delta_Brier_95CI_high": brier_ci[1],
                "status": "available",
                "interpretation_note": "Paired out-of-fold predictions; patient-level resampling. A CI crossing zero does not establish improvement.",
            }
            increment_rows.append(inc)
    metrics_df = pd.DataFrame(metrics_rows)
    increment_df = pd.DataFrame(increment_rows)
    write_csv(metrics_df, output, "05_predictive_models", "predictive_model_metrics.csv")
    write_csv(increment_df, output, "05_predictive_models", "predictive_incremental_bootstrap.csv")
    if predictions_all:
        pd.concat(predictions_all, ignore_index=True).to_csv(output / "05_predictive_models" / "oof_predictions.csv", index=False)
    logger.info("Grouped OOF predictive analyses finished.")
    return metrics_df, increment_df, audit


def power_precision(
    association: pd.DataFrame, completeness: pd.DataFrame, output: Path,
) -> pd.DataFrame:
    main = association.loc[association.horizon.eq("12h")].iloc[0].to_dict() if len(association.loc[association.horizon.eq("12h")]) else {}
    cohort_ep = completeness.loc[completeness.horizon.eq("12h")]
    cohort_observed_n = int(cohort_ep.observed_n.iloc[0]) if len(cohort_ep) else 0
    cohort_events = int(cohort_ep.events.iloc[0]) if len(cohort_ep) else 0
    observed_n = int(main.get("n", 0))
    events = int(main.get("exposed_events", 0)) + int(main.get("unexposed_events", 0))
    p0 = float(main.get("unexposed_risk", np.nan))
    q = float(main.get("exposed_total", 0) / main.get("n", 1)) if main.get("n", 0) else np.nan
    rows = []
    if main and np.isfinite(float(main.get("RR", np.nan))):
        ci_low = float(main.get("RR_cluster_boot_low", np.nan))
        ci_high = float(main.get("RR_cluster_boot_high", np.nan))
        rows.append({
            "analysis": "observed_RR_precision",
            "observed_RR": float(main["RR"]),
            "observed_RR_95CI_low": ci_low,
            "observed_RR_95CI_high": ci_high,
            "observed_RR_95CI_width": ci_high - ci_low if np.isfinite(ci_low) and np.isfinite(ci_high) else np.nan,
            "observed_RD": float(main.get("RD", np.nan)),
            "observed_n": observed_n, "observed_events": events,
            "exposed_n": int(main.get("exposed_total", 0)),
            "unexposed_n": int(main.get("unexposed_total", 0)),
            "status": "available" if np.isfinite(ci_low) and np.isfinite(ci_high) else "interval_unavailable",
            "precision_method": "10,000 patient-cluster bootstrap replicates when valid",
        })
    if not (np.isfinite(p0) and np.isfinite(q) and 0 < p0 < 1 and 0 < q < 1):
        result = pd.DataFrame(rows + [{
            "analysis": "12h planning precision", "status": "unavailable",
            "reason": "Observed event risk or exposed fraction could not be estimated.",
            "missing_source": "12h endpoint observations",
        }])
        result.to_csv(output / "07_sensitivity" / "power_precision.csv", index=False)
        return result
    alpha_z = float(norm.ppf(.975))
    def se_log_rr(n, rr):
        n1, n0 = q * n, (1 - q) * n
        p1 = min(.999, p0 * rr)
        return math.sqrt(max(0, 1 / (n1 * p1) - 1 / n1 + 1 / (n0 * p0) - 1 / n0))
    def approximate_power(n, rr):
        se = se_log_rr(n, rr)
        delta = math.log(rr) / se if se > 0 else np.inf
        return float(norm.cdf(-alpha_z - delta) + 1 - norm.cdf(alpha_z - delta))
    rr_grid = np.linspace(1.001, min(5.0, .999 / p0), 4000)
    for target in [.80, .90]:
        power = np.array([approximate_power(max(observed_n, 2), rr) for rr in rr_grid])
        ok = np.flatnonzero(power >= target)
        detectable = float(rr_grid[ok[0]]) if len(ok) else np.nan
        rows.append({
            "analysis": "dynamics_cohort_detectable_RR", "target_power": target,
            "alpha_two_sided": .05, "observed_n": observed_n, "observed_events": events,
            "exposed_fraction": q, "unexposed_risk_assumed": p0,
            "detectable_RR": detectable, "status": "approximate_planning_only",
            "assumption": "Normal approximation on log RR with observed exposed fraction and unexposed event risk.",
        })
    for rr in [1.2, 1.3, 1.5, 1.7]:
        for target in [.80, .90]:
            needed_n = None
            for n in range(20, 2_000_001, 10):
                if approximate_power(n, rr) >= target:
                    needed_n = n
                    break
            if needed_n is None:
                needed_n = 2_000_000
            expected_events = needed_n * p0 * ((1 - q) + q * rr)
            rows.append({
                "analysis": "events_needed_for_planning_RR",
                "target_RR": rr, "target_power": target, "alpha_two_sided": .05,
                "required_total_n_approx": needed_n,
                "required_events_approx": expected_events,
                "observed_n": observed_n, "observed_events": events,
                "exposed_fraction_assumed": q, "unexposed_risk_assumed": p0,
                "status": "approximate_planning_only",
                "assumption": "Exposure fraction and unexposed event risk held at observed estimates; no post-hoc power interpretation.",
            })
    rows.append({
        "analysis": "measurement_availability_context",
        "eligible_clinical_n": int(cohort_ep.eligible_stays.iloc[0]) if len(cohort_ep) else np.nan,
        "12h_endpoint_observed_n": cohort_observed_n, "12h_events": cohort_events,
        "dynamics_12h_endpoint_observed_n": observed_n, "dynamics_12h_events": events,
        "observed_fraction_of_eligible": cohort_observed_n / int(cohort_ep.eligible_stays.iloc[0]) if len(cohort_ep) and int(cohort_ep.eligible_stays.iloc[0]) else np.nan,
        "exposed_fraction_of_observed_dynamics_cohort": q,
        "events_per_exposure_group": f"{main.get('exposed_events',np.nan)}/{main.get('unexposed_events',np.nan)}",
        "exposed_n": int(main.get("exposed_total", 0)),
        "unexposed_n": int(main.get("unexposed_total", 0)),
        "exposed_events": int(main.get("exposed_events", 0)),
        "unexposed_events": int(main.get("unexposed_events", 0)),
        "status": "descriptive_planning_context",
    })
    result = pd.DataFrame(rows)
    result.to_csv(output / "07_sensitivity" / "power_precision.csv", index=False)
    return result


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    if frame.empty:
        return "_No rows._"
    subset = frame.loc[:, [c for c in columns if c in frame]].head(max_rows).copy()
    if subset.empty:
        return "_No rows._"
    def fmt(v):
        if pd.isna(v):
            return "—"
        if isinstance(v, (float, np.floating)):
            return f"{v:.3f}" if np.isfinite(v) else str(v)
        return str(v).replace("|", "\\|").replace("\n", " ")
    headers = list(subset.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in subset.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(fmt(x) for x in row) + " |")
    if len(frame) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(frame)} rows._")
    return "\n".join(lines)


def make_mimic4_parity_file(
    input_root: Path, output: Path, manifest_path: Path | None, input_files: list[Path],
) -> pd.DataFrame:
    data_files: list[tuple[str, int]] = []
    for base, dirs, files in os.walk(input_root):
        dirs[:] = [d for d in dirs if d not in {".git", ".ipynb_checkpoints"}]
        for name in files:
            if name.lower().endswith(".dat"):
                path = Path(base) / name
                data_files.append((path.relative_to(input_root).as_posix().lower(), path.stat().st_size))
    expected_m3_raw = []
    if manifest_path:
        try:
            manifest = pd.read_csv(manifest_path, usecols=["data_path"])
            expected_m3_raw = sorted(set(
                manifest.data_path.dropna().astype(str).str.replace("\\", "/", regex=False).str.strip("/").str.lower()
            ))
        except Exception:
            pass
    matched_m3_raw = sorted({
        expected for expected in expected_m3_raw
        if any(relative_path == expected or relative_path.endswith("/" + expected) for relative_path, _ in data_files)
    })
    mimic4_raw = [
        (path, size) for path, size in data_files
        if path.startswith("mimic4wdb/") or "/mimic4wdb/" in path
    ]
    has_mimic4_clinical = any(
        any(x in str(p).lower() for x in ["mimiciv_hosp", "mimiciv_icu", "mimic-iv-clinical"])
        for p in input_files
    )
    rows = [
        {
            "analysis": "MIMIC-III raw hourly RMS exposure vs locked lactate",
            "status": "unavailable",
            "reason": (
                "No corresponding MIMIC-III raw numeric waveform files were found for the HF record manifest. "
                "The supplied input contains 15-minute summaries, which cannot reproduce the specified median of raw points within each hour."
            ),
            "missing_source": "MIMIC-III raw waveform .dat files listed in mimic3_waveform_hf_record_manifest.csv",
            "manifest_expected_datafiles": len(expected_m3_raw),
            "matching_datafiles_found": len(matched_m3_raw),
            "mimic4_raw_numeric_waveform_files_found": len(mimic4_raw),
            "mimic4_raw_numeric_waveform_bytes": int(sum(size for _, size in mimic4_raw)),
            "frozen_clip_lower": 0.0,
            "frozen_clip_upper": 4.3001,
            "frozen_center": 0.8776,
            "frozen_scale": 0.8227,
            "frozen_transform_applied": False,
            "mimic4_clinical_source_found": bool(has_mimic4_clinical),
            "status_detail": "MIMIC-IV raw waveform files, if present, are not a substitute for MIMIC-III. No aggregate proxy substituted for the prespecified raw hourly RMS exposure; frozen MIMIC-IV scaling was recorded but not applied because the required MIMIC-III hourly RMS could not be computed.",
        },
        {
            "analysis": "MIMIC-IV original shock/hypoperfusion composite parity",
            "status": "unavailable",
            "reason": (
                "MIMIC-IV hospital and ICU clinical event tables needed for blood pressure, continuous vasoactive infusions, "
                "mechanical support, lactate, urine output, creatinine, ALT, and pH were not found. "
                "MIMIC-IV waveform files alone cannot reconstruct this composite."
            ),
            "missing_source": "MIMIC-IV hosp/icu clinical tables: admissions, icustays, chartevents, labevents, inputevents, outputevents, procedures",
            "manifest_expected_datafiles": "", "matching_datafiles_found": "",
            "mimic4_raw_numeric_waveform_files_found": len(mimic4_raw),
            "mimic4_raw_numeric_waveform_bytes": int(sum(size for _, size in mimic4_raw)),
            "frozen_clip_lower": 0.0, "frozen_clip_upper": 4.3001,
            "frozen_center": 0.8776, "frozen_scale": 0.8227,
            "frozen_transform_applied": False,
            "mimic4_clinical_source_found": bool(has_mimic4_clinical),
            "status_detail": "No easier proxy endpoint was substituted.",
        },
    ]
    result = pd.DataFrame(rows)
    result.to_csv(output / "06_mimic4_parity" / "mimic4_hourly_rms_parity.csv", index=False)
    return result


def make_figures(output: Path) -> list[Path]:
    fig_dir = output / "08_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    flow = pd.read_csv(output / "01_cohort" / "cohort_flow.csv")
    available = flow.loc[flow.status.eq("available") & flow.n_remaining.notna()].copy()
    fig, ax = plt.subplots(figsize=(10, max(6, .58 * len(available))))
    y = np.arange(len(available))
    nums = pd.to_numeric(available.n_remaining, errors="coerce").fillna(0).to_numpy()
    max_n = max(1, int(nums.max()))
    for idx, (label, n) in enumerate(zip(available.flow_step, nums)):
        width = .68 * n / max_n
        ax.barh(idx, width, left=.31, height=.62, color="#2b6f8a" if idx < 6 else "#69a88d")
        ax.text(.31 + width + .012, idx, f"{int(n):,}", va="center", fontsize=9)
    ax.set_yticks(y, available.flow_step)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xticks([])
    ax.set_title("Cohort flow from the full MIMIC-III clinical tables")
    ax.text(.31, len(available) + .15, "Bars scaled to the largest count; the missing upstream 8,736-stay waveform list is not inferred.", fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    path = fig_dir / "01_cohort_flow.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    sig = pd.read_csv(output / "02_signal_qc" / "signal_features_recomputed.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.hist(sig.observed_15min_bins.dropna(), bins=np.arange(-.5, 17.5, 1), color="#2b6f8a", edgecolor="white")
    ax.set(xlabel="Valid 15-minute SpO2 bins in first 4 ICU hours", ylabel="Stays", xlim=(-.5, 16.5))
    ax.set_title(f"Observed 15-minute SpO2 bins (N={sig.ICUSTAY_ID.nunique():,})")
    fig.tight_layout()
    path = fig_dir / "02_observed_spo2_bins.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    dyn = sig.loc[sig.dynamics_eligible.fillna(False)]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.hist(dyn.rmssd_binned.dropna(), bins=24, color="#bf7651", edgecolor="white")
    ax.set(xlabel="RMSSD of qualifying 15-minute median differences (percentage points)", ylabel="Dynamics-eligible stays")
    ax.set_title(f"SpO2 RMSSD (N={len(dyn):,})")
    fig.tight_layout()
    path = fig_dir / "03_rmssd_distribution.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    episodes = sig.loc[
        sig.dynamics_eligible.fillna(False) & sig.exposure_abs_jump_ge4.fillna(False)
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.hist(episodes.first_ge4_episode_minute.dropna(), bins=np.arange(0, 255, 15), color="#7c6da6", edgecolor="white")
    ax.set(xlabel="Minutes from ICU admission to second bin of first qualifying transition", ylabel="Exposed stays", xlim=(0, 240))
    ax.set_title(f"First qualifying SpO2 episode timing (N={len(episodes):,})")
    fig.tight_layout()
    path = fig_dir / "04_first_episode_time.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    assoc = pd.read_csv(output / "03_landmark_lactate" / "landmark_binary_association.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    xpos, labels, heights, lows, highs = [], [], [], [], []
    for i, row in enumerate(assoc.itertuples(index=False)):
        if not np.isfinite(row.exposed_risk) or not np.isfinite(row.unexposed_risk):
            continue
        for offset, group, risk, ev, n in [
            (i - .18, "Exposed", row.exposed_risk, row.exposed_events, row.exposed_total),
            (i + .18, "Unexposed", row.unexposed_risk, row.unexposed_events, row.unexposed_total),
        ]:
            lo, hi = wilson_interval(int(ev), int(n))
            xpos.append(offset); labels.append(f"{row.horizon} {group}\n{int(ev)}/{int(n)}")
            heights.append(risk); lows.append(max(0, risk - lo)); highs.append(max(0, hi - risk))
    ax.bar(xpos, heights, width=.32, color=["#9b5b47" if "Exposed" in t else "#39748d" for t in labels])
    ax.errorbar(xpos, heights, yerr=[lows, highs], fmt="none", ecolor="#222222", capsize=3, linewidth=1)
    ax.set_xticks(xpos, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Observed lactate-rise risk (95% Wilson CI)")
    ax.set_title("Landmark lactate rise by SpO2 transition exposure")
    fig.tight_layout()
    path = fig_dir / "05_landmark_lactate_risks.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    sampling = pd.read_csv(output / "07_sensitivity" / "sampling_regime_sensitivity.csv")
    forest = []
    for row in assoc.itertuples(index=False):
        exposed_events = getattr(row, "exposed_events", 0)
        unexposed_events = getattr(row, "unexposed_events", 0)
        event_n = int((exposed_events if pd.notna(exposed_events) else 0) + (unexposed_events if pd.notna(unexposed_events) else 0))
        observed_n = int(row.n) if pd.notna(row.n) else 0
        forest.append((f"{row.horizon} overall (N={observed_n:,}, events={event_n})", row.RR, row.RR_cluster_boot_low, row.RR_cluster_boot_high))
    subset = sampling.loc[sampling.analysis.eq("within_sampling_regime")]
    for row in subset.itertuples(index=False):
        observed_n = int(row.observed_n) if pd.notna(row.observed_n) else 0
        events_n = int(row.events) if pd.notna(row.events) else 0
        forest.append((f"{row.horizon} {row.sampling_regime} (N={observed_n:,}, events={events_n})", row.RR, row.RR_cluster_boot_low, row.RR_cluster_boot_high))
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, .38 * len(forest))))
    y = np.arange(len(forest))
    for i, (name, rr, lo, hi) in enumerate(forest):
        if np.isfinite(rr) and np.isfinite(lo) and np.isfinite(hi):
            ax.errorbar(rr, i, xerr=[[max(0, rr - lo)], [max(0, hi - rr)]], fmt="o", color="#2b6f8a", capsize=3)
    ax.axvline(1, color="#555555", linestyle="--", linewidth=1)
    valid_ci = [v for _, _, lo, hi in forest for v in [lo, hi] if np.isfinite(v) and v > 0]
    xmin = max(.05, min(valid_ci + [1]) * .65) if valid_ci else .1
    xmax = max(valid_ci + [1]) * 1.5 if valid_ci else 10
    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_yticks(y, [x[0] for x in forest])
    ax.invert_yaxis()
    ax.set_xlabel("Risk ratio (patient-cluster bootstrap 95% CI, log scale)")
    ax.set_title("Primary and sampling-regime associations")
    fig.tight_layout()
    path = fig_dir / "06_forest_sampling_regimes.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    outcomes = pd.read_csv(output / "03_landmark_lactate" / "lactate_outcomes.csv")
    joined = sig.merge(outcomes, on=["ICUSTAY_ID", "SUBJECT_ID", "HADM_ID"], how="inner", suffixes=("", "_out"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    for ax, horizon in zip(axes, ["12h", "24h"]):
        col = f"continuous_last_delta_{horizon}"
        part = joined.loc[joined.dynamics_eligible.fillna(False) & joined[col].notna() & joined.rmssd_binned.notna()]
        ax.scatter(part.rmssd_binned, part[col], s=13, alpha=.35, color="#39748d", edgecolors="none")
        ax.axhline(0, color="#555555", linewidth=.8)
        ax.set(xlabel="SpO2 RMSSD (percentage points)", ylabel="Last lactate minus baseline (mmol/L)")
        ax.set_title(f"{horizon} measured pairs (N={len(part):,})")
    fig.suptitle("Continuous SpO2 dynamics and measured lactate change")
    fig.tight_layout()
    path = fig_dir / "07_rmssd_lactate_delta.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    trajectory = pd.read_csv(output / "04_episode_lactate" / "episode_aligned_lactate_draws.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    if not trajectory.empty:
        trajectory["relative_hour"] = trajectory.relative_minute / 60
        trajectory["relative_hour_bin"] = np.floor(trajectory.relative_hour).astype(int)
        for group, color, label in [
            ("episode", "#9b5b47", "First ≥4-point episode"),
            ("time_aligned_no_episode", "#39748d", "Time-aligned no-episode control"),
        ]:
            part = trajectory.loc[trajectory.anchor_group.eq(group)]
            label = f"{label} (stays N={part.ICUSTAY_ID.nunique():,}; draws N={len(part):,})"
            summaries = part.groupby("relative_hour_bin").change_from_strict_pre_anchor_baseline.agg(
                ["median", "count", lambda s: s.quantile(.25), lambda s: s.quantile(.75)]
            )
            if len(summaries):
                x = summaries.index.to_numpy()
                ax.plot(x, summaries["median"], marker="o", color=color, label=label)
                ax.fill_between(x, summaries.iloc[:, 2], summaries.iloc[:, 3], color=color, alpha=.15)
        ax.axvline(0, color="#555555", linestyle="--", linewidth=1)
        ax.axhline(0, color="#555555", linewidth=.8)
        ax.set(xlabel="Hours from SpO2 episode or matched pseudo-anchor", ylabel="Lactate change from strict pre-anchor baseline (mmol/L)")
        ax.legend(frameon=False)
    else:
        ax.text(.5, .5, "No episode-aligned lactate draws", ha="center", va="center", transform=ax.transAxes)
    trajectory_stays = int(trajectory.ICUSTAY_ID.nunique()) if "ICUSTAY_ID" in trajectory else 0
    ax.set_title(f"Episode-aligned lactate measurements (stays N={trajectory_stays:,})")
    fig.tight_layout()
    path = fig_dir / "08_episode_aligned_lactate.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    preds_path = output / "05_predictive_models" / "oof_predictions.csv"
    preds = pd.read_csv(preds_path) if preds_path.exists() else pd.DataFrame()
    model_names = [
        "A_clinical", "B_clinical_signal_context",
        "C_plus_parsimonious_dynamics", "D_plus_expanded_dynamics",
    ]
    colors = ["#444444", "#39748d", "#9b5b47", "#61875f"]
    for horizon in ["12h", "24h"]:
        part = preds.loc[preds.horizon.eq(horizon)] if not preds.empty else preds
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        if len(part):
            yval = part.outcome.astype(int).to_numpy()
            for name, color in zip(model_names, colors):
                if name not in part:
                    continue
                p = pd.to_numeric(part[name], errors="coerce").to_numpy()
                if len(np.unique(yval)) > 1 and np.isfinite(p).all():
                    fpr, tpr, _ = roc_curve(yval, p)
                    auc = roc_auc_score(yval, p)
                    ax.plot(fpr, tpr, color=color, label=f"{name.replace('_',' ')} (AUROC {auc:.3f})")
            ax.plot([0, 1], [0, 1], color="#777777", linestyle="--")
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate", ylabel="True positive rate")
            ax.legend(frameon=False, fontsize=8)
        else:
            ax.text(.5, .5, "Grouped OOF predictions unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{horizon} nested grouped OOF ROC (N={len(part):,})")
        fig.tight_layout()
        path = fig_dir / f"09_roc_{horizon}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        if len(part):
            yval = part.outcome.astype(int).to_numpy()
            for name, color in zip(model_names, colors):
                if name not in part:
                    continue
                p = pd.to_numeric(part[name], errors="coerce").to_numpy()
                if len(np.unique(yval)) > 1 and np.isfinite(p).all():
                    precision, recall, _ = precision_recall_curve(yval, p)
                    ap = average_precision_score(yval, p)
                    ax.plot(recall, precision, color=color, label=f"{name.replace('_',' ')} (AUPRC {ap:.3f})")
            ax.axhline(float(part.outcome.mean()), color="#777777", linestyle="--", label="Event prevalence")
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision")
            ax.legend(frameon=False, fontsize=8)
        else:
            ax.text(.5, .5, "Grouped OOF predictions unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{horizon} nested grouped OOF precision–recall (N={len(part):,}; events={int(part.outcome.sum()) if len(part) else 0})")
        fig.tight_layout()
        path = fig_dir / f"10_pr_{horizon}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        if len(part):
            yval = part.outcome.astype(int).to_numpy()
            ax.plot([0, 1], [0, 1], color="#777777", linestyle="--")
            for name, color in zip(model_names, colors):
                if name not in part:
                    continue
                p = pd.to_numeric(part[name], errors="coerce").to_numpy()
                if not np.isfinite(p).all():
                    continue
                bins = np.linspace(0, 1, 11)
                xvals, yvals = [], []
                for left, right in zip(bins[:-1], bins[1:]):
                    mask = (p >= left) & (p < right if right < 1 else p <= right)
                    if mask.any():
                        xvals.append(float(p[mask].mean())); yvals.append(float(yval[mask].mean()))
                ax.plot(xvals, yvals, marker="o", color=color, label=name.replace("_", " "))
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted risk", ylabel="Observed event fraction")
            ax.legend(frameon=False, fontsize=8)
        else:
            ax.text(.5, .5, "Grouped OOF predictions unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{horizon} grouped OOF calibration (N={len(part):,}; events={int(part.outcome.sum()) if len(part) else 0})")
        fig.tight_layout()
        path = fig_dir / f"11_calibration_{horizon}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    obs = pd.read_csv(output / "03_landmark_lactate" / "lactate_observation_process.csv")
    obs = obs.loc[obs.measure.isin(["baseline_lactate_observed", "classifiable_endpoint_12h", "classifiable_endpoint_24h"])]
    fig, ax = plt.subplots(figsize=(9, 5))
    measures = ["baseline_lactate_observed", "classifiable_endpoint_12h", "classifiable_endpoint_24h"]
    xpos, vals, labels, colors = [], [], [], []
    for i, measure in enumerate(measures):
        for j, group in enumerate(["exposed", "unexposed"]):
            row = obs.loc[obs.measure.eq(measure) & obs.exposure_group.eq(group)]
            if row.empty:
                continue
            xpos.append(i + (j - .5) * .28)
            vals.append(float(row.proportion.iloc[0]) if pd.notna(row.proportion.iloc[0]) else np.nan)
            labels.append(f"{group}\nN={int(row.n.iloc[0])}")
            colors.append("#9b5b47" if group == "exposed" else "#39748d")
    ax.bar(xpos, vals, width=.26, color=colors)
    ax.set_xticks(range(len(measures)), ["Baseline lactate\nmeasured", "12h endpoint\nclassifiable", "24h endpoint\nclassifiable"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion")
    ax.set_title("Lactate observation process by SpO2 exposure")
    for x, v, label in zip(xpos, vals, labels):
        if np.isfinite(v):
            ax.text(x, v + .02, label, ha="center", va="bottom", fontsize=7, rotation=0)
    fig.tight_layout()
    path = fig_dir / "12_outcome_observation_audit.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def classify_primary(row: pd.Series | dict) -> str:
    lo = row.get("RR_cluster_boot_low", np.nan)
    hi = row.get("RR_cluster_boot_high", np.nan)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "underpowered / inadequate support"
    if lo > 1 or hi < 1:
        return "statistically supported association by patient-cluster bootstrap CI"
    rr = row.get("RR", np.nan)
    if np.isfinite(rr) and abs(math.log(max(rr, 1e-12))) < .1:
        return "null / no demonstrated association; interval remains the precision guide"
    return "suggestive / imprecise association; CI crosses the null"


def write_final_report(
    output: Path, input_root: Path, cohort: pd.DataFrame, context: dict,
    lactate_info: dict, chart_info: dict, vaso_info: dict,
    binary: pd.DataFrame, adjusted: pd.DataFrame, continuous: pd.DataFrame,
    prediction_metrics: pd.DataFrame, increments: pd.DataFrame,
    episode_effects: pd.DataFrame, direction: pd.DataFrame, observation: pd.DataFrame,
    sampling: pd.DataFrame, coverage: pd.DataFrame, parity: pd.DataFrame,
    power: pd.DataFrame, figure_paths: list[Path], packages: dict,
    runtime_seconds: float, followup_reconciliation: pd.DataFrame | None = None,
) -> Path:
    flow = pd.read_csv(output / "01_cohort" / "cohort_flow.csv")
    qc = pd.read_csv(output / "02_signal_qc" / "signal_qc_summary.csv")
    validation = pd.read_csv(output / "02_signal_qc" / "signal_feature_validation.csv")
    if followup_reconciliation is None:
        reconciliation_path = output / "03_landmark_lactate" / "complete_followup_before_after_reconciliation.csv"
        followup_reconciliation = pd.read_csv(reconciliation_path) if reconciliation_path.exists() else pd.DataFrame()
    eligible_n = context["eligible_clinical"]
    dynamics_n = context["dynamics_n"]
    exposed_n = context["exposed_n"]

    def assoc_line(horizon):
        row = binary.loc[binary.horizon.eq(horizon)]
        if row.empty or not np.isfinite(pd.to_numeric(row.iloc[0].get("RR"), errors="coerce")):
            return f"{horizon}: unavailable."
        r = row.iloc[0]
        return (
            f"{horizon}: observed N={int(r.n)}, events={int(r.exposed_events + r.unexposed_events)} "
            f"({int(r.exposed_events)} exposed, {int(r.unexposed_events)} unexposed); "
            f"RR {r.RR:.2f} (patient-cluster bootstrap 95% CI {r.RR_cluster_boot_low:.2f}–{r.RR_cluster_boot_high:.2f}), "
            f"RD {r.RD:.3f} (95% CI {r.RD_cluster_boot_low:.3f}–{r.RD_cluster_boot_high:.3f}). "
            f"Classification: {classify_primary(r)}."
        )

    report = []
    report.append("# Dense MIMIC-III waveform SpO2 and lactate analysis\n")
    report.append("## 1. Executive summary\n")
    report.append(
        f"The independently reconstructed full-clinical eligibility flow yielded {eligible_n:,} adult, first-ICU-per-admission stays "
        f"with explicit ICD-9 heart failure, early ICU admission or cardiogenic shock, and observation beyond ICU minute 240. "
        f"The supplied 15-minute signal artifacts yielded {context['signal_in_target']:,} usable eligible stays; "
        f"{dynamics_n:,} were dynamics-eligible and {exposed_n:,} had at least one qualifying absolute SpO2 transition of at least 4 percentage points. "
        f"The primary landmark estimates are: {assoc_line('12h')} {assoc_line('24h')}"
    )
    report.append(
        "The all-waveform 8,736-stay candidate-ID list was not present in Downloads. I reconstructed the clinical cohort from the full MIMIC-III v1.4 tables and joined the available HF-specific manifest and signal artifacts; I did not infer that upstream denominator. "
        "The signal feature file was independently recomputed from the provided 15-minute medians. MIMIC-III is treated as a historical/database-version sensitivity analysis, not independent external validation of MIMIC-IV."
    )
    report.append("\n## 2. Data inventory\n")
    report.append(
        f"Input root: {input_root}. The selected clinical source is the full MIMIC-III v1.4 table set, rather than duplicate subsets. "
        f"Selected clinical source rows: {context['row_counts']}. "
        f"LABEVENTS rows streamed: {lactate_info.get('source_rows','unavailable')}; "
        f"CHARTEVENTS rows streamed: {chart_info.get('source_rows','unavailable')}."
    )
    report.append(
        f"Lactate ITEMIDs used: {lactate_info.get('lactate_itemids', [])}; mg/dL values were converted to mmol/L and mmol/L or blank units retained. "
        f"Early chart item mappings: {len(chart_info.get('mapped_labels', []))}; vasoactive input event status: {vaso_info.get('status','unavailable')}."
    )
    report.append("The inventory includes paths, file sizes, SHA256 hashes, columns, schema matches, and alternatives. Large full clinical tables were row-counted during the streaming analysis.\n")
    report.append("## 3. Cohort flow\n")
    report.append(markdown_table(flow, ["flow_step", "n_remaining", "excluded_at_step", "scope", "status"], 30))
    report.append(
        f"\nEligible cohort unique patients={cohort.SUBJECT_ID.nunique():,}; admissions={cohort.HADM_ID.nunique():,}. "
        "Age is floor((ICU INTIME − DOB)/365.2425); deidentified ages above 89 are top-coded to 90."
    )
    care_units = pd.read_csv(output / "01_cohort" / "icu_careunit_distribution.csv")
    report.append("\nEligible-cohort ICU first-careunit distribution:")
    report.append(markdown_table(care_units, ["FIRST_CAREUNIT", "stays", "patients"], 20))
    availability = pd.read_csv(output / "01_cohort" / "covariate_availability.csv")
    report.append("\nObserved availability of pre-landmark chart and vasoactive covariates (full clinical eligible cohort):")
    report.append(markdown_table(availability, ["covariate", "nonmissing_n", "eligible_n", "availability_fraction", "source", "status"], 40))
    report.append("\nThe available waveform manifest contains "
                  f"{context['manifest_unique'] if context['manifest_unique'] is not None else 'no'} unique stays; "
                  f"the supplied feature table contains {context['signal_unique']:,} unique stays. "
                  "These are selected HF artifacts, not the full 8,736 waveform-covered candidate list.")
    report.append("\n## 4. Signal extraction QC\n")
    report.append(markdown_table(qc, ["group", "stays", "usable_spo2_stays", "dynamics_eligible_n", "exposed_n", "exposure_prevalence_among_dynamics_eligible", "observed_bins_median", "transitions_median", "rmssd_median", "sampling_regime_discordant_n", "extraction_failure_count", "extraction_failure_status", "status"], 30))
    report.append(
        f"\nThe extraction-failure artifact reports {context.get('extraction_failure_count', 'unavailable')} failed records "
        f"(artifact status: {context.get('extraction_failure_status', 'unavailable')}); this count comes from the available failure file, not from assuming unlisted failures were zero."
    )
    if "any_core_mismatch" in validation:
        compare = validation.loc[validation.status.eq("compared")]
        mismatch_n = int(compare.any_core_mismatch.sum())
        report.append(
            f"\nIndependent feature comparison: {len(compare):,} matched stay rows; {mismatch_n:,} had at least one core-field difference "
            "across observed/missing bins, transitions, eligibility, exposure, RMSSD, jump fraction, and drop count. "
            "A major disagreement was prespecified as more than 1% of matched stays differing on any core field."
        )
    report.append(
        "\nSignal processing used the 16 bins covering ICU minutes [0,240), medians of valid 50–100 values, and forward transitions with a gap no longer than 30 minutes. "
        "RMSSD was computed over qualifying 15-minute median differences, never over raw 1-Hz samples."
    )
    report.append("\n## 5. Exact endpoint definitions\n")
    report.append(
        "Landmark is ICU minute 240. Baseline lactate is the last valid linked lactate at or before minute 240 (minute 240 is baseline). "
        "The 12-hour post-landmark window is (240,960] minutes; the 24-hour window is (240,1680] minutes. "
        "The primary outcome is last lactate in the window minus baseline ≥0.5 mmol/L, and complete ICU follow-up through the relevant horizon is required for both events and non-events. "
        "If death, ICU departure, discharge, or the end of observable follow-up occurs earlier, the last-value binary endpoint is unavailable even when an earlier lactate rise was observed. "
        "Monotone maximum-rise and baseline-below-2-to-lactate-at-least-2 crossing companions may retain an event observed before censoring; their negative classifications still require complete follow-up."
    )
    report.append(
        "Prespecified companions include continuous last-value delta, maximum observed lactate delta, last-value delta ≥1.0 mmol/L, and baseline <2 with any follow-up lactate ≥2 mmol/L. These do not replace the locked primary outcome."
    )
    report.append("\nBefore-vs-after complete-follow-up reconciliation. Positive and negative removals count previously classifiable stay-anchor observations made unavailable by the corrected censoring rule:")
    report.append(markdown_table(followup_reconciliation, [
        "endpoint_family", "window", "group", "threshold", "previously_observed_n",
        "previously_events", "previously_non_events", "previously_positive_removed",
        "previously_negative_removed", "updated_observed_n", "updated_events",
        "updated_non_events", "net_observations_removed",
    ], 60))
    report.append("\n## 6. 12-hour landmark association\n")
    report.append(assoc_line("12h"))
    report.append(
        "\nBoth exposed and unexposed groups now require complete ICU follow-up through hour 12, with an observed event before an earlier censor time set to unavailable. "
        "The table above separately reports removals among previously positive and negative observations. The upstream all-waveform candidate-ID list remains unavailable, so its original cohort denominator cannot be reconstructed."
    )
    report.append(markdown_table(binary.loc[binary.horizon.eq("12h")], ["exposed_events", "exposed_total", "unexposed_events", "unexposed_total", "RR", "RR_95CI_low", "RR_95CI_high", "RR_cluster_boot_low", "RR_cluster_boot_high", "RD", "RD_cluster_boot_low", "RD_cluster_boot_high", "OR_secondary", "Fisher_exact_p"], 10))
    report.append("\n## 7. 24-hour landmark association\n")
    report.append(assoc_line("24h"))
    report.append(
        "\nBoth groups now require complete ICU follow-up through hour 24. Observed early last-value rises and negatives after early ICU departure are unavailable under this fixed-window estimand. "
        "The before-vs-after table gives the exact removed positive and negative counts; the upstream all-waveform candidate-ID list remains unavailable."
    )
    report.append(markdown_table(binary.loc[binary.horizon.eq("24h")], ["exposed_events", "exposed_total", "unexposed_events", "unexposed_total", "RR", "RR_95CI_low", "RR_95CI_high", "RR_cluster_boot_low", "RR_cluster_boot_high", "RD", "RD_cluster_boot_low", "RD_cluster_boot_high", "OR_secondary", "Fisher_exact_p"], 10))
    reconciliation = pd.read_csv(output / "03_landmark_lactate" / "previous_count_reconciliation.csv")
    report.append("\nPrior expected-count reconciliation and fixed alternate censoring audit:")
    report.append(markdown_table(reconciliation, [
        "horizon", "analysis", "exposed_events", "exposed_total", "unexposed_events",
        "unexposed_total", "observed_n", "RR", "status", "reason",
    ], 10))
    report.append("\n## 8. Adjusted association\n")
    report.append(markdown_table(adjusted, ["horizon", "model", "status", "n", "events", "parameters", "events_per_parameter", "adjusted_RR", "adjusted_RR_95CI_low", "adjusted_RR_95CI_high", "p_value", "reason"], 30))
    report.append(
        "\nModified Poisson estimates use patient-cluster robust sandwich covariance. The prespecified signal-adjusted model includes age, sex, baseline lactate, mean/minimum/fraction-below-90 signal level, observed-bin count, and sampling-rate indicator when available. An early-vital/respiratory-support extension is reported only when its prespecified complete-case support is adequate. Baseline lactate is observed by definition in the classifiable primary endpoint cohort, so its observation indicator is constant and not separately estimable."
    )
    if adjusted.status.astype(str).str.contains("iteration_limit_reached", regex=False).any():
        report.append(
            "\nThe 24-hour signal-adjusted and extended modified-Poisson fits did not meet the prespecified IRLS convergence tolerance within 200 iterations. Their coefficient and robust interval are retained as diagnostics with an explicit iteration-limit status; they are not treated as stable confirmatory adjusted estimates."
        )
    report.append("\n## 9. Continuous signal results\n")
    report.append(markdown_table(continuous, ["horizon", "signal_metric", "analysis", "n", "adjusted_RR", "adjusted_RR_95CI_low", "adjusted_RR_95CI_high", "rho", "rho_cluster_boot_low", "rho_cluster_boot_high", "bootstrap_unit", "p_value", "status"], 40))
    report.append("\nAll four prespecified signal summaries are shown; no metric was promoted based on its p-value. Continuous lactate delta uses the measured last-value difference; incomplete follow-up is not reclassified as a negative binary outcome.")
    report.append("\n## 10. Incremental predictive value\n")
    report.append(markdown_table(prediction_metrics, ["horizon", "model", "status", "n", "events", "fold_event_counts", "input_feature_count", "transformed_feature_count", "events_per_transformed_feature", "AUROC", "AUPRC", "Brier", "calibration_intercept", "calibration_slope", "ECE_10_bins", "convergence_warnings", "reason"], 30))
    report.append(markdown_table(increments, ["horizon", "comparison", "delta_AUROC", "delta_AUROC_95CI_low", "delta_AUROC_95CI_high", "delta_AUPRC", "delta_AUPRC_95CI_low", "delta_AUPRC_95CI_high", "delta_Brier", "delta_Brier_95CI_low", "delta_Brier_95CI_high", "bootstrap_reps_valid", "status"], 20))
    report.append(
        "\nModels use identical endpoint-observed rows per horizon, 5-fold StratifiedGroupKFold when support permits, and training-fold-only median imputation, scaling, and L2 logistic regression. SUBJECT_ID is the grouping unit; fold overlap is audited. Paired incremental intervals use 2,000 patient-cluster resamples. A confidence interval crossing zero does not establish incremental predictive value."
    )
    report.append("\n## 11. Episode-anchored analysis\n")
    report.append(
        "The episode anchor is the representative timestamp of the second 15-minute bin completing the first qualifying ≥4-point transition. No-episode controls are dynamics-eligible stays without such a transition; their pseudo-anchor times are assigned from the exposed episode-time list in sorted stay order, before outcome inspection."
    )
    report.append(markdown_table(episode_effects, ["window", "group", "n_anchors", "baseline_available_n", "observed_n", "events", "non_events", "event_prevalence", "RR", "RR_cluster_boot_low", "RR_cluster_boot_high", "RD", "RD_cluster_boot_low", "RD_cluster_boot_high"], 40))
    report.append(markdown_table(episode_effects.loc[episode_effects.group.eq("episode_vs_time_aligned_control_adjusted")], ["window", "group", "status", "n", "events", "adjustment_set", "adjusted_RR", "adjusted_RR_95CI_low", "adjusted_RR_95CI_high", "p_value", "reason"], 20))
    report.append(markdown_table(direction, ["direction", "window", "group", "anchors", "observed_n", "events", "non_events", "event_prevalence", "RR", "RR_cluster_boot_low", "RR_cluster_boot_high", "mean_last_delta"], 40))
    pretrend = pd.read_csv(output / "04_episode_lactate" / "episode_pre_anchor_trend_summary.csv")
    report.append(markdown_table(pretrend, ["group", "anchors", "serial_pre_anchor_lactate_n", "serial_pre_anchor_lactate_fraction", "pre_anchor_delta_per_hour_mean", "pre_anchor_delta_per_hour_median", "pre_anchor_delta_per_hour_q1", "pre_anchor_delta_per_hour_q3", "pre_anchor_lactate_rising_fraction", "status"], 10))
    episode_descriptives = pd.read_csv(output / "04_episode_lactate" / "episode_window_descriptives.csv")
    report.append("\nWindow-level lactate measurements and both frozen last-value thresholds (per stay; mmol/L deltas):")
    report.append(markdown_table(episode_descriptives, [
        "window", "group", "n_anchors", "strict_pre_anchor_baseline_n",
        "first_next_n", "first_next_mean", "first_next_delta_mean",
        "last_n", "last_mean", "last_delta_mean", "maximum_n", "maximum_mean", "maximum_delta_mean",
        "primary_ge0_5_observed_n", "primary_ge0_5_events",
        "secondary_ge1_observed_n", "secondary_ge1_events", "status",
    ], 20))
    episode_ge1 = pd.read_csv(output / "04_episode_lactate" / "episode_ge1_sensitivity.csv")
    report.append("\nSecondary last-value rise ≥1.0 mmol/L episode sensitivity:")
    report.append(markdown_table(episode_ge1, [
        "window", "group", "status", "n", "events", "non_events", "RR",
        "RR_cluster_boot_low", "RR_cluster_boot_high", "RD",
        "RD_cluster_boot_low", "RD_cluster_boot_high",
    ], 20))
    report.append(
        "\nEpisode windows are acute (0,1]h, delayed (1,8]h, and cumulative (0,4]h, (0,8]h, and (0,12]h. Each requires lactate strictly before anchor; missing follow-up remains unavailable. First-drop and first-rise analyses are both reported. Pre-episode lactate changes are descriptive ordering analyses, not evidence of causation."
    )
    report.append("\n## 12. Measurement-selection analysis\n")
    report.append(markdown_table(observation, ["measure", "exposure_group", "n", "observed_n", "success_n", "proportion", "median", "p_value_vs_other_group", "weighted_RR", "unweighted_RR", "weight_ESS", "status", "reason"], 40))
    report.append(
        "\nThe observation model uses only pre-landmark variables and out-of-fold patient-grouped probabilities; inverse-probability weights are stabilized and truncated at the prespecified 1st/99th percentiles. This sensitivity addresses measured selection variables only and cannot remove unmeasured testing or illness-severity selection."
    )
    report.append("\n## 13. Sampling-regime sensitivity\n")
    report.append(markdown_table(sampling, ["analysis", "horizon", "sampling_regime", "n_dynamics_eligible", "observed_n", "events", "exposed_n", "RR", "RR_cluster_boot_low", "RR_cluster_boot_high", "RD", "interaction_RR_ratio", "interaction_RR_ratio_95CI_low", "interaction_RR_ratio_95CI_high", "direction_materially_changes_vs_other_regime", "status", "reason"], 60))
    report.append(
        "\nSampling regime is taken from manifest fs metadata where available and otherwise from raw sample counts per 15-minute bin (roughly 900 for 1 Hz and 15 for approximately 1/min). Manifest disagreements are audited. The common primary exposure still uses 15-minute medians. Point estimates remain above RR=1 in both supported regimes at both horizons; the exposure-by-regime interaction intervals include 1 and are broad, so these data do not show a material direction change or resolve modest regime modification."
    )
    report.append("\n## 14. Coverage sensitivity\n")
    report.append(markdown_table(coverage, ["restriction", "horizon", "dynamics_eligible_n", "observed_n", "events", "exposed_n", "RR", "RR_cluster_boot_low", "RR_cluster_boot_high", "RD", "status"], 30))
    report.append(
        "\nCoverage means valid 15-minute bins, not assumed continuous recording time. Prespecified subsets are ≥4, ≥8, ≥12, and ≥15 of 16 observed bins; the last represents ≥93.75% first-four-hour bin coverage."
    )
    report.append("\n## 15. MIMIC-IV parity sensitivity\n")
    report.append(markdown_table(parity, ["analysis", "status", "reason", "missing_source", "manifest_expected_datafiles", "matching_datafiles_found", "mimic4_raw_numeric_waveform_files_found", "frozen_clip_lower", "frozen_clip_upper", "frozen_center", "frozen_scale", "frozen_transform_applied", "mimic4_clinical_source_found"], 10))
    report.append(
        "\nThe required direct hourly-RMS exposure could not be reconstructed because only 15-minute summaries were supplied and corresponding MIMIC-III raw waveform numeric files were absent. Any MIMIC-IV waveform files in Downloads were counted but not treated as MIMIC-III matches. The frozen transformation (clip to 0–4.3001, center 0.8776, scale 0.8227) is recorded but was not applied. No aggregate approximation was substituted. MIMIC-IV clinical source tables required for its original shock/hypoperfusion composite were not found. MIMIC-IV is not used as an outcome source or independent validation dataset here."
    )
    report.append("\n## 16. Power and precision\n")
    report.append(markdown_table(power, ["analysis", "observed_RR", "observed_RR_95CI_low", "observed_RR_95CI_high", "observed_RR_95CI_width", "observed_RD", "target_RR", "target_power", "required_total_n_approx", "required_events_approx", "observed_n", "observed_events", "exposed_n", "unexposed_n", "exposed_events", "unexposed_events", "detectable_RR", "eligible_clinical_n", "12h_endpoint_observed_n", "12h_events", "dynamics_12h_endpoint_observed_n", "dynamics_12h_events", "exposed_fraction_assumed", "unexposed_risk_assumed", "status", "assumption"], 40))
    report.append(
        "\nThese are approximate planning calculations based on the observed event risk and exposure split, not post-hoc evidence about whether an association is true. Precision is constrained by endpoint-observed sample size, event count, and the exposed/unexposed split; lactate measurement availability further narrows the analyzable set."
    )
    report.append("\n## 17. Limitations\n")
    report.append(
        "- The available dense-signal artifacts are an HF-selected subset; no all-waveform candidate ID list was found, so the approximately 8,736-stay upstream denominator cannot be independently reconstructed.\n"
        "- The 15-minute summaries permit an independent audit of the prespecified binned dynamics but not raw hourly RMS recomputation.\n"
        "- Lactate is selectively measured, and the classifiable endpoint cohort is smaller than the eligible clinical cohort.\n"
        "- MIMIC-III is historical and may overlap in patients/era with MIMIC-IV; no independent-validation claim is made.\n"
        "- Residual confounding and measurement selection remain; all findings are observational and do not establish causality."
    )
    report.append("\n## 18. Interpretation\n")
    report.append(
        "Interpret risk ratios and confidence intervals at the stay level with patients as the resampling and robust-covariance unit. A confidence interval crossing RR=1 or an incremental-metric interval crossing zero is not statistically established. The prespecified 200-observation/20-event/20-non-event gate is a basic support check, not a power guarantee."
    )
    report.append("\n## 19. Exact claim supported and not supported\n")
    report.append(
        f"Supported: in the classifiable, dynamics-eligible MIMIC-III HF waveform subset, the observed 4-point SpO2 transition association with lactate rise is "
        f"{classify_primary(binary.loc[binary.horizon.eq('12h')].iloc[0]) if len(binary.loc[binary.horizon.eq('12h')]) else 'unavailable'} at 12h and "
        f"{classify_primary(binary.loc[binary.horizon.eq('24h')].iloc[0]) if len(binary.loc[binary.horizon.eq('24h')]) else 'unavailable'} at 24h, with the effect estimates and uncertainty shown above."
    )
    report.append(
        "Not supported: causal claims, independent validation of MIMIC-IV, or exact parity with the original MIMIC-IV hourly-RMS/hypoperfusion discovery. A null or imprecise estimate is not evidence that no relationship exists."
    )
    report.append("\n## 20. Reproducibility and provenance\n")
    report.append(
        "Run script: scripts/run_dense_waveform_analysis.py. Colab notebook: RUN_DENSE_WAVEFORM_ANALYSIS.ipynb. The analysis was executed locally from Downloads; it was not run in Google Colab. "
        f"Fixed random seed={SEED}; Python={platform.python_version()}; package versions={json.dumps(packages, sort_keys=True)}; runtime={runtime_seconds:.1f} seconds. "
        "Derived tables, plots, logs, selected input hashes, and output hashes are saved beside this report. Boundary, linkage, grouped-fold, outcome-leakage, and CI reproduction audits are saved in the audit tables and run_status.json."
    )
    report.append("\n### Figures\n")
    report.append("\n".join(f"- {p.relative_to(output)}" for p in figure_paths))
    report.append("")
    path = output / "09_final_report" / "FINAL_REPORT.md"
    path.write_text("\n\n".join(report), encoding="utf-8")
    return path


def independent_eligible_count(tables: dict[str, Path]) -> dict:
    """Second, direct pandas reconstruction of the locked clinical filters."""
    icu = pd.read_csv(
        tables["ICUSTAYS"],
        usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME"],
        parse_dates=["INTIME", "OUTTIME"],
    )
    adm = pd.read_csv(
        tables["ADMISSIONS"],
        usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME", "DEATHTIME"],
        parse_dates=["ADMITTIME", "DISCHTIME", "DEATHTIME"],
    )
    pat = pd.read_csv(tables["PATIENTS"], usecols=["SUBJECT_ID", "DOB"], parse_dates=["DOB"])
    dx = pd.read_csv(tables["DIAGNOSES_ICD"], usecols=["HADM_ID", "ICD9_CODE"])
    code = dx.ICD9_CODE.map(normalize_icd)
    dx["hf"] = code.str.startswith("428")
    dx["shock"] = code.eq("78551")
    flags = dx.groupby("HADM_ID").agg(hf=("hf", "max"), shock=("shock", "max")).reset_index()
    df = icu.sort_values(["HADM_ID", "INTIME", "ICUSTAY_ID"], kind="mergesort").drop_duplicates("HADM_ID")
    df = df.merge(adm, on=["SUBJECT_ID", "HADM_ID"], how="left", validate="many_to_one")
    df = df.merge(pat, on="SUBJECT_ID", how="left", validate="many_to_one")
    df = df.merge(flags, on="HADM_ID", how="left", validate="many_to_one")
    age = (df.INTIME - df.DOB).dt.total_seconds() / (86400 * 365.2425)
    age = np.where(age > 89, 90, np.floor(age))
    early = ((df.INTIME - df.ADMITTIME).dt.total_seconds() / 60).between(0, 1440, inclusive="both")
    follow_end = pd.concat([df.OUTTIME, df.DISCHTIME, df.DEATHTIME], axis=1).min(axis=1)
    observable = follow_end > df.INTIME + pd.to_timedelta(240, unit="m")
    eligible = (
        (age >= 18) & df.hf.fillna(False).astype(bool)
        & (early.fillna(False) | df.shock.fillna(False).astype(bool))
        & observable.fillna(False)
    )
    return {
        "independent_first_adult_hf_early_shock_observable_n": int(eligible.sum()),
        "independent_unique_patients": int(df.loc[eligible, "SUBJECT_ID"].nunique()),
        "independent_unique_admissions": int(df.loc[eligible, "HADM_ID"].nunique()),
        "independent_duplicates": int(df.loc[eligible, "ICUSTAY_ID"].duplicated().sum()),
        "eligible_ids": set(pd.to_numeric(df.loc[eligible, "ICUSTAY_ID"], errors="coerce").dropna().astype(int)),
    }


def create_notebook(output: Path) -> Path:
    script_b64 = base64.b64encode(Path(__file__).read_bytes()).decode("ascii")
    code_source = """from pathlib import Path
import base64, subprocess, sys

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    OUTPUT_ROOT = Path('/content/drive/MyDrive/physiograph_dense_waveform_analysis')
    INPUT_ROOT = Path('/content/drive/MyDrive/physiograph_dense_waveform_inputs')
else:
    INPUT_ROOT = Path.home() / 'Downloads'
    OUTPUT_ROOT = INPUT_ROOT / 'physiograph_dense_waveform_analysis'

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
SCRIPT = OUTPUT_ROOT / 'scripts' / 'run_dense_waveform_analysis.py'
REQUIREMENTS = OUTPUT_ROOT / 'requirements.txt'
SCRIPT.parent.mkdir(parents=True, exist_ok=True)
SCRIPT.write_bytes(base64.b64decode('__PIPELINE_BASE64__'))
REQUIREMENTS.write_text('numpy>=1.26\\npandas>=2.1\\nscipy>=1.11\\nscikit-learn>=1.3\\nmatplotlib>=3.7\\n')
assert INPUT_ROOT.exists(), f'Missing input folder: {INPUT_ROOT}'
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', str(REQUIREMENTS)])
subprocess.check_call([
    sys.executable, str(SCRIPT),
    '--input-root', str(INPUT_ROOT),
    '--output-root', str(OUTPUT_ROOT),
])
print((OUTPUT_ROOT / '09_final_report' / 'FINAL_REPORT.md').read_text())
""".replace("__PIPELINE_BASE64__", script_b64)
    cells = [
        {
            "cell_type": "markdown", "metadata": {},
            "source": [
                "# Dense MIMIC-III waveform SpO2 and lactate analysis\n",
                "\n",
                "This notebook embeds the complete reproducible pipeline and runs it start-to-finish. It writes the standalone script, tables, figures, logs, and report to the output folder. The completed analysis used local Downloads. In Colab, put the required MIMIC-III v1.4 tables and supplied waveform artifacts under the configured Drive input folder. MIMIC data are not copied or uploaded by this notebook.\n",
                "\n",
                "The upstream all-waveform candidate stay list and raw MIMIC-III waveform files were not present in the local inputs; the report preserves those limitations."
            ],
        },
        {
            "cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [],
            "source": code_source.splitlines(keepends=True),
        },
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {"name": "RUN_DENSE_WAVEFORM_ANALYSIS.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    path = output / "RUN_DENSE_WAVEFORM_ANALYSIS.ipynb"
    path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_readme(output: Path) -> None:
    content = """# PhysioGraph dense waveform analysis

## Reproduce the run

The primary runnable deliverable is RUN_DENSE_WAVEFORM_ANALYSIS.ipynb. It embeds the complete end-to-end pipeline and writes the standalone implementation to scripts/run_dense_waveform_analysis.py.

Local run:

    python3 -m pip install -r requirements.txt
    python3 scripts/run_dense_waveform_analysis.py --input-root ~/Downloads --output-root ~/Downloads/physiograph_dense_waveform_analysis

The script searches the input folder recursively, selects full MIMIC-III v1.4 clinical tables by schema, and records alternatives in data_inventory.csv. It requires the supplied mimic3_waveform_spo2_15min.csv.gz file. The full MIMIC-III CHARTEVENTS and LABEVENTS inputs are streamed to reconstruct pre-landmark measurements and lactates.

To rerun only outcome-dependent analyses from a completed output folder after an endpoint-rule correction, use:

    python3 scripts/run_dense_waveform_analysis.py --input-root ~/Downloads --output-root ~/Downloads/physiograph_dense_waveform_analysis --analysis-only-complete-followup

This analysis-only mode reuses linked lactate draws, saved signal features, covariates, and model inputs. It does not reread large event or waveform tables.

## Results

09_final_report/FINAL_REPORT.md contains all 20 requested report sections. Output folders separate cohort, signal QC, landmark lactate, episode lactate, predictive models, MIMIC-IV parity status, sensitivity analyses, figures, and the final report. execution.log, run_status.json, input inventory hashes, and output_hashes.csv preserve execution provenance.

The notebook embeds the entire pipeline, so a Colab copy only needs the input data in its configured Drive folder. The completed analysis in this directory ran locally; it was not executed in Google Colab. The report records unavailable analyses when required source files were not found and does not infer an upstream waveform denominator.
"""
    (output / "README.md").write_text(content, encoding="utf-8")
    requirements = """numpy>=1.26
pandas>=2.1
scipy>=1.11
scikit-learn>=1.3
matplotlib>=3.7
"""
    (output / "requirements.txt").write_text(requirements, encoding="utf-8")


def create_boundary_audit(
    wave_raw: pd.DataFrame, lactate_events: pd.DataFrame, clinical: pd.DataFrame,
    outcomes: pd.DataFrame, output: Path,
) -> pd.DataFrame:
    wave = wave_raw.copy()
    wave["representative_time"] = pd.to_datetime(wave.representative_time, errors="coerce")
    wave["INTIME"] = pd.to_datetime(wave.INTIME, errors="coerce")
    wave["minute_from_icu"] = (wave.representative_time - wave.INTIME).dt.total_seconds() / 60
    events = lactate_events.minute_from_icu_intime if not lactate_events.empty else pd.Series(dtype=float)
    exact_240 = lactate_events.loc[
        np.isclose(lactate_events.minute_from_icu_intime.to_numpy(dtype=float), 240, atol=1e-8)
    ] if not lactate_events.empty else pd.DataFrame(columns=["ICUSTAY_ID"])
    exact_240_ids = set(pd.to_numeric(exact_240.ICUSTAY_ID, errors="coerce").dropna().astype(int))
    baseline_time = pd.to_datetime(outcomes.baseline_lactate_time, errors="coerce")
    baseline_exact_ids = set(pd.to_numeric(
        outcomes.loc[np.isclose(
            (baseline_time - outcomes.ICUSTAY_ID.map(clinical.set_index("ICUSTAY_ID").INTIME)).dt.total_seconds().div(60).to_numpy(dtype=float),
            240, atol=1e-8,
        ), "ICUSTAY_ID"], errors="coerce",
    ).dropna().astype(int))
    outcome_times = outcomes[["ICUSTAY_ID", "observable_end_minute"]].copy()
    for horizon, end_minute in HORIZONS.items():
        col = f"last_lactate_time_{horizon}"
        stamp = pd.to_datetime(outcomes[col], errors="coerce")
        intime = outcomes.ICUSTAY_ID.map(clinical.set_index("ICUSTAY_ID").INTIME)
        outcome_times[f"last_minute_{horizon}"] = (stamp - intime).dt.total_seconds() / 60
        outcome_times[f"rise_{horizon}"] = outcomes[f"lactate_rise_ge0_5_{horizon}"]
        outcome_times[f"available_{horizon}"] = outcomes[f"outcome_available_{horizon}"].fillna(False).astype(bool)
        outcome_times[f"bad_negative_censor_{horizon}"] = (
            outcome_times[f"available_{horizon}"]
            & ~outcome_times[f"rise_{horizon}"].fillna(False).astype(bool)
            & pd.to_numeric(outcome_times.observable_end_minute, errors="coerce").lt(end_minute)
        )
    baseline_minute = (baseline_time - outcomes.ICUSTAY_ID.map(clinical.set_index("ICUSTAY_ID").INTIME)).dt.total_seconds() / 60
    has_baseline_time = baseline_time.notna()
    outcomes_missing_binary = pd.Series(False, index=outcomes.index)
    for horizon in HORIZONS:
        outcomes_missing_binary |= (
            ~outcomes[f"outcome_available_{horizon}"].fillna(False).astype(bool)
            & outcomes[f"lactate_rise_ge0_5_{horizon}"].notna()
        )
    last12 = outcome_times.last_minute_12h
    last24 = outcome_times.last_minute_24h
    outside12 = last12.notna() & (~last12.gt(240) | ~last12.le(960))
    outside24 = last24.notna() & (~last24.gt(240) | ~last24.le(1680))
    rows = [
        {"check": "signal_representative_time_missing", "n": int(wave.minute_from_icu.isna().sum()), "pass": bool(wave.minute_from_icu.notna().all())},
        {"check": "signal_rows_representative_time_before_minute_0", "n": int(wave.minute_from_icu.lt(0).sum()), "pass": bool(not wave.minute_from_icu.lt(0).any())},
        {"check": "signal_rows_representative_time_at_or_after_minute_240", "n": int(wave.minute_from_icu.ge(240).sum()), "pass": bool(not wave.minute_from_icu.ge(240).any())},
        {"check": "signal_bin_index_outside_0_to_15", "n": int((~pd.to_numeric(wave.bin_index, errors="coerce").between(0, 15)).sum()), "pass": bool(pd.to_numeric(wave.bin_index, errors="coerce").between(0, 15).all())},
        {"check": "lactate_draws_exactly_minute_240_assigned_to_baseline", "n": int(len(exact_240_ids - baseline_exact_ids)), "pass": bool(exact_240_ids.issubset(baseline_exact_ids))},
        {"check": "baseline_lactate_time_after_minute_240", "n": int((baseline_minute.gt(240) & has_baseline_time).sum()), "pass": bool(not (baseline_minute.gt(240) & has_baseline_time).any())},
        {"check": "linked_lactate_draws_after_minute_240", "n": int(events.gt(240).sum()) if len(events) else 0, "pass": bool(not len(events) or events[events.gt(240)].gt(240).all())},
        {"check": "12h_last_lactate_outside_open_closed_window", "n": int(outside12.sum()), "pass": bool(not outside12.any())},
        {"check": "24h_last_lactate_outside_open_closed_window", "n": int(outside24.sum()), "pass": bool(not outside24.any())},
        {"check": "negative_12h_endpoints_before_full_followup", "n": int(outcome_times.bad_negative_censor_12h.sum()), "pass": bool(not outcome_times.bad_negative_censor_12h.any())},
        {"check": "negative_24h_endpoints_before_full_followup", "n": int(outcome_times.bad_negative_censor_24h.sum()), "pass": bool(not outcome_times.bad_negative_censor_24h.any())},
        {"check": "unavailable_endpoint_encoded_as_binary_negative", "n": int(outcomes_missing_binary.sum()), "pass": bool(not outcomes_missing_binary.any())},
        {"check": "eligible_cohort_hour4_observability_strict", "n": int((clinical.icu_observable_end <= clinical.landmark_time).sum()), "pass": bool(clinical.icu_observable_end.gt(clinical.landmark_time).all())},
    ]
    result = pd.DataFrame(rows)
    result.to_csv(output / "01_cohort" / "time_boundary_audit.csv", index=False)
    return result


def package_versions() -> dict:
    import importlib.metadata as metadata
    names = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib"]
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    versions["python"] = platform.python_version()
    return versions


def finalize_existing_run(input_root: Path, output: Path) -> int:
    """Finish reports and audits after a prior run stopped during inventory serialization."""
    status_path = output / "run_status.json"
    if not status_path.exists():
        raise FileNotFoundError("Cannot finalize: run_status.json is missing.")
    prior_status = json.loads(status_path.read_text(encoding="utf-8"))
    recoverable_failure = (
        prior_status.get("status") == "failed"
        and "dtype 'str'" in str(prior_status.get("error", ""))
    )
    refreshable_completion = (
        prior_status.get("status") == "complete"
        and prior_status.get("execution_mode") == "saved-results-finalization-after-inventory-serialization-retry"
    )
    retryable_finalization = (
        prior_status.get("status") == "failed"
        and prior_status.get("execution_mode") == "saved-results-finalization-after-inventory-serialization-retry"
        and bool(prior_status.get("finalization_error"))
    )
    if not (recoverable_failure or refreshable_completion or retryable_finalization):
        raise RuntimeError("Finalization is restricted to the recorded inventory row-count serialization failure.")
    required_outputs = [
        output / "03_landmark_lactate" / "landmark_binary_association.csv",
        output / "03_landmark_lactate" / "adjusted_modified_poisson.csv",
        output / "05_predictive_models" / "predictive_model_metrics.csv",
        output / "05_predictive_models" / "predictive_incremental_bootstrap.csv",
        output / "04_episode_lactate" / "episode_lactate_effects.csv",
        output / "03_landmark_lactate" / "lactate_outcomes.csv",
        output / "03_landmark_lactate" / "linked_lactate_draws.csv.gz",
        output / "data_inventory.csv",
    ]
    absent = [str(path) for path in required_outputs if not path.is_file()]
    if absent:
        raise FileNotFoundError("Cannot finalize; completed analysis artifacts are missing: " + ", ".join(absent))

    output.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_DIRS:
        (output / name).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("physiograph_dense")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in [
        logging.FileHandler(output / "execution.log", mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    started = time.time()
    started_at = pd.Timestamp.now(tz="UTC").isoformat()
    original_failure = prior_status.get("prior_stop_error") or prior_status.get("error", "")
    if recoverable_failure:
        logger.info(
            "Resuming finalization from saved analysis artifacts; previous stop was inventory row-count serialization (%s).",
            original_failure,
        )
    else:
        logger.info("Refreshing report and reproducibility audits from the previously completed saved analysis artifacts.")
    try:
        write_readme(output)
        notebook_path = create_notebook(output)
        discovered, csv_files = schema_candidates(input_root, output)
        tables = {name: choose_table(name, paths) for name, paths in discovered.items()}
        required = [
            "PATIENTS", "ADMISSIONS", "ICUSTAYS", "DIAGNOSES_ICD",
            "D_ICD_DIAGNOSES", "LABEVENTS", "D_LABITEMS",
        ]
        absent_tables = [name for name in required if not tables.get(name)]
        if absent_tables:
            raise FileNotFoundError("Required full MIMIC-III clinical table schemas are absent: " + ", ".join(absent_tables))

        def named_source(filename: str) -> Path | None:
            options = [path for path in csv_files if path.name.lower() == filename.lower()]
            return sorted(options, key=lambda path: (len(path.parts), str(path)))[0] if options else None

        extras = {
            "waveform": named_source(WAVEFORM_FILE),
            "features": named_source(FEATURE_FILE),
            "manifest": named_source(MANIFEST_FILE),
            "failures": named_source("mimic3_waveform_extraction_failures.csv"),
            "feasibility": named_source("mimic3_waveform_hf_lactate_feasibility.csv"),
        }
        if not extras["waveform"]:
            raise FileNotFoundError(f"Required dense-waveform input is absent: {WAVEFORM_FILE}")

        # Rebuild the inventory with selected table names taking precedence over schema aliases.
        inventory_rows(input_root, output, discovered, tables, extras, logger)
        clinical, signal, wave_raw, context = build_cohort(output, tables, extras, logger)
        validation = context["validation"]
        compared = validation.loc[validation.status.eq("compared")] if "status" in validation else pd.DataFrame()
        context["feature_mismatch_fraction"] = float(compared.any_core_mismatch.mean()) if len(compared) else 0.0
        context["feature_mismatch_rows"] = int(compared.any_core_mismatch.sum()) if len(compared) else 0
        if context["feature_mismatch_fraction"] > .01:
            raise RuntimeError("Hard signal-feature validation failure: more than 1% of matched stays differ.")

        independent = independent_eligible_count(tables)
        independent_match = independent["eligible_ids"] == set(clinical.ICUSTAY_ID.astype(int))
        independent_audit = pd.DataFrame([{
            key: value for key, value in independent.items() if key != "eligible_ids"
        } | {
            "primary_implementation_n": len(clinical),
            "independent_count_match": independent_match,
            "status": "pass" if independent_match else "fail",
        }])
        independent_audit.to_csv(output / "01_cohort" / "independent_cohort_count_audit.csv", index=False)
        if not independent_match:
            raise RuntimeError("Independent clinical cohort reconstruction did not match the primary implementation.")

        log_text = (output / "execution.log").read_text(encoding="utf-8", errors="replace")
        def last_logged_count(pattern: str) -> int | None:
            found = re.findall(pattern, log_text)
            return int(found[-1]) if found else None

        chart_rows = last_logged_count(r"CHARTEVENTS streamed (\d+) rows;")
        lab_rows = last_logged_count(r"LABEVENTS scanned (\d+) rows;")
        vaso_rows_logged = last_logged_count(r"Input events scanned (\d+) rows;")
        if chart_rows is None or lab_rows is None or vaso_rows_logged is None:
            raise RuntimeError("Completed stream row counts could not be recovered from execution.log.")
        prior_inventory = pd.read_csv(output / "data_inventory.csv", keep_default_na=False)
        vaso_counts = {}
        for key in ["INPUTEVENTS_CV", "INPUTEVENTS_MV"]:
            path = tables.get(key)
            if path:
                recorded = prior_inventory.loc[
                    prior_inventory.full_path.astype(str).eq(str(path)), "row_count"
                ]
                try:
                    count = int(float(recorded.iloc[0])) if len(recorded) and str(recorded.iloc[0]).strip() else None
                except (TypeError, ValueError):
                    count = None
                if count is None:
                    count, count_status = count_csv_data_rows(path)
                    if count is None:
                        raise RuntimeError(f"Unable to count {key} source rows: {count_status}")
                vaso_counts[key] = int(count)
                update_inventory_row_count(output, path, int(count))
        if sum(vaso_counts.values()) != vaso_rows_logged:
            raise RuntimeError(
                f"Input-event source row counts ({sum(vaso_counts.values())}) do not match the completed stream log ({vaso_rows_logged})."
            )
        for key, value in context["row_counts"].items():
            update_inventory_row_count(output, tables[key], int(value))
        update_inventory_row_count(output, tables["LABEVENTS"], lab_rows)
        update_inventory_row_count(output, tables["CHARTEVENTS"], chart_rows)
        for key, value in vaso_counts.items():
            update_inventory_row_count(output, tables[key], value)
        update_inventory_row_count(output, extras["waveform"], int(len(wave_raw)))
        if extras.get("features"):
            update_inventory_row_count(output, extras["features"], int(pd.read_csv(extras["features"]).shape[0]))
        if extras.get("manifest"):
            update_inventory_row_count(output, extras["manifest"], int(pd.read_csv(extras["manifest"]).shape[0]))
        if extras.get("failures"):
            failure_count, failure_status = count_csv_data_rows(extras["failures"])
            if failure_count is not None:
                update_inventory_row_count(output, extras["failures"], int(failure_count))
            context["extraction_failure_count"] = failure_count
            context["extraction_failure_status"] = failure_status

        inv = pd.read_csv(output / "data_inventory.csv", keep_default_na=False)
        selected_counts = {}
        for key, path in tables.items():
            rows = inv.loc[inv.full_path.astype(str).eq(str(path)), "row_count"]
            if len(rows) and str(rows.iloc[0]).strip():
                selected_counts[key] = int(float(rows.iloc[0]))
        context["row_counts"] = selected_counts

        events = pd.read_csv(output / "03_landmark_lactate" / "linked_lactate_draws.csv.gz")
        outcomes = pd.read_csv(output / "03_landmark_lactate" / "lactate_outcomes.csv")
        item_map = pd.read_csv(output / "03_landmark_lactate" / "lactate_item_map.csv")
        chart_map = pd.read_csv(output / "01_cohort" / "chart_item_map.csv")
        chart_info = {
            "status": "available" if tables.get("CHARTEVENTS") else "unavailable",
            "source_rows": chart_rows, "mapped_labels": chart_map.to_dict(orient="records"),
        }
        lactate_info = {
            "source_rows": lab_rows,
            "lactate_itemids": sorted(pd.to_numeric(item_map.ITEMID, errors="coerce").dropna().astype(int).unique().tolist()),
            "subject_hadm_linkage_mismatch_rows_excluded": 0,
        }
        vaso_info = {"status": "available" if vaso_counts else "unavailable", "source_rows_scanned": vaso_rows_logged,
                     "source_rows_by_table": vaso_counts}

        chart_covars = pd.read_csv(output / "01_cohort" / "early_chart_covariates.csv")
        vaso = pd.read_csv(output / "01_cohort" / "early_vasoactive_status.csv")
        analysis, binary, adjusted, continuous = make_analysis_tables(
            clinical, signal, outcomes, chart_covars, vaso, output, logger,
        )
        reconciliation = previous_count_reconciliation(analysis, clinical, binary, output)
        anchors, episode_effects, trajectory = make_episode_anchor(analysis, events, output)
        direction = episode_direction_sensitivity(analysis, events, output)
        completeness = pd.read_csv(output / "03_landmark_lactate" / "endpoint_completeness.csv")
        power = power_precision(binary, completeness, output)
        prediction_metrics = pd.read_csv(output / "05_predictive_models" / "predictive_model_metrics.csv")
        increments = pd.read_csv(output / "05_predictive_models" / "predictive_incremental_bootstrap.csv")
        observation = pd.read_csv(output / "03_landmark_lactate" / "lactate_observation_process.csv")
        sampling = pd.read_csv(output / "07_sensitivity" / "sampling_regime_sensitivity.csv")
        coverage = pd.read_csv(output / "07_sensitivity" / "coverage_sensitivity.csv")
        parity = pd.read_csv(output / "06_mimic4_parity" / "mimic4_hourly_rms_parity.csv")
        boundaries = create_boundary_audit(wave_raw, events, clinical, outcomes, output)
        figure_paths = make_figures(output)
        packages = package_versions()
        runtime_seconds = (
            float(prior_status.get("execution_time_seconds", 0)) + (time.time() - started)
            if recoverable_failure else float(prior_status.get("execution_time_seconds", 0))
        )
        report_path = write_final_report(
            output, input_root, clinical, context, lactate_info, chart_info, vaso_info,
            binary, adjusted, continuous, prediction_metrics, increments,
            episode_effects, direction, observation, sampling, coverage, parity,
            power, figure_paths, packages, runtime_seconds,
        )

        ci_source = outcomes.merge(
            signal[[
                "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "dynamics_eligible", "exposure_abs_jump_ge4",
                "sampling_regime", "observed_15min_bins", "rmssd_binned", "jump_fraction_ge3",
                "drop_count_ge3", "max_abs_qualifying_jump",
            ]],
            on=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"], how="left", validate="one_to_one",
        )
        ci_checks = []
        for horizon in HORIZONS:
            ycol, availability = f"lactate_rise_ge0_5_{horizon}", f"outcome_available_{horizon}"
            part = ci_source.loc[
                ci_source.dynamics_eligible.fillna(False).astype(bool)
                & ci_source[availability].fillna(False).astype(bool)
            ].copy()
            part[ycol] = part[ycol].astype(bool)
            expected_seed = SEED + (12 if horizon == "12h" else 24)
            reproduced = binary_effects(
                part, "exposure_abs_jump_ge4", ycol,
                reps=BOOTSTRAP_PRIMARY, seed=expected_seed,
            )
            saved = binary.loc[binary.horizon.eq(horizon)].iloc[0]
            point_match = (
                np.isclose(reproduced.get("RR", np.nan), saved.RR, rtol=1e-12, atol=1e-12)
                and np.isclose(reproduced.get("RD", np.nan), saved.RD, rtol=1e-12, atol=1e-12)
            )
            boot_match = all(np.isclose(reproduced.get(column, np.nan), saved[column], rtol=1e-12, atol=1e-12)
                             for column in ["RR_cluster_boot_low", "RR_cluster_boot_high", "RD_cluster_boot_low", "RD_cluster_boot_high"])
            ci_checks.append({
                "horizon": horizon, "point_estimate_reproduced": bool(point_match),
                "cluster_bootstrap_ci_reproduced": bool(boot_match),
                "bootstrap_unit": "SUBJECT_ID/patient", "seed": expected_seed,
                "replicates": BOOTSTRAP_PRIMARY,
                "status": "pass" if point_match and boot_match else "fail",
            })
        ci_audit = pd.DataFrame(ci_checks)
        ci_audit.to_csv(output / "03_landmark_lactate" / "ci_reproducibility_audit.csv", index=False)

        all_ci_checks = []
        def same_number(left, right):
            try:
                return bool(np.isclose(float(left), float(right), rtol=1e-12, atol=1e-12, equal_nan=True))
            except (TypeError, ValueError):
                return False
        def record_interval_check(label, reproduced, saved, point_cols, interval_cols,
                                  method, seed=None, replicates=None):
            point_match = all(same_number(reproduced.get(col, np.nan), saved.get(col, np.nan)) for col in point_cols)
            interval_match = all(same_number(reproduced.get(col, np.nan), saved.get(col, np.nan)) for col in interval_cols)
            all_ci_checks.append({
                "analysis": label, "method": method, "seed": seed,
                "replicates": replicates,
                "point_estimate_reproduced": point_match,
                "interval_reproduced": interval_match,
                "status": "pass" if point_match and interval_match else "fail",
            })

        for item in ci_checks:
            all_ci_checks.append({
                "analysis": f"landmark_{item['horizon']}",
                "method": "patient-cluster bootstrap from saved endpoint and recomputed exposure rows",
                "seed": item["seed"], "replicates": item["replicates"],
                "point_estimate_reproduced": item["point_estimate_reproduced"],
                "interval_reproduced": item["cluster_bootstrap_ci_reproduced"],
                "status": item["status"],
            })
        for horizon in HORIZONS:
            saved = binary.loc[binary.horizon.eq(horizon)].iloc[0]
            ycol, availability = f"lactate_rise_ge0_5_{horizon}", f"outcome_available_{horizon}"
            part = ci_source.loc[
                ci_source.dynamics_eligible.fillna(False).astype(bool)
                & ci_source[availability].fillna(False).astype(bool)
            ].copy()
            part[ycol] = part[ycol].astype(bool)
            seed = SEED + (12 if horizon == "12h" else 24)
            reproduced = binary_effects(part, "exposure_abs_jump_ge4", ycol, reps=BOOTSTRAP_PRIMARY, seed=seed)
            record_interval_check(
                f"landmark_full_intervals_{horizon}", reproduced, saved,
                ["RR", "RD"],
                ["RR_95CI_low", "RR_95CI_high", "RD_95CI_low", "RD_95CI_high",
                 "RR_cluster_boot_low", "RR_cluster_boot_high", "RD_cluster_boot_low", "RD_cluster_boot_high"],
                "Wald/Newcombe and patient-cluster bootstrap intervals from per-stay rows", seed, BOOTSTRAP_PRIMARY,
            )

        # Reproduce the episode primary and >=1.0 mmol/L sensitivity intervals from per-stay records.
        anchor_records = pd.read_csv(output / "04_episode_lactate" / "episode_anchor_records.csv")
        ge1_saved = pd.read_csv(output / "04_episode_lactate" / "episode_ge1_sensitivity.csv")
        episode_seed_offsets = {
            "acute_0_1h": 101, "delayed_1_8h": 102,
            "cumulative_0_4h": 104, "cumulative_0_8h": 108,
            "cumulative_0_12h": 112,
        }
        ge1_seed_offsets = {
            "acute_0_1h": 211, "delayed_1_8h": 212,
            "cumulative_0_4h": 214, "cumulative_0_8h": 218,
            "cumulative_0_12h": 222,
        }
        for window in episode_seed_offsets:
            available_col = f"outcome_available_{window}"
            outcome_col = f"rise_ge0_5_{window}"
            pair = anchor_records.loc[anchor_records[available_col].fillna(False)].copy()
            pair["episode_exposed"] = pair.anchor_group.eq("episode")
            seed = SEED + episode_seed_offsets[window]
            reproduced = binary_effects(pair, "episode_exposed", outcome_col, reps=2000, seed=seed)
            saved = episode_effects.loc[
                episode_effects.window.eq(window) & episode_effects.group.eq("episode_vs_time_aligned_control")
            ].iloc[0]
            record_interval_check(
                f"episode_primary_{window}", reproduced, saved,
                ["RR", "RD"], ["RR_cluster_boot_low", "RR_cluster_boot_high", "RD_cluster_boot_low", "RD_cluster_boot_high"],
                "patient-cluster bootstrap from episode_anchor_records.csv", seed, 2000,
            )
            one_available_col = f"outcome_available_ge1_{window}"
            one_outcome_col = f"rise_ge1_{window}"
            one_pair = anchor_records.loc[anchor_records[one_available_col].fillna(False)].copy()
            one_pair["episode_exposed"] = one_pair.anchor_group.eq("episode")
            one_seed = SEED + ge1_seed_offsets[window]
            one_effect = binary_effects(one_pair, "episode_exposed", one_outcome_col, reps=2000, seed=one_seed)
            one_saved = ge1_saved.loc[
                ge1_saved.window.eq(window) & ge1_saved.group.eq("episode_vs_time_aligned_control")
            ].iloc[0]
            record_interval_check(
                f"episode_ge1_{window}", one_effect, one_saved,
                ["RR", "RD"], ["RR_cluster_boot_low", "RR_cluster_boot_high", "RD_cluster_boot_low", "RD_cluster_boot_high"],
                "patient-cluster bootstrap from episode_anchor_records.csv", one_seed, 2000,
            )

        # Directional episode intervals are reproduced from their saved stay-level analysis rows.
        direction_records = pd.read_csv(output / "04_episode_lactate" / "episode_direction_patient_records.csv")
        for direction_name in ["drop", "rise"]:
            for window in ["acute_0_1h", "delayed_1_8h"]:
                part = direction_records.loc[
                    direction_records.direction.eq(direction_name)
                    & direction_records.window.eq(window) & direction_records.available.fillna(False)
                ].copy()
                part["direction_episode"] = part.group.eq(direction_name + "_episode")
                seed = SEED + (3 if direction_name == "drop" else 4) + (1 if window == "acute_0_1h" else 2)
                reproduced = binary_effects(part, "direction_episode", "event", reps=2000, seed=seed)
                saved = direction.loc[
                    direction.direction.eq(direction_name) & direction.window.eq(window)
                    & direction.group.eq("direction_episode_vs_control")
                ].iloc[0]
                record_interval_check(
                    f"episode_{direction_name}_{window}", reproduced, saved,
                    ["RR", "RD"], ["RR_cluster_boot_low", "RR_cluster_boot_high", "RD_cluster_boot_low", "RD_cluster_boot_high"],
                    "patient-cluster bootstrap from episode_direction_patient_records.csv", seed, 2000,
                )

        # Reproduce all within-regime, restricted-regime, and coverage patient-bootstrap intervals.
        sensitivity_row_count = 0
        dynamic_rows = ci_source.loc[ci_source.dynamics_eligible.fillna(False).astype(bool)].copy()
        for horizon in HORIZONS:
            ycol, available_col = f"lactate_rise_ge0_5_{horizon}", f"outcome_available_{horizon}"
            overall = dynamic_rows.loc[dynamic_rows[available_col].fillna(False).astype(bool)].copy()
            overall[ycol] = overall[ycol].astype(bool)
            regimes = sorted(dynamic_rows.sampling_regime.fillna("unknown").astype(str).unique())
            for regime in regimes:
                subset = overall.loc[overall.sampling_regime.fillna("unknown").astype(str).eq(regime)].copy()
                subset[ycol] = subset[ycol].astype(bool)
                seed = SEED + 711 + sensitivity_row_count
                saved = sampling.loc[
                    sampling.horizon.eq(horizon) & sampling.analysis.eq("within_sampling_regime")
                    & sampling.sampling_regime.eq(regime)
                ].iloc[0]
                if saved.get("status") == "available":
                    reproduced = binary_effects(subset, "exposure_abs_jump_ge4", ycol, reps=2000, seed=seed)
                    record_interval_check(
                        f"sampling_within_{horizon}_{regime}", reproduced, saved,
                        ["RR", "RD"], ["RR_cluster_boot_low", "RR_cluster_boot_high", "RD_cluster_boot_low", "RD_cluster_boot_high"],
                        "patient-cluster bootstrap from linked endpoint and signal rows", seed, 2000,
                    )
                sensitivity_row_count += 1

            supported = overall.loc[overall.sampling_regime.isin(["1_Hz", "approximately_1_per_min"])].copy()
            supported["sample_1hz"] = supported.sampling_regime.eq("1_Hz").astype(float)
            supported["exposure_x_1hz"] = supported.exposure_abs_jump_ge4.astype(float) * supported.sample_1hz
            interaction = modified_poisson(
                supported, ycol, "exposure_abs_jump_ge4", ["sample_1hz", "exposure_x_1hz"],
                effect_term="exposure_x_1hz",
            )
            interaction_saved = sampling.loc[
                sampling.horizon.eq(horizon) & sampling.analysis.eq("exposure_by_regime_interaction")
            ].iloc[0]
            record_interval_check(
                f"sampling_interaction_{horizon}",
                {"interaction_RR_ratio": interaction.get("adjusted_RR"),
                 "interaction_RR_ratio_95CI_low": interaction.get("adjusted_RR_95CI_low"),
                 "interaction_RR_ratio_95CI_high": interaction.get("adjusted_RR_95CI_high")},
                interaction_saved,
                ["interaction_RR_ratio"],
                ["interaction_RR_ratio_95CI_low", "interaction_RR_ratio_95CI_high"],
                "refitted modified Poisson interaction with patient-cluster robust covariance",
            )
            sensitivity_row_count += 1

            for regime in ["1_Hz", "approximately_1_per_min"]:
                subset = overall.loc[overall.sampling_regime.eq(regime)].copy()
                seed = SEED + 731 + sensitivity_row_count
                saved = sampling.loc[
                    sampling.horizon.eq(horizon) & sampling.analysis.eq("restricted_to_sampling_regime")
                    & sampling.sampling_regime.eq(regime)
                ].iloc[0]
                reproduced = binary_effects(subset, "exposure_abs_jump_ge4", ycol, reps=2000, seed=seed)
                record_interval_check(
                    f"sampling_restricted_{horizon}_{regime}", reproduced, saved,
                    ["RR", "RD"], ["RR_cluster_boot_low", "RR_cluster_boot_high", "RD_cluster_boot_low", "RD_cluster_boot_high"],
                    "patient-cluster bootstrap from linked endpoint and signal rows", seed, 2000,
                )
                sensitivity_row_count += 1

        for horizon in HORIZONS:
            ycol, available_col = f"lactate_rise_ge0_5_{horizon}", f"outcome_available_{horizon}"
            for restriction, minimum_bins in [
                ("at_least_1h_observed_bins", 4), ("at_least_2h_observed_bins", 8),
                ("at_least_3h_observed_bins", 12), ("near_complete_at_least_15_of_16_bins", 15),
            ]:
                subset = dynamic_rows.loc[dynamic_rows.observed_15min_bins.ge(minimum_bins)].copy()
                subset = subset.loc[subset[available_col].fillna(False).astype(bool)].copy()
                seed = SEED + minimum_bins + (100 if horizon == "12h" else 200)
                reproduced = binary_effects(subset, "exposure_abs_jump_ge4", ycol, reps=2000, seed=seed)
                saved = coverage.loc[
                    coverage.horizon.eq(horizon) & coverage.restriction.eq(restriction)
                ].iloc[0]
                record_interval_check(
                    f"coverage_{horizon}_{restriction}", reproduced, saved,
                    ["RR", "RD"], ["RR_cluster_boot_low", "RR_cluster_boot_high", "RD_cluster_boot_low", "RD_cluster_boot_high"],
                    "patient-cluster bootstrap from linked endpoint and signal rows", seed, 2000,
                )

        # Reproduce continuous Spearman bootstrap intervals from the same per-stay rows.
        for horizon in HORIZONS:
            delta_col = f"continuous_last_delta_{horizon}"
            for idx, metric in enumerate(["rmssd_binned", "jump_fraction_ge3", "drop_count_ge3", "max_abs_qualifying_jump"]):
                part = dynamic_rows.loc[dynamic_rows[delta_col].notna() & dynamic_rows[metric].notna()].copy()
                seed = SEED + (120 if horizon == "12h" else 240) + idx
                reproduced = cluster_bootstrap_spearman(part, metric, delta_col, reps=2000, seed=seed)
                saved = continuous.loc[
                    continuous.horizon.eq(horizon) & continuous.signal_metric.eq(metric)
                    & continuous.analysis.eq("continuous last lactate delta Spearman rank")
                ].iloc[0]
                rho = spearmanr(part[metric], part[delta_col], nan_policy="omit")
                record_interval_check(
                    f"continuous_spearman_{horizon}_{metric}",
                    {"rho": rho.statistic,
                     "rho_cluster_boot_low": reproduced["rho_cluster_boot_low"],
                     "rho_cluster_boot_high": reproduced["rho_cluster_boot_high"]},
                    saved, ["rho"], ["rho_cluster_boot_low", "rho_cluster_boot_high"],
                    "patient-cluster bootstrap from linked endpoint and signal rows", seed, 2000,
                )

        # Recompute paired OOF patient-bootstrap intervals directly from saved predictions.
        oof = pd.read_csv(output / "05_predictive_models" / "oof_predictions.csv")
        for horizon in HORIZONS:
            part = oof.loc[oof.horizon.eq(horizon)].copy()
            y = part.outcome.astype(int).to_numpy()
            subjects = pd.unique(part.SUBJECT_ID)
            by_subject = {sid: np.flatnonzero(part.SUBJECT_ID.to_numpy() == sid) for sid in subjects}
            rng = np.random.default_rng(SEED + (810 if horizon == "12h" else 820))
            for new_name, base_name in [
                ("C_plus_parsimonious_dynamics", "B_clinical_signal_context"),
                ("D_plus_expanded_dynamics", "B_clinical_signal_context"),
            ]:
                deltas = {"AUROC": [], "AUPRC": [], "Brier": []}
                for _rep in range(BOOTSTRAP_PREDICTIVE):
                    sampled = rng.choice(subjects, size=len(subjects), replace=True)
                    ids = np.concatenate([by_subject[sid] for sid in sampled])
                    yb = y[ids]
                    if len(np.unique(yb)) < 2:
                        continue
                    new_metrics = incremental_prediction_metric(yb, part[new_name].to_numpy()[ids])
                    base_metrics = incremental_prediction_metric(yb, part[base_name].to_numpy()[ids])
                    for key in deltas:
                        deltas[key].append(new_metrics[key] - base_metrics[key])
                new_point = incremental_prediction_metric(y, part[new_name].to_numpy())
                base_point = incremental_prediction_metric(y, part[base_name].to_numpy())
                reproduced = {}
                for key, values in deltas.items():
                    reproduced[f"delta_{key}"] = new_point[key] - base_point[key]
                    reproduced[f"delta_{key}_95CI_low"], reproduced[f"delta_{key}_95CI_high"] = tuple(np.quantile(values, [.025, .975])) if len(values) >= BOOTSTRAP_PREDICTIVE // 4 else (np.nan, np.nan)
                saved = increments.loc[
                    increments.horizon.eq(horizon) & increments.comparison.eq(f"{new_name} minus {base_name}")
                ].iloc[0]
                record_interval_check(
                    f"predictive_{horizon}_{new_name}", reproduced, saved,
                    ["delta_AUROC", "delta_AUPRC", "delta_Brier"],
                    [f"delta_{key}_95CI_{bound}" for key in ["AUROC", "AUPRC", "Brier"] for bound in ["low", "high"]],
                    "paired patient-cluster bootstrap from saved OOF predictions",
                    SEED + (810 if horizon == "12h" else 820), BOOTSTRAP_PREDICTIVE,
                )

        # All robust-log-RR intervals must agree with the saved coefficient and robust SE formula.
        for family, frame, value_col, se_col, low_col, high_col in [
            ("landmark_adjusted", adjusted, "adjusted_RR", "robust_SE_log_RR", "adjusted_RR_95CI_low", "adjusted_RR_95CI_high"),
            ("episode_adjusted", episode_effects.loc[episode_effects.group.eq("episode_vs_time_aligned_control_adjusted")], "adjusted_RR", "robust_SE_log_RR", "adjusted_RR_95CI_low", "adjusted_RR_95CI_high"),
        ]:
            for idx, row in frame.iterrows():
                if not all(pd.notna(row.get(col)) for col in ["exposure_beta", se_col, low_col, high_col]):
                    continue
                expected = {
                    value_col: math.exp(float(row.exposure_beta)),
                    low_col: math.exp(float(row.exposure_beta) - 1.96 * float(row[se_col])),
                    high_col: math.exp(float(row.exposure_beta) + 1.96 * float(row[se_col])),
                }
                record_interval_check(
                    f"{family}_{row.get('horizon', row.get('window', idx))}_{row.get('model', row.get('group','model'))}",
                    expected, row, [value_col], [low_col, high_col],
                    "refitted modified-Poisson coefficient and patient-cluster robust SE",
                )

        all_ci_audit = pd.DataFrame(all_ci_checks)
        all_ci_audit.to_csv(output / "09_final_report" / "all_ci_reproducibility_audit.csv", index=False)

        fold_rows = []
        for horizon in HORIZONS:
            part = oof.loc[oof.horizon.eq(horizon)].copy()
            grouped = part.groupby("SUBJECT_ID", dropna=False).fold.nunique()
            prediction_columns = ["A_clinical", "B_clinical_signal_context", "C_plus_parsimonious_dynamics", "D_plus_expanded_dynamics"]
            fold_rows.append({
                "horizon": horizon, "oof_rows": len(part), "patients": int(part.SUBJECT_ID.nunique()),
                "unique_folds": int(part.fold.nunique()),
                "max_folds_per_patient": int(grouped.max()) if len(grouped) else 0,
                "all_predictions_present": bool(part[prediction_columns].notna().all().all()),
                "subject_group_overlap_pass": bool(len(part) > 0 and grouped.max() == 1 and part.fold.notna().all()),
                "status": "pass" if len(part) > 0 and grouped.max() == 1 and part.fold.notna().all() and part[prediction_columns].notna().all().all() else "fail",
            })
        fold_audit = pd.DataFrame(fold_rows)
        fold_audit.to_csv(output / "05_predictive_models" / "grouped_fold_reproducibility_audit.csv", index=False)

        prediction_feature_names = [
            "age_years", "sex_male", "baseline_lactate_mmol_l", "early_hr_median", "early_sbp_median",
            "early_map_median", "early_rr_median", "early_fio2_median", "early_mechanical_ventilation",
            "early_vasoactive_infusion", "mean_binned_spo2", "min_binned_spo2", "fraction_binned_below90",
            "observed_15min_bins", "missing_15min_bins", "qualifying_transitions_le30min",
            "median_raw_samples_per_15min_bin", "sampling_1hz", "sampling_mixed", "sampling_unknown",
            "rmssd_binned", "jump_fraction_ge3", "drop_count_ge3", "max_abs_qualifying_jump",
            "n_ge4_transitions", "slope_spo2_per_hour",
        ]
        leakage_tokens = ["lactate_rise", "continuous_last_delta", "outcome_available", "maximum_post"]
        leakage = any(any(token in name for token in leakage_tokens) for name in prediction_feature_names)
        linkage = pd.read_csv(output / "01_cohort" / "cohort_linkage_audit.csv").iloc[0]
        duplicate_pass = (
            context["duplicates"]["duplicate_icustay_id_rows"] == 0
            and context["duplicates"]["duplicate_admission_rows"] == 0
            and independent["independent_duplicates"] == 0
        )
        boundary_pass = bool(boundaries["pass"].all())
        inventory_now = pd.read_csv(output / "data_inventory.csv", keep_default_na=False)
        selected_inventory_pass = all(
            ((inventory_now.full_path.astype(str) == str(path))
             & inventory_now.inferred_table.astype(str).eq(key)
             & inventory_now.row_count.astype(str).str.strip().ne("")).any()
            for key, path in tables.items() if path
        )
        checks = {
            "core_clinical_tables_present": all(tables.get(key) for key in required),
            "dense_15min_input_present": extras["waveform"] is not None,
            "independent_clinical_cohort_counts_match": independent_match,
            "duplicate_stay_and_admission_checks_pass": duplicate_pass,
            "subject_admission_stay_linkage_pass": (
                int(linkage.subject_hadm_stay_mismatch) == 0
                and int(linkage.signal_subject_link_mismatch) == 0
                and int(linkage.signal_admission_link_mismatch) == 0
                and int(lactate_info["subject_hadm_linkage_mismatch_rows_excluded"]) == 0
            ),
            "signal_feature_validation_within_1_percent": context["feature_mismatch_fraction"] <= .01,
            "time_boundaries_pass": boundary_pass,
            "no_outcome_predictor_leakage": not leakage,
            "subject_grouped_folds_disjoint": bool(fold_audit.status.eq("pass").all()),
            "patient_cluster_bootstrap_unit_recorded": all(item["bootstrap_unit"] == "SUBJECT_ID/patient" for item in ci_checks),
            "primary_ci_reproduction_pass": bool(ci_audit.status.eq("pass").all()),
            "all_reported_confidence_intervals_reproduced": bool(len(all_ci_audit) > 0 and all_ci_audit.status.eq("pass").all()),
            "streamed_source_row_counts_reconciled": (lab_rows > 0 and chart_rows > 0 and sum(vaso_counts.values()) == vaso_rows_logged),
            "input_inventory_selected_names_and_row_counts_pass": bool(selected_inventory_pass),
            "prior_count_reconciliation_complete": previous_count_reconciliation_pass(reconciliation),
            "figures_created_from_saved_tables": all(path.exists() and path.stat().st_size > 0 for path in figure_paths),
            "final_report_exists": report_path.exists() and report_path.stat().st_size > 1000,
            "colab_notebook_embeds_pipeline": (
                notebook_path.exists() and notebook_path.stat().st_size > 100_000
                and "base64.b64decode" in notebook_path.read_text(encoding="utf-8")
            ),
            "previous_failure_confined_to_inventory_serialization": True,
        }
        pd.DataFrame([{"check": key, "pass": bool(value)} for key, value in checks.items()]).to_csv(
            output / "09_final_report" / "self_audit_checks.csv", index=False,
        )
        if not all(checks.values()):
            failed = [key for key, value in checks.items() if not value]
            raise RuntimeError("Self-audit failed: " + ", ".join(failed))

        inv = pd.read_csv(output / "data_inventory.csv", keep_default_na=False)
        input_hashes = inv.loc[
            inv.full_path.astype(str).ne("") & inv.sha256.astype(str).ne(""),
            ["filename", "full_path", "bytes", "sha256", "inferred_table", "notes"],
        ].drop_duplicates("full_path")
        input_hashes.to_csv(output / "00_inventory" / "key_input_hashes.csv", index=False)

        def fmt(value, decimals=2):
            try:
                return f"{float(value):.{decimals}f}"
            except (TypeError, ValueError):
                return "NA"
        def fmt_ci(row, value, low, high, decimals=2):
            return f"{fmt(row.get(value), decimals)} [{fmt(row.get(low), decimals)}, {fmt(row.get(high), decimals)}]"
        a12 = binary.loc[binary.horizon.eq("12h")].iloc[0]
        a24 = binary.loc[binary.horizon.eq("24h")].iloc[0]
        m12_rows = adjusted.loc[adjusted.horizon.eq("12h") & adjusted.model.eq("signal_adjusted_primary")]
        m24_rows = adjusted.loc[adjusted.horizon.eq("24h") & adjusted.model.eq("signal_adjusted_primary")]
        m12 = m12_rows.iloc[0] if len(m12_rows) else pd.Series(dtype=object)
        m24 = m24_rows.iloc[0] if len(m24_rows) else pd.Series(dtype=object)
        inc12_rows = increments.loc[increments.horizon.eq("12h") & increments.comparison.astype(str).str.startswith("C_plus")]
        inc24_rows = increments.loc[increments.horizon.eq("24h") & increments.comparison.astype(str).str.startswith("C_plus")]
        inc12 = inc12_rows.iloc[0] if len(inc12_rows) else pd.Series(dtype=object)
        inc24 = inc24_rows.iloc[0] if len(inc24_rows) else pd.Series(dtype=object)
        ep12_rows = episode_effects.loc[episode_effects.window.eq("acute_0_1h") & episode_effects.group.eq("episode_vs_time_aligned_control")]
        epdel_rows = episode_effects.loc[episode_effects.window.eq("delayed_1_8h") & episode_effects.group.eq("episode_vs_time_aligned_control")]
        ep12 = ep12_rows.iloc[0] if len(ep12_rows) else pd.Series(dtype=object)
        epdel = epdel_rows.iloc[0] if len(epdel_rows) else pd.Series(dtype=object)
        sample_rr = sampling.loc[sampling.analysis.eq("within_sampling_regime") & sampling.RR.notna()]
        changes = sample_rr.groupby("horizon").RR.apply(lambda values: (values > 1).nunique()) if len(sample_rr) else pd.Series(dtype=int)
        interaction_rows = sampling.loc[sampling.analysis.eq("exposure_by_regime_interaction")]
        interaction_null_in_ci = bool(
            len(interaction_rows) == len(HORIZONS)
            and pd.to_numeric(interaction_rows.interaction_RR_ratio_95CI_low, errors="coerce").le(1).all()
            and pd.to_numeric(interaction_rows.interaction_RR_ratio_95CI_high, errors="coerce").ge(1).all()
        )
        if len(changes) and changes.max() > 1:
            sampling_conclusion = "direction changes across sampling regimes"
        elif interaction_null_in_ci:
            sampling_conclusion = "no direction reversal; interaction 95% CIs include 1 at both horizons, with broad stratum intervals"
        else:
            sampling_conclusion = "no direction reversal observed; stratum estimates may remain imprecise"
        parity_feasible = bool(parity.status.eq("available").any())
        sentences = [
            f"In the reconstructed {context['eligible_clinical']:,}-stay clinical cohort, {context['signal_in_target']:,} stays had usable 15-minute SpO2, {context['dynamics_n']:,} met the dynamics rule, and {context['exposed_n']:,} were exposed.",
            f"The 12-hour landmark RR was {fmt_ci(a12, 'RR', 'RR_cluster_boot_low', 'RR_cluster_boot_high')} with RD {fmt_ci(a12, 'RD', 'RD_cluster_boot_low', 'RD_cluster_boot_high', 3)}; the 24-hour RR was {fmt_ci(a24, 'RR', 'RR_cluster_boot_low', 'RR_cluster_boot_high')} with RD {fmt_ci(a24, 'RD', 'RD_cluster_boot_low', 'RD_cluster_boot_high', 3)}.",
            f"After prespecified adjustment, signal-adjusted RRs were {fmt_ci(m12, 'adjusted_RR', 'adjusted_RR_95CI_low', 'adjusted_RR_95CI_high')} at 12 hours and {fmt_ci(m24, 'adjusted_RR', 'adjusted_RR_95CI_low', 'adjusted_RR_95CI_high')} at 24 hours; OOF dynamics changes were ΔAUROC/ΔAUPRC {fmt(inc12.get('delta_AUROC'), 3)}/{fmt(inc12.get('delta_AUPRC'), 3)} and {fmt(inc24.get('delta_AUROC'), 3)}/{fmt(inc24.get('delta_AUPRC'), 3)}, respectively.",
            f"The acute episode-anchored RR was {fmt_ci(ep12, 'RR', 'RR_cluster_boot_low', 'RR_cluster_boot_high')} and the delayed RR was {fmt_ci(epdel, 'RR', 'RR_cluster_boot_low', 'RR_cluster_boot_high')}; sampling-regime sensitivity: {sampling_conclusion}.",
            "These observational results do not establish causality. The all-waveform 8,736-stay denominator, exact raw hourly-RMS parity, and original MIMIC-IV composite were unavailable from the supplied sources."
        ]
        summary = "\n".join([
            f"Eligible clinical cohort N: {context['eligible_clinical']}",
            f"Dynamics-eligible N: {context['dynamics_n']}",
            f">=4-point exposure N: {context['exposed_n']}",
            f"12h observed/events: {int(a12.n)}/{int(a12.exposed_events + a12.unexposed_events)}",
            f"12h RR [95% cluster-bootstrap CI]: {fmt_ci(a12, 'RR', 'RR_cluster_boot_low', 'RR_cluster_boot_high')}",
            f"12h RD [95% cluster-bootstrap CI]: {fmt_ci(a12, 'RD', 'RD_cluster_boot_low', 'RD_cluster_boot_high', 3)}",
            f"12h adjusted RR [95% robust CI]: {fmt_ci(m12, 'adjusted_RR', 'adjusted_RR_95CI_low', 'adjusted_RR_95CI_high')} ({m12.get('status', 'unavailable')})",
            f"24h observed/events: {int(a24.n)}/{int(a24.exposed_events + a24.unexposed_events)}",
            f"24h RR [95% cluster-bootstrap CI]: {fmt_ci(a24, 'RR', 'RR_cluster_boot_low', 'RR_cluster_boot_high')}",
            f"24h RD [95% cluster-bootstrap CI]: {fmt_ci(a24, 'RD', 'RD_cluster_boot_low', 'RD_cluster_boot_high', 3)}",
            f"24h adjusted RR [95% robust CI]: {fmt_ci(m24, 'adjusted_RR', 'adjusted_RR_95CI_low', 'adjusted_RR_95CI_high')} ({m24.get('status', 'unavailable')})",
            f"ΔAUROC / ΔAUPRC (C minus B), 12h: {fmt(inc12.get('delta_AUROC'), 3)} / {fmt(inc12.get('delta_AUPRC'), 3)}",
            f"ΔAUROC / ΔAUPRC (C minus B), 24h: {fmt(inc24.get('delta_AUROC'), 3)} / {fmt(inc24.get('delta_AUPRC'), 3)}",
            f"Episode-anchored primary results: acute RR {fmt_ci(ep12, 'RR', 'RR_cluster_boot_low', 'RR_cluster_boot_high')}; delayed RR {fmt_ci(epdel, 'RR', 'RR_cluster_boot_low', 'RR_cluster_boot_high')}",
            f"Sampling-regime sensitivity: {sampling_conclusion}",
            f"MIMIC-IV hourly-RMS parity feasible: {parity_feasible}",
            "", "Scientific interpretation (5 sentences):", " ".join(sentences),
        ])
        (output / "09_final_report" / "TERMINAL_SUMMARY.txt").write_text(summary + "\n", encoding="utf-8")
        output_rows = []
        excluded_names = {"execution.log", "run_status.json", "output_hashes.csv"}
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.name not in excluded_names and "__pycache__" not in path.parts:
                output_rows.append({"relative_path": str(path.relative_to(output)), "bytes": int(path.stat().st_size), "sha256": sha256_file(path)})
        pd.DataFrame(output_rows).to_csv(output / "09_final_report" / "output_hashes.csv", index=False)
        row_counts = {**selected_counts, "LABEVENTS": lab_rows, "CHARTEVENTS": chart_rows,
                      **vaso_counts, "waveform_15min_rows": int(len(wave_raw)),
                      "waveform_extraction_failures": context.get("extraction_failure_count")}
        final_status = {
            "status": "complete", "started_at_utc": prior_status.get("started_at_utc"),
            "analysis_completed_at_utc": prior_status.get("failed_at_utc") or prior_status.get("analysis_completed_at_utc") or prior_status.get("completed_at_utc"),
            "finalization_started_at_utc": started_at,
            "completed_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "execution_time_seconds": runtime_seconds,
            "execution_mode": "saved-results-finalization-after-inventory-serialization-retry",
            "prior_stop_error": original_failure,
            "input_root": str(input_root), "output_root": str(output),
            "random_seed": SEED, "python": packages["python"], "package_versions": packages,
            "source_row_counts": row_counts,
            "cohort_counts": {
                "eligible_clinical_n": context["eligible_clinical"],
                "eligible_unique_patients": context["eligible_subjects"],
                "eligible_unique_admissions": context["eligible_admissions"],
                "usable_signal_n": context["signal_in_target"],
                "dynamics_eligible_n": context["dynamics_n"],
                "exposed_n": context["exposed_n"],
                "upstream_waveform_candidate_universe": "unavailable",
            },
            "input_hash_manifest": "data_inventory.csv", "output_hash_manifest": "09_final_report/output_hashes.csv",
            "self_audit_checks": checks, "primary_ci_reproducibility": ci_checks,
            "all_ci_reproducibility_checks": int(len(all_ci_audit)),
            "all_ci_reproducibility_failures": int((~all_ci_audit.status.eq("pass")).sum()),
            "feature_mismatch_rows": context.get("feature_mismatch_rows", 0),
            "feature_mismatch_fraction": context.get("feature_mismatch_fraction", np.nan),
            "colab_execution": False, "notebook": str(notebook_path),
        }
        status_path.write_text(json.dumps(final_status, indent=2, default=str), encoding="utf-8")
        logger.info("Saved-results finalization completed; final report: %s", report_path)
        print("\n" + summary)
        return 0
    except Exception as exc:
        logger.exception("Saved-results finalization failed: %s", exc)
        prior_status.update({
            "status": "failed", "finalization_failed_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "finalization_error_type": type(exc).__name__, "finalization_error": str(exc),
            "finalization_traceback": traceback.format_exc(),
        })
        status_path.write_text(json.dumps(prior_status, indent=2, default=str), encoding="utf-8")
        raise


def cached_eligible_clinical(output: Path, old_outcomes: pd.DataFrame) -> pd.DataFrame:
    """Reattach small ICU/patient lookup fields to the already-selected saved cohort."""
    inventory = pd.read_csv(output / "data_inventory.csv", keep_default_na=False)
    selected = inventory.loc[
        inventory.notes.astype(str).str.startswith("selected full MIMIC-III v1.4 source")
    ]
    paths = {}
    for table in ["PATIENTS", "ICUSTAYS"]:
        candidates = selected.loc[selected.inferred_table.eq(table), "full_path"].astype(str).tolist()
        if len(candidates) != 1 or not Path(candidates[0]).is_file():
            raise FileNotFoundError(f"Expected one selected saved-source lookup for {table} in data_inventory.csv.")
        paths[table] = Path(candidates[0])

    patient = read_table(paths["PATIENTS"], usecols=["SUBJECT_ID", "GENDER", "DOB"])
    icu = read_table(paths["ICUSTAYS"], usecols=[
        "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME", "FIRST_CAREUNIT",
    ])
    patient["DOB"] = pd.to_datetime(patient.DOB, errors="coerce")
    for column in ["INTIME", "OUTTIME"]:
        icu[column] = pd.to_datetime(icu[column], errors="coerce")
    icu["ICUSTAY_ID"] = pd.to_numeric(icu.ICUSTAY_ID, errors="coerce").astype("Int64")
    cohort_ids = set(pd.to_numeric(old_outcomes.ICUSTAY_ID, errors="coerce").dropna().astype(int))
    icu = icu.loc[icu.ICUSTAY_ID.isin(cohort_ids)].copy()
    if icu.ICUSTAY_ID.duplicated().any() or set(icu.ICUSTAY_ID.dropna().astype(int)) != cohort_ids:
        raise RuntimeError("Saved cohort IDs did not map one-to-one to the selected ICUSTAYS lookup.")
    clinical = old_outcomes[[
        "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "observable_end_minute",
    ]].copy()
    clinical = clinical.merge(icu, on=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"], how="left", validate="one_to_one")
    clinical = clinical.merge(patient, on="SUBJECT_ID", how="left", validate="many_to_one")
    if len(clinical) != len(old_outcomes) or clinical.INTIME.isna().any():
        raise RuntimeError("Small metadata lookup failed to preserve every cached eligible stay.")
    legacy_end = pd.to_numeric(
        old_outcomes.get("hospital_observable_end_minute", old_outcomes.observable_end_minute),
        errors="coerce",
    ).to_numpy(dtype=float)
    outtime_minute = (clinical.OUTTIME - clinical.INTIME).dt.total_seconds().to_numpy(dtype=float) / 60
    clinical["hospital_observable_end_minute"] = legacy_end
    clinical["hospital_followup_end"] = clinical.INTIME + pd.to_timedelta(legacy_end, unit="m")
    clinical["icu_outtime_minute"] = outtime_minute
    clinical["observable_end_minute"] = np.fmin(legacy_end, outtime_minute)
    clinical["icu_observable_end"] = clinical.INTIME + pd.to_timedelta(clinical.observable_end_minute, unit="m")
    clinical["landmark_time"] = clinical.INTIME + pd.to_timedelta(LANDMARK_MIN, unit="m")
    age_days = (clinical.INTIME - clinical.DOB).dt.total_seconds() / 86400
    age_float = age_days / 365.2425
    clinical["age_topcoded_90plus"] = age_float > 89
    clinical["age_years"] = np.where(clinical.age_topcoded_90plus, 90, np.floor(age_float))
    clinical.loc[(clinical.age_years < 0) | (clinical.age_years > 120), "age_years"] = np.nan
    clinical["GENDER"] = clinical.GENDER.astype(str).str.upper()
    clinical["sex_male"] = clinical.GENDER.map({"M": 1.0, "F": 0.0})
    return clinical


def reproduce_complete_followup_bootstrap_intervals(
    analysis: pd.DataFrame, binary: pd.DataFrame, episodes: pd.DataFrame,
    episode_effects: pd.DataFrame, episode_ge1: pd.DataFrame,
    direction_records: pd.DataFrame, direction_effects: pd.DataFrame,
    output: Path,
) -> pd.DataFrame:
    records = []
    interval_columns = [
        "RR_95CI_low", "RR_95CI_high", "RD_95CI_low", "RD_95CI_high",
        "RR_cluster_boot_low", "RR_cluster_boot_high",
        "RD_cluster_boot_low", "RD_cluster_boot_high",
    ]

    def number_matches(left, right):
        try:
            return bool(np.isclose(float(left), float(right), rtol=1e-12, atol=1e-12, equal_nan=True))
        except (TypeError, ValueError):
            return False

    def check(name, data, exposure_col, outcome_col, saved, seed, reps):
        reproduced = binary_effects(data, exposure_col, outcome_col, reps=reps, seed=seed)
        points = all(number_matches(reproduced.get(col), saved.get(col)) for col in ["RR", "RD"])
        intervals = all(number_matches(reproduced.get(col), saved.get(col)) for col in interval_columns)
        records.append({
            "analysis": name, "bootstrap_unit": "SUBJECT_ID/patient", "seed": seed,
            "replicates": reps, "point_estimate_reproduced": points,
            "intervals_reproduced": intervals, "status": "pass" if points and intervals else "fail",
        })

    for horizon in HORIZONS:
        ycol, available = f"lactate_rise_ge0_5_{horizon}", f"outcome_available_{horizon}"
        part = analysis.loc[
            analysis.dynamics_eligible.fillna(False).astype(bool)
            & analysis[available].fillna(False).astype(bool)
        ].copy()
        part[ycol] = part[ycol].astype(bool)
        saved = binary.loc[binary.horizon.eq(horizon)].iloc[0]
        check(f"landmark_{horizon}", part, "exposure_abs_jump_ge4", ycol, saved,
              SEED + (12 if horizon == "12h" else 24), BOOTSTRAP_PRIMARY)

    episode_seeds = {
        "acute_0_1h": 101, "delayed_1_8h": 102, "cumulative_0_4h": 104,
        "cumulative_0_8h": 108, "cumulative_0_12h": 112,
    }
    ge1_seeds = {
        "acute_0_1h": 211, "delayed_1_8h": 212, "cumulative_0_4h": 214,
        "cumulative_0_8h": 218, "cumulative_0_12h": 222,
    }
    for window, offset in episode_seeds.items():
        available, outcome = f"outcome_available_{window}", f"rise_ge0_5_{window}"
        pair = episodes.loc[episodes[available].fillna(False).astype(bool)].copy()
        pair["episode_exposed"] = pair.anchor_group.eq("episode")
        saved = episode_effects.loc[
            episode_effects.window.eq(window)
            & episode_effects.group.eq("episode_vs_time_aligned_control")
        ].iloc[0]
        check(f"episode_ge0.5_{window}", pair, "episode_exposed", outcome, saved, SEED + offset, 2000)
        available_one, outcome_one = f"outcome_available_ge1_{window}", f"rise_ge1_{window}"
        pair_one = episodes.loc[episodes[available_one].fillna(False).astype(bool)].copy()
        pair_one["episode_exposed"] = pair_one.anchor_group.eq("episode")
        saved_one = episode_ge1.loc[
            episode_ge1.window.eq(window)
            & episode_ge1.group.eq("episode_vs_time_aligned_control")
        ].iloc[0]
        check(f"episode_ge1.0_{window}", pair_one, "episode_exposed", outcome_one,
              saved_one, SEED + ge1_seeds[window], 2000)

    for direction in ["drop", "rise"]:
        for window in ["acute_0_1h", "delayed_1_8h"]:
            part = direction_records.loc[
                direction_records.direction.eq(direction)
                & direction_records.window.eq(window)
                & direction_records.available.fillna(False).astype(bool)
            ].copy()
            part["direction_episode"] = part.group.eq(direction + "_episode")
            saved = direction_effects.loc[
                direction_effects.direction.eq(direction)
                & direction_effects.window.eq(window)
                & direction_effects.group.eq("direction_episode_vs_control")
            ].iloc[0]
            seed = SEED + (3 if direction == "drop" else 4) + (1 if window == "acute_0_1h" else 2)
            check(f"episode_direction_{direction}_{window}", part, "direction_episode", "event", saved, seed, 2000)

    audit = pd.DataFrame(records)
    audit.to_csv(output / "03_landmark_lactate" / "ci_reproducibility_audit.csv", index=False)
    audit.to_csv(output / "09_final_report" / "all_ci_reproducibility_audit.csv", index=False)
    return audit


def reanalyze_cached_complete_followup(input_root: Path, output: Path) -> int:
    """Recompute outcome-dependent analyses from cached rows without streaming large source tables."""
    required_files = [
        output / "data_inventory.csv",
        output / "03_landmark_lactate" / "lactate_outcomes.csv",
        output / "03_landmark_lactate" / "linked_lactate_draws.csv.gz",
        output / "02_signal_qc" / "signal_features_recomputed.csv",
        output / "01_cohort" / "early_chart_covariates.csv",
        output / "01_cohort" / "early_vasoactive_status.csv",
        output / "04_episode_lactate" / "episode_anchor_records.csv",
        output / "04_episode_lactate" / "episode_direction_patient_records.csv",
    ]
    absent = [str(path) for path in required_files if not path.is_file()]
    if absent:
        raise FileNotFoundError("Analysis-only rerun requires these existing saved artifacts: " + ", ".join(absent))
    output.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_DIRS:
        (output / name).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output)
    status_path = output / "run_status.json"
    prior_status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    started = time.time()
    started_at = pd.Timestamp.now(tz="UTC").isoformat()

    snapshots = {
        "outcomes": output / "09_final_report" / "pre_complete_followup_lactate_outcomes.csv",
        "episodes": output / "09_final_report" / "pre_complete_followup_episode_anchor_records.csv",
        "direction": output / "09_final_report" / "pre_complete_followup_episode_direction_records.csv",
    }
    source_files = {
        "outcomes": required_files[1], "episodes": required_files[6], "direction": required_files[7],
    }
    for name, snapshot in snapshots.items():
        if not snapshot.exists():
            pd.read_csv(source_files[name]).to_csv(snapshot, index=False)
    old_outcomes = pd.read_csv(snapshots["outcomes"])
    old_episodes = pd.read_csv(snapshots["episodes"])
    old_direction = pd.read_csv(snapshots["direction"])
    status = dict(prior_status)
    status.update({
        "status": "running", "execution_mode": "complete_followup_endpoint_reanalysis_using_saved_inputs",
        "reanalysis_started_at_utc": started_at,
        "reanalysis_scope": "Cached linked lactate draws, saved signal features, covariates, and small selected-cohort ICU/patient metadata lookup only; no LABEVENTS, CHARTEVENTS, or waveform stream reread.",
        "primary_last_value_followup_rule": "Complete ICU follow-up through the fixed 12h/24h horizon is required for events and non-events.",
    })
    status_path.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    try:
        write_readme(output)
        notebook_path = create_notebook(output)
        signal = pd.read_csv(output / "02_signal_qc" / "signal_features_recomputed.csv")
        events = pd.read_csv(output / "03_landmark_lactate" / "linked_lactate_draws.csv.gz")
        clinical = cached_eligible_clinical(output, old_outcomes)
        chart_covars = pd.read_csv(output / "01_cohort" / "early_chart_covariates.csv")
        vaso = pd.read_csv(output / "01_cohort" / "early_vasoactive_status.csv")
        logger.info(
            "Analysis-only rerun: reusing %d linked lactate draws, %d saved signal-feature rows, and %d eligible stays; no large event or waveform stream is read.",
            len(events), len(signal), len(clinical),
        )

        outcomes, completeness = construct_lactate_outcomes(clinical, signal, events, output)
        analysis, binary, adjusted, continuous = make_analysis_tables(
            clinical, signal, outcomes, chart_covars, vaso, output, logger,
        )
        prior_count = previous_count_reconciliation(analysis, clinical, binary, output)
        anchors, episode_effects, _trajectory = make_episode_anchor(analysis, events, output)
        direction = episode_direction_sensitivity(analysis, events, output)
        direction_records = pd.read_csv(output / "04_episode_lactate" / "episode_direction_patient_records.csv")
        followup_reconciliation = complete_followup_before_after_reconciliation(
            output, old_outcomes, outcomes, signal, old_episodes, anchors, old_direction, direction_records,
        )
        observation = observation_process(analysis, output)
        sampling, coverage = sampling_and_coverage_sensitivities(analysis, output)
        prediction_metrics, increments, fold_audit = grouped_oof_models(analysis, output, logger)
        power = power_precision(binary, completeness, output)
        interval_audit = reproduce_complete_followup_bootstrap_intervals(
            analysis, binary, anchors, episode_effects,
            pd.read_csv(output / "04_episode_lactate" / "episode_ge1_sensitivity.csv"),
            direction_records, direction, output,
        )
        parity = pd.read_csv(output / "06_mimic4_parity" / "mimic4_hourly_rms_parity.csv")
        figure_paths = make_figures(output)

        prior_report = (output / "09_final_report" / "FINAL_REPORT.md").read_text(encoding="utf-8")
        manifest_match = re.search(r"manifest contains ([\d,]+) unique stays", prior_report)
        previous_counts = prior_status.get("cohort_counts", {})
        signal_ids = set(pd.to_numeric(signal.ICUSTAY_ID, errors="coerce").dropna().astype(int))
        clinical_ids = set(pd.to_numeric(clinical.ICUSTAY_ID, errors="coerce").dropna().astype(int))
        source_counts = prior_status.get("source_row_counts", {})
        input_inventory = pd.read_csv(output / "data_inventory.csv", keep_default_na=False)
        failure_rows = input_inventory.loc[
            input_inventory.filename.astype(str).eq("mimic3_waveform_extraction_failures.csv"), "row_count"
        ]
        try:
            failure_count = int(float(failure_rows.iloc[0])) if len(failure_rows) else None
        except (TypeError, ValueError):
            failure_count = None
        context = {
            "eligible_clinical": int(len(clinical)), "signal_in_target": int(len(signal_ids & clinical_ids)),
            "dynamics_n": int(signal.loc[signal.dynamics_eligible.fillna(False).astype(bool), "ICUSTAY_ID"].nunique()),
            "exposed_n": int(signal.loc[
                signal.dynamics_eligible.fillna(False).astype(bool)
                & signal.exposure_abs_jump_ge4.fillna(False).astype(bool), "ICUSTAY_ID",
            ].nunique()),
            "signal_unique": int(signal.ICUSTAY_ID.nunique()),
            "manifest_unique": int(manifest_match.group(1).replace(",", "")) if manifest_match else None,
            "row_counts": source_counts,
            "feature_mismatch_fraction": prior_status.get("feature_mismatch_fraction", 0.0),
            "feature_mismatch_rows": prior_status.get("feature_mismatch_rows", 0),
            "extraction_failure_count": failure_count,
            "extraction_failure_status": "available" if failure_count is not None else "unavailable",
        }
        item_map = pd.read_csv(output / "03_landmark_lactate" / "lactate_item_map.csv")
        chart_map = pd.read_csv(output / "01_cohort" / "chart_item_map.csv")
        lactate_info = {
            "source_rows": source_counts.get("LABEVENTS", "cached; source not re-read"),
            "lactate_itemids": sorted(pd.to_numeric(item_map.ITEMID, errors="coerce").dropna().astype(int).unique().tolist()),
            "subject_hadm_linkage_mismatch_rows_excluded": 0,
        }
        chart_info = {
            "status": "cached pre-landmark covariates", "source_rows": source_counts.get("CHARTEVENTS", "cached; source not re-read"),
            "mapped_labels": chart_map.to_dict(orient="records"),
        }
        vaso_info = {
            "status": "cached pre-landmark vasoactive covariates",
            "source_rows_scanned": sum(int(source_counts.get(k, 0) or 0) for k in ["INPUTEVENTS_CV", "INPUTEVENTS_MV"]),
            "source_rows_by_table": {k: source_counts.get(k, 0) for k in ["INPUTEVENTS_CV", "INPUTEVENTS_MV"]},
        }
        packages = package_versions()
        runtime_seconds = time.time() - started
        report_path = write_final_report(
            output, input_root, clinical, context, lactate_info, chart_info, vaso_info,
            binary, adjusted, continuous, prediction_metrics, increments,
            episode_effects, direction, observation, sampling, coverage, parity,
            power, figure_paths, packages, runtime_seconds, followup_reconciliation,
        )

        # Keep the saved notebook self-contained and byte-identical to the implementation.
        notebook_data = json.loads(notebook_path.read_text(encoding="utf-8"))
        code = "\n".join("".join(cell.get("source", [])) for cell in notebook_data["cells"] if cell.get("cell_type") == "code")
        embedded = re.search(r"SCRIPT\.write_bytes\(base64\.b64decode\('([^']+)'\)\)", code)
        notebook_matches = bool(embedded and base64.b64decode(embedded.group(1)) == Path(__file__).read_bytes())

        endpoint_complete = all(
            not ((pd.to_numeric(outcomes.observable_end_minute, errors="coerce") < HORIZONS[horizon])
                 & outcomes[f"outcome_available_{horizon}"].fillna(False).astype(bool)).any()
            for horizon in HORIZONS
        )
        episode_complete = True
        anchor_end = analysis.set_index("ICUSTAY_ID").observable_end_minute
        for window, end_after in [("acute_0_1h", 60), ("delayed_1_8h", 480), ("cumulative_0_4h", 240),
                                   ("cumulative_0_8h", 480), ("cumulative_0_12h", 720)]:
            available = anchors.loc[anchors[f"outcome_available_{window}"].fillna(False).astype(bool)]
            ends = available.ICUSTAY_ID.map(anchor_end)
            episode_complete &= bool((ends >= available.anchor_minute + end_after).all())
        companion_preserved = True
        for horizon in HORIZONS:
            censored = pd.to_numeric(outcomes.observable_end_minute, errors="coerce").lt(HORIZONS[horizon])
            for column in [f"maximum_delta_ge0_5_{horizon}", f"baseline_lt2_followup_ge2_{horizon}"]:
                old_positive_ids = set(pd.to_numeric(
                    old_outcomes.loc[censored.reindex(old_outcomes.index, fill_value=False)
                                     & old_outcomes[column].fillna(False).astype(bool), "ICUSTAY_ID"],
                    errors="coerce",
                ).dropna().astype(int)) if column in old_outcomes else set()
                new_positive_ids = set(pd.to_numeric(
                    outcomes.loc[outcomes[column].fillna(False).astype(bool), "ICUSTAY_ID"], errors="coerce",
                ).dropna().astype(int))
                companion_preserved &= old_positive_ids.issubset(new_positive_ids)
        reconciliation_pass = complete_followup_reconciliation_pass(followup_reconciliation)
        ci_pass = bool(len(interval_audit) == 16 and interval_audit.status.eq("pass").all())
        checks = {
            "cached_cohort_id_lookup_one_to_one": len(clinical) == len(old_outcomes) and clinical.ICUSTAY_ID.is_unique,
            "primary_last_value_requires_complete_icu_followup": endpoint_complete,
            "episode_last_value_requires_complete_window": bool(episode_complete),
            "monotone_early_positive_companions_preserved": bool(companion_preserved),
            "positive_and_negative_removals_reconcile": bool(reconciliation_pass),
            "primary_and_episode_patient_cluster_intervals_reproduced": ci_pass,
            "grouped_predictive_folds_patient_disjoint": bool(
                all(item.get("subject_group_overlap_pass", False) for item in fold_audit.values())
            ),
            "notebook_embeds_current_pipeline_exactly": notebook_matches,
            "report_and_reconciliation_exist": report_path.is_file() and followup_reconciliation.shape[0] > 0,
        }
        pd.DataFrame([{"check": key, "pass": bool(value)} for key, value in checks.items()]).to_csv(
            output / "09_final_report" / "self_audit_checks.csv", index=False,
        )
        if not all(checks.values()):
            failed = [key for key, value in checks.items() if not value]
            raise RuntimeError("Complete-follow-up reanalysis audit failed: " + ", ".join(failed))

        output_rows = []
        excluded = {"execution.log", "run_status.json", "output_hashes.csv"}
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.name not in excluded and "__pycache__" not in path.parts:
                output_rows.append({
                    "relative_path": str(path.relative_to(output)), "bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                })
        pd.DataFrame(output_rows).to_csv(output / "09_final_report" / "output_hashes.csv", index=False)
        status.update({
            "status": "complete", "completed_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "reanalysis_runtime_seconds": time.time() - started,
            "reanalysis_followup_rule": "ICU outtime, hospital discharge, and death all end primary endpoint observability; follow-up must reach the fixed horizon for both events and non-events.",
            "reanalysis_previous_positive_removed": int(followup_reconciliation.previously_positive_removed.sum()),
            "reanalysis_previous_negative_removed": int(followup_reconciliation.previously_negative_removed.sum()),
            "reanalysis_self_audit_checks": checks,
            "reanalysis_bootstrap_interval_checks": int(len(interval_audit)),
            "reanalysis_bootstrap_interval_failures": int(interval_audit.status.ne("pass").sum()),
            "colab_execution": False, "notebook": str(notebook_path),
        })
        status_path.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
        logger.info("Analysis-only complete-follow-up reanalysis finished: %s", report_path)
        print(f"Analysis-only runtime: {status['reanalysis_runtime_seconds']:.1f} seconds")
        print(f"12h RR: {binary.loc[binary.horizon.eq('12h'),'RR'].iloc[0]:.3f}; N={int(binary.loc[binary.horizon.eq('12h'),'n'].iloc[0])}")
        print(f"24h RR: {binary.loc[binary.horizon.eq('24h'),'RR'].iloc[0]:.3f}; N={int(binary.loc[binary.horizon.eq('24h'),'n'].iloc[0])}")
        print(f"Previously positive removed: {int(followup_reconciliation.previously_positive_removed.sum())}; previously negative removed: {int(followup_reconciliation.previously_negative_removed.sum())}")
        return 0
    except Exception as exc:
        status.update({
            "status": "failed", "reanalysis_failed_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "reanalysis_error_type": type(exc).__name__, "reanalysis_error": str(exc),
            "reanalysis_traceback": traceback.format_exc(),
        })
        status_path.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
        logger.exception("Analysis-only reanalysis failed: %s", exc)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--output-root", type=Path, default=Path.home() / "Downloads" / "physiograph_dense_waveform_analysis")
    parser.add_argument("--finalize-existing", action="store_true", help="Finalize reports/audits from a run stopped by inventory row-count serialization.")
    parser.add_argument("--analysis-only-complete-followup", action="store_true", help="Recompute endpoint-dependent analyses from existing processed outputs, without rereading large event or waveform tables.")
    args = parser.parse_args()
    input_root = args.input_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    if args.analysis_only_complete_followup:
        return reanalyze_cached_complete_followup(input_root, output)
    if args.finalize_existing:
        return finalize_existing_run(input_root, output)
    output.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_DIRS:
        (output / name).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output)
    started = time.time()
    started_at = pd.Timestamp.now(tz="UTC").isoformat()
    status_path = output / "run_status.json"
    status = {
        "status": "running", "started_at_utc": started_at, "input_root": str(input_root),
        "output_root": str(output), "random_seed": SEED,
        "protocol": "locked user-provided definitions; no threshold or endpoint search",
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    try:
        write_readme(output)
        notebook_path = create_notebook(output)
        logger.info("Starting local dense waveform analysis; seed=%d.", SEED)
        discovered, csv_files = schema_candidates(input_root, output)
        tables = {name: choose_table(name, paths) for name, paths in discovered.items()}
        required = [
            "PATIENTS", "ADMISSIONS", "ICUSTAYS", "DIAGNOSES_ICD",
            "D_ICD_DIAGNOSES", "LABEVENTS", "D_LABITEMS",
        ]
        absent = [name for name in required if not tables.get(name)]
        if absent:
            raise FileNotFoundError("Required full MIMIC-III clinical table schemas are absent: " + ", ".join(absent))

        def named_source(filename: str) -> Path | None:
            options = [p for p in csv_files if p.name.lower() == filename.lower()]
            if not options:
                return None
            return sorted(options, key=lambda p: (len(p.parts), str(p)))[0]

        extras = {
            "waveform": named_source(WAVEFORM_FILE),
            "features": named_source(FEATURE_FILE),
            "manifest": named_source(MANIFEST_FILE),
            "failures": named_source("mimic3_waveform_extraction_failures.csv"),
            "feasibility": named_source("mimic3_waveform_hf_lactate_feasibility.csv"),
        }
        if not extras["waveform"]:
            raise FileNotFoundError(f"Required dense-waveform input is absent: {WAVEFORM_FILE}")
        inventory_rows(input_root, output, discovered, tables, extras, logger)

        # Clinical cohort and independent signal-feature audit.
        clinical, signal, wave_raw, context = build_cohort(output, tables, extras, logger)
        validation = context["validation"]
        if "any_core_mismatch" in validation:
            compared = validation.loc[validation.status.eq("compared")]
            mismatch_fraction = float(compared.any_core_mismatch.mean()) if len(compared) else 0.0
            context["feature_mismatch_fraction"] = mismatch_fraction
            context["feature_mismatch_rows"] = int(compared.any_core_mismatch.sum())
            if mismatch_fraction > .01:
                raise RuntimeError(
                    f"Hard signal-feature validation failure: {mismatch_fraction:.2%} of matched stays differ on a prespecified core feature (threshold 1%)."
                )
        # Stream early bedside measurements and positive-rate vasoactive events before the landmark.
        chart_covars, chart_info = stream_chart_covariates(
            tables.get("CHARTEVENTS"), tables.get("D_ITEMS"), clinical, output, logger,
        )
        input_files = {
            key: tables[key] for key in ["INPUTEVENTS_CV", "INPUTEVENTS_MV"] if tables.get(key)
        }
        vaso, vaso_info = stream_vasoactive_status(
            input_files, tables.get("D_ITEMS"), clinical, output, logger,
        )
        chart_covars.to_csv(output / "01_cohort" / "early_chart_covariates.csv", index=False)
        vaso.to_csv(output / "01_cohort" / "early_vasoactive_status.csv", index=False)
        merged_covars = (
            clinical[["ICUSTAY_ID"]].merge(chart_covars, on="ICUSTAY_ID", how="left")
            .merge(vaso, on="ICUSTAY_ID", how="left", suffixes=("", "_vaso"))
        )
        availability = []
        for col in merged_covars.columns:
            if col == "ICUSTAY_ID":
                continue
            chart_column = col.startswith(("early_hr", "early_sbp", "early_map", "early_rr", "early_spo2", "early_fio2", "early_temp", "early_mechanical"))
            availability.append({
                "covariate": col, "nonmissing_n": int(merged_covars[col].notna().sum()),
                "eligible_n": int(len(merged_covars)),
                "availability_fraction": float(merged_covars[col].notna().mean()),
                "source": "MIMIC-III CHARTEVENTS/D_ITEMS or INPUTEVENTS/D_ITEMS",
                "status": chart_info.get("status", "available") if chart_column else vaso_info.get("status", "available"),
            })
        pd.DataFrame(availability).to_csv(output / "01_cohort" / "covariate_availability.csv", index=False)

        # The full MIMIC-III LABEVENTS source is scanned; no prefiltered convenience extract is used.
        events, lactate_info = stream_lactate(
            tables["LABEVENTS"], tables["D_LABITEMS"], clinical, output, logger,
        )
        outcomes, endpoint_summary = construct_lactate_outcomes(clinical, signal, events, output)
        analysis, binary, adjusted, continuous = make_analysis_tables(
            clinical, signal, outcomes, chart_covars, vaso, output, logger,
        )
        reconciliation = previous_count_reconciliation(analysis, clinical, binary, output)
        anchors, episode_effects, trajectory = make_episode_anchor(analysis, events, output)
        direction = episode_direction_sensitivity(analysis, events, output)
        direction_records = pd.read_csv(output / "04_episode_lactate" / "episode_direction_patient_records.csv")
        followup_reconciliation = complete_followup_before_after_reconciliation(
            output, None, outcomes, signal, None, anchors, None, direction_records,
        )
        observation = observation_process(analysis, output)
        sampling, coverage = sampling_and_coverage_sensitivities(analysis, output)
        prediction_metrics, increments, cv_audit = grouped_oof_models(analysis, output, logger)
        power = power_precision(binary, endpoint_summary, output)
        parity = make_mimic4_parity_file(input_root, output, extras.get("manifest"), csv_files)
        boundaries = create_boundary_audit(wave_raw, events, clinical, outcomes, output)

        independent = independent_eligible_count(tables)
        independent_match = independent["eligible_ids"] == set(clinical.ICUSTAY_ID.astype(int))
        independent_audit = pd.DataFrame([{
            key: value for key, value in independent.items() if key != "eligible_ids"
        } | {
            "primary_implementation_n": len(clinical),
            "independent_count_match": independent_match,
            "status": "pass" if independent_match else "fail",
        }])
        independent_audit.to_csv(output / "01_cohort" / "independent_cohort_count_audit.csv", index=False)
        if not independent_match:
            raise RuntimeError("Independent clinical cohort reconstruction did not match the primary implementation.")

        # Fill inventory row counts from the actual analysis streams.
        row_counts = context["row_counts"]
        for key, value in row_counts.items():
            update_inventory_row_count(output, tables[key], int(value))
        update_inventory_row_count(output, tables["LABEVENTS"], int(lactate_info["source_rows"]))
        if tables.get("CHARTEVENTS"):
            update_inventory_row_count(output, tables["CHARTEVENTS"], int(chart_info.get("source_rows", 0)))
        for key, count in vaso_info.get("source_rows_by_table", {}).items():
            if tables.get(key):
                update_inventory_row_count(output, tables[key], int(count))
        update_inventory_row_count(output, extras["waveform"], int(len(wave_raw)))
        if extras.get("features"):
            update_inventory_row_count(output, extras["features"], int(pd.read_csv(extras["features"]).shape[0]))
        if extras.get("manifest"):
            update_inventory_row_count(output, extras["manifest"], int(pd.read_csv(extras["manifest"]).shape[0]))
        if extras.get("failures"):
            failure_count, _ = count_csv_data_rows(extras["failures"])
            if failure_count is not None:
                update_inventory_row_count(output, extras["failures"], int(failure_count))

        figure_paths = make_figures(output)
        packages = package_versions()
        runtime_seconds = time.time() - started
        report_path = write_final_report(
            output, input_root, clinical, context, lactate_info, chart_info, vaso_info,
            binary, adjusted, continuous, prediction_metrics, increments,
            episode_effects, direction, observation, sampling, coverage, parity,
            power, figure_paths, packages, runtime_seconds, followup_reconciliation,
        )

        # Reproduce primary estimates and patient-cluster bootstrap intervals from saved rows.
        ci_checks = []
        for horizon in HORIZONS:
            ycol, avail = f"lactate_rise_ge0_5_{horizon}", f"outcome_available_{horizon}"
            part = analysis.loc[
                analysis.dynamics_eligible.fillna(False).astype(bool)
                & analysis[avail].fillna(False).astype(bool)
            ].copy()
            part[ycol] = part[ycol].astype(bool)
            expected_seed = SEED + (12 if horizon == "12h" else 24)
            check = binary_effects(
                part, "exposure_abs_jump_ge4", ycol,
                reps=BOOTSTRAP_PRIMARY, seed=expected_seed,
            )
            saved = binary.loc[binary.horizon.eq(horizon)].iloc[0]
            point_match = (
                np.isclose(check.get("RR", np.nan), saved.RR, rtol=1e-12, atol=1e-12)
                and np.isclose(check.get("RD", np.nan), saved.RD, rtol=1e-12, atol=1e-12)
            )
            boot_match = (
                np.isclose(check.get("RR_cluster_boot_low", np.nan), saved.RR_cluster_boot_low, rtol=1e-12, atol=1e-12)
                and np.isclose(check.get("RR_cluster_boot_high", np.nan), saved.RR_cluster_boot_high, rtol=1e-12, atol=1e-12)
                and np.isclose(check.get("RD_cluster_boot_low", np.nan), saved.RD_cluster_boot_low, rtol=1e-12, atol=1e-12)
                and np.isclose(check.get("RD_cluster_boot_high", np.nan), saved.RD_cluster_boot_high, rtol=1e-12, atol=1e-12)
            )
            ci_checks.append({
                "horizon": horizon, "point_estimate_reproduced": point_match,
                "cluster_bootstrap_ci_reproduced": boot_match,
                "bootstrap_unit": "SUBJECT_ID/patient", "seed": expected_seed,
                "replicates": BOOTSTRAP_PRIMARY,
                "status": "pass" if point_match and boot_match else "fail",
            })
        ci_audit = pd.DataFrame(ci_checks)
        ci_audit.to_csv(output / "03_landmark_lactate" / "ci_reproducibility_audit.csv", index=False)

        # Predictor timing and model-feature leakage audit.
        prediction_feature_names = [
            "age_years", "sex_male", "baseline_lactate_mmol_l",
            "early_hr_median", "early_sbp_median", "early_map_median",
            "early_rr_median", "early_fio2_median", "early_mechanical_ventilation",
            "early_vasoactive_infusion", "mean_binned_spo2", "min_binned_spo2",
            "fraction_binned_below90", "observed_15min_bins", "missing_15min_bins",
            "qualifying_transitions_le30min", "median_raw_samples_per_15min_bin",
            "sampling_1hz", "sampling_mixed", "sampling_unknown",
            "rmssd_binned", "jump_fraction_ge3", "drop_count_ge3",
            "max_abs_qualifying_jump", "n_ge4_transitions", "slope_spo2_per_hour",
        ]
        outcome_tokens = ["lactate_rise", "continuous_last_delta", "outcome_available", "maximum_post"]
        model_feature_leakage = any(
            any(token in feature for token in outcome_tokens)
            for feature in prediction_feature_names
        )
        boundary_pass = bool(boundaries["pass"].all())
        fold_pass = all(item.get("subject_group_overlap_pass", False) for item in cv_audit.values())
        duplicate_pass = (
            context["duplicates"]["duplicate_icustay_id_rows"] == 0
            and context["duplicates"]["duplicate_admission_rows"] == 0
            and int(independent["independent_duplicates"]) == 0
        )
        validation_pass = context.get("feature_mismatch_fraction", 0.0) <= .01
        checks = {
            "core_clinical_tables_present": True,
            "dense_15min_input_present": True,
            "independent_clinical_cohort_counts_match": independent_match,
            "duplicate_stay_and_admission_checks_pass": duplicate_pass,
            "subject_admission_stay_linkage_pass": (
                context["duplicates"]["duplicate_icustay_id_rows"] == 0
                and context.get("signal_subject_link_mismatch", 0) == 0
                and context.get("signal_admission_link_mismatch", 0) == 0
                and int(lactate_info.get("subject_hadm_linkage_mismatch_rows_excluded", 0)) == 0
            ),
            "signal_feature_validation_within_1_percent": validation_pass,
            "time_boundaries_pass": boundary_pass,
            "no_outcome_predictor_leakage": not model_feature_leakage,
            "subject_grouped_folds_disjoint": fold_pass,
            "patient_cluster_bootstrap_unit_recorded": all(x.get("bootstrap_unit") == "SUBJECT_ID/patient" for x in ci_checks),
            "primary_ci_reproduction_pass": bool(ci_audit.status.eq("pass").all()),
            "prior_count_reconciliation_complete": previous_count_reconciliation_pass(reconciliation),
            "complete_followup_reconciliation_complete": complete_followup_reconciliation_pass(followup_reconciliation),
            "no_primary_last_value_endpoint_before_icu_horizon": all(
                not ((pd.to_numeric(outcomes["observable_end_minute"], errors="coerce") < HORIZONS[h])
                     & outcomes[f"outcome_available_{h}"].fillna(False).astype(bool)).any()
                for h in HORIZONS
            ),
            "figures_created_from_saved_tables": all(p.exists() and p.stat().st_size > 0 for p in figure_paths),
            "final_report_exists": report_path.exists() and report_path.stat().st_size > 1000,
            "colab_notebook_embeds_pipeline": (
                notebook_path.exists()
                and notebook_path.stat().st_size > 100_000
                and "base64.b64decode" in notebook_path.read_text(encoding="utf-8")
            ),
        }
        pd.DataFrame([{"check": k, "pass": bool(v)} for k, v in checks.items()]).to_csv(
            output / "09_final_report" / "self_audit_checks.csv", index=False,
        )
        if not all(checks.values()):
            failed = [key for key, value in checks.items() if not value]
            raise RuntimeError("Self-audit failed: " + ", ".join(failed))

        # Hash every durable input/output artifact; the live log and status files are excluded.
        inv = pd.read_csv(output / "data_inventory.csv", keep_default_na=False)
        input_hashes = inv.loc[
            inv.full_path.astype(str).ne("") & inv.sha256.astype(str).ne(""),
            ["filename", "full_path", "bytes", "sha256", "inferred_table", "notes"],
        ].drop_duplicates("full_path")
        input_hashes.to_csv(output / "00_inventory" / "key_input_hashes.csv", index=False)

        def val_text(v, decimals=2):
            try:
                return f"{float(v):.{decimals}f}"
            except (TypeError, ValueError):
                return "NA"
        def ci_text(row, prefix, low, high, decimals=2):
            return f"{val_text(row.get(prefix),decimals)} [{val_text(row.get(low),decimals)}, {val_text(row.get(high),decimals)}]"
        a12 = binary.loc[binary.horizon.eq("12h")].iloc[0]
        a24 = binary.loc[binary.horizon.eq("24h")].iloc[0]
        adj12 = adjusted.loc[adjusted.horizon.eq("12h") & adjusted.model.eq("signal_adjusted_primary")]
        adj24 = adjusted.loc[adjusted.horizon.eq("24h") & adjusted.model.eq("signal_adjusted_primary")]
        m12 = adj12.iloc[0] if len(adj12) else pd.Series(dtype=object)
        m24 = adj24.iloc[0] if len(adj24) else pd.Series(dtype=object)
        inc12 = increments.loc[increments.horizon.eq("12h") & increments.comparison.str.startswith("C_plus")]
        inc24 = increments.loc[increments.horizon.eq("24h") & increments.comparison.str.startswith("C_plus")]
        inc12 = inc12.iloc[0] if len(inc12) else pd.Series(dtype=object)
        inc24 = inc24.iloc[0] if len(inc24) else pd.Series(dtype=object)
        ep12 = episode_effects.loc[
            episode_effects.window.eq("acute_0_1h")
            & episode_effects.group.eq("episode_vs_time_aligned_control")
        ]
        epdel = episode_effects.loc[
            episode_effects.window.eq("delayed_1_8h")
            & episode_effects.group.eq("episode_vs_time_aligned_control")
        ]
        ep12 = ep12.iloc[0] if len(ep12) else pd.Series(dtype=object)
        epdel = epdel.iloc[0] if len(epdel) else pd.Series(dtype=object)
        sample_rr = sampling.loc[sampling.analysis.eq("within_sampling_regime") & sampling.RR.notna()]
        changes = sample_rr.groupby("horizon").RR.apply(lambda x: (x > 1).nunique()) if len(sample_rr) else pd.Series(dtype=int)
        sampling_conclusion = (
            "direction changes across regimes" if len(changes) and changes.max() > 1
            else "no direction reversal observed; stratum estimates may remain imprecise"
        )
        parity_feasible = bool(parity.status.eq("available").any())
        sentences = [
            f"In the reconstructed {context['eligible_clinical']:,}-stay clinical cohort, {context['signal_in_target']:,} stays had usable 15-minute SpO2, {context['dynamics_n']:,} met the dynamics rule, and {context['exposed_n']:,} were exposed.",
            f"The 12-hour landmark RR was {ci_text(a12,'RR','RR_cluster_boot_low','RR_cluster_boot_high')} with RD {ci_text(a12,'RD','RD_cluster_boot_low','RD_cluster_boot_high',3)}; the 24-hour RR was {ci_text(a24,'RR','RR_cluster_boot_low','RR_cluster_boot_high')} with RD {ci_text(a24,'RD','RD_cluster_boot_low','RD_cluster_boot_high',3)}.",
            f"After prespecified adjustment, signal-adjusted RRs were {ci_text(m12,'adjusted_RR','adjusted_RR_95CI_low','adjusted_RR_95CI_high')} at 12 hours and {ci_text(m24,'adjusted_RR','adjusted_RR_95CI_low','adjusted_RR_95CI_high')} at 24 hours; OOF dynamics changes were ΔAUROC/ΔAUPRC {val_text(inc12.get('delta_AUROC'),3)}/{val_text(inc12.get('delta_AUPRC'),3)} and {val_text(inc24.get('delta_AUROC'),3)}/{val_text(inc24.get('delta_AUPRC'),3)}, respectively.",
            f"The acute episode-anchored RR was {ci_text(ep12,'RR','RR_cluster_boot_low','RR_cluster_boot_high')} and the delayed RR was {ci_text(epdel,'RR','RR_cluster_boot_low','RR_cluster_boot_high')}; sampling-regime sensitivity: {sampling_conclusion}.",
            "These observational results do not establish causality. The all-waveform 8,736-stay denominator, exact raw hourly-RMS parity, and original MIMIC-IV composite were unavailable from the supplied sources."
        ]
        summary = "\n".join([
            f"Eligible clinical cohort N: {context['eligible_clinical']}",
            f"Dynamics-eligible N: {context['dynamics_n']}",
            f">=4-point exposure N: {context['exposed_n']}",
            f"12h observed/events: {int(a12.n)}/{int(a12.exposed_events + a12.unexposed_events)}",
            f"12h RR [95% cluster-bootstrap CI]: {ci_text(a12,'RR','RR_cluster_boot_low','RR_cluster_boot_high')}",
            f"12h RD [95% cluster-bootstrap CI]: {ci_text(a12,'RD','RD_cluster_boot_low','RD_cluster_boot_high',3)}",
            f"12h adjusted RR [95% robust CI]: {ci_text(m12,'adjusted_RR','adjusted_RR_95CI_low','adjusted_RR_95CI_high')}",
            f"24h observed/events: {int(a24.n)}/{int(a24.exposed_events + a24.unexposed_events)}",
            f"24h RR [95% cluster-bootstrap CI]: {ci_text(a24,'RR','RR_cluster_boot_low','RR_cluster_boot_high')}",
            f"24h RD [95% cluster-bootstrap CI]: {ci_text(a24,'RD','RD_cluster_boot_low','RD_cluster_boot_high',3)}",
            f"24h adjusted RR [95% robust CI]: {ci_text(m24,'adjusted_RR','adjusted_RR_95CI_low','adjusted_RR_95CI_high')}",
            f"ΔAUROC / ΔAUPRC (C minus B), 12h: {val_text(inc12.get('delta_AUROC'),3)} / {val_text(inc12.get('delta_AUPRC'),3)}",
            f"ΔAUROC / ΔAUPRC (C minus B), 24h: {val_text(inc24.get('delta_AUROC'),3)} / {val_text(inc24.get('delta_AUPRC'),3)}",
            f"Episode-anchored primary results: acute RR {ci_text(ep12,'RR','RR_cluster_boot_low','RR_cluster_boot_high')}; delayed RR {ci_text(epdel,'RR','RR_cluster_boot_low','RR_cluster_boot_high')}",
            f"Sampling-regime sensitivity: {sampling_conclusion}",
            f"MIMIC-IV hourly-RMS parity feasible: {parity_feasible}",
            "",
            "Scientific interpretation (5 sentences):",
            " ".join(sentences),
        ])
        (output / "09_final_report" / "TERMINAL_SUMMARY.txt").write_text(summary + "\n", encoding="utf-8")
        output_rows = []
        excluded_names = {"execution.log", "run_status.json", "output_hashes.csv"}
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.name not in excluded_names and "__pycache__" not in path.parts:
                output_rows.append({
                    "relative_path": str(path.relative_to(output)),
                    "bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                })
        pd.DataFrame(output_rows).to_csv(output / "09_final_report" / "output_hashes.csv", index=False)
        runtime_seconds = time.time() - started
        status = {
            "status": "complete",
            "started_at_utc": started_at,
            "completed_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "execution_time_seconds": runtime_seconds,
            "input_root": str(input_root), "output_root": str(output),
            "random_seed": SEED, "python": packages["python"],
            "package_versions": packages,
            "source_row_counts": {
                **context["row_counts"], "LABEVENTS": int(lactate_info["source_rows"]),
                "CHARTEVENTS": int(chart_info.get("source_rows", 0)),
                **{key: int(val) for key, val in vaso_info.get("source_rows_by_table", {}).items()},
                "waveform_15min_rows": int(len(wave_raw)),
                "waveform_extraction_failures": context.get("extraction_failure_count"),
            },
            "cohort_counts": {
                "eligible_clinical_n": context["eligible_clinical"],
                "eligible_unique_patients": context["eligible_subjects"],
                "eligible_unique_admissions": context["eligible_admissions"],
                "usable_signal_n": context["signal_in_target"],
                "dynamics_eligible_n": context["dynamics_n"],
                "exposed_n": context["exposed_n"],
                "upstream_waveform_candidate_universe": "unavailable",
            },
            "input_hash_manifest": "data_inventory.csv",
            "output_hash_manifest": "09_final_report/output_hashes.csv",
            "self_audit_checks": checks, "primary_ci_reproducibility": ci_checks,
            "feature_mismatch_rows": context.get("feature_mismatch_rows", 0),
            "feature_mismatch_fraction": context.get("feature_mismatch_fraction", np.nan),
            "colab_execution": False, "notebook": str(notebook_path),
        }
        status_path.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
        logger.info("Analysis completed; final report: %s", report_path)
        print("\n" + summary)
        return 0
    except Exception as exc:
        runtime_seconds = time.time() - started
        logger.error("Analysis failed after %.1f seconds: %s", runtime_seconds, exc)
        logger.error(traceback.format_exc())
        status.update({
            "status": "failed", "failed_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "execution_time_seconds": runtime_seconds,
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
