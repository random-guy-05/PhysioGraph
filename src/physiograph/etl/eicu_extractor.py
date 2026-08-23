"""eICU-CRD data extraction: token-based cohort and chunked event streaming.

Uses token-based text matching for diagnoses, infusions, and treatments.
Chunked CSV reading (250K rows per chunk) for lab.csv and vitalPeriodic.csv
to avoid loading multi-GB files entirely into memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .audit import AuditLogger
from .shared import (
    EVENT_REQUIRED_COLUMNS,
    LANDMARK_MINUTES,
    OUTCOME_WINDOW_END_MINUTES,
    SourceExtraction,
    _concat_or_empty,
    _finalize_events,
    _iter_csv_chunks,
    build_death_events,
    build_event_frame,
    event_stay_count,
    normalize_text,
    series_contains_any,
)

if TYPE_CHECKING:
    from typing import Any

# ──────────────────────────────────────────────────────────────────────
# eICU Token Maps and Column Sets
# ──────────────────────────────────────────────────────────────────────

EICU_LAB_NAME_MAP: dict[str, str] = {
    "lactate": "lactate",
    "ph": "ph",
    "creatinine": "creatinine",
    "total bilirubin": "bilirubin_total",
}

EICU_VITAL_COLUMN_MAP: dict[str, str] = {
    "heartrate": "hr",
    "systemicsystolic": "sbp",
    "systemicmean": "map",
    "respiration": "resp_rate",
    "sao2": "spo2",
    "temperature": "temp",
}

EICU_INFUSION_DRUG_TOKENS: list[str] = [
    "norepinephrine",
    "epinephrine",
    "dopamine",
    "dobutamine",
    "milrinone",
    "vasopressin",
    "phenylephrine",
]

VIS_WEIGHTS_EICU: dict[str, float] = {
    "norepinephrine": 100.0,
    "epinephrine": 100.0,
    "dopamine": 1.0,
    "dobutamine": 1.0,
    "milrinone": 10.0,
    "vasopressin": 10000.0,
    "phenylephrine": 0.0, # Not in standard calculation
}

EICU_TREATMENT_TOKENS: list[str] = [
    "intraaortic balloon pump",
    "lvad",
    "bivad",
    "rvad",
]

EICU_DIAGNOSIS_TOKENS: dict[str, list[str]] = {
    "cohort_hf_flag": ["congestive heart failure", "heart failure"],
    "shock_icd_flag": ["cardiogenic shock"],
    "cardiomyopathy_flag": ["cardiomyopathy"],
    "acute_mi_flag": ["acute myocardial infarction"],
}

EICU_COHORT_FLAG_COLUMNS: list[str] = [
    "cohort_hf_flag",
    "shock_icd_flag",
    "cardiomyopathy_flag",
    "acute_mi_flag",
]

EICU_PATIENT_COLUMNS: list[str] = [
    "patientunitstayid",
    "patienthealthsystemstayid",
    "gender",
    "age",
    "hospitaladmitoffset",
    "unitdischargeoffset",
    "unitdischargelocation",
    "unitdischargestatus",
    "hospitaldischargeoffset",
    "hospitaldischargelocation",
    "hospitaldischargestatus",
    "hospitaldischargeyear",
]

CHUNK_SIZE: int = 250_000


# ──────────────────────────────────────────────────────────────────────
# eICU Cohort Helpers
# ──────────────────────────────────────────────────────────────────────


def parse_eicu_age(value: object) -> float:
    """Parse eICU age values, handling deidentification patterns (e.g., '>89').

    Args:
        value: Raw age value from patient.csv.

    Returns:
        Parsed float age, or NaN.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return np.nan
    text = str(value).strip()
    if text.startswith(">"):
        text = text[1:]
    return float(pd.to_numeric(text, errors="coerce"))


def derive_death_offset_minutes(patient_row: pd.Series) -> float | None:
    """Determine the earlier of unit or hospital death in ICU-offset minutes.

    Checks unit and hospital discharge status/location for expired/death
    tokens, returning the smallest positive offset.

    Args:
        patient_row: Row from patients DataFrame.

    Returns:
        Death offset in minutes from ICU admission, or None.
    """
    candidates: list[float] = []
    unit_status = normalize_text(patient_row.get("unitdischargestatus"))
    unit_location = normalize_text(patient_row.get("unitdischargelocation"))
    hospital_status = normalize_text(patient_row.get("hospitaldischargestatus"))
    hospital_location = normalize_text(
        patient_row.get("hospitaldischargelocation")
    )

    if any(
        token in unit_status or token in unit_location
        for token in ["expired", "death"]
    ):
        unit_offset = pd.to_numeric(
            patient_row.get("unitdischargeoffset"), errors="coerce"
        )
        if pd.notna(unit_offset):
            candidates.append(float(unit_offset))
    if any(
        token in hospital_status or token in hospital_location
        for token in ["expired", "death"]
    ):
        hospital_offset = pd.to_numeric(
            patient_row.get("hospitaldischargeoffset"), errors="coerce"
        )
        hospital_admit_offset = pd.to_numeric(
            patient_row.get("hospitaladmitoffset"), errors="coerce"
        )
        if pd.notna(hospital_offset):
            candidates.append(
                float(hospital_offset - hospital_admit_offset)
                if pd.notna(hospital_admit_offset)
                else float(hospital_offset)
            )

    return min(candidates) if candidates else None


def _load_eicu_patients(root: Path, audit: AuditLogger) -> pd.DataFrame:
    """Load the eICU patient table with graceful column handling.

    Only loads columns that exist in the file, filling missing ones with NA.

    Args:
        root: eICU data directory.
        audit: AuditLogger for recording step counts.

    Returns:
        Patients DataFrame.
    """
    available_columns = set(
        pd.read_csv(root / "patient.csv", nrows=0).columns.tolist()
    )
    patients = pd.read_csv(
        root / "patient.csv",
        usecols=[c for c in EICU_PATIENT_COLUMNS if c in available_columns],
    )
    for column in EICU_PATIENT_COLUMNS:
        if column not in patients.columns:
            patients[column] = pd.NA
    audit.log(
        "eicu_patients_loaded",
        row_count=len(patients),
        stay_count=patients["patientunitstayid"].nunique(),
    )
    return patients


def _build_eicu_diagnosis_flags(
    root: Path, patients: pd.DataFrame
) -> pd.DataFrame:
    """Build diagnosis flag columns from diagnosis and admissionDx tables.

    Uses token-based matching to set cohort, shock, cardiomyopathy, and
    acute MI flags based on EICU_DIAGNOSIS_TOKENS.

    Args:
        root: eICU data directory.
        patients: Patients DataFrame.

    Returns:
        DataFrame with stay_id and flag columns.
    """
    diagnosis = pd.read_csv(
        root / "diagnosis.csv",
        usecols=["patientunitstayid", "diagnosisstring"],
    )
    admission_dx = pd.read_csv(
        root / "admissionDx.csv",
        usecols=["patientunitstayid", "admitdxname", "admitdxtext"],
    )

    diagnosis_flags = pd.DataFrame(
        {
            "stay_id": pd.Index(
                patients["patientunitstayid"].astype(int).unique(), dtype=int
            )
        }
    )
    if not diagnosis.empty:
        diagnosis["text"] = diagnosis["diagnosisstring"].fillna("")
    if not admission_dx.empty:
        admission_dx["text"] = (
            admission_dx[["admitdxname", "admitdxtext"]]
            .fillna("")
            .agg(" ".join, axis=1)
        )

    for flag_column, tokens in EICU_DIAGNOSIS_TOKENS.items():
        flagged_stays: set[int] = set()
        if not diagnosis.empty:
            flagged_stays |= set(
                diagnosis.loc[
                    series_contains_any(diagnosis["text"], tokens),
                    "patientunitstayid",
                ]
                .astype(int)
                .tolist()
            )
        if not admission_dx.empty:
            flagged_stays |= set(
                admission_dx.loc[
                    series_contains_any(admission_dx["text"], tokens),
                    "patientunitstayid",
                ]
                .astype(int)
                .tolist()
            )
        diagnosis_flags[flag_column] = (
            diagnosis_flags["stay_id"].isin(flagged_stays).astype(int)
        )
    return diagnosis_flags


def _build_eicu_cohort(
    patients: pd.DataFrame,
    diagnosis_flags: pd.DataFrame,
    audit: AuditLogger,
    *,
    max_stays: int | None,
) -> tuple[pd.DataFrame, list[int]]:
    """Assemble the eICU cohort from patients and diagnosis flags.

    Filters to stays with any diagnosis flag set, computes derived fields
    (age, sex, death offset), and builds the cohort DataFrame.

    Args:
        patients: Patients DataFrame.
        diagnosis_flags: Diagnosis flags DataFrame.
        audit: AuditLogger for recording step counts.
        max_stays: Optional cap on cohort size.

    Returns:
        Tuple of (cohort_df, cohort_ids).
    """
    cohort_ids = (
        diagnosis_flags.loc[
            diagnosis_flags[EICU_COHORT_FLAG_COLUMNS].any(axis=1),
            "stay_id",
        ]
        .astype(int)
        .tolist()
    )
    if max_stays is not None:
        cohort_ids = sorted(cohort_ids)[:max_stays]

    patients = patients.loc[
        patients["patientunitstayid"].isin(cohort_ids)
    ].copy()
    diagnosis_flags = diagnosis_flags.loc[
        diagnosis_flags["stay_id"].isin(cohort_ids)
    ].copy()
    audit.log(
        "eicu_candidate_cohort",
        stay_count=len(cohort_ids),
        details={"max_stays": max_stays},
    )

    patients["age"] = patients["age"].map(parse_eicu_age)
    patients["is_male"] = (
        patients["gender"].fillna("").str.lower().eq("male").astype(int)
    )
    patients["death_offset_minutes"] = patients.apply(
        derive_death_offset_minutes, axis=1
    )

    cohort_df = pd.DataFrame(
        {
            "dataset": "eicu",
            "stay_id": patients["patientunitstayid"].astype(int),
            "person_id": patients["patienthealthsystemstayid"].astype(int),
            "admit_time": pd.NaT,
            "admit_year": pd.to_numeric(
                patients["hospitaldischargeyear"], errors="coerce"
            ),
            "age": patients["age"],
            "is_male": patients["is_male"],
            "cohort_hf_flag": patients["patientunitstayid"]
            .isin(
                diagnosis_flags.loc[
                    diagnosis_flags["cohort_hf_flag"] == 1, "stay_id"
                ]
            )
            .astype(int),
            "shock_icd_flag": patients["patientunitstayid"]
            .isin(
                diagnosis_flags.loc[
                    diagnosis_flags["shock_icd_flag"] == 1, "stay_id"
                ]
            )
            .astype(int),
            "early_icu_flag": 1,
            "death_offset_minutes": patients["death_offset_minutes"],
            "excluded_before_landmark_flag": 0,
            "exclusion_reason": "",
        }
    )
    return cohort_df, cohort_ids


# ──────────────────────────────────────────────────────────────────────
# eICU Event Extraction (Chunked)
# ──────────────────────────────────────────────────────────────────────


def _stream_eicu_infusion_events(
    root: Path,
    cohort_ids: list[int],
    *,
    chunk_size: int,
    max_chunks: int | None,
) -> pd.DataFrame:
    """Stream infusionDrug.csv in chunks, filtering to cohort and vasopressors.

    Matches drug names against EICU_INFUSION_DRUG_TOKENS using
    token-based text matching. Extracts numeric infusion rates.

    Args:
        root: eICU data directory.
        cohort_ids: List of included patientunitstayid values.
        chunk_size: Rows per chunk.
        max_chunks: Optional cap on chunks read.

    Returns:
        Event DataFrame with pressor rows.
    """
    frames: list[pd.DataFrame] = []
    for chunk in _iter_csv_chunks(
        root / "infusionDrug.csv",
        usecols=[
            "patientunitstayid",
            "infusionoffset",
            "drugname",
            "drugrate",
            "infusionrate",
        ],
        chunksize=chunk_size,
        max_chunks=max_chunks,
    ):
        filtered = chunk.loc[
            chunk["patientunitstayid"].isin(cohort_ids)
        ].copy()
        filtered = filtered.loc[
            series_contains_any(filtered["drugname"], EICU_INFUSION_DRUG_TOKENS)
        ].copy()
        if filtered.empty:
            continue
        numeric_rate = pd.to_numeric(
            filtered["drugrate"], errors="coerce"
        ).combine_first(
            pd.to_numeric(filtered["infusionrate"], errors="coerce")
        )

        pressor_events = build_event_frame(
            dataset="eicu",
            stay_id=filtered["patientunitstayid"].astype(int),
            event_family="intervention",
            concept="pressor",
            source_table="infusionDrug.csv",
            raw_name=filtered["drugname"].astype(str),
            offset_minutes=pd.to_numeric(
                filtered["infusionoffset"], errors="coerce"
            ),
            value_numeric=numeric_rate,
            value_text=filtered["drugname"].astype(str),
            unit=pd.NA,
            is_intervention=1,
        )

        # Calculate VIS score
        drug_names_lower = filtered["drugname"].fillna("").astype(str).str.lower()
        vis_weights = pd.Series(0.0, index=filtered.index)
        for token, weight in VIS_WEIGHTS_EICU.items():
            mask = drug_names_lower.str.contains(token, regex=False)
            vis_weights.loc[mask] = weight

        vis_score = numeric_rate * vis_weights

        vis_events = build_event_frame(
            dataset="eicu",
            stay_id=filtered["patientunitstayid"].astype(int),
            event_family="derived",
            concept="vis",
            source_table="infusionDrug.csv",
            raw_name=filtered["drugname"].astype(str),
            offset_minutes=pd.to_numeric(
                filtered["infusionoffset"], errors="coerce"
            ),
            value_numeric=vis_score,
            value_text=pd.NA,
            unit=pd.NA,
            is_intervention=0,
        )
        frames.append(pd.concat([pressor_events, vis_events], ignore_index=True))
    return _concat_or_empty(frames, EVENT_REQUIRED_COLUMNS)

def _stream_eicu_urine_output_events(
    root: Path,
    cohort_ids: list[int],
    *,
    chunk_size: int,
    max_chunks: int | None,
) -> pd.DataFrame:
    """Stream intakeOutput.csv in chunks, filtering to cohort and urine output.

    Args:
        root: eICU data directory.
        cohort_ids: List of included patientunitstayid values.
        chunk_size: Rows per chunk.
        max_chunks: Optional cap on chunks read.

    Returns:
        Event DataFrame with urine output rows.
    """
    intake_path = root / "intakeOutput.csv"
    if not intake_path.exists():
        return pd.DataFrame(columns=EVENT_REQUIRED_COLUMNS)

    frames: list[pd.DataFrame] = []
    for chunk in _iter_csv_chunks(
        intake_path,
        usecols=[
            "patientunitstayid",
            "intakeoutputoffset",
            "cellpath",
            "cellvaluenumeric",
        ],
        chunksize=chunk_size,
        max_chunks=max_chunks,
    ):
        filtered = chunk.loc[
            chunk["patientunitstayid"].isin(cohort_ids)
        ].copy()

        # Filter for urine output
        cellpath_lower = filtered["cellpath"].fillna("").astype(str).str.lower()
        filtered = filtered.loc[
            cellpath_lower.str.contains("urine", regex=False)
        ].copy()

        if filtered.empty:
            continue

        numeric_val = pd.to_numeric(
            filtered["cellvaluenumeric"], errors="coerce"
        )
        filtered["cellvaluenumeric"] = numeric_val
        filtered = filtered.dropna(subset=["cellvaluenumeric", "intakeoutputoffset"])

        if filtered.empty:
            continue

        frames.append(
            build_event_frame(
                dataset="eicu",
                stay_id=filtered["patientunitstayid"].astype(int),
                event_family="vital",
                concept="urine_output",
                source_table="intakeOutput.csv",
                raw_name=filtered["cellpath"].astype(str),
                offset_minutes=pd.to_numeric(
                    filtered["intakeoutputoffset"], errors="coerce"
                ),
                value_numeric=filtered["cellvaluenumeric"],
                value_text=pd.NA,
                unit=pd.NA,
                is_intervention=0,
            )
        )
    return _concat_or_empty(frames)


def _load_eicu_treatment_events(
    root: Path, cohort_ids: list[int]
) -> pd.DataFrame:
    """Extract MCS treatment events via token-based matching on treatment.csv.

    The treatment table is small enough to load entirely.

    Args:
        root: eICU data directory.
        cohort_ids: List of included patientunitstayid values.

    Returns:
        Event DataFrame with MCS rows.
    """
    treatments = pd.read_csv(
        root / "treatment.csv",
        usecols=[
            "patientunitstayid",
            "treatmentoffset",
            "treatmentstring",
        ],
    )
    treatments = treatments.loc[
        treatments["patientunitstayid"].isin(cohort_ids)
    ].copy()
    treatments = treatments.loc[
        series_contains_any(
            treatments["treatmentstring"], EICU_TREATMENT_TOKENS
        )
    ].copy()
    return build_event_frame(
        dataset="eicu",
        stay_id=treatments["patientunitstayid"].astype(int),
        event_family="intervention",
        concept="mcs",
        source_table="treatment.csv",
        raw_name=treatments["treatmentstring"].astype(str),
        offset_minutes=pd.to_numeric(
            treatments["treatmentoffset"], errors="coerce"
        ),
        value_numeric=pd.NA,
        value_text=treatments["treatmentstring"].astype(str),
        unit=pd.NA,
        is_intervention=1,
    )


def _stream_eicu_lab_events(
    root: Path,
    cohort_ids: list[int],
    *,
    chunk_size: int,
    max_chunks: int | None,
) -> pd.DataFrame:
    """Stream lab.csv in chunks, filtering to cohort and mapped lab names.

    Uses normalized text matching against EICU_LAB_NAME_MAP. Lab results
    outside [0, 1680] minutes are excluded.

    Args:
        root: eICU data directory.
        cohort_ids: List of included patientunitstayid values.
        chunk_size: Rows per chunk.
        max_chunks: Optional cap on chunks read.

    Returns:
        Event DataFrame with lab measurement rows.
    """
    normalized_map = {
        normalize_text(key): value
        for key, value in EICU_LAB_NAME_MAP.items()
    }
    normalized_keys = set(normalized_map)
    frames: list[pd.DataFrame] = []
    for chunk in _iter_csv_chunks(
        root / "lab.csv",
        usecols=[
            "patientunitstayid",
            "labresultoffset",
            "labname",
            "labresult",
        ],
        chunksize=chunk_size,
        max_chunks=max_chunks,
    ):
        filtered = chunk.loc[
            chunk["patientunitstayid"].isin(cohort_ids)
        ].copy()
        filtered["labname_normalized"] = filtered["labname"].map(
            normalize_text
        )
        filtered = filtered.loc[
            filtered["labname_normalized"].isin(normalized_keys)
        ].copy()
        if filtered.empty:
            continue
        filtered["concept"] = filtered["labname_normalized"].map(
            normalized_map
        )
        filtered["labresult"] = pd.to_numeric(
            filtered["labresult"], errors="coerce"
        )
        offsets = pd.to_numeric(
            filtered["labresultoffset"], errors="coerce"
        )
        filtered["labresultoffset"] = offsets
        filtered = filtered.dropna(subset=["labresult", "labresultoffset"])
        filtered = filtered.loc[
            (filtered["labresultoffset"] >= 0)
            & (filtered["labresultoffset"] <= 1680)
        ].copy()
        frames.append(
            build_event_frame(
                dataset="eicu",
                stay_id=filtered["patientunitstayid"].astype(int),
                event_family="lab",
                concept=filtered["concept"],
                source_table="lab.csv",
                raw_name=filtered["labname"].astype(str),
                offset_minutes=filtered["labresultoffset"],
                value_numeric=filtered["labresult"],
                value_text=pd.NA,
                unit=pd.NA,
                is_intervention=0,
            )
        )
    return _concat_or_empty(frames)


def _stream_eicu_vital_events(
    root: Path,
    cohort_ids: list[int],
    *,
    chunk_size: int,
    max_chunks: int | None,
) -> pd.DataFrame:
    """Stream vitalPeriodic.csv in chunks, filtering to cohort and vital columns.

    Vitals are melted from wide (column per vital sign) to long format
    (concept/value per row). Only observation-window vitals (0-240 min)
    are retained to limit memory usage.

    Args:
        root: eICU data directory.
        cohort_ids: List of included patientunitstayid values.
        chunk_size: Rows per chunk.
        max_chunks: Optional cap on chunks read.

    Returns:
        Event DataFrame with vital measurement rows.
    """
    # NOTE: vitalPeriodic.csv is ~7.39 GB. NEVER load it fully.
    # This function streams it in 250K-row chunks and filters
    # aggressively to the cohort + observation window.
    vital_columns = [
        "patientunitstayid",
        "observationoffset",
        *EICU_VITAL_COLUMN_MAP.keys(),
    ]
    frames: list[pd.DataFrame] = []
    for chunk in _iter_csv_chunks(
        root / "vitalPeriodic.csv",
        usecols=vital_columns,
        chunksize=chunk_size,
        max_chunks=max_chunks,
    ):
        filtered = chunk.loc[
            chunk["patientunitstayid"].isin(cohort_ids)
        ].copy()
        if filtered.empty:
            continue
        filtered["observationoffset"] = pd.to_numeric(
            filtered["observationoffset"], errors="coerce"
        )
        filtered = filtered.loc[
            (filtered["observationoffset"] >= 0)
            & (filtered["observationoffset"] < 240)
        ].copy()
        melted = filtered.melt(
            id_vars=["patientunitstayid", "observationoffset"],
            value_vars=list(EICU_VITAL_COLUMN_MAP),
            var_name="raw_name",
            value_name="value_numeric",
        )
        melted["value_numeric"] = pd.to_numeric(
            melted["value_numeric"], errors="coerce"
        )
        melted = melted.dropna(subset=["value_numeric"])
        if melted.empty:
            continue
        melted["concept"] = melted["raw_name"].map(EICU_VITAL_COLUMN_MAP)
        frames.append(
            build_event_frame(
                dataset="eicu",
                stay_id=melted["patientunitstayid"].astype(int),
                event_family="vital",
                concept=melted["concept"],
                source_table="vitalPeriodic.csv",
                raw_name=melted["raw_name"].astype(str),
                offset_minutes=melted["observationoffset"],
                value_numeric=melted["value_numeric"],
                value_text=pd.NA,
                unit=pd.NA,
                is_intervention=0,
            )
        )
    return _concat_or_empty(frames)


def extract_eicu(
    root: Path,
    audit: AuditLogger,
    *,
    max_stays: int | None = None,
    max_chunks: int | None = None,
    chunk_size: int = CHUNK_SIZE,
) -> SourceExtraction:
    """Extract eICU-CRD cohort and clinical events via chunked streaming.

    Performs token-based cohort identification, pressor infusion and MCS
    treatment extraction, chunked lab and vital sign streaming, and death
    event construction. Prefilters to the cohort before streaming large
    tables (lab.csv ~2.2 GB, vitalPeriodic.csv ~7.4 GB).

    Args:
        root: Path to eICU data directory.
        audit: AuditLogger for recording step counts.
        max_stays: Optional cap on cohort size.
        max_chunks: Optional cap on chunks per streaming table.
        chunk_size: Rows per chunk for CSV streaming (default: 250,000).

    Returns:
        SourceExtraction with cohort_df and events_df.
    """
    patients = _load_eicu_patients(root, audit)
    diagnosis_flags = _build_eicu_diagnosis_flags(root, patients)
    cohort_df, cohort_ids = _build_eicu_cohort(
        patients, diagnosis_flags, audit, max_stays=max_stays
    )

    infusion_events = _stream_eicu_infusion_events(
        root, cohort_ids, chunk_size=chunk_size, max_chunks=max_chunks
    )
    audit.log(
        "eicu_infusions_streamed",
        row_count=len(infusion_events),
        stay_count=event_stay_count(infusion_events),
    )

    treatment_events = _load_eicu_treatment_events(root, cohort_ids)
    audit.log(
        "eicu_treatments_loaded",
        row_count=len(treatment_events),
        stay_count=event_stay_count(treatment_events),
    )

    uo_events = _stream_eicu_urine_output_events(
        root, cohort_ids, chunk_size=chunk_size, max_chunks=max_chunks
    )
    audit.log(
        "eicu_uo_extracted",
        row_count=len(uo_events),
        stay_count=event_stay_count(uo_events),
    )

    lab_events = _stream_eicu_lab_events(
        root, cohort_ids, chunk_size=chunk_size, max_chunks=max_chunks
    )
    audit.log(
        "eicu_labs_streamed",
        row_count=len(lab_events),
        stay_count=event_stay_count(lab_events),
    )

    vital_events = _stream_eicu_vital_events(
        root, cohort_ids, chunk_size=chunk_size, max_chunks=max_chunks
    )
    audit.log(
        "eicu_vitals_streamed",
        row_count=len(vital_events),
        stay_count=event_stay_count(vital_events),
    )

    death_events = build_death_events(
        cohort_df,
        dataset="eicu",
        source_table="patient.csv",
        raw_name="discharge_status",
    )
    events_df = _finalize_events(
        pd.concat(
            [
                infusion_events,
                treatment_events,
                uo_events,
                lab_events,
                vital_events,
                death_events,
            ],
            ignore_index=True,
        )
    )
    return SourceExtraction(cohort_df=cohort_df, events_df=events_df)
