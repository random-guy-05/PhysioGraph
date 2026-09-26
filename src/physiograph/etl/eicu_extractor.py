"""eICU-CRD data extraction: token-based cohort and chunked event streaming.

Uses token-based text matching for diagnoses, infusions, and treatments.
Chunked CSV reading (250K rows per chunk) for lab.csv and vitalPeriodic.csv
to avoid loading multi-GB files entirely into memory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from physiograph.constants import (
    EICU_DIAGNOSIS_TOKENS as CANONICAL_EICU_DIAGNOSIS_TOKENS,
)
from physiograph.constants import (
    EICU_INFUSION_DRUG_TOKENS as CANONICAL_EICU_INFUSION_DRUG_TOKENS,
)
from physiograph.constants import (
    EICU_LAB_NAME_MAP as CANONICAL_EICU_LAB_NAME_MAP,
)
from physiograph.constants import (
    EICU_TREATMENT_TOKENS as CANONICAL_EICU_TREATMENT_TOKENS,
)
from physiograph.constants import (
    EICU_VITAL_COLUMN_MAP as CANONICAL_EICU_VITAL_COLUMN_MAP,
)

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

# ──────────────────────────────────────────────────────────────────────
# eICU Token Maps and Column Sets
# ──────────────────────────────────────────────────────────────────────

EICU_LAB_NAME_MAP: dict[str, str] = dict(CANONICAL_EICU_LAB_NAME_MAP)
EICU_VITAL_COLUMN_MAP: dict[str, str] = dict(CANONICAL_EICU_VITAL_COLUMN_MAP)
EICU_INFUSION_DRUG_TOKENS: list[str] = list(CANONICAL_EICU_INFUSION_DRUG_TOKENS)
EICU_TREATMENT_TOKENS: list[str] = list(CANONICAL_EICU_TREATMENT_TOKENS)
EICU_DIAGNOSIS_TOKENS: dict[str, list[str]] = {
    flag: list(tokens) for flag, tokens in CANONICAL_EICU_DIAGNOSIS_TOKENS.items()
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
    "uniquepid",
    "hospitalid",
    "wardid",
    "gender",
    "age",
    "ethnicity",
    "unittype",
    "unitadmitsource",
    "unitvisitnumber",
    "unitstaytype",
    "hospitaladmitoffset",
    "unitdischargeoffset",
    "unitdischargelocation",
    "unitdischargestatus",
    "hospitaldischargeoffset",
    "hospitaldischargelocation",
    "hospitaldischargestatus",
    "hospitaldischargeyear",
    "admissionweight",
]

CHUNK_SIZE: int = 250_000

EICU_NURSE_NIBP_MAP: dict[tuple[str, str, str], str] = {
    ("Vital Signs", "Non-Invasive BP", "Non-Invasive BP Systolic"): "sbp",
    ("Vital Signs", "Non-Invasive BP", "Non-Invasive BP Diastolic"): "dbp",
    ("Vital Signs", "Non-Invasive BP", "Non-Invasive BP Mean"): "map",
}


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
    text = text.removeprefix(">")
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
        if pd.notna(hospital_offset):
            # Every eICU event offset, including hospital discharge, is already
            # centered on this unit admission.  Subtracting the (usually
            # negative) hospital-admit offset double-counts pre-ICU time.
            candidates.append(float(hospital_offset))

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
    root: Path,
    patients: pd.DataFrame,
    diagnosis_flags: pd.DataFrame,
    audit: AuditLogger,
    *,
    max_stays: int | None,
    preferred_stay_ids: set[int] | None = None,
) -> tuple[pd.DataFrame, list[int]]:
    """Assemble an adult HF ICU cohort harmonized to the MIMIC phenotype."""
    patients["age"] = patients["age"].map(parse_eicu_age)
    patients["is_male"] = np.where(
        patients["gender"].fillna("").str.lower().eq("male"),
        1.0,
        np.where(
            patients["gender"].fillna("").str.lower().eq("female"),
            0.0,
            np.nan,
        ),
    )
    patients["sex_unknown_flag"] = patients["is_male"].isna().astype(int)
    patients["death_offset_minutes"] = patients.apply(
        derive_death_offset_minutes, axis=1
    )
    merged = patients.merge(
        diagnosis_flags,
        left_on="patientunitstayid",
        right_on="stay_id",
        how="inner",
        validate="one_to_one",
    )
    # eICU unit transfers share ``patienthealthsystemstayid``.  Propagate the
    # encounter phenotype, then retain one ICU anchor per hospital encounter,
    # matching MIMIC's first-ICU-per-admission design while preserving repeat
    # hospitalizations for the same person.
    encounter = merged["patienthealthsystemstayid"].astype("string")
    merged["_encounter_key"] = encounter.where(
        encounter.notna() & encounter.str.strip().ne(""),
        "stay:" + merged["patientunitstayid"].astype("string"),
    )
    for flag in EICU_COHORT_FLAG_COLUMNS:
        merged[flag] = merged.groupby("_encounter_key", sort=False)[flag].transform("max")
    if preferred_stay_ids is not None:
        merged = merged.loc[merged["patientunitstayid"].isin(preferred_stay_ids)].copy()
    merged["_unit_visit_order"] = pd.to_numeric(
        merged["unitvisitnumber"], errors="coerce"
    ).fillna(np.inf)
    merged = (
        merged.sort_values(
            ["_encounter_key", "_unit_visit_order", "patientunitstayid"]
        )
        .drop_duplicates("_encounter_key", keep="first")
        .copy()
    )
    merged["early_icu_flag"] = (
        -pd.to_numeric(merged["hospitaladmitoffset"], errors="coerce")
    ).between(0, 24 * 60, inclusive="both").fillna(False).astype(int)
    merged = merged.loc[
        pd.to_numeric(merged["age"], errors="coerce").ge(18)
        & merged["cohort_hf_flag"].eq(1)
        & (merged["early_icu_flag"].eq(1) | merged["shock_icd_flag"].eq(1))
    ].copy()
    merged = merged.sort_values("patientunitstayid").reset_index(drop=True)
    if max_stays is not None:
        merged = merged.head(max_stays).copy()
    cohort_ids = merged["patientunitstayid"].astype(int).tolist()
    person = merged["uniquepid"].astype("string")
    encounter = merged["patienthealthsystemstayid"].astype("string")
    person = person.where(
        person.notna() & person.str.strip().ne(""),
        "encounter:" + encounter,
    )
    person = person.where(
        person.notna()
        & person.str.strip().ne("")
        & ~person.str.contains(r"<na>$", case=False, regex=True, na=True),
        "stay:" + merged["patientunitstayid"].astype("string"),
    )
    merged["person_id_normalized"] = person
    merged["first_stay_per_person_flag"] = (
        merged.sort_values(["person_id_normalized", "unitvisitnumber", "patientunitstayid"])
        .groupby("person_id_normalized", sort=False)
        .cumcount()
        .eq(0)
        .astype(int)
    )
    merged["followup_end_offset_minutes"] = pd.to_numeric(
        merged["unitdischargeoffset"], errors="coerce"
    )
    audit.log(
        "eicu_candidate_cohort",
        stay_count=len(cohort_ids),
        details={
            "max_stays": max_stays,
            "preferred_stay_sampling": preferred_stay_ids is not None,
            "adult_only": True,
            "phenotype": "heart_failure_and_early_icu_or_cardiogenic_shock",
            "person_identity": "uniquePID",
        },
    )

    cohort_df = pd.DataFrame(
        {
            "dataset": "eicu",
            "stay_id": merged["patientunitstayid"].astype(int),
            "person_id": merged["person_id_normalized"],
            "hospital_encounter_id": merged["patienthealthsystemstayid"],
            "admit_time": pd.NaT,
            "admit_year": pd.to_numeric(
                merged["hospitaldischargeyear"], errors="coerce"
            ),
            "age": merged["age"],
            "is_male": merged["is_male"],
            "sex_unknown_flag": merged["sex_unknown_flag"],
            "race_ethnicity": merged["ethnicity"].fillna("UNKNOWN").astype(str),
            "hospital_id": merged["hospitalid"].astype("string"),
            "ward_id": merged["wardid"].astype("string"),
            "icu_type": merged["unittype"].fillna("UNKNOWN").astype(str),
            "unit_admit_source": merged["unitadmitsource"].fillna("UNKNOWN").astype(str),
            "cohort_hf_flag": merged["cohort_hf_flag"].astype(int),
            "shock_icd_flag": merged["shock_icd_flag"].astype(int),
            "cardiomyopathy_flag": merged["cardiomyopathy_flag"].astype(int),
            "acute_mi_flag": merged["acute_mi_flag"].astype(int),
            "early_icu_flag": merged["early_icu_flag"].astype(int),
            "death_offset_minutes": merged["death_offset_minutes"],
            "unit_discharge_offset_minutes": pd.to_numeric(
                merged["unitdischargeoffset"], errors="coerce"
            ),
            "hospital_discharge_offset_minutes": (
                pd.to_numeric(merged["hospitaldischargeoffset"], errors="coerce")
            ),
            "followup_end_offset_minutes": merged["followup_end_offset_minutes"],
            "first_stay_per_person_flag": merged["first_stay_per_person_flag"],
            "admission_weight_kg": pd.to_numeric(
                merged["admissionweight"], errors="coerce"
            ),
            "mcs_source_available": int((root / "treatment.csv").exists()),
            "pressor_source_available": int((root / "infusionDrug.csv").exists()),
            # Quantitative eICU rates are not assumed standardized.  Pressor
            # presence is retained, but VIS stays unavailable by protocol.
            "vis_source_available": 0,
            "urine_output_source_available": int((root / "intakeOutput.csv").exists()),
            "respiratory_source_available": int(
                (root / "respiratoryCharting.csv").exists()
                or (root / "treatment.csv").exists()
            ),
            "rrt_source_available": int((root / "treatment.csv").exists()),
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
    raw_frames: list[pd.DataFrame] = []
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
        filtered["numeric_rate"] = pd.to_numeric(
            filtered["drugrate"], errors="coerce"
        ).combine_first(
            pd.to_numeric(filtered["infusionrate"], errors="coerce")
        )
        filtered["infusionoffset"] = pd.to_numeric(
            filtered["infusionoffset"], errors="coerce"
        )
        filtered = filtered.loc[
            filtered["infusionoffset"].between(0, OUTCOME_WINDOW_END_MINUTES)
            & filtered["numeric_rate"].gt(0)
        ].copy()
        if not filtered.empty:
            raw_frames.append(filtered)
    if not raw_frames:
        return pd.DataFrame(columns=EVENT_REQUIRED_COLUMNS)
    infusions = pd.concat(raw_frames, ignore_index=True).sort_values(
        ["patientunitstayid", "infusionoffset", "drugname"]
    )
    frames: list[pd.DataFrame] = [
        build_event_frame(
            dataset="eicu",
            stay_id=infusions["patientunitstayid"].astype(int),
            event_family="intervention",
            concept="pressor",
            source_table="infusionDrug.csv",
            raw_name=infusions["drugname"].astype(str),
            offset_minutes=infusions["infusionoffset"],
            value_numeric=infusions["numeric_rate"],
            value_text=(
                infusions["drugname"].astype(str)
                + " | drugrate="
                + infusions["drugrate"].astype(str)
                + " | infusionrate="
                + infusions["infusionrate"].astype(str)
            ),
            unit="unstandardized_eicu_rate",
            is_intervention=0,
        )
    ]
    status_rows: list[dict[str, object]] = []
    initiation_rows: list[pd.Series] = []
    for stay_id, group in infusions.groupby("patientunitstayid", sort=False):
        baseline = group.loc[group["infusionoffset"].le(LANDMARK_MINUTES)]
        status_rows.append(
            {
                "stay_id": int(stay_id),
                "offset_minutes": float(LANDMARK_MINUTES),
                "active": int(not baseline.empty),
                "drugs": "+".join(sorted(baseline["drugname"].astype(str).unique())),
            }
        )
        if baseline.empty:
            post = group.loc[group["infusionoffset"].gt(LANDMARK_MINUTES)]
            if not post.empty:
                initiation_rows.append(post.iloc[0])
    if status_rows:
        status = pd.DataFrame(status_rows)
        frames.append(
            build_event_frame(
                dataset="eicu",
                stay_id=status["stay_id"],
                event_family="intervention_status",
                concept="pressor_active",
                source_table="infusionDrug.csv",
                raw_name=status["drugs"],
                offset_minutes=status["offset_minutes"],
                value_numeric=status["active"],
                value_text=status["drugs"],
                unit="binary",
                is_intervention=0,
            )
        )
    if initiation_rows:
        initiations = pd.DataFrame(initiation_rows)
        frames.append(
            build_event_frame(
                dataset="eicu",
                stay_id=initiations["patientunitstayid"].astype(int),
                event_family="intervention",
                concept="pressor_initiation",
                source_table="infusionDrug.csv",
                raw_name=initiations["drugname"].astype(str),
                offset_minutes=initiations["infusionoffset"],
                value_numeric=1.0,
                value_text=initiations["drugname"].astype(str),
                unit="binary",
                is_intervention=1,
            )
        )
    return _concat_or_empty(frames)


def _load_eicu_treatment_events(
    root: Path,
    cohort_ids: list[int],
    *,
    chunk_size: int,
    max_chunks: int | None,
) -> pd.DataFrame:
    """Extract MCS, RRT, ventilation, airway and oxygen treatment events."""
    path = root / "treatment.csv"
    if not path.exists():
        return pd.DataFrame(columns=EVENT_REQUIRED_COLUMNS)
    selected_chunks: list[pd.DataFrame] = []
    for chunk in _iter_csv_chunks(
        path,
        usecols=[
            "patientunitstayid",
            "treatmentoffset",
            "treatmentstring",
        ],
        chunksize=chunk_size,
        max_chunks=max_chunks,
    ):
        selected = chunk.loc[chunk["patientunitstayid"].isin(cohort_ids)].copy()
        if not selected.empty:
            selected_chunks.append(selected)
    if not selected_chunks:
        return pd.DataFrame(columns=EVENT_REQUIRED_COLUMNS)
    treatments = pd.concat(selected_chunks, ignore_index=True)
    text = treatments["treatmentstring"].fillna("").astype(str)
    normalized = text.map(normalize_text)
    treatments["_normalized_text"] = normalized
    # Device removal/explant/discontinuation documents the end of support, not
    # a new MCS exposure.  Without this exclusion, an IABP-removal row can be
    # misclassified as incident MCS when it is the first matching row retained
    # for a stay.
    mcs_end_documentation = normalized.str.contains(
        r"\bremov(?:al|e|ed|ing)?\b|\bexplant(?:ation|ed|ing)?\b|"
        r"\bdiscontinu(?:e|ed|ation|ing)?\b",
        regex=True,
        na=False,
    )
    masks: list[tuple[str, pd.Series, str, int]] = [
        (
            "mcs",
            series_contains_any(text, EICU_TREATMENT_TOKENS)
            & ~mcs_end_documentation,
            "intervention",
            1,
        ),
        (
            "rrt",
            normalized.str.contains(
                r"\bcrrt\b|\bcvvh\b|\bcvvhd\b|dialysis|hemofiltration|hemodialysis",
                regex=True,
                na=False,
            )
            & ~normalized.str.contains(
                r"catheter|vascular access|access site|dressing",
                regex=True,
                na=False,
            ),
            "intervention_status",
            0,
        ),
        (
            "mechanical_ventilation",
            normalized.str.contains(
                r"\bmechanical ventilation\b",
                regex=True,
                na=False,
            )
            & ~normalized.str.contains(
                r"\bnon invasive ventilation\b|\bventilator weaning\b",
                regex=True,
                na=False,
            ),
            "respiratory_support",
            0,
        ),
        (
            "noninvasive_ventilation",
            normalized.str.contains(r"\bbipap\b|\bcpap\b|noninvasive", regex=True, na=False),
            "respiratory_support",
            0,
        ),
        (
            "airway_device",
            normalized.str.contains(r"intubat|endotracheal|tracheost", regex=True, na=False),
            "respiratory_support",
            0,
        ),
        (
            "oxygen_device",
            normalized.str.contains(
                r"nasal cannula|high flow|hfnc|nonrebreather|oxygen mask|supplemental oxygen",
                regex=True,
                na=False,
            ),
            "respiratory_support",
            0,
        ),
    ]
    frames: list[pd.DataFrame] = []
    offsets = pd.to_numeric(treatments["treatmentoffset"], errors="coerce")
    for concept, mask, event_family, is_intervention in masks:
        selected = treatments.loc[mask].copy()
        if selected.empty:
            continue
        if concept == "mcs":
            # eICU treatment rows are documentation events, not reliable device
            # start/stop intervals.  Repeated charting must not be counted as
            # repeated MCS initiation.  Retain only the first documentation of
            # each device family, and use an explicit landmark status row for
            # devices already documented by hour 4.
            device_text = selected["_normalized_text"].fillna("").astype(str)
            selected["_device_family"] = np.select(
                [
                    device_text.str.contains(
                        r"\biabp\b|\bintra ?aortic balloon\b", regex=True
                    ),
                    device_text.str.contains(r"\bimpella\b", regex=True),
                    device_text.str.contains(r"\becmo\b|extracorporeal membrane", regex=True),
                    device_text.str.contains(r"\blvad\b|left ventricular assist", regex=True),
                    device_text.str.contains(r"\brvad\b|right ventricular assist", regex=True),
                    device_text.str.contains(r"\bbivad\b|biventricular assist", regex=True),
                ],
                ["iabp", "impella", "ecmo", "lvad", "rvad", "bivad"],
                default="other_mcs",
            )
            selected["_offset"] = pd.to_numeric(
                selected["treatmentoffset"], errors="coerce"
            )
            selected = selected.dropna(subset=["_offset"]).sort_values(
                ["patientunitstayid", "_device_family", "_offset"]
            )
            first_device = selected.drop_duplicates(
                ["patientunitstayid", "_device_family"], keep="first"
            )
            initiation = first_device.loc[first_device["_offset"].gt(LANDMARK_MINUTES)]
            if not initiation.empty:
                frames.append(
                    build_event_frame(
                        dataset="eicu",
                        stay_id=initiation["patientunitstayid"].astype(int),
                        event_family="intervention",
                        concept="mcs",
                        source_table="treatment.csv",
                        raw_name=initiation["treatmentstring"].astype(str),
                        offset_minutes=initiation["_offset"],
                        value_numeric=1.0,
                        value_text=(
                            initiation["_device_family"].astype(str)
                            + "|first_documentation"
                        ),
                        unit="binary_documented_initiation_proxy",
                        is_intervention=1,
                    )
                )
            baseline = selected.loc[selected["_offset"].le(LANDMARK_MINUTES)]
            if not baseline.empty:
                status = (
                    baseline.groupby("patientunitstayid", as_index=False)["_device_family"]
                    .agg(lambda values: "+".join(sorted(set(values.astype(str)))))
                )
                frames.append(
                    build_event_frame(
                        dataset="eicu",
                        stay_id=status["patientunitstayid"].astype(int),
                        event_family="intervention_status",
                        concept="mcs_active",
                        source_table="treatment.csv",
                        raw_name=status["_device_family"].astype(str),
                        offset_minutes=pd.Series(float(LANDMARK_MINUTES), index=status.index),
                        value_numeric=1.0,
                        value_text=status["_device_family"].astype(str),
                        unit="binary_documented_by_landmark",
                        is_intervention=0,
                    )
                )
            continue
        frames.append(
            build_event_frame(
                dataset="eicu",
                stay_id=selected["patientunitstayid"].astype(int),
                event_family=event_family,
                concept=concept,
                source_table="treatment.csv",
                raw_name=selected["treatmentstring"].astype(str),
                offset_minutes=offsets.loc[selected.index],
                value_numeric=1.0,
                value_text=selected["treatmentstring"].astype(str),
                unit="binary",
                is_intervention=is_intervention,
            )
        )
    return _concat_or_empty(frames)


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
            "labmeasurenamesystem",
            "labmeasurenameinterface",
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
                unit=(
                    filtered["labmeasurenamesystem"].astype("string").fillna("")
                    + "|"
                    + filtered["labmeasurenameinterface"].astype("string").fillna("")
                ).str.strip("|"),
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


def _stream_eicu_aperiodic_vital_events(
    root: Path,
    cohort_ids: list[int],
    *,
    chunk_size: int,
    max_chunks: int | None,
) -> pd.DataFrame:
    """Stream noninvasive blood pressure from ``vitalAperiodic.csv``.

    ``vitalPeriodic.systemicsystolic`` is predominantly invasive arterial
    pressure and is absent for most eICU stays.  The aperiodic table contains
    cuff blood pressure and is therefore required for an unbiased SBP/MAP
    comparator.  The source is optional so installations with a partial eICU
    export still run, but source availability remains explicit in the manifest.
    """
    path = root / "vitalAperiodic.csv"
    if not path.is_file():
        return pd.DataFrame(columns=EVENT_REQUIRED_COLUMNS)
    value_map = {
        "noninvasivesystolic": "sbp",
        "noninvasivediastolic": "dbp",
        "noninvasivemean": "map",
    }
    frames: list[pd.DataFrame] = []
    for chunk in _iter_csv_chunks(
        path,
        usecols=[
            "patientunitstayid",
            "observationoffset",
            *value_map,
        ],
        chunksize=chunk_size,
        max_chunks=max_chunks,
    ):
        filtered = chunk.loc[chunk["patientunitstayid"].isin(cohort_ids)].copy()
        if filtered.empty:
            continue
        filtered["observationoffset"] = pd.to_numeric(
            filtered["observationoffset"], errors="coerce"
        )
        filtered = filtered.loc[
            filtered["observationoffset"].ge(0)
            & filtered["observationoffset"].lt(LANDMARK_MINUTES)
        ]
        if filtered.empty:
            continue
        melted = filtered.melt(
            id_vars=["patientunitstayid", "observationoffset"],
            value_vars=list(value_map),
            var_name="raw_name",
            value_name="value_numeric",
        )
        melted["value_numeric"] = pd.to_numeric(
            melted["value_numeric"], errors="coerce"
        )
        melted = melted.dropna(subset=["value_numeric"])
        if melted.empty:
            continue
        melted["concept"] = melted["raw_name"].map(value_map)
        frames.append(
            build_event_frame(
                dataset="eicu",
                stay_id=melted["patientunitstayid"].astype(int),
                event_family="vital",
                concept=melted["concept"],
                source_table="vitalAperiodic.csv",
                raw_name=melted["raw_name"],
                offset_minutes=melted["observationoffset"],
                value_numeric=melted["value_numeric"],
                value_text=pd.NA,
                unit="mmHg",
                is_intervention=0,
            )
        )
    return _concat_or_empty(frames)


def _stream_eicu_nurse_bp_events(
    root: Path,
    cohort_ids: list[int],
    *,
    chunk_size: int,
    max_chunks: int | None,
) -> pd.DataFrame:
    """Stream only exact, prospectively validated nurse-charted NIBP labels."""

    path = root / "nurseCharting.csv"
    if not path.is_file():
        return pd.DataFrame(columns=EVENT_REQUIRED_COLUMNS)
    label_columns = [
        "nursingchartcelltypecat",
        "nursingchartcelltypevallabel",
        "nursingchartcelltypevalname",
    ]
    frames: list[pd.DataFrame] = []
    for chunk in _iter_csv_chunks(
        path,
        usecols=[
            "patientunitstayid",
            "nursingchartoffset",
            *label_columns,
            "nursingchartvalue",
        ],
        chunksize=chunk_size,
        max_chunks=max_chunks,
    ):
        filtered = chunk.loc[chunk["patientunitstayid"].isin(cohort_ids)].copy()
        if filtered.empty:
            continue
        filtered["nursingchartoffset"] = pd.to_numeric(
            filtered["nursingchartoffset"], errors="coerce"
        )
        filtered = filtered.loc[
            filtered["nursingchartoffset"].ge(0)
            & filtered["nursingchartoffset"].lt(LANDMARK_MINUTES)
        ].copy()
        if filtered.empty:
            continue
        for column in label_columns:
            filtered[column] = filtered[column].astype("string").str.strip()
        keys = list(filtered[label_columns].itertuples(index=False, name=None))
        filtered["concept"] = [EICU_NURSE_NIBP_MAP.get(key) for key in keys]
        filtered = filtered.loc[filtered["concept"].notna()].copy()
        filtered["value_numeric"] = pd.to_numeric(
            filtered["nursingchartvalue"], errors="coerce"
        )
        filtered = filtered.dropna(subset=["value_numeric"])
        if filtered.empty:
            continue
        raw_name = (
            filtered["nursingchartcelltypevallabel"].astype(str)
            + " | "
            + filtered["nursingchartcelltypevalname"].astype(str)
        )
        frames.append(
            build_event_frame(
                dataset="eicu",
                stay_id=filtered["patientunitstayid"].astype(int),
                event_family="vital",
                concept=filtered["concept"],
                source_table="nurseCharting.csv",
                raw_name=raw_name,
                offset_minutes=filtered["nursingchartoffset"],
                value_numeric=filtered["value_numeric"],
                value_text=pd.NA,
                unit="mmHg",
                is_intervention=0,
            )
        )
    return _concat_or_empty(frames)


def _stream_eicu_respiratory_events(
    root: Path,
    cohort_ids: list[int],
    *,
    chunk_size: int,
    max_chunks: int | None,
) -> pd.DataFrame:
    """Extract structured FiO2, ventilator mode, airway and oxygen context."""
    path = root / "respiratoryCharting.csv"
    if not path.exists():
        return pd.DataFrame(columns=EVENT_REQUIRED_COLUMNS)
    frames: list[pd.DataFrame] = []
    for chunk in _iter_csv_chunks(
        path,
        usecols=[
            "patientunitstayid",
            "respchartoffset",
            "respcharttypecat",
            "respchartvaluelabel",
            "respchartvalue",
        ],
        chunksize=chunk_size,
        max_chunks=max_chunks,
    ):
        filtered = chunk.loc[chunk["patientunitstayid"].isin(cohort_ids)].copy()
        filtered["respchartoffset"] = pd.to_numeric(
            filtered["respchartoffset"], errors="coerce"
        )
        filtered = filtered.loc[
            filtered["respchartoffset"].between(0, LANDMARK_MINUTES, inclusive="left")
        ].copy()
        if filtered.empty:
            continue
        label = (
            filtered["respcharttypecat"].fillna("").astype(str)
            + " "
            + filtered["respchartvaluelabel"].fillna("").astype(str)
        ).map(normalize_text)
        value_text = filtered["respchartvalue"].fillna("").astype(str)
        combined = (label + " " + value_text.map(normalize_text)).str.strip()
        numeric = pd.to_numeric(
            value_text.str.extract(r"([-+]?[0-9]*\.?[0-9]+)", expand=False),
            errors="coerce",
        )
        concept_masks: list[tuple[str, pd.Series, pd.Series]] = []
        fio2_mask = label.str.contains(
            r"\bfio2\b|fraction inspired oxygen|inspired o2|oxygen concentration",
            regex=True,
            na=False,
        )
        fio2 = numeric.where(~numeric.between(0.20, 1.0), numeric * 100.0)
        concept_masks.append(("fio2", fio2_mask & fio2.between(21, 100), fio2))
        concept_masks.extend(
            [
                (
                    "mechanical_ventilation",
                    combined.str.contains(
                        r"ventilator mode|assist control|pressure control|volume control|\bsimv\b|mechanical ventilation",
                        regex=True,
                        na=False,
                    )
                    & ~combined.str.contains(
                        r"non[ -]?invasive ventilation|\bbipap\b|\bcpap\b",
                        regex=True,
                        na=False,
                    ),
                    pd.Series(1.0, index=filtered.index),
                ),
                (
                    "noninvasive_ventilation",
                    combined.str.contains(r"\bbipap\b|\bcpap\b|noninvasive", regex=True, na=False),
                    pd.Series(1.0, index=filtered.index),
                ),
                (
                    "oxygen_device",
                    combined.str.contains(
                        r"nasal cannula|high flow|hfnc|nonrebreather|oxygen mask|o2 device|oxygen device",
                        regex=True,
                        na=False,
                    ),
                    pd.Series(1.0, index=filtered.index),
                ),
                (
                    "airway_device",
                    combined.str.contains(r"intubat|endotracheal|tracheost", regex=True, na=False),
                    pd.Series(1.0, index=filtered.index),
                ),
            ]
        )
        for concept, mask, values in concept_masks:
            selected = filtered.loc[mask].copy()
            if selected.empty:
                continue
            frames.append(
                build_event_frame(
                    dataset="eicu",
                    stay_id=selected["patientunitstayid"].astype(int),
                    event_family="respiratory_support",
                    concept=concept,
                    source_table="respiratoryCharting.csv",
                    raw_name=selected["respchartvaluelabel"].astype(str),
                    offset_minutes=selected["respchartoffset"],
                    value_numeric=values.loc[selected.index],
                    value_text=selected["respchartvalue"].astype(str),
                    unit="percent" if concept == "fio2" else "binary",
                    is_intervention=0,
                )
            )
    return _concat_or_empty(frames)


def _discover_eicu_spo2_stays(
    root: Path,
    *,
    chunk_size: int,
    max_chunks: int | None,
) -> set[int] | None:
    """Return plausible-SpO2 stays from the same chunks used by a pilot."""
    if max_chunks is None:
        return None
    stays: set[int] = set()
    for chunk in _iter_csv_chunks(
        root / "vitalPeriodic.csv",
        usecols=["patientunitstayid", "observationoffset", "sao2"],
        chunksize=chunk_size,
        max_chunks=max_chunks,
    ):
        offset = pd.to_numeric(chunk["observationoffset"], errors="coerce")
        spo2 = pd.to_numeric(chunk["sao2"], errors="coerce")
        mask = offset.between(0, LANDMARK_MINUTES, inclusive="left") & spo2.between(50, 100)
        stays.update(chunk.loc[mask, "patientunitstayid"].dropna().astype(int).tolist())
    return stays


def _stream_eicu_urine_output_events(
    root: Path,
    cohort_ids: list[int],
    *,
    chunk_size: int,
    max_chunks: int | None,
) -> pd.DataFrame:
    """Extract plausible urine-output volumes from intakeOutput.csv."""
    path = root / "intakeOutput.csv"
    if not path.exists():
        return pd.DataFrame(columns=EVENT_REQUIRED_COLUMNS)
    frames: list[pd.DataFrame] = []
    for chunk in _iter_csv_chunks(
        path,
        usecols=[
            "patientunitstayid",
            "intakeoutputoffset",
            "intakeoutputentryoffset",
            "celllabel",
            "cellpath",
            "cellvaluenumeric",
        ],
        chunksize=chunk_size,
        max_chunks=max_chunks,
    ):
        filtered = chunk.loc[chunk["patientunitstayid"].isin(cohort_ids)].copy()
        text = (
            filtered["celllabel"].fillna("").astype(str)
            + " "
            + filtered["cellpath"].fillna("").astype(str)
        )
        filtered = filtered.loc[
            text.str.contains(r"urine|foley|voided|nephrostomy|urostomy", case=False, regex=True)
            & ~text.str.contains(r"culture|specimen|appearance|color", case=False, regex=True)
            & ~text.str.contains(
                r"urine\s*(?:count|occurrence)|incontin|mixed\s+urine\s*/?\s*stool",
                case=False,
                regex=True,
            )
            & ~text.str.contains(
                r"\btotal\b|subtotal|cumulative|24\s*hour|daily|net\s*(?:balance|output)",
                case=False,
                regex=True,
            )
        ].copy()
        if filtered.empty:
            continue
        filtered["cellvaluenumeric"] = pd.to_numeric(
            filtered["cellvaluenumeric"], errors="coerce"
        )
        # ``intakeOutputOffset`` is the clinical observation time;
        # ``intakeOutputEntryOffset`` is when it was entered into the system.
        # Outcome windows must use observation time to avoid documentation-lag
        # misclassification around the four-hour landmark.
        filtered["event_offset"] = pd.to_numeric(
            filtered["intakeoutputoffset"], errors="coerce"
        ).combine_first(
            pd.to_numeric(filtered["intakeoutputentryoffset"], errors="coerce")
        )
        filtered = filtered.loc[
            filtered["cellvaluenumeric"].ge(0)
            & filtered["cellvaluenumeric"].le(5000)
            & filtered["event_offset"].between(0, OUTCOME_WINDOW_END_MINUTES)
        ].copy()
        if filtered.empty:
            continue
        frames.append(
            build_event_frame(
                dataset="eicu",
                stay_id=filtered["patientunitstayid"].astype(int),
                event_family="output",
                concept="urine_output",
                source_table="intakeOutput.csv",
                raw_name=filtered["celllabel"].astype(str),
                offset_minutes=filtered["event_offset"],
                value_numeric=filtered["cellvaluenumeric"],
                value_text=filtered["cellpath"].astype(str),
                unit="mL",
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
    preferred_stays = _discover_eicu_spo2_stays(
        root,
        chunk_size=chunk_size,
        max_chunks=max_chunks,
    )
    cohort_df, cohort_ids = _build_eicu_cohort(
        root,
        patients,
        diagnosis_flags,
        audit,
        max_stays=max_stays,
        preferred_stay_ids=preferred_stays,
    )

    infusion_events = _stream_eicu_infusion_events(
        root, cohort_ids, chunk_size=chunk_size, max_chunks=max_chunks
    )
    audit.log(
        "eicu_infusions_streamed",
        row_count=len(infusion_events),
        stay_count=event_stay_count(infusion_events),
    )

    treatment_events = _load_eicu_treatment_events(
        root,
        cohort_ids,
        chunk_size=chunk_size,
        max_chunks=max_chunks,
    )
    audit.log(
        "eicu_treatments_loaded",
        row_count=len(treatment_events),
        stay_count=event_stay_count(treatment_events),
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
    aperiodic_vital_events = _stream_eicu_aperiodic_vital_events(
        root, cohort_ids, chunk_size=chunk_size, max_chunks=max_chunks
    )
    audit.log(
        "eicu_aperiodic_vitals_streamed",
        row_count=len(aperiodic_vital_events),
        stay_count=event_stay_count(aperiodic_vital_events),
        details={"source_available": (root / "vitalAperiodic.csv").is_file()},
    )
    nurse_bp_events = _stream_eicu_nurse_bp_events(
        root, cohort_ids, chunk_size=chunk_size, max_chunks=max_chunks
    )
    audit.log(
        "eicu_nurse_bp_streamed",
        row_count=len(nurse_bp_events),
        stay_count=event_stay_count(nurse_bp_events),
        details={"source_available": (root / "nurseCharting.csv").is_file()},
    )
    if (
        max_chunks is None
        and len(cohort_df) > 0
        and vital_events.loc[
            vital_events["concept"].eq("spo2")
            & pd.to_numeric(vital_events["value_numeric"], errors="coerce").between(50, 100)
        ].empty
    ):
        raise RuntimeError(
            "eICU full extraction produced no plausible cohort-window SpO2 rows; "
            "this is a fatal ID/window invariant."
        )

    respiratory_events = _stream_eicu_respiratory_events(
        root,
        cohort_ids,
        chunk_size=chunk_size,
        max_chunks=max_chunks,
    )
    audit.log(
        "eicu_respiratory_context_streamed",
        row_count=len(respiratory_events),
        stay_count=event_stay_count(respiratory_events),
    )

    urine_events = _stream_eicu_urine_output_events(
        root,
        cohort_ids,
        chunk_size=chunk_size,
        max_chunks=max_chunks,
    )
    audit.log(
        "eicu_urine_output_streamed",
        row_count=len(urine_events),
        stay_count=event_stay_count(urine_events),
    )

    weight_rows = cohort_df.loc[
        pd.to_numeric(cohort_df.get("admission_weight_kg"), errors="coerce").gt(0),
        ["stay_id", "admission_weight_kg"],
    ]
    weight_events = (
        build_event_frame(
            dataset="eicu",
            stay_id=weight_rows["stay_id"].astype(int),
            event_family="demographic_measurement",
            concept="weight",
            source_table="patient.csv",
            raw_name="admissionweight",
            offset_minutes=pd.Series(0.0, index=weight_rows.index),
            value_numeric=weight_rows["admission_weight_kg"],
            value_text=pd.NA,
            unit="kg",
            is_intervention=0,
        )
        if not weight_rows.empty
        else pd.DataFrame(columns=EVENT_REQUIRED_COLUMNS)
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
                lab_events,
                vital_events,
                aperiodic_vital_events,
                nurse_bp_events,
                respiratory_events,
                urine_events,
                weight_events,
                death_events,
            ],
            ignore_index=True,
        )
    )
    return SourceExtraction(cohort_df=cohort_df, events_df=events_df)
