"""eICU cohort builder using the cross-database harmonized HF phenotype.

Implementation notes
--------------------
- Diagnosis matching uses **token-based** string matching on
  ``diagnosisstring`` and ``admitdxname``/``admitdxtext`` rather than
  ICD code prefixes.
- The ICU anchor is ``patientunitstayid`` with offset zero (no
  separate ICU stay table).
- Age parsing handles the eICU de-identification pattern (``">89"``
  becomes ``89.0``).
- Death offset is derived from discharge status/location fields rather
  than a single ``deathtime`` column.
- Cardiomyopathy, acute MI, and shock matches are retained as covariates;
  they never substitute for an explicit heart-failure match.

The public entry-point is :func:`build_cohort`, which returns a
:class:`CohortResult`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from physiograph.config import load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default eICU diagnosis tokens (from Cell 8 / Cell 33)
# ---------------------------------------------------------------------------

_EICU_DIAGNOSIS_TOKENS: dict[str, list[str]] = {
    "cohort_hf_flag": ["congestive heart failure", "heart failure"],
    "shock_icd_flag": ["cardiogenic shock"],
    "cardiomyopathy_flag": ["cardiomyopathy"],
    "acute_mi_flag": ["acute myocardial infarction"],
}

_EICU_COHORT_FLAG_COLUMNS: list[str] = [
    "cohort_hf_flag",
    "shock_icd_flag",
    "cardiomyopathy_flag",
    "acute_mi_flag",
]

_EICU_PATIENT_COLUMNS: list[str] = [
    "patientunitstayid",
    "patienthealthsystemstayid",
    "uniquepid",
    "unitvisitnumber",
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


# ---------------------------------------------------------------------------
# Text normalisation and token matching
# ---------------------------------------------------------------------------

def normalize_text(value: object) -> str:
    """Lowercase and strip whitespace for eICU text matching.

    Parameters
    ----------
    value:
        Raw text value (may be None or NaN).

    Returns
    -------
    str
        Lowercased, stripped string.  Empty string for null-ish input.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip().lower()


def series_contains_any(series: pd.Series, tokens: list[str]) -> pd.Series:
    """Check whether each element of *series* contains any of *tokens*.

    Matching is case-insensitive substring search after normalisation.

    Parameters
    ----------
    series:
        Series of raw text values.
    tokens:
        List of substring tokens to search for.

    Returns
    -------
    pd.Series
        Boolean mask — ``True`` where any token is found.
    """
    normalized = series.fillna("").astype(str).str.lower().str.strip()
    pattern = "|".join(re.escape(t) for t in tokens)
    return normalized.str.contains(pattern, regex=True, na=False)


# ---------------------------------------------------------------------------
# Age parsing
# ---------------------------------------------------------------------------

def parse_eicu_age(value: object) -> float:
    """Parse eICU age field, handling the ``">89"`` de-identification pattern.

    Parameters
    ----------
    value:
        Raw age value from the patient table.

    Returns
    -------
    float
        Parsed numeric age, or ``NaN`` if unparseable.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return np.nan
    text = str(value).strip()
    if text.startswith(">"):
        text = text[1:]
    return float(pd.to_numeric(text, errors="coerce"))


# ---------------------------------------------------------------------------
# Death offset derivation
# ---------------------------------------------------------------------------

def derive_death_offset_minutes(patient_row: pd.Series) -> float | None:
    """Derive death offset in minutes from eICU patient discharge fields.

    Checks both unit-level and hospital-level discharge status/location
    for death indicators (``"expired"`` or ``"death"``).  Returns the
    earliest offset found, or ``None`` if no death indicator is present.

    Parameters
    ----------
    patient_row:
        A single row from the eICU patient table.

    Returns
    -------
    float or None
        Minutes from unit admission to death, or ``None``.
    """
    candidates: list[float] = []
    unit_status = normalize_text(patient_row.get("unitdischargestatus"))
    unit_location = normalize_text(patient_row.get("unitdischargelocation"))
    hospital_status = normalize_text(patient_row.get("hospitaldischargestatus"))
    hospital_location = normalize_text(patient_row.get("hospitaldischargelocation"))

    if any(token in unit_status or token in unit_location for token in ["expired", "death"]):
        unit_offset = pd.to_numeric(patient_row.get("unitdischargeoffset"), errors="coerce")
        if pd.notna(unit_offset):
            candidates.append(float(unit_offset))

    if any(token in hospital_status or token in hospital_location for token in ["expired", "death"]):
        hospital_offset = pd.to_numeric(patient_row.get("hospitaldischargeoffset"), errors="coerce")
        if pd.notna(hospital_offset):
            candidates.append(float(hospital_offset))

    return min(candidates) if candidates else None


# ---------------------------------------------------------------------------
# Result container (reuses same structure as MIMIC)
# ---------------------------------------------------------------------------

@dataclass
class CohortResult:
    """Container for eICU cohort extraction results.

    Attributes
    ----------
    cohort_df:
        Full cohort DataFrame with one row per ICU stay,
        including demographics, flags, and exclusion status.
    valid_stay_ids:
        Array of ``patientunitstayid`` values that passed all
        exclusion criteria.
    anchors:
        DataFrame mapping ``stay_id`` → anchor information (for eICU,
        this is typically empty since the anchor is offset zero).
    """

    cohort_df: pd.DataFrame
    valid_stay_ids: np.ndarray
    anchors: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------------------
# Cohort builder
# ---------------------------------------------------------------------------

class EICUCohortBuilder:
    """Build the production-aligned eICU HF-shock cohort from raw CSV tables.

    This reproduces the phenotype definition from Cells 33–35 of the
    PhysioGraph External Pipeline notebook.  The logic is:

    1. Load patient table and parse demographics.
    2. Build diagnosis flags via token-based matching on
       ``diagnosisstring`` and ``admissionDx`` text fields.
    3. Propagate diagnosis flags across a hospital encounter and retain its
       first ICU unit.
    4. Include adults with HF and early ICU care, or cardiogenic shock under
       the production phenotype, then exclude loss of follow-up/death by 4 h.

    Parameters
    ----------
    data_root:
        Path to the eICU CSV directory (containing ``patient.csv``,
        ``diagnosis.csv``, etc.).
    config:
        Optional pre-loaded config dict.  If ``None``,
        :func:`~physiograph.config.load_config` is called with
        ``dataset="eicu"``.
    max_stays:
        If set, cap the cohort to this many stays (for testing).
    """

    def __init__(
        self,
        data_root: str | Path,
        config: dict[str, Any] | None = None,
        max_stays: int | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.config = config or load_config("eicu")
        self.max_stays = max_stays

        # Allow config overrides for diagnosis tokens
        cfg_tokens = self.config.get("eicu", {}).get("diagnosis_tokens", _EICU_DIAGNOSIS_TOKENS)
        self.diagnosis_tokens: dict[str, list[str]] = cfg_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_cohort(self) -> CohortResult:
        """Execute the full eICU cohort extraction pipeline.

        Returns
        -------
        CohortResult
            Container with ``cohort_df``, ``valid_stay_ids``, and
            ``anchors``.
        """
        from physiograph.etl.audit import AuditLogger
        from physiograph.etl.eicu_extractor import (
            _build_eicu_cohort,
            _build_eicu_diagnosis_flags,
            _load_eicu_patients,
        )

        audit = AuditLogger(dataset="eicu")
        patients = _load_eicu_patients(self.data_root, audit)
        diagnosis_flags = _build_eicu_diagnosis_flags(self.data_root, patients)
        cohort_df, cohort_ids = _build_eicu_cohort(
            self.data_root,
            patients,
            diagnosis_flags,
            audit,
            max_stays=self.max_stays,
            preferred_stay_ids=None,
        )
        death = pd.to_numeric(cohort_df["death_offset_minutes"], errors="coerce")
        followup = pd.to_numeric(
            cohort_df["followup_end_offset_minutes"], errors="coerce"
        )
        early_death = death.between(0, 240, inclusive="both")
        early_discharge = followup.le(240)
        cohort_df["excluded_before_landmark_flag"] = (
            early_death | early_discharge
        ).astype(int)
        cohort_df["exclusion_reason"] = [
            ";".join(
                reason
                for condition, reason in (
                    (bool(death_flag), "death_before_or_at_4h"),
                    (bool(discharge_flag), "icu_discharge_before_or_at_4h"),
                )
                if condition
            )
            for death_flag, discharge_flag in zip(early_death, early_discharge)
        ]
        valid_ids = cohort_df.loc[
            cohort_df["excluded_before_landmark_flag"].eq(0), "stay_id"
        ].astype(int).to_numpy()

        logger.info(
            "eICU cohort: %d candidate stays, %d after exclusions",
            len(cohort_ids),
            len(valid_ids),
        )

        return CohortResult(
            cohort_df=cohort_df,
            valid_stay_ids=valid_ids,
            anchors=pd.DataFrame(),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_patients(self) -> pd.DataFrame:
        """Load and return the eICU patient table.

        Returns
        -------
        pd.DataFrame
            Patient table with standardised column names.
        """
        available_columns = set(
            pd.read_csv(self.data_root / "patient.csv", nrows=0).columns.tolist()
        )
        usecols = [c for c in _EICU_PATIENT_COLUMNS if c in available_columns]
        patients = pd.read_csv(self.data_root / "patient.csv", usecols=usecols)

        # Ensure all expected columns exist (fill with NA if missing)
        for c in _EICU_PATIENT_COLUMNS:
            if c not in patients.columns:
                patients[c] = pd.NA

        logger.info("Loaded %d eICU patient rows", len(patients))
        return patients

    def _build_diagnosis_flags(self, patients: pd.DataFrame) -> pd.DataFrame:
        """Build binary diagnosis flags via token-based matching.

        Parameters
        ----------
        patients:
            Patient table (used for the set of unique stay IDs).

        Returns
        -------
        pd.DataFrame
            DataFrame with ``stay_id`` and one column per diagnosis
            flag (``cohort_hf_flag``, ``shock_icd_flag``, etc.).
        """
        diagnosis = pd.read_csv(
            self.data_root / "diagnosis.csv",
            usecols=["patientunitstayid", "diagnosisstring"],
        )
        admission_dx = pd.read_csv(
            self.data_root / "admissionDx.csv",
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
            admission_dx["text"] = admission_dx[["admitdxname", "admitdxtext"]].fillna("").agg(
                " ".join, axis=1
            )

        for flag_column, tokens in self.diagnosis_tokens.items():
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
            diagnosis_flags[flag_column] = diagnosis_flags["stay_id"].isin(flagged_stays).astype(int)

        return diagnosis_flags

    def _build_cohort_df(
        self,
        patients: pd.DataFrame,
        diagnosis_flags: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[int]]:
        """Assemble the candidate cohort DataFrame.

        Parameters
        ----------
        patients:
            Patient table.
        diagnosis_flags:
            Diagnosis flags DataFrame from :meth:`_build_diagnosis_flags`.

        Returns
        -------
        tuple[pd.DataFrame, list[int]]
            ``(cohort_df, cohort_ids)`` — the cohort DataFrame and
            the list of included stay IDs.
        """
        merged = patients.merge(
            diagnosis_flags,
            left_on="patientunitstayid",
            right_on="stay_id",
            how="inner",
            validate="one_to_one",
        )
        merged["age"] = merged["age"].map(parse_eicu_age)
        merged["is_male"] = np.where(
            merged["gender"].fillna("").str.lower().eq("male"),
            1.0,
            np.where(
                merged["gender"].fillna("").str.lower().eq("female"),
                0.0,
                np.nan,
            ),
        )
        encounter = merged["patienthealthsystemstayid"].astype("string")
        merged["_encounter"] = encounter.where(
            encounter.notna() & encounter.str.strip().ne(""),
            "stay:" + merged["patientunitstayid"].astype("string"),
        )
        for flag in _EICU_COHORT_FLAG_COLUMNS:
            if flag not in merged:
                merged[flag] = 0
            merged[flag] = merged.groupby("_encounter", sort=False)[flag].transform(
                "max"
            )
        visit = pd.to_numeric(
            merged.get(
                "unitvisitnumber",
                pd.Series(np.nan, index=merged.index),
            ),
            errors="coerce",
        )
        merged["_visit"] = visit.fillna(np.inf)
        merged = (
            merged.sort_values(["_encounter", "_visit", "patientunitstayid"])
            .drop_duplicates("_encounter", keep="first")
            .copy()
        )
        merged["early_icu_flag"] = (
            -pd.to_numeric(merged["hospitaladmitoffset"], errors="coerce")
        ).between(0, 24 * 60, inclusive="both").fillna(False).astype(int)
        # Identical to MIMIC: adult + explicit HF + (early ICU or shock).
        merged = merged.loc[
            pd.to_numeric(merged["age"], errors="coerce").ge(18)
            & merged["cohort_hf_flag"].eq(1)
            & (merged["early_icu_flag"].eq(1) | merged["shock_icd_flag"].eq(1))
        ].copy()
        if self.max_stays is not None:
            merged = merged.sort_values("patientunitstayid").head(self.max_stays)
        merged["death_offset_minutes"] = merged.apply(
            derive_death_offset_minutes, axis=1
        )
        cohort_ids = merged["patientunitstayid"].astype(int).tolist()

        cohort_df = pd.DataFrame(
            {
                "dataset": "eicu",
                "stay_id": merged["patientunitstayid"].astype(int),
                "person_id": merged["patienthealthsystemstayid"].astype("string"),
                "hospital_encounter_id": merged["patienthealthsystemstayid"],
                "admit_time": pd.NaT,
                "admit_year": pd.to_numeric(merged["hospitaldischargeyear"], errors="coerce"),
                "age": merged["age"],
                "is_male": merged["is_male"],
                "cohort_hf_flag": merged["cohort_hf_flag"].astype(int),
                "shock_icd_flag": merged["shock_icd_flag"].astype(int),
                "cardiomyopathy_flag": merged["cardiomyopathy_flag"].astype(int),
                "acute_mi_flag": merged["acute_mi_flag"].astype(int),
                "early_icu_flag": merged["early_icu_flag"].astype(int),
                "death_offset_minutes": merged["death_offset_minutes"],
                "unitdischargeoffset": pd.to_numeric(
                    merged["unitdischargeoffset"], errors="coerce"
                ),
                "excluded_before_landmark_flag": 0,
                "exclusion_reason": "",
            }
        )

        return cohort_df, cohort_ids

    def _apply_exclusions(self, cohort_df: pd.DataFrame) -> pd.DataFrame:
        """Apply adult, four-hour follow-up, and pre-landmark-death exclusions.

        Parameters
        ----------
        cohort_df:
            Candidate cohort DataFrame.

        Returns
        -------
        pd.DataFrame
            Cohort DataFrame with ``excluded_before_landmark_flag`` and
            ``exclusion_reason`` columns populated.
        """
        reasons: list[str] = []
        excluded: list[int] = []

        for idx, row in cohort_df.iterrows():
            r: list[str] = []

            # Adult cohort rule (normally already enforced by the builder).
            age = row.get("age")
            if pd.isna(age) or float(age) < 18:
                r.append("age_lt_18_or_missing")

            # Four-hour landmark requires follow-up beyond minute 240.
            unit_discharge = pd.to_numeric(row.get("unitdischargeoffset", pd.NA), errors="coerce")
            if pd.notna(unit_discharge) and float(unit_discharge) <= 240:
                r.append("icu_discharge_before_or_at_4h")

            # Pre-landmark death exclusion
            death_offset = row.get("death_offset_minutes")
            if pd.notna(death_offset) and float(death_offset) >= 0 and float(death_offset) <= 240:
                r.append("death_before_or_at_4h")

            reasons.append(";".join(r))
            excluded.append(1 if r else 0)

        cohort_df = cohort_df.copy()
        cohort_df["excluded_before_landmark_flag"] = excluded
        cohort_df["exclusion_reason"] = reasons

        return cohort_df


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def build_cohort(
    data_root: str | Path,
    config: dict[str, Any] | None = None,
    max_stays: int | None = None,
) -> CohortResult:
    """Build the eICU HF-shock cohort.

    This is a convenience wrapper around :class:`EICUCohortBuilder`.

    Parameters
    ----------
    data_root:
        Path to the eICU CSV directory.
    config:
        Optional pre-loaded config dict.
    max_stays:
        If set, cap the cohort to this many stays (for testing).

    Returns
    -------
    CohortResult
        Container with ``cohort_df``, ``valid_stay_ids``, and
        ``anchors``.
    """
    builder = EICUCohortBuilder(data_root=data_root, config=config, max_stays=max_stays)
    return builder.build_cohort()
