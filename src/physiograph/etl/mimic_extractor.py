"""MIMIC-IV data extraction: cohort definition and event extraction.

ItemID-based extraction for labs, vitals, pressors, and MCS events.
Uses chunked CSV reading for large tables (chartevents, labevents).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .audit import AuditLogger
from .shared import (
    EVENT_REQUIRED_COLUMNS,
    OBSERVATION_HOURS,
    OUTCOME_WINDOW_END_HOURS,
    SourceExtraction,
    _concat_or_empty,
    _finalize_events,
    _iter_csv_chunks,
    build_death_events,
    build_event_frame,
    event_stay_count,
    match_icd_prefix,
)

if TYPE_CHECKING:
    from typing import Any

# ──────────────────────────────────────────────────────────────────────
# MIMIC Item IDs and ICD Patterns
# ──────────────────────────────────────────────────────────────────────

MIMIC_LAB_IDS: dict[str, list[int]] = {
    "lactate": [50813],
    "ph": [50820],
    "creatinine": [50912],
    "bun": [51006],
    "alt": [50861],
    "ast": [50878],
    "bilirubin_total": [50885],
}

MIMIC_VITAL_IDS: dict[str, list[int]] = {
    "hr": [220045],
    "sbp": [220050, 220179],
    "map": [220052, 220181],
    "resp_rate": [220210],
    "spo2": [220277],
    "temp": [223761],
    "weight": [226512],
}

HF_ICD_PATTERNS: dict[int, list[str]] = {9: [r"^428"], 10: [r"^I50"]}

CARDIOGENIC_SHOCK_ICD_PATTERNS: dict[int, list[str]] = {
    9: [r"^78551"],
    10: [r"^R570"],
}

PRESSOR_ITEMID_TO_DRUG: dict[int, str] = {
    221906: "norepinephrine",
    221289: "epinephrine",
    221662: "dopamine",
    221653: "dobutamine",
    221749: "phenylephrine",
    222315: "vasopressin",
    221986: "milrinone",
}
PRESSOR_ITEMIDS: list[int] = list(PRESSOR_ITEMID_TO_DRUG)
VIS_COEFFICIENTS: dict[str, float] = {
    "dopamine": 1.0,
    "dobutamine": 1.0,
    "epinephrine": 100.0,
    "norepinephrine": 100.0,
    "milrinone": 10.0,
    "vasopressin": 10_000.0,
    "phenylephrine": 10.0,
}

MCS_PROCEDUREEVENT_IDS: dict[str, list[int]] = {
    "iabp": [224272],
    "impella": [228169],
    "ecmo": [229529, 229530],
}

CHUNK_SIZE: int = 250_000


def _load_mimic_cohort(
    root: Path,
    audit: AuditLogger,
    *,
    max_stays: int | None,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Build MIMIC-IV HF/shock cohort with ICU anchor times.

    Identifies heart-failure admissions with cardiogenic shock ICD codes
    admitted to ICU within 24 hours of hospital arrival. Anchors the
    timeline to the first ICU admission time.

    Args:
        root: Path to MIMIC data directory.
        audit: AuditLogger for recording step counts.
        max_stays: Optional cap on cohort size (for testing).

    Returns:
        Tuple of (cohort_df, cohort_hadms, anchors):
        - cohort_df: Per-stay metadata DataFrame.
        - cohort_hadms: Array of included hospital admission IDs.
        - anchors: DataFrame mapping hadm_id to anchor_time.
    """
    dx = pd.read_csv(
        root / "diagnoses_icd.csv",
        usecols=["hadm_id", "icd_code", "icd_version"],
    )
    hf_hadms = dx.loc[
        match_icd_prefix(dx["icd_code"], dx["icd_version"], HF_ICD_PATTERNS),
        "hadm_id",
    ].unique()
    shock_hadms = dx.loc[
        match_icd_prefix(
            dx["icd_code"], dx["icd_version"], CARDIOGENIC_SHOCK_ICD_PATTERNS
        ),
        "hadm_id",
    ].unique()
    audit.log(
        "mimic_diagnoses_loaded",
        row_count=len(dx),
        stay_count=len(pd.unique(dx["hadm_id"])),
    )

    admissions = pd.read_csv(
        root / "admissions.csv",
        usecols=[
            "hadm_id",
            "subject_id",
            "admittime",
            "deathtime",
            "hospital_expire_flag",
        ],
        parse_dates=["admittime", "deathtime"],
    )
    admissions = admissions.loc[admissions["hadm_id"].isin(hf_hadms)].copy()
    admissions["admit_year"] = admissions["admittime"].dt.year.astype(int)

    patients = pd.read_csv(
        root / "patients.csv",
        usecols=["subject_id", "gender", "anchor_age", "anchor_year"],
    )
    admissions = admissions.merge(patients, on="subject_id", how="left")
    admissions["age"] = admissions["anchor_age"] + (
        admissions["admit_year"] - admissions["anchor_year"]
    )
    admissions["is_male"] = (admissions["gender"] == "M").astype(int)

    icu = pd.read_csv(
        root / "icustays.csv",
        usecols=["hadm_id", "intime"],
        parse_dates=["intime"],
    )
    icu = icu.merge(
        admissions[["hadm_id", "admittime"]], on="hadm_id", how="inner"
    )
    icu["hrs_to_icu"] = (
        (icu["intime"] - icu["admittime"]).dt.total_seconds() / 3600.0
    )
    icu = icu.sort_values(["hadm_id", "intime"])

    first_icu = icu.drop_duplicates(subset=["hadm_id"], keep="first").copy()
    early_icu = first_icu.loc[
        (first_icu["hrs_to_icu"] >= 0) & (first_icu["hrs_to_icu"] <= 24)
    ].copy()
    early_icu_hadms = early_icu["hadm_id"].unique()
    icu_anchor = (
        first_icu[["hadm_id", "intime"]]
        .rename(columns={"intime": "anchor_time"})
    )
    icu_anchor_hadms = icu_anchor["hadm_id"].unique()

    cohort_hadms = np.intersect1d(
        hf_hadms, np.union1d(early_icu_hadms, shock_hadms)
    )
    cohort_hadms = np.intersect1d(cohort_hadms, icu_anchor_hadms)
    if max_stays is not None:
        cohort_hadms = np.array(
            sorted(cohort_hadms)[:max_stays], dtype=int
        )

    admissions = admissions.loc[
        admissions["hadm_id"].isin(cohort_hadms)
    ].copy()
    admissions = admissions.merge(icu_anchor, on="hadm_id", how="inner")
    admissions["death_offset_minutes"] = (
        (admissions["deathtime"] - admissions["anchor_time"]).dt.total_seconds()
        / 60.0
    )
    audit.log(
        "mimic_candidate_cohort",
        stay_count=len(cohort_hadms),
        details={"max_stays": max_stays},
    )

    cohort_df = pd.DataFrame(
        {
            "dataset": "mimic",
            "stay_id": admissions["hadm_id"].astype(int),
            "person_id": admissions["subject_id"].astype(int),
            "admit_time": admissions["anchor_time"],
            "admit_year": admissions["admit_year"],
            "age": pd.to_numeric(admissions["age"], errors="coerce"),
            "is_male": admissions["is_male"].astype(int),
            "cohort_hf_flag": 1,
            "shock_icd_flag": admissions["hadm_id"]
            .isin(shock_hadms)
            .astype(int),
            "early_icu_flag": admissions["hadm_id"]
            .isin(early_icu_hadms)
            .astype(int),
            "death_offset_minutes": admissions["death_offset_minutes"],
            "excluded_before_landmark_flag": 0,
            "exclusion_reason": "",
        }
    )
    anchors = admissions[["hadm_id", "anchor_time"]].copy()
    return cohort_df, cohort_hadms, anchors


def _extract_mimic_pressor_events(
    root: Path,
    cohort_hadms: np.ndarray,
    anchors: pd.DataFrame,
) -> pd.DataFrame:
    """Extract sustained vasopressor events from inputevents.csv.

    Only includes infusions with rate > 0 and duration >= 1.0 hour.

    Args:
        root: MIMIC data directory.
        cohort_hadms: Array of included hospital admission IDs.
        anchors: DataFrame with hadm_id → anchor_time mapping.

    Returns:
        Event DataFrame with pressor rows.
    """
    inputevents = pd.read_csv(
        root / "inputevents.csv",
        usecols=[
            "hadm_id",
            "starttime",
            "endtime",
            "itemid",
            "rate",
            "rateuom",
            "patientweight",
        ],
    )
    inputevents = inputevents.loc[
        inputevents["hadm_id"].isin(cohort_hadms)
    ].copy()
    inputevents = inputevents.loc[
        inputevents["itemid"].isin(PRESSOR_ITEMIDS)
    ].dropna(subset=["rate"])
    inputevents = inputevents.loc[
        pd.to_numeric(inputevents["rate"], errors="coerce") > 0
    ].copy()
    inputevents["starttime"] = pd.to_datetime(
        inputevents["starttime"], errors="coerce"
    )
    inputevents["endtime"] = pd.to_datetime(
        inputevents["endtime"], errors="coerce"
    )
    inputevents = inputevents.merge(anchors, on="hadm_id", how="left")
    inputevents["duration_hrs"] = (
        (inputevents["endtime"] - inputevents["starttime"]).dt.total_seconds()
        / 3600.0
    )
    inputevents = inputevents.loc[
        inputevents["duration_hrs"] >= 1.0
    ].copy()
    offsets = (
        (inputevents["starttime"] - inputevents["anchor_time"]).dt.total_seconds()
        / 60.0
    )
    inputevents["drug"] = inputevents["itemid"].map(PRESSOR_ITEMID_TO_DRUG)
    pressor_frame = build_event_frame(
        dataset="mimic",
        stay_id=inputevents["hadm_id"].astype(int),
        event_family="intervention",
        concept="pressor",
        source_table="inputevents.csv",
        raw_name=inputevents["drug"].astype(str),
        offset_minutes=offsets,
        value_numeric=pd.to_numeric(inputevents["rate"], errors="coerce"),
        value_text=inputevents["drug"].astype(str),
        unit=inputevents["rateuom"].astype("string"),
        is_intervention=1,
    )

    rate = pd.to_numeric(inputevents["rate"], errors="coerce")
    weight = pd.to_numeric(inputevents["patientweight"], errors="coerce")
    unit = inputevents["rateuom"].fillna("").astype(str).str.lower().str.replace(" ", "", regex=False)
    dose = pd.Series(np.nan, index=inputevents.index, dtype=float)
    non_vasopressin = inputevents["drug"].ne("vasopressin")
    dose.loc[non_vasopressin & unit.str.contains("mcg/kg/min", regex=False)] = rate
    dose.loc[non_vasopressin & unit.str.fullmatch(r"mcg/min|mcgpermin", na=False) & weight.gt(0)] = (
        rate / weight
    )
    vasopressin = inputevents["drug"].eq("vasopressin")
    dose.loc[vasopressin & unit.str.contains("units/kg/min", regex=False)] = rate
    dose.loc[vasopressin & unit.str.fullmatch(r"units?/min", na=False) & weight.gt(0)] = rate / weight
    dose.loc[vasopressin & unit.str.fullmatch(r"units?/(hour|hr)", na=False) & weight.gt(0)] = rate / (60.0 * weight)
    dose.loc[vasopressin & unit.str.fullmatch(r"units?/kg/(hour|hr)", na=False)] = rate / 60.0
    inputevents["vis_component"] = dose * inputevents["drug"].map(VIS_COEFFICIENTS)

    vis_rows: list[dict[str, object]] = []
    compatible = inputevents.dropna(subset=["vis_component", "starttime", "endtime", "anchor_time"])
    for hadm_id, group in compatible.groupby("hadm_id", sort=False):
        group = group.sort_values("starttime")
        for _, infusion in group.iterrows():
            active = group.loc[
                group["starttime"].le(infusion["starttime"])
                & group["endtime"].gt(infusion["starttime"])
            ]
            vis_rows.append(
                {
                    "hadm_id": int(hadm_id),
                    "offset_minutes": float(
                        (infusion["starttime"] - infusion["anchor_time"]).total_seconds() / 60.0
                    ),
                    "vis": float(active["vis_component"].sum()),
                    "active_drugs": "+".join(sorted(active["drug"].astype(str).unique())),
                }
            )
    if not vis_rows:
        return pressor_frame
    vis = pd.DataFrame(vis_rows).drop_duplicates(["hadm_id", "offset_minutes", "vis"])
    vis_frame = build_event_frame(
        dataset="mimic",
        stay_id=vis["hadm_id"],
        event_family="intervention_intensity",
        concept="vis",
        source_table="inputevents.csv",
        raw_name=vis["active_drugs"],
        offset_minutes=vis["offset_minutes"],
        value_numeric=vis["vis"],
        value_text=vis["active_drugs"],
        unit="VIS",
        is_intervention=0,
    )
    return pd.concat([pressor_frame, vis_frame], ignore_index=True)


def _stream_mimic_urine_output_events(
    root: Path,
    cohort_hadms: np.ndarray,
    anchors: pd.DataFrame,
    *,
    chunk_size: int,
    max_chunks: int | None,
) -> pd.DataFrame:
    """Extract plausible urine-output volumes using dictionary labels."""
    output_path = root / "outputevents.csv"
    dictionary_path = root / "d_items.csv"
    if not output_path.exists() or not dictionary_path.exists():
        return pd.DataFrame(columns=EVENT_REQUIRED_COLUMNS)
    dictionary = pd.read_csv(dictionary_path, usecols=["itemid", "label"])
    labels = dictionary["label"].fillna("").astype(str)
    urine_items = dictionary.loc[
        labels.str.contains(r"urine|foley|voided|nephrostomy|urostomy", case=False, regex=True)
        & ~labels.str.contains(r"culture|specimen|appearance|color", case=False, regex=True),
        ["itemid", "label"],
    ]
    item_to_label = urine_items.set_index("itemid")["label"].to_dict()
    frames: list[pd.DataFrame] = []
    for chunk in _iter_csv_chunks(
        output_path,
        usecols=["hadm_id", "charttime", "itemid", "value", "valueuom"],
        chunksize=chunk_size,
        max_chunks=max_chunks,
    ):
        filtered = chunk.loc[
            chunk["hadm_id"].isin(cohort_hadms) & chunk["itemid"].isin(item_to_label)
        ].copy()
        if filtered.empty:
            continue
        filtered["value"] = pd.to_numeric(filtered["value"], errors="coerce")
        filtered = filtered.loc[filtered["value"].gt(0) & filtered["value"].le(5000)].copy()
        filtered["charttime"] = pd.to_datetime(filtered["charttime"], errors="coerce")
        filtered = filtered.merge(anchors, on="hadm_id", how="left")
        filtered["offset_minutes"] = (
            (filtered["charttime"] - filtered["anchor_time"]).dt.total_seconds() / 60.0
        )
        filtered = filtered.loc[filtered["offset_minutes"].between(0, OUTCOME_WINDOW_END_HOURS * 60)]
        if filtered.empty:
            continue
        frames.append(
            build_event_frame(
                dataset="mimic",
                stay_id=filtered["hadm_id"].astype(int),
                event_family="output",
                concept="urine_output",
                source_table="outputevents.csv",
                raw_name=filtered["itemid"].map(item_to_label).astype(str),
                offset_minutes=filtered["offset_minutes"],
                value_numeric=filtered["value"],
                value_text=pd.NA,
                unit=filtered["valueuom"].astype("string"),
                is_intervention=0,
            )
        )
    return _concat_or_empty(frames)


def _extract_mimic_mcs_events(
    root: Path,
    cohort_hadms: np.ndarray,
    anchors: pd.DataFrame,
) -> pd.DataFrame:
    """Extract mechanical circulatory support events from procedureevents.csv.

    Args:
        root: MIMIC data directory.
        cohort_hadms: Array of included hospital admission IDs.
        anchors: DataFrame with hadm_id → anchor_time mapping.

    Returns:
        Event DataFrame with MCS rows (iabp, impella, ecmo).
    """
    procedures = pd.read_csv(
        root / "procedureevents.csv",
        usecols=["hadm_id", "starttime", "itemid"],
    )
    procedures = procedures.loc[
        procedures["hadm_id"].isin(cohort_hadms)
    ].copy()
    procedures["starttime"] = pd.to_datetime(
        procedures["starttime"], errors="coerce"
    )
    procedures = procedures.merge(anchors, on="hadm_id", how="left")
    mcs_lookup: dict[int, str] = {
        itemid: concept
        for concept, itemids in MCS_PROCEDUREEVENT_IDS.items()
        for itemid in itemids
    }
    procedures = procedures.loc[
        procedures["itemid"].isin(mcs_lookup)
    ].copy()
    offsets = (
        (procedures["starttime"] - procedures["anchor_time"]).dt.total_seconds()
        / 60.0
    )
    return build_event_frame(
        dataset="mimic",
        stay_id=procedures["hadm_id"].astype(int),
        event_family="intervention",
        concept="mcs",
        source_table="procedureevents.csv",
        raw_name=procedures["itemid"].map(mcs_lookup).astype(str),
        offset_minutes=offsets,
        value_numeric=pd.NA,
        value_text=procedures["itemid"].astype(str),
        unit=pd.NA,
        is_intervention=1,
    )


def _stream_mimic_measurement_events(
    root: Path,
    *,
    file_name: str,
    source_table: str,
    item_map: dict[int, str],
    event_family: str,
    cohort_hadms: np.ndarray,
    anchors: pd.DataFrame,
    offset_min: float,
    offset_max: float,
    upper_inclusive: bool,
    chunk_size: int,
    max_chunks: int | None,
) -> pd.DataFrame:
    """Stream measurement events (vitals/labs) in chunks from large CSV files.

    Each chunk is filtered to the cohort, mapped from item IDs to concepts,
    and windowed to the specified offset range.

    Args:
        root: MIMIC data directory.
        file_name: CSV filename (e.g., "chartevents.csv").
        source_table: Source table name for provenance.
        item_map: Mapping from itemid to concept name.
        event_family: "vital" or "lab".
        cohort_hadms: Array of included hospital admission IDs.
        anchors: DataFrame with hadm_id → anchor_time mapping.
        offset_min: Minimum offset in minutes (inclusive).
        offset_max: Maximum offset in minutes.
        upper_inclusive: If True, include events at exactly offset_max.
        chunk_size: Rows per chunk.
        max_chunks: Optional cap on chunks read.

    Returns:
        Event DataFrame with measurement rows.
    """
    frames: list[pd.DataFrame] = []
    for chunk in _iter_csv_chunks(
        root / file_name,
        usecols=["hadm_id", "itemid", "charttime", "valuenum"],
        chunksize=chunk_size,
        max_chunks=max_chunks,
    ):
        filtered = chunk.loc[
            chunk["hadm_id"].isin(cohort_hadms)
            & chunk["itemid"].isin(item_map)
        ].copy()
        if filtered.empty:
            continue
        filtered["valuenum"] = pd.to_numeric(
            filtered["valuenum"], errors="coerce"
        )
        filtered = filtered.dropna(subset=["valuenum"])
        filtered["charttime"] = pd.to_datetime(
            filtered["charttime"], errors="coerce"
        )
        filtered = filtered.merge(anchors, on="hadm_id", how="left")
        filtered["offset_minutes"] = (
            (filtered["charttime"] - filtered["anchor_time"]).dt.total_seconds()
            / 60.0
        )
        if upper_inclusive:
            filtered = filtered.loc[
                (filtered["offset_minutes"] >= offset_min)
                & (filtered["offset_minutes"] <= offset_max)
            ].copy()
        else:
            filtered = filtered.loc[
                (filtered["offset_minutes"] >= offset_min)
                & (filtered["offset_minutes"] < offset_max)
            ].copy()
        if filtered.empty:
            continue
        frames.append(
            build_event_frame(
                dataset="mimic",
                stay_id=filtered["hadm_id"].astype(int),
                event_family=event_family,
                concept=filtered["itemid"].map(item_map),
                source_table=source_table,
                raw_name=filtered["itemid"].astype(str),
                offset_minutes=filtered["offset_minutes"],
                value_numeric=filtered["valuenum"],
                value_text=pd.NA,
                unit=pd.NA,
                is_intervention=0,
            )
        )
    return _concat_or_empty(frames)


def extract_mimic(
    root: Path,
    audit: AuditLogger,
    *,
    max_stays: int | None = None,
    max_chunks: int | None = None,
    chunk_size: int = CHUNK_SIZE,
) -> SourceExtraction:
    """Extract MIMIC-IV cohort and clinical events.

    Performs ICD-based cohort identification (HF + shock), pressor/MCS
    intervention extraction, chunked vitals/labs streaming, and death
    event construction.

    Args:
        root: Path to MIMIC data directory.
        audit: AuditLogger for recording step counts.
        max_stays: Optional cap on cohort size.
        max_chunks: Optional cap on chunks per streaming table.
        chunk_size: Rows per chunk for CSV streaming (default: 250,000).

    Returns:
        SourceExtraction with cohort_df and events_df.
    """
    cohort_df, cohort_hadms, anchors = _load_mimic_cohort(
        root, audit, max_stays=max_stays
    )

    pressor_events = _extract_mimic_pressor_events(root, cohort_hadms, anchors)
    audit.log(
        "mimic_pressors_extracted",
        row_count=len(pressor_events),
        stay_count=event_stay_count(pressor_events),
    )

    mcs_events = _extract_mimic_mcs_events(root, cohort_hadms, anchors)
    audit.log(
        "mimic_mcs_extracted",
        row_count=len(mcs_events),
        stay_count=event_stay_count(mcs_events),
    )

    vital_item_map: dict[int, str] = {
        itemid: concept
        for concept, itemids in MIMIC_VITAL_IDS.items()
        for itemid in itemids
    }
    vital_events = _stream_mimic_measurement_events(
        root,
        file_name="chartevents.csv",
        source_table="chartevents.csv",
        item_map=vital_item_map,
        event_family="vital",
        cohort_hadms=cohort_hadms,
        anchors=anchors,
        offset_min=0.0,
        offset_max=240.0,
        upper_inclusive=False,
        chunk_size=chunk_size,
        max_chunks=max_chunks,
    )
    audit.log(
        "mimic_vitals_streamed",
        row_count=len(vital_events),
        stay_count=event_stay_count(vital_events),
    )

    lab_item_map: dict[int, str] = {
        itemid: concept
        for concept, itemids in MIMIC_LAB_IDS.items()
        for itemid in itemids
    }
    lab_events = _stream_mimic_measurement_events(
        root,
        file_name="labevents.csv",
        source_table="labevents.csv",
        item_map=lab_item_map,
        event_family="lab",
        cohort_hadms=cohort_hadms,
        anchors=anchors,
        offset_min=0.0,
        offset_max=1680.0,
        upper_inclusive=True,
        chunk_size=chunk_size,
        max_chunks=max_chunks,
    )
    audit.log(
        "mimic_labs_streamed",
        row_count=len(lab_events),
        stay_count=event_stay_count(lab_events),
    )

    urine_events = _stream_mimic_urine_output_events(
        root,
        cohort_hadms,
        anchors,
        chunk_size=chunk_size,
        max_chunks=max_chunks,
    )
    audit.log(
        "mimic_urine_output_streamed",
        row_count=len(urine_events),
        stay_count=event_stay_count(urine_events),
    )

    death_events = build_death_events(
        cohort_df,
        dataset="mimic",
        source_table="admissions.csv",
        raw_name="deathtime",
    )
    events_df = _finalize_events(
        pd.concat(
            [
                pressor_events,
                mcs_events,
                vital_events,
                lab_events,
                urine_events,
                death_events,
            ],
            ignore_index=True,
        )
    )
    return SourceExtraction(cohort_df=cohort_df, events_df=events_df)
