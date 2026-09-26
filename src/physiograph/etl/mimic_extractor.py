"""MIMIC-IV data extraction: cohort definition and event extraction.

ItemID-based extraction for labs, vitals, pressors, MCS, and ventilation events.
Uses chunked CSV reading for large tables (chartevents, labevents).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from physiograph.constants import (
    CARDIOGENIC_SHOCK_ICD_PATTERNS as CANONICAL_CARDIOGENIC_SHOCK_ICD_PATTERNS,
    HF_ICD_PATTERNS as CANONICAL_HF_ICD_PATTERNS,
    MCS_PROCEDUREEVENT_IDS as CANONICAL_MCS_PROCEDUREEVENT_IDS,
    MIMIC_LAB_IDS as CANONICAL_MIMIC_LAB_IDS,
    MIMIC_VITAL_IDS as CANONICAL_MIMIC_VITAL_IDS,
)

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
    normalize_text,
)

if TYPE_CHECKING:
    from typing import Any

# ──────────────────────────────────────────────────────────────────────
# MIMIC Item IDs and ICD Patterns
# ──────────────────────────────────────────────────────────────────────

MIMIC_LAB_IDS: dict[str, list[int]] = {
    concept: list(itemids) for concept, itemids in CANONICAL_MIMIC_LAB_IDS.items()
}
MIMIC_VITAL_IDS: dict[str, list[int]] = {
    concept: list(itemids) for concept, itemids in CANONICAL_MIMIC_VITAL_IDS.items()
}
HF_ICD_PATTERNS: dict[int, list[str]] = {
    int(version): list(patterns)
    for version, patterns in CANONICAL_HF_ICD_PATTERNS.items()
}
CARDIOGENIC_SHOCK_ICD_PATTERNS: dict[int, list[str]] = {
    int(version): list(patterns)
    for version, patterns in CANONICAL_CARDIOGENIC_SHOCK_ICD_PATTERNS.items()
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
    concept: list(itemids)
    for concept, itemids in CANONICAL_MCS_PROCEDUREEVENT_IDS.items()
}

CHUNK_SIZE: int = 250_000
PRESSOR_RESTART_GRACE_MINUTES: float = 5.0

# Direct MetaVision procedure records.  ``Invasive Ventilation`` is an
# interval and therefore supplies both a true start time and landmark-active
# status; ``Intubation`` is retained as a more specific onset sensitivity.
# Dictionary discovery below is authoritative and these IDs are fallbacks for
# installations with a reduced ``d_items`` export.
MIMIC_INVASIVE_VENTILATION_PROCEDURE_IDS: set[int] = {225792}
MIMIC_INTUBATION_PROCEDURE_IDS: set[int] = {224385}


def _load_mimic_cohort(
    root: Path,
    audit: AuditLogger,
    *,
    max_stays: int | None,
    preferred_stay_ids: set[int] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Build an adult, first-ICU-per-admission MIMIC-IV HF cohort.

    The canonical identifier is the real ICU ``stay_id``.  Hospital admission
    identifiers are retained only as context and for joining labevents.  This
    prevents measurements from later ICU stays in the same admission being
    mixed into the landmark window.
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
            "dischtime",
            "deathtime",
            "hospital_expire_flag",
            "race",
        ],
        parse_dates=["admittime", "dischtime", "deathtime"],
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
    admissions["is_male"] = np.where(
        admissions["gender"].eq("M"), 1.0,
        np.where(admissions["gender"].eq("F"), 0.0, np.nan),
    )
    admissions["sex_unknown_flag"] = admissions["gender"].isna().astype(int)

    icu = pd.read_csv(
        root / "icustays.csv",
        usecols=[
            "subject_id",
            "hadm_id",
            "stay_id",
            "first_careunit",
            "last_careunit",
            "intime",
            "outtime",
            "los",
        ],
        parse_dates=["intime", "outtime"],
    )
    icu = icu.merge(
        admissions[["hadm_id", "admittime"]], on="hadm_id", how="inner"
    )
    icu["hrs_to_icu"] = (
        (icu["intime"] - icu["admittime"]).dt.total_seconds() / 3600.0
    )
    icu = icu.sort_values(["hadm_id", "intime", "stay_id"])

    first_icu = icu.drop_duplicates(subset=["hadm_id"], keep="first").copy()
    early_icu = first_icu.loc[
        (first_icu["hrs_to_icu"] >= 0) & (first_icu["hrs_to_icu"] <= 24)
    ].copy()
    first_icu["early_icu_flag"] = (
        first_icu["hrs_to_icu"].between(0, 24, inclusive="both")
    ).astype(int)
    first_icu["shock_icd_flag"] = first_icu["hadm_id"].isin(shock_hadms).astype(int)
    first_icu = first_icu.loc[
        first_icu["hadm_id"].isin(hf_hadms)
        & ((first_icu["early_icu_flag"] == 1) | (first_icu["shock_icd_flag"] == 1))
    ].copy()
    first_icu = first_icu.merge(
        admissions,
        on=["hadm_id", "subject_id", "admittime"],
        how="inner",
        validate="one_to_one",
    )
    first_icu = first_icu.loc[pd.to_numeric(first_icu["age"], errors="coerce").ge(18)].copy()
    if preferred_stay_ids is not None:
        first_icu = first_icu.loc[first_icu["stay_id"].isin(preferred_stay_ids)].copy()
    first_icu = first_icu.sort_values(["stay_id"]).reset_index(drop=True)
    if max_stays is not None:
        first_icu = first_icu.head(max_stays).copy()

    first_icu["anchor_time"] = first_icu["intime"]
    for source, target in (
        ("deathtime", "death_offset_minutes"),
        ("outtime", "unit_discharge_offset_minutes"),
        ("dischtime", "hospital_discharge_offset_minutes"),
    ):
        first_icu[target] = (
            (first_icu[source] - first_icu["anchor_time"]).dt.total_seconds() / 60.0
        )
    first_icu["followup_end_offset_minutes"] = first_icu[
        ["unit_discharge_offset_minutes", "hospital_discharge_offset_minutes"]
    ].min(axis=1, skipna=True)
    first_icu["first_stay_per_person_flag"] = (
        first_icu.sort_values(["subject_id", "anchor_time", "stay_id"])
        .groupby("subject_id", sort=False)
        .cumcount()
        .eq(0)
        .astype(int)
    )
    cohort_stays = first_icu["stay_id"].astype(int).to_numpy()
    audit.log(
        "mimic_candidate_cohort",
        stay_count=len(cohort_stays),
        details={
            "max_stays": max_stays,
            "preferred_stay_sampling": preferred_stay_ids is not None,
            "adult_only": True,
            "identity": "icustays.stay_id",
        },
    )

    cohort_df = pd.DataFrame(
        {
            "dataset": "mimic",
            "stay_id": first_icu["stay_id"].astype(int),
            "person_id": first_icu["subject_id"].astype(int),
            "hadm_id": first_icu["hadm_id"].astype(int),
            "admit_time": first_icu["anchor_time"],
            "admit_year": first_icu["admit_year"],
            "age": pd.to_numeric(first_icu["age"], errors="coerce"),
            "is_male": first_icu["is_male"],
            "sex_unknown_flag": first_icu["sex_unknown_flag"],
            "race_ethnicity": first_icu["race"].fillna("UNKNOWN").astype(str),
            "hospital_id": "mimic_single_center",
            "icu_type": first_icu["first_careunit"].fillna("UNKNOWN").astype(str),
            "cohort_hf_flag": 1,
            "shock_icd_flag": first_icu["shock_icd_flag"].astype(int),
            "early_icu_flag": first_icu["early_icu_flag"].astype(int),
            "death_offset_minutes": first_icu["death_offset_minutes"],
            "unit_discharge_offset_minutes": first_icu["unit_discharge_offset_minutes"],
            "hospital_discharge_offset_minutes": first_icu["hospital_discharge_offset_minutes"],
            "followup_end_offset_minutes": first_icu["followup_end_offset_minutes"],
            "first_stay_per_person_flag": first_icu["first_stay_per_person_flag"],
            "mcs_source_available": int(
                (root / "procedureevents.csv").exists()
                or (root / "chartevents.csv").exists()
            ),
            "pressor_source_available": int((root / "inputevents.csv").exists()),
            "vis_source_available": int((root / "inputevents.csv").exists()),
            "urine_output_source_available": int((root / "outputevents.csv").exists()),
            "respiratory_source_available": int(
                (root / "chartevents.csv").exists()
                or (root / "procedureevents.csv").exists()
            ),
            "rrt_source_available": int(
                (root / "procedureevents.csv").exists()
                or (root / "chartevents.csv").exists()
            ),
            "excluded_before_landmark_flag": 0,
            "exclusion_reason": "",
        }
    )
    anchors = first_icu[["stay_id", "hadm_id", "anchor_time"]].copy()
    return cohort_df, cohort_stays, anchors


def _extract_mimic_pressor_events(
    root: Path,
    cohort_stays: np.ndarray,
    anchors: pd.DataFrame,
    *,
    chunk_size: int,
    max_chunks: int | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Extract pressor intervals and a change-point VIS time series.

    Infusions are keyed by ICU stay, include short infusions, and generate VIS
    at every start/end plus the landmark/horizon boundaries.  This preserves
    drugs already running at the landmark and prevents an unchanged infusion
    from being mistaken for VIS=0 in the outcome window.
    """
    frames: list[pd.DataFrame] = []
    raw_pressor_rows = 0
    cohort_pressor_rows = 0
    path = root / "inputevents.csv"
    if not path.exists():
        return pd.DataFrame(columns=EVENT_REQUIRED_COLUMNS), {
            "raw_pressor_rows": 0,
            "cohort_pressor_rows": 0,
            "vis_compatible_rows": 0,
            "vis_incompatible_rows": 0,
            "vis_incompatible_stays": 0,
        }
    for chunk in _iter_csv_chunks(
        path,
        usecols=[
            "stay_id",
            "starttime",
            "endtime",
            "itemid",
            "rate",
            "rateuom",
            "patientweight",
        ],
        chunksize=chunk_size,
        max_chunks=max_chunks,
    ):
        pressor_mask = chunk["itemid"].isin(PRESSOR_ITEMIDS)
        raw_pressor_rows += int(pressor_mask.sum())
        filtered = chunk.loc[
            pressor_mask & chunk["stay_id"].isin(cohort_stays)
        ].copy()
        cohort_pressor_rows += len(filtered)
        if not filtered.empty:
            frames.append(filtered)
    if not frames:
        return pd.DataFrame(columns=EVENT_REQUIRED_COLUMNS), {
            "raw_pressor_rows": raw_pressor_rows,
            "cohort_pressor_rows": cohort_pressor_rows,
            "vis_compatible_rows": 0,
            "vis_incompatible_rows": 0,
            "vis_incompatible_stays": 0,
        }
    inputevents = pd.concat(frames, ignore_index=True)
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
    inputevents = inputevents.merge(
        anchors[["stay_id", "anchor_time"]],
        on="stay_id",
        how="inner",
        validate="many_to_one",
    )
    inputevents["start_offset"] = (
        (inputevents["starttime"] - inputevents["anchor_time"]).dt.total_seconds()
        / 60.0
    )
    inputevents["end_offset"] = (
        (inputevents["endtime"] - inputevents["anchor_time"]).dt.total_seconds()
        / 60.0
    )
    inputevents = inputevents.loc[
        inputevents["start_offset"].notna()
        & inputevents["end_offset"].gt(inputevents["start_offset"])
        & inputevents["end_offset"].gt(0)
        & inputevents["start_offset"].le(OUTCOME_WINDOW_END_HOURS * 60)
    ].copy()
    inputevents["drug"] = inputevents["itemid"].map(PRESSOR_ITEMID_TO_DRUG)
    pressor_frame = build_event_frame(
        dataset="mimic",
        stay_id=inputevents["stay_id"].astype(int),
        event_family="intervention",
        concept="pressor",
        source_table="inputevents.csv",
        raw_name=inputevents["drug"].astype(str),
        offset_minutes=inputevents["start_offset"],
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
    dose.loc[non_vasopressin & unit.str.fullmatch(r"mg/kg/(hour|hr)", na=False)] = (
        rate * 1000.0 / 60.0
    )
    dose.loc[
        non_vasopressin
        & unit.str.fullmatch(r"mg/(hour|hr)", na=False)
        & weight.gt(0)
    ] = rate * 1000.0 / (60.0 * weight)
    vasopressin = inputevents["drug"].eq("vasopressin")
    dose.loc[vasopressin & unit.str.contains("units/kg/min", regex=False)] = rate
    dose.loc[vasopressin & unit.str.fullmatch(r"units?/min", na=False) & weight.gt(0)] = rate / weight
    dose.loc[vasopressin & unit.str.fullmatch(r"units?/(hour|hr)", na=False) & weight.gt(0)] = rate / (60.0 * weight)
    dose.loc[vasopressin & unit.str.fullmatch(r"units?/kg/(hour|hr)", na=False)] = rate / 60.0
    inputevents["vis_component"] = dose * inputevents["drug"].map(VIS_COEFFICIENTS)
    compatible = inputevents.dropna(subset=["vis_component", "start_offset", "end_offset"])
    vis_rows: list[dict[str, object]] = []
    initiation_rows: list[dict[str, object]] = []
    active_rows: list[dict[str, object]] = []
    boundary_points = {0.0, 240.0, 960.0, 1680.0}
    for stay_id, all_group in inputevents.groupby("stay_id", sort=False):
        all_group = all_group.sort_values(["start_offset", "end_offset"])
        transition_points = sorted(
            {
                float(value)
                for value in pd.concat(
                    [all_group["start_offset"], all_group["end_offset"]],
                    ignore_index=True,
                ).dropna()
                if 0 <= float(value) <= OUTCOME_WINDOW_END_HOURS * 60
            }
        )
        for point in transition_points:
            before = all_group.loc[
                all_group["start_offset"].lt(point)
                & all_group["end_offset"].ge(
                    point - PRESSOR_RESTART_GRACE_MINUTES
                )
            ]
            after = all_group.loc[
                all_group["start_offset"].le(point)
                & all_group["end_offset"].gt(point)
            ]
            if before.empty and not after.empty:
                initiation_rows.append(
                    {
                        "stay_id": int(stay_id),
                        "offset_minutes": point,
                        "active_drugs": "+".join(sorted(after["drug"].astype(str).unique())),
                    }
                )
        for point in (0.0, 240.0):
            active = all_group.loc[
                all_group["start_offset"].le(point)
                & all_group["end_offset"].gt(point)
            ]
            active_rows.append(
                {
                    "stay_id": int(stay_id),
                    "offset_minutes": point,
                    "active": int(not active.empty),
                    "active_drugs": "+".join(sorted(active["drug"].astype(str).unique())),
                }
            )
        group = compatible.loc[compatible["stay_id"].eq(stay_id)]
        if group.empty:
            continue
        points = sorted(
            boundary_points
            | {
                float(value)
                for value in pd.concat([group["start_offset"], group["end_offset"]]).dropna()
                if 0 <= float(value) <= OUTCOME_WINDOW_END_HOURS * 60
            }
        )
        for point in points:
            active = group.loc[
                group["start_offset"].le(point)
                & group["end_offset"].gt(point)
            ]
            vis_rows.append(
                {
                    "stay_id": int(stay_id),
                    "offset_minutes": point,
                    "vis": float(active["vis_component"].sum()) if not active.empty else 0.0,
                    "active_drugs": "+".join(sorted(active["drug"].astype(str).unique())),
                }
            )
    event_frames = [pressor_frame]
    incompatible = inputevents.loc[inputevents["vis_component"].isna()].copy()
    if not incompatible.empty:
        # Preserve the fact that a vasoactive exposure could not be converted
        # to the prespecified weight-normalized VIS scale.  Downstream outcome
        # code uses this marker to make VIS unavailable for that stay instead
        # of silently treating the unconverted component as zero.
        event_frames.append(
            build_event_frame(
                dataset="mimic",
                stay_id=incompatible["stay_id"].astype(int),
                event_family="data_quality",
                concept="vis_rate_unstandardized",
                source_table="inputevents.csv",
                raw_name=incompatible["drug"].astype(str),
                offset_minutes=incompatible["start_offset"],
                value_numeric=pd.to_numeric(incompatible["rate"], errors="coerce"),
                value_text=incompatible["rateuom"].astype("string"),
                unit=incompatible["rateuom"].astype("string"),
                is_intervention=0,
            )
        )
    if initiation_rows:
        initiations = pd.DataFrame(initiation_rows).drop_duplicates(
            ["stay_id", "offset_minutes"]
        )
        event_frames.append(
            build_event_frame(
                dataset="mimic",
                stay_id=initiations["stay_id"],
                event_family="intervention",
                concept="pressor_initiation",
                source_table="inputevents.csv",
                raw_name=initiations["active_drugs"],
                offset_minutes=initiations["offset_minutes"],
                value_numeric=1.0,
                value_text=initiations["active_drugs"],
                unit="binary",
                is_intervention=1,
            )
        )
    if active_rows:
        active = pd.DataFrame(active_rows).drop_duplicates(
            ["stay_id", "offset_minutes"]
        )
        event_frames.append(
            build_event_frame(
                dataset="mimic",
                stay_id=active["stay_id"],
                event_family="intervention_status",
                concept="pressor_active",
                source_table="inputevents.csv",
                raw_name=active["active_drugs"],
                offset_minutes=active["offset_minutes"],
                value_numeric=active["active"],
                value_text=active["active_drugs"],
                unit="binary",
                is_intervention=0,
            )
        )
    if not vis_rows:
        return pd.concat(event_frames, ignore_index=True), {
            "raw_pressor_rows": raw_pressor_rows,
            "cohort_pressor_rows": cohort_pressor_rows,
            "vis_compatible_rows": int(compatible.shape[0]),
            "vis_incompatible_rows": int(inputevents["vis_component"].isna().sum()),
            "vis_incompatible_stays": int(incompatible["stay_id"].nunique()),
        }
    vis = pd.DataFrame(vis_rows).drop_duplicates(["stay_id", "offset_minutes", "vis"])
    vis_frame = build_event_frame(
        dataset="mimic",
        stay_id=vis["stay_id"],
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
    event_frames.append(vis_frame)
    return pd.concat(event_frames, ignore_index=True), {
        "raw_pressor_rows": raw_pressor_rows,
        "cohort_pressor_rows": cohort_pressor_rows,
        "vis_compatible_rows": int(compatible.shape[0]),
        "vis_incompatible_rows": int(inputevents["vis_component"].isna().sum()),
        "vis_incompatible_stays": int(incompatible["stay_id"].nunique()),
    }


def _stream_mimic_urine_output_events(
    root: Path,
    cohort_stays: np.ndarray,
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
        usecols=["stay_id", "charttime", "itemid", "value", "valueuom"],
        chunksize=chunk_size,
        max_chunks=max_chunks,
    ):
        filtered = chunk.loc[
            chunk["stay_id"].isin(cohort_stays) & chunk["itemid"].isin(item_to_label)
        ].copy()
        if filtered.empty:
            continue
        filtered["value"] = pd.to_numeric(filtered["value"], errors="coerce")
        # Preserve true zero output: zero is clinically meaningful anuria, not
        # missingness.  Negative corrections and impossible volumes are removed.
        filtered = filtered.loc[filtered["value"].ge(0) & filtered["value"].le(5000)].copy()
        filtered["charttime"] = pd.to_datetime(filtered["charttime"], errors="coerce")
        filtered = filtered.merge(
            anchors[["stay_id", "anchor_time"]],
            on="stay_id",
            how="inner",
            validate="many_to_one",
        )
        filtered["offset_minutes"] = (
            (filtered["charttime"] - filtered["anchor_time"]).dt.total_seconds() / 60.0
        )
        filtered = filtered.loc[filtered["offset_minutes"].between(0, OUTCOME_WINDOW_END_HOURS * 60)]
        if filtered.empty:
            continue
        frames.append(
            build_event_frame(
                dataset="mimic",
                stay_id=filtered["stay_id"].astype(int),
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
    cohort_stays: np.ndarray,
    anchors: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Extract intervention and respiratory procedure intervals by ICU stay.

    This deliberately uses the direct ``procedureevents`` Invasive Ventilation
    and Intubation rows.  Ventilator-setting charting is useful baseline
    context but is not a defensible proxy for a new ventilation start.
    """
    path = root / "procedureevents.csv"
    dictionary_path = root / "d_items.csv"
    if not path.exists() or not dictionary_path.exists():
        return pd.DataFrame(columns=EVENT_REQUIRED_COLUMNS), {
            "mcs_rows": 0,
            "rrt_rows": 0,
            "mechanical_ventilation_rows": 0,
            "intubation_rows": 0,
        }
    dictionary = pd.read_csv(
        dictionary_path,
        usecols=["itemid", "label", "linksto", "category"],
    )
    labels = dictionary["label"].fillna("").astype(str).str.lower()
    links = dictionary["linksto"].fillna("").astype(str).str.lower()
    mcs_mask = links.eq("procedureevents") & labels.str.contains(
        r"\biabp\b|intra.?aortic.*balloon|impella|\becmo\b|extracorporeal membrane|ventricular assist|\blvad\b|\brvad\b|\bbivad\b",
        regex=True,
        na=False,
    )
    configured_mcs = {
        int(itemid)
        for itemids in MCS_PROCEDUREEVENT_IDS.values()
        for itemid in itemids
    }
    mcs_items = set(dictionary.loc[mcs_mask, "itemid"].astype(int)) | configured_mcs
    rrt_mask = links.eq("procedureevents") & labels.str.contains(
        r"\bcrrt\b|\bcvvh\b|\bcvvhd\b|\bdialysis\b|hemofiltration|hemodialysis",
        regex=True,
        na=False,
    ) & ~labels.str.contains(r"catheter|access|site|dressing", regex=True, na=False)
    rrt_items = set(dictionary.loc[rrt_mask, "itemid"].astype(int))
    invasive_ventilation_mask = links.eq("procedureevents") & labels.str.fullmatch(
        r"invasive ventilation", na=False
    )
    invasive_ventilation_items = (
        set(dictionary.loc[invasive_ventilation_mask, "itemid"].astype(int))
        | MIMIC_INVASIVE_VENTILATION_PROCEDURE_IDS
    )
    intubation_mask = links.eq("procedureevents") & labels.str.fullmatch(
        r"intubation", na=False
    )
    intubation_items = (
        set(dictionary.loc[intubation_mask, "itemid"].astype(int))
        | MIMIC_INTUBATION_PROCEDURE_IDS
    )
    item_sets = {
        "mcs": mcs_items,
        "rrt": rrt_items,
        "mechanical_ventilation": invasive_ventilation_items,
        "intubation": intubation_items,
    }
    label_map = dictionary.set_index("itemid")["label"].fillna("").astype(str).to_dict()
    procedures = pd.read_csv(
        path,
        usecols=["stay_id", "starttime", "endtime", "itemid"],
    )
    procedures = procedures.loc[
        procedures["stay_id"].isin(cohort_stays)
        & procedures["itemid"].isin(set().union(*item_sets.values()))
    ].copy()
    if procedures.empty:
        return pd.DataFrame(columns=EVENT_REQUIRED_COLUMNS), {
            f"{concept}_rows": 0 for concept in item_sets
        }
    procedures["starttime"] = pd.to_datetime(
        procedures["starttime"], errors="coerce"
    )
    procedures["endtime"] = pd.to_datetime(procedures["endtime"], errors="coerce")
    procedures = procedures.merge(
        anchors[["stay_id", "anchor_time"]],
        on="stay_id",
        how="inner",
        validate="many_to_one",
    )
    procedures["start_offset"] = (
        (procedures["starttime"] - procedures["anchor_time"]).dt.total_seconds()
        / 60.0
    )
    procedures["end_offset"] = (
        (procedures["endtime"] - procedures["anchor_time"]).dt.total_seconds()
        / 60.0
    )
    procedures["end_offset"] = procedures["end_offset"].where(
        procedures["end_offset"].gt(procedures["start_offset"]),
        procedures["start_offset"],
    )
    procedures["label"] = procedures["itemid"].map(label_map).fillna(
        procedures["itemid"].astype(str)
    )
    frames: list[pd.DataFrame] = []
    counts: dict[str, int] = {}
    for concept, itemids in item_sets.items():
        selected = procedures.loc[procedures["itemid"].isin(itemids)].copy()
        counts[f"{concept}_rows"] = int(len(selected))
        if selected.empty:
            continue
        frames.append(
            build_event_frame(
                dataset="mimic",
                stay_id=selected["stay_id"].astype(int),
                event_family=(
                    "respiratory_support_procedure"
                    if concept in {"mechanical_ventilation", "intubation"}
                    else "intervention"
                ),
                concept=concept,
                source_table="procedureevents.csv",
                raw_name=selected["label"],
                offset_minutes=selected["start_offset"],
                value_numeric=1.0,
                value_text=selected["label"],
                unit="binary",
                is_intervention=int(
                    concept in {"mcs", "mechanical_ventilation", "intubation"}
                ),
            )
        )
        if concept == "intubation":
            continue
        status_rows: list[dict[str, object]] = []
        for stay_id, group in selected.groupby("stay_id", sort=False):
            for point in (0.0, 240.0):
                active = group.loc[
                    group["start_offset"].le(point)
                    & group["end_offset"].ge(point)
                ]
                status_rows.append(
                    {
                        "stay_id": int(stay_id),
                        "offset_minutes": point,
                        "active": int(not active.empty),
                        "labels": "+".join(sorted(active["label"].astype(str).unique())),
                    }
                )
        if status_rows:
            status = pd.DataFrame(status_rows)
            frames.append(
                build_event_frame(
                    dataset="mimic",
                    stay_id=status["stay_id"],
                    event_family="intervention_status",
                    concept=f"{concept}_active",
                    source_table="procedureevents.csv",
                    raw_name=status["labels"],
                    offset_minutes=status["offset_minutes"],
                    value_numeric=status["active"],
                    value_text=status["labels"],
                    unit="binary",
                    is_intervention=0,
                )
            )
    return _concat_or_empty(frames), counts


def extract_mimic_respiratory_procedure_events(
    root: str | Path,
    cohort_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Extract only direct invasive-ventilation/intubation procedure events.

    The helper supports a fast analysis-only endpoint repair: it scans the
    relatively small MIMIC ``procedureevents`` table and does not rebuild or
    mutate the fingerprinted cohort/event artifacts.
    """
    root = Path(root)
    required = {"stay_id", "admit_time"}
    missing = required.difference(cohort_df.columns)
    if missing:
        raise ValueError(
            "MIMIC respiratory procedure extraction requires cohort columns: "
            + ", ".join(sorted(missing))
        )
    mimic = cohort_df.loc[
        cohort_df.get("dataset", pd.Series("mimic", index=cohort_df.index))
        .astype(str)
        .str.lower()
        .eq("mimic")
    ].copy()
    anchors = (
        mimic[["stay_id", "admit_time"]]
        .drop_duplicates("stay_id")
        .rename(columns={"admit_time": "anchor_time"})
    )
    anchors["anchor_time"] = pd.to_datetime(anchors["anchor_time"], errors="coerce")
    anchors = anchors.dropna(subset=["stay_id", "anchor_time"])
    events, counts = _extract_mimic_mcs_events(
        root,
        anchors["stay_id"].astype(int).to_numpy(),
        anchors,
    )
    respiratory_concepts = {
        "mechanical_ventilation",
        "mechanical_ventilation_active",
        "intubation",
    }
    if events.empty:
        return events, {
            key: value
            for key, value in counts.items()
            if key in {"mechanical_ventilation_rows", "intubation_rows"}
        }
    return (
        events.loc[events["concept"].isin(respiratory_concepts)].reset_index(drop=True),
        {
            key: value
            for key, value in counts.items()
            if key in {"mechanical_ventilation_rows", "intubation_rows"}
        },
    )


def _mimic_chartevent_item_map(root: Path) -> dict[int, str]:
    """Build the vital and structured respiratory/treatment context map."""
    item_map: dict[int, str] = {
        int(itemid): concept
        for concept, itemids in MIMIC_VITAL_IDS.items()
        for itemid in itemids
    }
    dictionary_path = root / "d_items.csv"
    if not dictionary_path.exists():
        return item_map
    dictionary = pd.read_csv(
        dictionary_path,
        usecols=["itemid", "label", "linksto", "category"],
    )
    dictionary = dictionary.loc[
        dictionary["linksto"].fillna("").astype(str).str.lower().eq("chartevents")
    ].copy()
    label = dictionary["label"].fillna("").astype(str).str.lower()
    category = dictionary["category"].fillna("").astype(str).str.lower()
    extracorporeal_context = (
        category.str.contains(r"\becmo\b|tandem heart", regex=True, na=False)
        | label.str.contains(r"\((?:ecmo|ch)\)", regex=True, na=False)
    )
    rules: list[tuple[pd.Series, str]] = [
        (
            label.str.contains(
                r"inspired o2 fraction|\bfio2\b", regex=True, na=False
            )
            & ~extracorporeal_context,
            "fio2",
        ),
        (label.str.contains(r"^o2 flow$|oxygen flow", regex=True, na=False), "oxygen_flow"),
        (label.str.contains(r"oxygen delivery|o2 delivery|oxygen device", regex=True, na=False), "oxygen_device"),
        (label.str.contains(r"ventilator mode|ventilation mode", regex=True, na=False), "ventilator_mode"),
        (label.str.contains(r"airway type|endotracheal tube|ett placement", regex=True, na=False), "airway_device"),
        (
            category.eq("dialysis")
            & ~label.str.contains(r"site|appearance|dressing", regex=True, na=False),
            "rrt_context",
        ),
    ]
    for mask, concept in rules:
        for itemid in dictionary.loc[mask, "itemid"].dropna().astype(int):
            item_map.setdefault(int(itemid), concept)

    # Operational device measurements provide documentation of support even
    # when a dedicated procedure-line row is absent (notably LVAD/RVAD).  Map
    # only operational settings/flows, not dressing, removal, site, alarm, or
    # placement-check rows, which are much weaker evidence of active support.
    device_text = (label + " " + category).str.strip()
    operational = label.str.contains(
        r"\bflow\b|\bspeed\b|\bsweep\b|\bfio2\b|performance level|pump setting|"
        r"balloon pump setting|\biabp setting\b|assisted systole|"
        r"augmented diastole|\bbaedp\b|\biabp mean\b|balloon waveform|"
        r"plateau pressure|motor current|circuit configuration|purge pressure|"
        r"purge solution flow|cardiac power output|cardiac output",
        regex=True,
        na=False,
    )
    excluded_documentation = label.str.contains(
        r"discontinu|remov|explant|dressing|site|alarm|power source|"
        r"emergency equipment|visually inspected|tubing change|cap change|"
        r"zero.?calibrat|reposition|placement confirmed|suction event",
        regex=True,
        na=False,
    )
    family_masks: list[tuple[pd.Series, str]] = [
        (
            device_text.str.contains(
                r"\bbivad\b|biventricular assist", regex=True, na=False
            ),
            "mcs_context_bivad",
        ),
        (
            device_text.str.contains(
                r"\blvad\b|left ventricular assi[st]+ device",
                regex=True,
                na=False,
            ),
            "mcs_context_lvad",
        ),
        (
            device_text.str.contains(
                r"\brvad\b|right ventricular assi[st]+ device",
                regex=True,
                na=False,
            ),
            "mcs_context_rvad",
        ),
        (
            device_text.str.contains(
                r"\biabp\b|intra.?aortic.*balloon", regex=True, na=False
            ),
            "mcs_context_iabp",
        ),
        (
            device_text.str.contains(r"\bimpella\b", regex=True, na=False),
            "mcs_context_impella",
        ),
        (
            device_text.str.contains(
                r"\becmo\b|extracorporeal membrane|tandem heart",
                regex=True,
                na=False,
            ),
            "mcs_context_ecmo",
        ),
    ]
    for family_mask, concept in family_masks:
        mask = family_mask & operational & ~excluded_documentation
        for itemid in dictionary.loc[mask, "itemid"].dropna().astype(int):
            item_map.setdefault(int(itemid), concept)
    return item_map


def _mimic_lab_item_map(root: Path) -> dict[int, str]:
    """Resolve all blood-lab ItemIDs, including newer MIMIC dictionary IDs.

    MIMIC-IV dictionary revisions added parallel IDs for several core labs.
    Static IDs remain a fallback, while the mounted ``d_labitems`` dictionary
    prevents silent loss when a newer release is supplied.
    """
    item_map: dict[int, str] = {
        int(itemid): concept
        for concept, itemids in MIMIC_LAB_IDS.items()
        for itemid in itemids
    }
    dictionary_path = root / "d_labitems.csv"
    if not dictionary_path.exists():
        return item_map
    dictionary = pd.read_csv(
        dictionary_path,
        usecols=["itemid", "label", "fluid"],
    )
    dictionary = dictionary.loc[
        dictionary["fluid"].fillna("").astype(str).str.strip().str.lower().eq("blood")
    ].copy()
    label = dictionary["label"].fillna("").astype(str).str.strip().str.lower()
    rules: list[tuple[pd.Series, str]] = [
        (label.eq("lactate"), "lactate"),
        (label.eq("ph"), "ph"),
        (
            label.isin(["creatinine", "creatinine, whole blood"]),
            "creatinine",
        ),
        (label.eq("urea nitrogen"), "bun"),
        (label.str.contains(r"^alanine aminotransferase(?: \(alt\))?$", regex=True), "alt"),
        (label.str.contains(r"^(?:asparate|aspartate) aminotransferase(?: \(ast\))?$", regex=True), "ast"),
        (label.eq("bilirubin, total"), "bilirubin_total"),
        (label.eq("troponin t"), "troponin_t"),
        (label.eq("troponin i"), "troponin_i"),
        (label.eq("platelet count"), "platelets"),
        (label.eq("inr(pt)"), "inr"),
    ]
    for mask, concept in rules:
        for itemid in dictionary.loc[mask, "itemid"].dropna().astype(int):
            item_map[int(itemid)] = concept
    return item_map


def _stream_mimic_measurement_events(
    root: Path,
    *,
    file_name: str,
    source_table: str,
    item_map: dict[int, str],
    event_family: str,
    cohort_ids: np.ndarray,
    anchors: pd.DataFrame,
    id_column: str,
    offset_min: float,
    offset_max: float,
    upper_inclusive: bool,
    extended_context_concepts: set[str] | None = None,
    chunk_size: int,
    max_chunks: int | None,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Stream measurements with a concept-level extraction funnel."""
    frames: list[pd.DataFrame] = []
    funnel: dict[str, dict[str, int]] = {
        concept: {
            "raw_item_rows": 0,
            "cohort_rows": 0,
            "quality_pass_rows": 0,
            "numeric_rows": 0,
            "window_rows": 0,
            "plausible_rows": 0,
        }
        for concept in sorted(set(item_map.values()))
    }
    header = pd.read_csv(root / file_name, nrows=0).columns.tolist()
    requested = [id_column, "itemid", "charttime", "valuenum", "value", "valueuom", "warning"]
    usecols = [column for column in requested if column in header]
    nonnumeric_concepts = {
        "oxygen_device",
        "ventilator_mode",
        "airway_device",
        "rrt_context",
    }
    anchor_lookup = anchors.drop_duplicates(id_column).set_index(id_column)
    for chunk in _iter_csv_chunks(
        root / file_name,
        usecols=usecols,
        chunksize=chunk_size,
        max_chunks=max_chunks,
    ):
        mapped_mask = chunk["itemid"].isin(item_map)
        for concept, count in chunk.loc[mapped_mask, "itemid"].map(item_map).value_counts().items():
            funnel[str(concept)]["raw_item_rows"] += int(count)
        filtered = chunk.loc[mapped_mask & chunk[id_column].isin(cohort_ids)].copy()
        if filtered.empty:
            continue
        filtered["concept"] = filtered["itemid"].map(item_map)
        for concept, count in filtered["concept"].value_counts().items():
            funnel[str(concept)]["cohort_rows"] += int(count)
        if "warning" in filtered:
            warning = pd.to_numeric(filtered["warning"], errors="coerce").fillna(0)
            filtered = filtered.loc[warning.ne(1)].copy()
        for concept, count in filtered["concept"].value_counts().items():
            funnel[str(concept)]["quality_pass_rows"] += int(count)
        filtered["valuenum"] = pd.to_numeric(filtered["valuenum"], errors="coerce")
        text_value = (
            filtered["value"].astype("string")
            if "value" in filtered
            else pd.Series(pd.NA, index=filtered.index, dtype="string")
        )
        nonnumeric_context = filtered["concept"].isin(nonnumeric_concepts) | (
            filtered["concept"].astype(str).str.startswith("mcs_context_")
        )
        keep = filtered["valuenum"].notna() | (
            nonnumeric_context & text_value.notna()
        )
        filtered = filtered.loc[keep].copy()
        text_value = text_value.loc[filtered.index]
        for concept, count in filtered.loc[filtered["valuenum"].notna(), "concept"].value_counts().items():
            funnel[str(concept)]["numeric_rows"] += int(count)
        filtered["charttime"] = pd.to_datetime(filtered["charttime"], errors="coerce")
        filtered["anchor_time"] = filtered[id_column].map(anchor_lookup["anchor_time"])
        filtered["stay_id_normalized"] = (
            filtered[id_column]
            if id_column == "stay_id"
            else filtered[id_column].map(anchor_lookup["stay_id"])
        )
        filtered["offset_minutes"] = (
            (filtered["charttime"] - filtered["anchor_time"]).dt.total_seconds() / 60.0
        )
        if upper_inclusive:
            primary_window = filtered["offset_minutes"].between(
                offset_min, offset_max, inclusive="both"
            )
        else:
            primary_window = filtered["offset_minutes"].ge(offset_min) & filtered[
                "offset_minutes"
            ].lt(offset_max)
        extended_window = pd.Series(False, index=filtered.index)
        if extended_context_concepts:
            extended_window = (
                filtered["concept"].isin(extended_context_concepts)
                & filtered["offset_minutes"].ge(0)
                & filtered["offset_minutes"].le(OUTCOME_WINDOW_END_HOURS * 60)
            )
        filtered = filtered.loc[primary_window | extended_window].copy()
        if filtered.empty:
            continue
        mcs_context = filtered["concept"].astype(str).str.startswith(
            "mcs_context_"
        )
        if mcs_context.any():
            context_numeric = pd.to_numeric(
                filtered["valuenum"], errors="coerce"
            )
            context_text = text_value.reindex(filtered.index).fillna("").map(
                normalize_text
            )
            inactive_context = context_text.str.contains(
                r"\bnone\b|\boff\b|discontinu|remov|explant|not in use|"
                r"not applicable|standby",
                regex=True,
                na=False,
            )
            filtered = filtered.loc[
                ~mcs_context
                | (
                    ~inactive_context
                    & (context_numeric.isna() | context_numeric.gt(0))
                )
            ].copy()
            text_value = text_value.reindex(filtered.index)
            if filtered.empty:
                continue
        for concept, count in filtered["concept"].value_counts().items():
            funnel[str(concept)]["window_rows"] += int(count)
        spo2_values = filtered.loc[filtered["concept"].eq("spo2"), "valuenum"]
        if "spo2" in funnel:
            funnel["spo2"]["plausible_rows"] += int(
                spo2_values.between(50, 100, inclusive="both").sum()
            )
        fio2_mask = filtered["concept"].eq("fio2")
        if fio2_mask.any():
            fio2 = filtered.loc[fio2_mask, "valuenum"].astype(float)
            fio2 = fio2.where(~fio2.between(0.20, 1.0), fio2 * 100.0)
            filtered.loc[fio2_mask, "valuenum"] = fio2
            funnel["fio2"]["plausible_rows"] += int(fio2.between(21, 100).sum())
        # Harmonize temperature and weight before either enters a cross-dataset
        # adjustment model.  MIMIC stores both Fahrenheit/Celsius and kg/lb IDs.
        temp_f_mask = filtered["concept"].eq("temp") & (
            filtered["itemid"].eq(223761)
            | filtered.get("valueuom", pd.Series("", index=filtered.index))
            .fillna("")
            .astype(str)
            .str.lower()
            .str.contains(r"°?f|fahrenheit", regex=True)
        )
        if temp_f_mask.any():
            filtered.loc[temp_f_mask, "valuenum"] = (
                filtered.loc[temp_f_mask, "valuenum"] - 32.0
            ) * (5.0 / 9.0)
        weight_lb_mask = filtered["concept"].eq("weight") & (
            filtered["itemid"].eq(226531)
            | filtered.get("valueuom", pd.Series("", index=filtered.index))
            .fillna("")
            .astype(str)
            .str.lower()
            .str.contains(r"lb|pound", regex=True)
        )
        if weight_lb_mask.any():
            filtered.loc[weight_lb_mask, "valuenum"] = (
                filtered.loc[weight_lb_mask, "valuenum"] * 0.45359237
            )
        for concept in set(filtered["concept"].astype(str)) - {"spo2", "fio2"}:
            funnel[concept]["plausible_rows"] += int(filtered["concept"].eq(concept).sum())
        filtered["value_text"] = text_value.reindex(filtered.index)
        filtered["unit"] = (
            filtered["valueuom"].astype("string")
            if "valueuom" in filtered
            else pd.Series(pd.NA, index=filtered.index, dtype="string")
        )
        filtered = filtered.dropna(subset=["stay_id_normalized", "offset_minutes"])
        filtered = filtered.sort_values(
            ["stay_id_normalized", "concept", "offset_minutes", "itemid"]
        ).drop_duplicates(
            ["stay_id_normalized", "concept", "offset_minutes", "valuenum", "value_text"],
            keep="first",
        )
        frames.append(
            build_event_frame(
                dataset="mimic",
                stay_id=filtered["stay_id_normalized"].astype(int),
                event_family=event_family,
                concept=filtered["concept"],
                source_table=source_table,
                raw_name=filtered["itemid"].astype(str),
                offset_minutes=filtered["offset_minutes"],
                value_numeric=filtered["valuenum"],
                value_text=filtered["value_text"],
                unit=filtered["unit"],
                is_intervention=0,
            )
        )
    return _concat_or_empty(frames), funnel


def _discover_mimic_spo2_stays(
    root: Path,
    *,
    chunk_size: int,
    max_chunks: int | None,
) -> set[int] | None:
    """Return SpO2-bearing stays in a deliberately truncated pilot scan.

    A pilot must sample cohort IDs from the same physical chunks it reads;
    independently selecting the lowest cohort IDs is the bug that produced the
    historical all-zero MIMIC result.
    """
    if max_chunks is None:
        return None
    stays: set[int] = set()
    for chunk in _iter_csv_chunks(
        root / "chartevents.csv",
        usecols=["stay_id", "itemid", "valuenum", "warning"],
        chunksize=chunk_size,
        max_chunks=max_chunks,
    ):
        values = pd.to_numeric(chunk["valuenum"], errors="coerce")
        warning = pd.to_numeric(chunk["warning"], errors="coerce").fillna(0)
        mask = (
            chunk["itemid"].eq(220277)
            & values.between(50, 100, inclusive="both")
            & warning.ne(1)
        )
        stays.update(chunk.loc[mask, "stay_id"].dropna().astype(int).tolist())
    return stays


def extract_mimic(
    root: Path,
    audit: AuditLogger,
    *,
    max_stays: int | None = None,
    max_chunks: int | None = None,
    chunk_size: int = CHUNK_SIZE,
) -> SourceExtraction:
    """Extract MIMIC-IV cohort and clinical events.

    Performs harmonized ICD-based HF cohort identification, pressor/MCS and
    direct respiratory-procedure extraction, chunked vitals/labs streaming,
    and death event construction.

    Args:
        root: Path to MIMIC data directory.
        audit: AuditLogger for recording step counts.
        max_stays: Optional cap on cohort size.
        max_chunks: Optional cap on chunks per streaming table.
        chunk_size: Rows per chunk for CSV streaming (default: 250,000).

    Returns:
        SourceExtraction with cohort_df and events_df.
    """
    preferred_stays = _discover_mimic_spo2_stays(
        root,
        chunk_size=chunk_size,
        max_chunks=max_chunks,
    )
    cohort_df, cohort_stays, anchors = _load_mimic_cohort(
        root,
        audit,
        max_stays=max_stays,
        preferred_stay_ids=preferred_stays,
    )

    pressor_events, pressor_audit = _extract_mimic_pressor_events(
        root,
        cohort_stays,
        anchors,
        chunk_size=chunk_size,
        max_chunks=max_chunks,
    )
    audit.log(
        "mimic_pressors_extracted",
        row_count=len(pressor_events),
        stay_count=event_stay_count(pressor_events),
        details=pressor_audit,
    )

    mcs_events, procedure_audit = _extract_mimic_mcs_events(
        root, cohort_stays, anchors
    )
    audit.log(
        "mimic_procedures_extracted",
        row_count=len(mcs_events),
        stay_count=event_stay_count(mcs_events),
        details=procedure_audit,
    )

    vital_item_map = _mimic_chartevent_item_map(root)
    vital_events, vital_funnel = _stream_mimic_measurement_events(
        root,
        file_name="chartevents.csv",
        source_table="chartevents.csv",
        item_map=vital_item_map,
        event_family="vital",
        cohort_ids=cohort_stays,
        anchors=anchors,
        id_column="stay_id",
        offset_min=0.0,
        offset_max=240.0,
        upper_inclusive=False,
        extended_context_concepts={
            concept
            for concept in vital_item_map.values()
            if str(concept).startswith("mcs_context_")
        },
        chunk_size=chunk_size,
        max_chunks=max_chunks,
    )
    audit.log(
        "mimic_vitals_streamed",
        row_count=len(vital_events),
        stay_count=event_stay_count(vital_events),
        details={"concept_funnel": vital_funnel},
    )
    spo2_funnel = vital_funnel.get("spo2", {})
    if (
        max_chunks is None
        and len(cohort_df) > 0
        and spo2_funnel.get("raw_item_rows", 0) > 0
        and spo2_funnel.get("plausible_rows", 0) == 0
    ):
        raise RuntimeError(
            "MIMIC full extraction found raw itemid 220277 rows but no plausible "
            "cohort-window SpO2 rows. This is a fatal ID/anchor/join invariant."
        )

    lab_item_map = _mimic_lab_item_map(root)
    lab_events, lab_funnel = _stream_mimic_measurement_events(
        root,
        file_name="labevents.csv",
        source_table="labevents.csv",
        item_map=lab_item_map,
        event_family="lab",
        cohort_ids=anchors["hadm_id"].astype(int).unique(),
        anchors=anchors,
        id_column="hadm_id",
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
        details={"concept_funnel": lab_funnel},
    )

    urine_events = _stream_mimic_urine_output_events(
        root,
        cohort_stays,
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
