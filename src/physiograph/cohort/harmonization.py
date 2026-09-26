"""Analysis-time cohort harmonization for cross-database HF validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


HARMONIZED_HF_PHENOTYPE = (
    "age_ge_18_and_explicit_heart_failure_and_"
    "(icu_within_24h_of_hospital_admission_or_cardiogenic_shock)"
)


def _binary(frame: pd.DataFrame, column: str, *, default: int = 0) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype="int64")
    return (
        pd.to_numeric(frame[column], errors="coerce")
        .fillna(default)
        .eq(1)
        .astype(int)
    )


def _encounter_key(frame: pd.DataFrame) -> pd.Series:
    """Return a database-qualified hospital-encounter identifier."""
    candidates = (
        "hospital_encounter_id",
        "hadm_id",
        "patienthealthsystemstayid",
    )
    identifier = pd.Series(pd.NA, index=frame.index, dtype="string")
    for column in candidates:
        if column not in frame:
            continue
        values = frame[column].astype("string")
        valid = values.notna() & values.str.strip().ne("")
        identifier = identifier.where(identifier.notna(), values.where(valid))
    identifier = identifier.where(
        identifier.notna(), "stay:" + frame["stay_id"].astype("string")
    )
    return frame["dataset"].astype("string") + ":" + identifier


def harmonize_hf_cohort(
    cohort_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply one fail-closed HF phenotype to MIMIC and eICU.

    Encounter-level diagnosis flags are propagated before selecting the first
    ICU anchor.  This prevents a diagnosis documented during a later eICU unit
    transfer from being lost while ensuring that transfers do not contribute
    multiple analysis rows.
    """
    required = {"dataset", "stay_id", "age", "cohort_hf_flag"}
    missing = required.difference(cohort_df.columns)
    if missing:
        raise ValueError(
            "Harmonized HF validation requires cohort columns: "
            + ", ".join(sorted(missing))
        )
    frame = cohort_df.copy()
    frame["dataset"] = frame["dataset"].astype(str).str.lower()
    unsupported = sorted(set(frame["dataset"]).difference({"mimic", "eicu"}))
    if unsupported:
        raise ValueError(
            "Harmonized HF validation received unsupported datasets: "
            + ", ".join(unsupported)
        )
    frame["_encounter_key"] = _encounter_key(frame)
    frame["_input_order"] = np.arange(len(frame))
    if "unitvisitnumber" in frame:
        frame["_icu_order"] = pd.to_numeric(
            frame["unitvisitnumber"], errors="coerce"
        )
    elif "unit_visit_number" in frame:
        frame["_icu_order"] = pd.to_numeric(
            frame["unit_visit_number"], errors="coerce"
        )
    else:
        frame["_icu_order"] = np.nan
    frame["_icu_order"] = frame["_icu_order"].fillna(frame["_input_order"])
    for column in (
        "cohort_hf_flag",
        "shock_icd_flag",
        "early_icu_flag",
        "excluded_before_landmark_flag",
    ):
        frame[column] = _binary(frame, column)
    # Diagnosis labels may occur on any unit transfer in the same hospital
    # encounter.  An exclusion, once present, is also encounter-level.
    for column in (
        "cohort_hf_flag",
        "shock_icd_flag",
        "early_icu_flag",
        "excluded_before_landmark_flag",
    ):
        frame[column] = frame.groupby("_encounter_key", sort=False)[column].transform(
            "max"
        )
    frame["_adult"] = pd.to_numeric(frame["age"], errors="coerce").ge(18)
    frame["_explicit_hf"] = frame["cohort_hf_flag"].eq(1)
    frame["_early_or_shock"] = frame["early_icu_flag"].eq(1) | frame[
        "shock_icd_flag"
    ].eq(1)
    frame["_not_excluded"] = frame["excluded_before_landmark_flag"].eq(0)

    audit_rows: list[dict[str, Any]] = []
    for dataset, group in frame.groupby("dataset", sort=True):
        first = group.sort_values(["_icu_order", "_input_order", "stay_id"]).drop_duplicates(
            "_encounter_key", keep="first"
        )
        included = (
            first["_adult"]
            & first["_explicit_hf"]
            & first["_early_or_shock"]
            & first["_not_excluded"]
        )
        audit_rows.append(
            {
                "dataset": dataset,
                "phenotype": HARMONIZED_HF_PHENOTYPE,
                "input_rows": int(len(group)),
                "input_unique_stays": int(group["stay_id"].nunique()),
                "hospital_encounters": int(first["_encounter_key"].nunique()),
                "transfer_rows_removed": int(len(group) - len(first)),
                "adult_encounters": int(first["_adult"].sum()),
                "explicit_hf_encounters": int(first["_explicit_hf"].sum()),
                "early_icu_or_shock_encounters": int(first["_early_or_shock"].sum()),
                "excluded_before_landmark_encounters": int(
                    (~first["_not_excluded"]).sum()
                ),
                "excluded_no_explicit_hf": int((~first["_explicit_hf"]).sum()),
                "excluded_not_early_icu_or_shock": int(
                    (first["_explicit_hf"] & ~first["_early_or_shock"]).sum()
                ),
                "included_encounters": int(included.sum()),
            }
        )

    first = frame.sort_values(["_icu_order", "_input_order", "stay_id"]).drop_duplicates(
        "_encounter_key", keep="first"
    )
    keep = (
        first["_adult"]
        & first["_explicit_hf"]
        & first["_early_or_shock"]
        & first["_not_excluded"]
    )
    internal = {
        "_encounter_key",
        "_input_order",
        "_icu_order",
        "_adult",
        "_explicit_hf",
        "_early_or_shock",
        "_not_excluded",
    }
    harmonized = first.loc[keep].drop(
        columns=[column for column in internal if column in first]
    )
    harmonized = harmonized.reset_index(drop=True)
    if harmonized.duplicated(["dataset", "stay_id"]).any():
        raise RuntimeError("Harmonized HF cohort contains duplicate ICU stays")
    return harmonized, pd.DataFrame(audit_rows)


def filter_to_harmonized_cohort(
    frame: pd.DataFrame,
    cohort_df: pd.DataFrame,
) -> pd.DataFrame:
    """Restrict an event/analysis table to harmonized dataset-stay keys."""
    if frame.empty:
        return frame.copy()
    required = {"dataset", "stay_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Cohort filtering requires columns: " + ", ".join(sorted(missing))
        )
    keys = cohort_df[["dataset", "stay_id"]].drop_duplicates()
    return frame.merge(keys, on=["dataset", "stay_id"], how="inner").reset_index(
        drop=True
    )


__all__ = [
    "HARMONIZED_HF_PHENOTYPE",
    "filter_to_harmonized_cohort",
    "harmonize_hf_cohort",
]
