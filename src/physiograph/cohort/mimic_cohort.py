"""MIMIC-III heart-failure / cardiogenic-shock cohort builder.

Extracts the HF-shock phenotype definition from the PhysioGraph
HF_Shock notebook (Cell 4) and the External Pipeline notebook
(Cell 37).  The logic is reproduced *exactly* as-is from the
original notebooks — no new exclusion criteria or phenotype changes.

Key steps
---------
1. Match ICD-9/10 codes for heart failure and cardiogenic shock.
2. Identify admissions with early ICU care (≤ 24 h) or shock ICD.
3. Anchor to the **first ICU stay** per admission (landmark time).
4. Exclude patients who received a pressor/MCS intervention or died
   before the 4-hour landmark.

The public entry-point is :func:`build_cohort`, which returns a
:class:`CohortResult` containing the cohort DataFrame, the array of
valid stay IDs, and the anchor mapping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from physiograph.config import load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ICD pattern matching (extracted from Cell 2 / Cell 37)
# ---------------------------------------------------------------------------

# Default ICD patterns — can be overridden via config YAML
_HF_ICD_PATTERNS: dict[int, list[str]] = {
    9: [r"^428"],
    10: [r"^I50"],
}

_CARDIOGENIC_SHOCK_ICD_PATTERNS: dict[int, list[str]] = {
    9: [r"^78551"],
    10: [r"^R570"],
}

# Comorbidity ICD patterns (from Cell 4)
_COMORBIDITY_ICD_PATTERNS: dict[str, dict[int, list[str]]] = {
    "diabetes": {9: [r"^250"], 10: [r"^E1[0-4]"]},
    "ckd": {9: [r"^585"], 10: [r"^N18"]},
    "prior_mi": {9: [r"^412"], 10: [r"^I252"]},
}

# Pressor / MCS item IDs (from Cell 2)
_PRESSOR_ITEMIDS: list[int] = [221906, 221289, 221662, 221653, 221749, 222315]

_MCS_PROCEDUREEVENT_IDS: dict[str, list[int]] = {
    "iabp": [224272],
    "impella": [228169],
    "ecmo": [229529, 229530],
}


# ---------------------------------------------------------------------------
# ICD matching utility
# ---------------------------------------------------------------------------

def match_icd_prefix(
    icd_codes: pd.Series,
    icd_versions: pd.Series,
    pattern_map: dict[int, list[str]],
) -> pd.Series:
    """Match ICD codes against version-specific prefix patterns.

    Parameters
    ----------
    icd_codes:
        Series of ICD code strings (may contain NaN).
    icd_versions:
        Series of ICD version integers (9 or 10).
    pattern_map:
        Mapping from ICD version to a list of regex prefix patterns,
        e.g. ``{9: [r'^428'], 10: [r'^I50']}``.

    Returns
    -------
    pd.Series
        Boolean mask — ``True`` where any pattern matches.
    """
    icd_codes = icd_codes.fillna("").astype(str)
    out = pd.Series(False, index=icd_codes.index)
    for version, patterns in pattern_map.items():
        version_mask = icd_versions == version
        if not patterns:
            continue
        pattern = "|".join(f"(?:{p})" for p in patterns)
        out = out | (version_mask & icd_codes.str.match(pattern, na=False))
    return out


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CohortResult:
    """Container for MIMIC cohort extraction results.

    Attributes
    ----------
    cohort_df:
        Full cohort DataFrame with one row per hospital admission,
        including demographics, flags, and exclusion status.
    valid_stay_ids:
        Array of ``hadm_id`` values that passed all exclusion criteria.
    anchors:
        DataFrame mapping ``hadm_id`` → ``anchor_time`` (first ICU
        admission timestamp) for offset calculations.
    """

    cohort_df: pd.DataFrame
    valid_stay_ids: np.ndarray
    anchors: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------------------
# Cohort builder
# ---------------------------------------------------------------------------

class MIMICCohortBuilder:
    """Build the MIMIC-III HF-shock cohort from raw CSV tables.

    This reproduces the phenotype definition from Cell 4 of the
    PhysioGraph_HF_Shock notebook and Cell 37 of the External Pipeline
    notebook.  The logic is:

    1. Identify HF admissions via ICD-9/10 pattern matching.
    2. Identify cardiogenic-shock admissions via ICD patterns.
    3. Find admissions with early ICU care (first ICU stay within 24 h
       of hospital admission).
    4. Intersect HF admissions with (early ICU ∪ shock) admissions.
    5. Anchor each admission to its first ICU stay time.
    6. Exclude stays with pre-landmark pressor/MCS or death.

    Parameters
    ----------
    data_root:
        Path to the MIMIC-III CSV directory (containing
        ``diagnoses_icd.csv``, ``admissions.csv``, etc.).
    config:
        Optional pre-loaded config dict.  If ``None``,
        :func:`~physiograph.config.load_config` is called with
        ``dataset="mimic"``.
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
        self.config = config or load_config("mimic")
        self.max_stays = max_stays

        # Allow config overrides for ICD patterns
        cfg_icd = self.config.get("hf_icd_patterns", _HF_ICD_PATTERNS)
        self.hf_icd_patterns = {
            int(k): v for k, v in cfg_icd.items()
        }
        cfg_shock = self.config.get("cardiogenic_shock_icd_patterns", _CARDIOGENIC_SHOCK_ICD_PATTERNS)
        self.cardiogenic_shock_icd_patterns = {
            int(k): v for k, v in cfg_shock.items()
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_cohort(self) -> CohortResult:
        """Execute the full MIMIC cohort extraction pipeline.

        Returns
        -------
        CohortResult
            Container with ``cohort_df``, ``valid_stay_ids``, and
            ``anchors``.
        """
        dx = self._load_diagnoses()
        hf_hadms, shock_hadms = self._identify_phenotype_admissions(dx)

        admissions = self._load_admissions(hf_hadms)
        early_icu_hadms, icu_anchor, icu_anchor_hadms = self._compute_icu_anchors(admissions)

        # Cohort = HF ∩ (early ICU ∪ shock) ∩ has_anchor
        cohort_hadms = np.intersect1d(
            hf_hadms,
            np.union1d(early_icu_hadms, shock_hadms),
        )
        cohort_hadms = np.intersect1d(cohort_hadms, icu_anchor_hadms)

        if self.max_stays is not None:
            cohort_hadms = np.array(sorted(cohort_hadms)[: self.max_stays], dtype=int)

        admissions = admissions.loc[admissions["hadm_id"].isin(cohort_hadms)].copy()
        admissions = admissions.merge(icu_anchor, on="hadm_id", how="inner")

        # Compute death offset relative to ICU anchor
        admissions["death_offset_minutes"] = (
            (admissions["deathtime"] - admissions["anchor_time"])
            .dt.total_seconds()
            / 60.0
        )

        # Comorbidities
        admissions = self._add_comorbidities(admissions, dx, cohort_hadms)

        # Pre-landmark exclusions
        early_interv_ids, early_death_ids = self._find_pre_landmark_exclusions(
            admissions, cohort_hadms
        )
        invalid_before_landmark = np.union1d(early_interv_ids, early_death_ids)
        valid_hadms = np.setdiff1d(cohort_hadms, invalid_before_landmark)

        logger.info(
            "MIMIC cohort: %d HF admissions, %d early ICU, %d shock ICD, "
            "%d after exclusions",
            len(hf_hadms),
            len(early_icu_hadms),
            len(shock_hadms),
            len(valid_hadms),
        )

        # Build cohort DataFrame
        cohort_df = self._build_cohort_df(
            admissions=admissions,
            cohort_hadms=cohort_hadms,
            shock_hadms=shock_hadms,
            early_icu_hadms=early_icu_hadms,
            early_interv_ids=early_interv_ids,
            early_death_ids=early_death_ids,
        )

        anchors = admissions[["hadm_id", "anchor_time"]].copy()

        return CohortResult(
            cohort_df=cohort_df,
            valid_stay_ids=valid_hadms,
            anchors=anchors,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_diagnoses(self) -> pd.DataFrame:
        """Load and return the diagnoses_icd table."""
        path = self.data_root / "diagnoses_icd.csv"
        logger.info("Loading diagnoses from %s", path)
        return pd.read_csv(path, usecols=["hadm_id", "icd_code", "icd_version"])

    def _identify_phenotype_admissions(
        self, dx: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Identify HF and cardiogenic-shock admissions via ICD patterns.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(hf_hadms, shock_hadms)`` — arrays of unique ``hadm_id``
            values matching each phenotype.
        """
        dx["icd_code"] = dx["icd_code"].fillna("").astype(str)

        hf_mask = match_icd_prefix(dx["icd_code"], dx["icd_version"], self.hf_icd_patterns)
        shock_mask = match_icd_prefix(
            dx["icd_code"], dx["icd_version"], self.cardiogenic_shock_icd_patterns
        )

        hf_hadms = dx.loc[hf_mask, "hadm_id"].unique()
        shock_hadms = dx.loc[shock_mask, "hadm_id"].unique()

        logger.info(
            "Identified %d HF admissions, %d shock admissions",
            len(hf_hadms),
            len(shock_hadms),
        )
        return hf_hadms, shock_hadms

    def _load_admissions(self, hf_hadms: np.ndarray) -> pd.DataFrame:
        """Load admissions and patients, filtered to HF admissions.

        Parameters
        ----------
        hf_hadms:
            Array of hospital admission IDs with HF diagnosis.

        Returns
        -------
        pd.DataFrame
            Merged admissions + patients DataFrame with computed
            ``age`` and ``is_male`` columns.
        """
        adm = pd.read_csv(
            self.data_root / "admissions.csv",
            usecols=["hadm_id", "subject_id", "admittime", "deathtime", "hospital_expire_flag"],
            parse_dates=["admittime", "deathtime"],
        )
        adm = adm.loc[adm["hadm_id"].isin(hf_hadms)].copy()
        adm["admit_year"] = adm["admittime"].dt.year.astype(int)

        patients = pd.read_csv(
            self.data_root / "patients.csv",
            usecols=["subject_id", "gender", "anchor_age", "anchor_year"],
        )
        adm = adm.merge(patients, on="subject_id", how="left")
        adm["age"] = adm["anchor_age"] + (adm["admit_year"] - adm["anchor_year"])
        adm["is_male"] = (adm["gender"] == "M").astype(int)

        return adm

    def _compute_icu_anchors(
        self, admissions: pd.DataFrame
    ) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
        """Compute first-ICU-stay anchors for each admission.

        Returns
        -------
        tuple[np.ndarray, pd.DataFrame, np.ndarray]
            ``(early_icu_hadms, icu_anchor_df, icu_anchor_hadms)``
            where ``icu_anchor_df`` has columns ``[hadm_id, anchor_time]``.
        """
        icu = pd.read_csv(
            self.data_root / "icustays.csv",
            usecols=["hadm_id", "intime"],
            parse_dates=["intime"],
        )
        icu = icu.merge(admissions[["hadm_id", "admittime"]], on="hadm_id", how="inner")
        icu["hrs_to_icu"] = (icu["intime"] - icu["admittime"]).dt.total_seconds() / 3600.0
        icu = icu.sort_values(["hadm_id", "intime"])

        first_icu = icu.drop_duplicates(subset=["hadm_id"], keep="first").copy()
        early_icu = first_icu.loc[
            (first_icu["hrs_to_icu"] >= 0) & (first_icu["hrs_to_icu"] <= 24)
        ].copy()
        early_icu_hadms = early_icu["hadm_id"].unique()

        icu_anchor = first_icu[["hadm_id", "intime"]].rename(
            columns={"intime": "anchor_time"}
        )
        icu_anchor_hadms = icu_anchor["hadm_id"].unique()

        return early_icu_hadms, icu_anchor, icu_anchor_hadms

    def _add_comorbidities(
        self,
        admissions: pd.DataFrame,
        dx: pd.DataFrame,
        cohort_hadms: np.ndarray,
    ) -> pd.DataFrame:
        """Add comorbidity flags (diabetes, CKD, prior MI) to admissions.

        Parameters
        ----------
        admissions:
            Admissions DataFrame (will be modified in-place).
        dx:
            Full diagnoses_icd DataFrame.
        cohort_hadms:
            Array of cohort admission IDs.

        Returns
        -------
        pd.DataFrame
            Admissions with ``cmb_diabetes``, ``cmb_ckd``, ``cmb_prior_mi``
            columns added.
        """
        dx_all = dx.loc[dx["hadm_id"].isin(cohort_hadms)].copy()
        dx_all["icd_code"] = dx_all["icd_code"].fillna("").astype(str)

        for cmb_name, patterns in _COMORBIDITY_ICD_PATTERNS.items():
            mask = match_icd_prefix(dx_all["icd_code"], dx_all["icd_version"], patterns)
            cmb_hadms = dx_all.loc[mask, "hadm_id"].unique()
            admissions[f"cmb_{cmb_name}"] = admissions["hadm_id"].isin(cmb_hadms).astype(int)

        return admissions

    def _find_pre_landmark_exclusions(
        self,
        admissions: pd.DataFrame,
        cohort_hadms: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find stays with pressor/MCS or death before the 4-hour landmark.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(early_interv_ids, early_death_ids)`` — arrays of
            ``hadm_id`` values excluded for each reason.
        """
        observation_hours = self.config.get("observation_hours", 4.0)

        # Pressor exclusions
        try:
            inputs = pd.read_csv(
                self.data_root / "inputevents.csv",
                usecols=["hadm_id", "starttime", "endtime", "itemid", "rate"],
            )
        except ValueError:
            inputs = pd.read_csv(self.data_root / "inputevents.csv")

        inputs = inputs.loc[inputs["hadm_id"].isin(cohort_hadms)].copy()
        inputs = inputs.loc[inputs["itemid"].isin(_PRESSOR_ITEMIDS)].dropna(subset=["rate"])
        inputs = inputs.loc[inputs["rate"] > 0.0].copy()
        inputs["starttime"] = pd.to_datetime(inputs["starttime"], errors="coerce")
        inputs["endtime"] = pd.to_datetime(inputs["endtime"], errors="coerce")
        inputs["duration_hrs"] = (inputs["endtime"] - inputs["starttime"]).dt.total_seconds() / 3600.0
        sustained = inputs.loc[inputs["duration_hrs"] >= 1.0].copy()

        # MCS exclusions
        proc = pd.read_csv(
            self.data_root / "procedureevents.csv",
            usecols=["hadm_id", "starttime", "itemid"],
        )
        proc = proc.loc[proc["hadm_id"].isin(cohort_hadms)].copy()
        proc["starttime"] = pd.to_datetime(proc["starttime"], errors="coerce")
        mcs_itemids = [iid for ids in _MCS_PROCEDUREEVENT_IDS.values() for iid in ids]
        mcs_events = proc.loc[proc["itemid"].isin(mcs_itemids)].copy()

        # Merge with anchor times for offset calculation
        anchor_map = admissions[["hadm_id", "anchor_time"]].copy()
        sustained = sustained.merge(anchor_map, on="hadm_id", how="left")
        mcs_events = mcs_events.merge(anchor_map, on="hadm_id", how="left")

        sustained["hrs_in"] = (
            (sustained["starttime"] - sustained["anchor_time"]).dt.total_seconds() / 3600.0
        )
        mcs_events["hrs_in"] = (
            (mcs_events["starttime"] - mcs_events["anchor_time"]).dt.total_seconds() / 3600.0
        )

        early_interv_ids = np.union1d(
            sustained.loc[sustained["hrs_in"] <= observation_hours, "hadm_id"].unique(),
            mcs_events.loc[mcs_events["hrs_in"] <= observation_hours, "hadm_id"].unique(),
        )

        # Death exclusions
        admissions["hrs_to_death"] = (
            (admissions["deathtime"] - admissions["anchor_time"]).dt.total_seconds() / 3600.0
        )
        early_death_ids = admissions.loc[
            admissions["hrs_to_death"].notna()
            & (admissions["hrs_to_death"] >= 0)
            & (admissions["hrs_to_death"] <= observation_hours),
            "hadm_id",
        ].unique()

        return early_interv_ids, early_death_ids

    def _build_cohort_df(
        self,
        admissions: pd.DataFrame,
        cohort_hadms: np.ndarray,
        shock_hadms: np.ndarray,
        early_icu_hadms: np.ndarray,
        early_interv_ids: np.ndarray,
        early_death_ids: np.ndarray,
    ) -> pd.DataFrame:
        """Assemble the final cohort DataFrame.

        Parameters
        ----------
        admissions:
            Admissions with demographics, comorbidities, and anchor times.
        cohort_hadms:
            All cohort admission IDs (before exclusions).
        shock_hadms:
            Admissions with cardiogenic shock ICD codes.
        early_icu_hadms:
            Admissions with early ICU care.
        early_interv_ids:
            Admissions excluded for pre-landmark intervention.
        early_death_ids:
            Admissions excluded for pre-landmark death.

        Returns
        -------
        pd.DataFrame
            Cohort DataFrame with the standard column schema.
        """
        invalid_before_landmark = np.union1d(early_interv_ids, early_death_ids)

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
                "shock_icd_flag": admissions["hadm_id"].isin(shock_hadms).astype(int),
                "early_icu_flag": admissions["hadm_id"].isin(early_icu_hadms).astype(int),
                "death_offset_minutes": admissions["death_offset_minutes"],
                "excluded_before_landmark_flag": admissions["hadm_id"].isin(invalid_before_landmark).astype(int),
                "exclusion_reason": "",
            }
        )

        # Fill exclusion reasons
        interv_set = set(early_interv_ids)
        death_set = set(early_death_ids)
        reasons: list[str] = []
        for hadm_id in cohort_df["stay_id"]:
            r: list[str] = []
            if hadm_id in interv_set:
                r.append("intervention_before_or_at_4h")
            if hadm_id in death_set:
                r.append("death_before_or_at_4h")
            reasons.append(";".join(r))
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
    """Build the MIMIC-III HF-shock cohort.

    This is a convenience wrapper around :class:`MIMICCohortBuilder`.

    Parameters
    ----------
    data_root:
        Path to the MIMIC-III CSV directory.
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
    builder = MIMICCohortBuilder(data_root=data_root, config=config, max_stays=max_stays)
    return builder.build_cohort()