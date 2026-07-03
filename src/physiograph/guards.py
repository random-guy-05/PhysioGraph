"""Runtime guards and precondition checks.

Data leakage prevention is the single most common failure mode in
clinical ML pipelines. This module provides systematic guards that
enforce:

1. No patient overlap between train/test splits
2. No post-landmark features in training data
3. No outcome columns in feature matrices
4. Fit-on-train / transform-on-test preprocessing (YAIB pattern)
5. PROBAST+AI Domain 4 compliance (no temporal leakage, no
   patient overlap, no outcome contamination)

These guards are extracted and hardened from the existing
notebook-based analysis (PhysioGraph_HF_Shock.ipynb and
PhysioGraph_External_Pipeline.ipynb).

References
----------
- PROBAST+AI: Collins et al., BMJ 2024
  (https://doi.org/10.1136/bmj-2023-077876)
- YAIB: Röhrl et al., Scientific Reports 2023
  (https://doi.org/10.1038/s41598-023-42433-8)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ANALYSIS_ONLY_CONTEXT_COLUMNS: list[str] = [
    "landmark_lactate",
    "post_landmark_lactate_last",
    "lactate_clearance_24h_pct",
    "complete_lactate_clearance_24h_flag",
    "clearance_ge_64_24h_flag",
    "mortality_24h_flag",
]

OUTCOME_FLAG_COLUMNS: list[str] = [
    "pressor_24h_flag",
    "mcs_24h_flag",
    "escalation_24h_flag",
    "renal_injury_24h_flag",
    "hypoperfusion_24h_flag",
    "hepatic_injury_24h_flag",
    "end_organ_24h_flag",
    "shock_progression_24h_flag",
    "target",
]

FORBIDDEN_FEATURE_COLUMNS: frozenset[str] = frozenset(
    ANALYSIS_ONLY_CONTEXT_COLUMNS + OUTCOME_FLAG_COLUMNS
)


def _check_column_overlap(
    columns: set[str],
    forbidden: frozenset[str],
    description: str,
    context: str = "",
) -> None:
    """Raise ValueError if *columns* intersects *forbidden*.

    Parameters
    ----------
    columns : set[str]
        Column names to check.
    forbidden : frozenset[str]
        Columns that must not appear.
    description : str
        Human-readable label for the forbidden category (e.g.
        "post-landmark analysis").
    context : str
        Optional location hint included in the error message.
    """
    overlap = sorted(forbidden & columns)
    if not overlap:
        return
    prefix = f"[{context}] " if context else ""
    raise ValueError(
        f"{prefix}Forbidden {description} columns found: {overlap}."
        f"\nThese columns contain information that is not available at"
        f" prediction time and would constitute data leakage."
    )


class LeakageGuard:
    """Systematic data-leakage guard for clinical ML pipelines.

    Enforces PROBAST+AI Domain 4 compliance by checking:

    * No patient overlap between train and test splits.
    * No post-landmark (analysis-only) columns in feature matrices.
    * No outcome-flag columns in feature matrices.
    * Observation-window-only events when a ``window`` column is
      present in the events DataFrame.

    Parameters
    ----------
    landmark_hours : float
        Hours after ICU admission that define the prediction
        landmark.  Features must use only data observed before this
        time.
    outcome_column : str
        Name of the primary outcome column that must never appear in
        features.
    patient_id_column : str
        Name of the column that uniquely identifies patients across
        rows.
    timestamp_column : str
        Name of the column holding the event timestamp (offset from
        admission in minutes or hours).

    Attributes
    ----------
    landmark_hours : float
    outcome_column : str
    patient_id_column : str
    timestamp_column : str
    """

    def __init__(
        self,
        landmark_hours: float,
        outcome_column: str = "target",
        patient_id_column: str = "stay_id",
        timestamp_column: str = "offset_minutes",
    ) -> None:
        self.landmark_hours = landmark_hours
        self.outcome_column = outcome_column
        self.patient_id_column = patient_id_column
        self.timestamp_column = timestamp_column

    def assert_no_patient_overlap(
        self,
        train_ids: np.ndarray | pd.Index,
        test_ids: np.ndarray | pd.Index,
        context: str = "",
    ) -> None:
        """Verify no patient identifier appears in both train and test.

        Parameters
        ----------
        train_ids : ndarray or pd.Index
            Patient identifiers assigned to training.
        test_ids : ndarray or pd.Index
            Patient identifiers assigned to testing.
        context : str
            Optional label included in error messages.

        Raises
        ------
        ValueError
            If any patient appears in both sets.
        """
        train_set = set(map(int, train_ids))
        test_set = set(map(int, test_ids))
        overlap: set[int] = train_set & test_set
        if overlap:
            prefix = f"[{context}] " if context else ""
            raise ValueError(
                f"{prefix}{len(overlap)} patient(s) appear in both "
                f"train and test sets.  Patient overlap would inflate "
                f"performance estimates and invalidate external "
                f"validity (PROBAST+AI Domain 4).\n"
                f"Examples: {sorted(overlap)[:10]}"
            )

    def assert_no_post_landmark_features(
        self,
        features_df: pd.DataFrame,
        context: str = "",
    ) -> None:
        """Verify the feature matrix contains no post-landmark columns.

        Post-landmark columns are defined by
        :data:`ANALYSIS_ONLY_CONTEXT_COLUMNS`.  They describe
        outcomes or events that occur *after* the landmark time and
        must never be available to a model at prediction time.

        Parameters
        ----------
        features_df : pd.DataFrame
            Feature matrix to validate.
        context : str
            Optional label included in error messages.

        Raises
        ------
        ValueError
            If any analysis-only column is present in *features_df*.
        """
        _check_column_overlap(
            set(features_df.columns),
            frozenset(ANALYSIS_ONLY_CONTEXT_COLUMNS),
            description="post-landmark (analysis-only)",
            context=context,
        )

    def assert_no_outcome_in_features(
        self,
        features_df: pd.DataFrame,
        context: str = "",
    ) -> None:
        """Verify the feature matrix contains no outcome-flag columns.

        Outcome flags (e.g. mortality, pressor, MCS, escalation) must
        not leak into the predictor set.

        Parameters
        ----------
        features_df : pd.DataFrame
            Feature matrix to validate.
        context : str
            Optional label included in error messages.

        Raises
        ------
        ValueError
            If any outcome-flag column is present in *features_df*.
        """
        _check_column_overlap(
            set(features_df.columns),
            frozenset(OUTCOME_FLAG_COLUMNS),
            description="outcome-flag",
            context=context,
        )

    def assert_observation_only(
        self,
        events_df: pd.DataFrame,
        context: str = "",
    ) -> None:
        """Verify that ``events_df`` contains only observation-window rows.

        When a ``"window"`` column exists, every row must have
        ``window == "observation"``.  Other windows (``"outcome"``,
        etc.) represent post-landmark data that would leak future
        information into training.

        Parameters
        ----------
        events_df : pd.DataFrame
            Events DataFrame to validate.
        context : str
            Optional label included in error messages.

        Raises
        ------
        ValueError
            If non-observation ``window`` values are found.
        """
        if "window" not in events_df.columns:
            return
        non_obs = sorted(
            events_df.loc[
                events_df["window"] != "observation", "window"
            ]
            .dropna()
            .unique()
            .tolist()
        )
        if non_obs:
            prefix = f"[{context}] " if context else ""
            raise ValueError(
                f"{prefix}Primary events must use observation-only "
                f"data (window='observation').  Found non-observation "
                f"windows: {non_obs}.\n"
                f"Post-landmark events would leak future clinical "
                f"information into training features."
            )

    def assert_no_feature_leakage(
        self,
        features_df: pd.DataFrame,
        context: str = "",
    ) -> None:
        """Run both post-landmark and outcome-flag column checks.

        This is a convenience method that calls
        :meth:`assert_no_post_landmark_features` and
        :meth:`assert_no_outcome_in_features`, catching all
        forbidden-column violations in a single call.

        Parameters
        ----------
        features_df : pd.DataFrame
            Feature matrix to validate.
        context : str
            Optional label included in error messages.

        Raises
        ------
        ValueError
            If any forbidden columns appear.
        """
        _check_column_overlap(
            set(features_df.columns),
            FORBIDDEN_FEATURE_COLUMNS,
            description="post-landmark or outcome",
            context=context,
        )

    def check_probast_domain4(
        self,
        features_df: pd.DataFrame,
        train_ids: np.ndarray | pd.Index | None = None,
        test_ids: np.ndarray | pd.Index | None = None,
        events_df: pd.DataFrame | None = None,
        context: str = "PROBAST+AI-D4",
    ) -> list[str]:
        """Run all PROBAST+AI Domain 4 leakage checks and return findings.

        PROBAST+AI Domain 4 requires that:
        1. No temporal leakage -- features use only pre-landmark data.
        2. No patient overlap between train and test splits.
        3. No outcome contamination in the feature matrix.

        Parameters
        ----------
        features_df : pd.DataFrame
            Feature matrix to validate.
        train_ids : ndarray or pd.Index, optional
            Training patient identifiers for overlap check.
        test_ids : ndarray or pd.Index, optional
            Test patient identifiers for overlap check.
        events_df : pd.DataFrame, optional
            Events DataFrame for observation-only window check.
        context : str
            Label included in error messages.  Defaults to
            ``"PROBAST+AI-D4"``.

        Returns
        -------
        list[str]
            List of violations found (empty if all checks pass).

        Raises
        ------
        ValueError
            If any check fails (identical behaviour to the individual
            assertion methods).
        """
        issues: list[str] = []

        try:
            self.assert_no_post_landmark_features(features_df, context)
        except ValueError as e:
            issues.append(str(e))

        try:
            self.assert_no_outcome_in_features(features_df, context)
        except ValueError as e:
            issues.append(str(e))

        if train_ids is not None and test_ids is not None:
            try:
                self.assert_no_patient_overlap(
                    train_ids, test_ids, context
                )
            except ValueError as e:
                issues.append(str(e))

        if events_df is not None:
            try:
                self.assert_observation_only(events_df, context)
            except ValueError as e:
                issues.append(str(e))

        if issues:
            raise ValueError(
                f"PROBAST+AI Domain 4 violations ({len(issues)}):\n"
                + "\n".join(f"  {i+1}. {msg}" for i, msg in enumerate(issues))
            )

        return issues


class Preprocessor:
    """YAIB-style preprocessor with fit-transform enforcement.

    Stores means, scales, and fill values computed exclusively from
    training data.  Calling :meth:`transform` before :meth:`fit`
    raises a :class:`RuntimeError`.

    Implements the YAIB convention: fit on training data only,
    transform validation and test data using the stored training
    statistics.  This prevents the most common preprocessing leak
    where statistics are inadvertently computed from the full
    dataset.

    Parameters
    ----------
    continuous_columns : list[str]
        Columns to standardise with z-score normalisation.
    binary_columns : list[str]
        Binary flag columns filled with the training mode.
    categorical_columns : list[str], optional
        Categorical columns encoded via one-hot expansion.
        Default: ``[]``.

    Attributes
    ----------
    means_ : dict[str, float] | None
        Per-column training means (populated after :meth:`fit`).
    scales_ : dict[str, float] | None
        Per-column training standard deviations.
    fill_values_ : dict[str, float | int] | None
        Per-column fill values for missing data.
    categories_ : dict[str, list[str]] | None
        Per-column category lists (for one-hot encoding).
    continuous_columns : list[str]
    binary_columns : list[str]
    categorical_columns : list[str]
    _fitted : bool
        Whether :meth:`fit` has been called.
    """

    def __init__(
        self,
        continuous_columns: list[str],
        binary_columns: list[str],
        categorical_columns: list[str] | None = None,
    ) -> None:
        self.continuous_columns = list(continuous_columns)
        self.binary_columns = list(binary_columns)
        self.categorical_columns = list(categorical_columns or [])
        self._fitted: bool = False
        self.means_: dict[str, float] | None = None
        self.scales_: dict[str, float] | None = None
        self.fill_values_: dict[str, float | int] | None = None
        self.categories_: dict[str, list[str]] | None = None
        self._feature_names_out_: list[str] | None = None

    def fit(self, train_df: pd.DataFrame) -> "Preprocessor":
        """Compute statistics from training data.

        For each continuous column: median fill value, mean, standard
        deviation.

        For each binary column: mode fill value.

        For each categorical column: unique categories, fill value.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training DataFrame containing all columns listed in
            ``continuous_columns``, ``binary_columns``, and
            ``categorical_columns``.

        Returns
        -------
        Preprocessor
            Self (enables chaining).

        Raises
        ------
        ValueError
            If a required column is missing from *train_df*.
        """
        self.means_ = {}
        self.scales_ = {}
        self.fill_values_ = {}
        self.categories_ = {}
        names: list[str] = []

        for col in self.continuous_columns:
            if col not in train_df.columns:
                raise ValueError(
                    f"Continuous column '{col}' not found in training data."
                )
            series = train_df[col]
            fill_val = float(series.median()) if len(series) else 0.0
            if not np.isfinite(fill_val):
                fill_val = 0.0
            self.fill_values_[col] = fill_val

            filled = series.fillna(fill_val)
            mean_val = float(filled.mean())
            std_val = float(filled.std())
            if not np.isfinite(std_val) or std_val == 0.0:
                std_val = 1.0
            self.means_[col] = mean_val
            self.scales_[col] = std_val
            names.append(col)

        for col in self.binary_columns:
            if col not in train_df.columns:
                raise ValueError(
                    f"Binary column '{col}' not found in training data."
                )
            mode_vals = train_df[col].mode(dropna=True)
            fill_val = int(mode_vals.iloc[0]) if len(mode_vals) > 0 else 0
            self.fill_values_[col] = fill_val
            names.append(col)

        for col in self.categorical_columns:
            if col not in train_df.columns:
                raise ValueError(
                    f"Categorical column '{col}' not found in training data."
                )
            raw = train_df[col].astype("object").fillna("missing").astype(str)
            categories = sorted(raw.unique().tolist())
            if not categories:
                categories = ["missing"]
            self.categories_[col] = categories
            self.fill_values_[col] = "missing"
            for cat in categories[1:]:
                names.append(f"{col}__{cat}")

        self._feature_names_out_ = names
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply stored training statistics to *df*.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with the same columns seen during :meth:`fit`.

        Returns
        -------
        ndarray
            Transformed feature matrix (float64).

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called first.
        ValueError
            If a required column is missing or a continuous column
            still contains NaN after filling.
        """
        if not self._fitted:
            raise RuntimeError(
                "Preprocessor.transform() called before fit().  "
                "Fit on training data first to compute normalisation "
                "statistics without leakage."
            )

        columns: list[np.ndarray] = []

        for col in self.continuous_columns:
            if col not in df.columns:
                raise ValueError(
                    f"Continuous column '{col}' not found in data."
                )
            filled = df[col].fillna(self.fill_values_[col])
            if filled.isna().any():
                raise ValueError(
                    f"Continuous column '{col}' still contains "
                    f"missing values after fill."
                )
            scaled = (filled.to_numpy(dtype=float) - self.means_[col]) / self.scales_[col]
            columns.append(scaled)

        for col in self.binary_columns:
            if col not in df.columns:
                raise ValueError(
                    f"Binary column '{col}' not found in data."
                )
            filled = df[col].fillna(self.fill_values_[col]).to_numpy(dtype=float)
            columns.append(filled)

        for col in self.categorical_columns:
            raw = df[col].astype("object").fillna(self.fill_values_[col]).astype(str)
            raw = raw.where(raw.isin(self.categories_[col]), self.categories_[col][0])
            for cat in self.categories_[col][1:]:
                columns.append((raw == cat).to_numpy(dtype=float))

        if not columns:
            return np.empty((len(df), 0), dtype=float)
        return np.column_stack(columns).astype(float)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit on *df* and transform it in a single call.

        Useful only for the training set -- never call this on
        validation or test data.

        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame.

        Returns
        -------
        ndarray
            Transformed feature matrix.
        """
        return self.fit(df).transform(df)

    @property
    def fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return self._fitted

    @property
    def feature_names_out(self) -> list[str] | None:
        """Column names produced by :meth:`transform`.

        Returns ``None`` before :meth:`fit` has been called.
        """
        return self._feature_names_out_


def assert_no_feature_leakage_columns(
    columns: list[str] | pd.Index,
) -> None:
    """Raise ValueError if any forbidden column appears in *columns*.

    This reproduces the ``assert_no_feature_leakage_columns``
    function from ``PhysioGraph_External_Pipeline.ipynb`` Cell 17.

    Parameters
    ----------
    columns : list[str] or pd.Index
        Column names to check against
        :data:`FORBIDDEN_FEATURE_COLUMNS`.

    Raises
    ------
    ValueError
        If overlap is found.
    """
    overlap = sorted(FORBIDDEN_FEATURE_COLUMNS & set(columns))
    if overlap:
        raise ValueError(
            f"Post-landmark or outcome columns leaked into primary "
            f"features: {overlap}"
        )


def assert_observation_only(
    events_df: pd.DataFrame,
) -> None:
    """Raise ValueError if *events_df* contains non-observation windows.

    This reproduces the ``assert_observation_only`` function from
    ``PhysioGraph_External_Pipeline.ipynb`` Cell 17.

    Parameters
    ----------
    events_df : pd.DataFrame
        Events DataFrame with a ``"window"`` column.

    Raises
    ------
    ValueError
        If any row has ``window`` not equal to ``"observation"``.
    """
    non_observation = sorted(
        events_df.loc[
            events_df["window"] != "observation", "window"
        ]
        .dropna()
        .unique()
        .tolist()
    )
    if non_observation:
        raise ValueError(
            f"Primary features must use observation-only events, "
            f"found windows: {non_observation}"
        )


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    name: str,
) -> None:
    """Raise ValueError if *df* is missing any required column.

    This reproduces the ``require_columns`` function from
    ``PhysioGraph_External_Pipeline.ipynb`` Cell 19.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check.
    columns : list[str]
        Required column names.
    name : str
        Human-readable name for error messages.

    Raises
    ------
    ValueError
        If any required column is missing.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
