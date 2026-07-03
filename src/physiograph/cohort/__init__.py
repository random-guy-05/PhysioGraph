"""Cohort selection and patient filtering for clinical datasets.

This sub-package provides cohort builders for MIMIC-III and eICU that
reproduce the HF-shock phenotype definitions from the original
PhysioGraph notebooks, along with contract validators for data
integrity checks at pipeline boundaries.

Quick start::

    from physiograph.cohort import build_cohort, MIMICCohortBuilder, EICUCohortBuilder

    # MIMIC-III
    result = build_cohort("mimic", data_root="/path/to/mimic/csvs")
    print(result.cohort_df.shape, len(result.valid_stay_ids))

    # eICU
    result = build_cohort("eicu", data_root="/path/to/eicu/csvs")
    print(result.cohort_df.shape, len(result.valid_stay_ids))
"""

from physiograph.cohort.eicu_cohort import (
    CohortResult as EICUCohortResult,
    EICUCohortBuilder,
    build_cohort as build_eicu_cohort,
    derive_death_offset_minutes,
    normalize_text,
    parse_eicu_age,
    series_contains_any,
)
from physiograph.cohort.mimic_cohort import (
    CohortResult,
    MIMICCohortBuilder,
    build_cohort as build_mimic_cohort,
    match_icd_prefix,
)
from physiograph.cohort.validators import (
    assert_cohort_size,
    assert_flag_values,
    assert_no_duplicate_stays,
    assert_no_nulls_in_required,
    assert_required_columns,
)

__all__ = [
    # Builders
    "MIMICCohortBuilder",
    "EICUCohortBuilder",
    # Results
    "CohortResult",
    "EICUCohortResult",
    # Convenience functions
    "build_cohort",
    # ICD matching
    "match_icd_prefix",
    # eICU text utilities
    "normalize_text",
    "series_contains_any",
    "parse_eicu_age",
    "derive_death_offset_minutes",
    # Validators
    "assert_required_columns",
    "assert_no_duplicate_stays",
    "assert_cohort_size",
    "assert_flag_values",
    "assert_no_nulls_in_required",
]


def build_cohort(
    dataset: str,
    data_root: str | None = None,
    config: dict | None = None,
    max_stays: int | None = None,
):
    """Dispatch to the appropriate cohort builder based on *dataset*.

    Parameters
    ----------
    dataset:
        Dataset identifier — ``"mimic"`` or ``"eicu"``.
    data_root:
        Path to the raw CSV directory.  If ``None``, the path is
        resolved from the config.
    config:
        Optional pre-loaded config dict.
    max_stays:
        If set, cap the cohort to this many stays (for testing).

    Returns
    -------
    CohortResult
        Container with ``cohort_df``, ``valid_stay_ids``, and
        ``anchors``.

    Raises
    ------
    ValueError
        If *dataset* is not ``"mimic"`` or ``"eicu"``.
    """
    from pathlib import Path

    from physiograph.config import load_config

    if config is None:
        config = load_config(dataset)

    if data_root is None:
        resolved = config.get("resolved_paths", {})
        key = f"{dataset}_root"
        if key not in resolved:
            raise ValueError(
                f"Cannot resolve data root for '{dataset}'. "
                f"Provide data_root explicitly or ensure config paths are resolved."
            )
        data_root = resolved[key]

    data_root = Path(data_root)

    if dataset == "mimic":
        builder = MIMICCohortBuilder(data_root=data_root, config=config, max_stays=max_stays)
        return builder.build_cohort()
    elif dataset == "eicu":
        builder = EICUCohortBuilder(data_root=data_root, config=config, max_stays=max_stays)
        return builder.build_cohort()
    else:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. Expected 'mimic' or 'eicu'."
        )