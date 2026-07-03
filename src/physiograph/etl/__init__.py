"""Extract-Transform-Load pipelines for clinical datasets (MIMIC-IV, eICU).

Public API:
- AuditLogger: Structured audit logging at every pipeline step.
- SourceExtraction: Result container for cohort + events DataFrames.
- extract_mimic: MIMIC-IV cohort and event extraction.
- extract_eicu: eICU-CRD cohort and event extraction (chunked streaming).
- Shared utilities: build_event_frame, _finalize_events,
  build_death_events, event_stay_count, match_icd_prefix,
  classify_offset_minutes, and time-window functions.
"""

from .audit import AuditLogger
from .eicu_extractor import extract_eicu
from .mimic_extractor import extract_mimic
from .shared import (
    COHORT_REQUIRED_COLUMNS,
    EVENT_REQUIRED_COLUMNS,
    LABEL_REQUIRED_COLUMNS,
    SourceExtraction,
    _finalize_events,
    build_death_events,
    build_event_frame,
    classify_offset_minutes,
    contains_any_token,
    event_stay_count,
    is_observation_offset_minutes,
    is_outcome_offset_minutes,
    is_pre_or_at_landmark,
    match_icd_prefix,
    normalize_text,
    observation_time_bin,
    ordered_columns,
    require_columns,
    sanitize_events,
    series_contains_any,
)

__all__ = [
    "AuditLogger",
    "SourceExtraction",
    "extract_mimic",
    "extract_eicu",
    "build_event_frame",
    "_finalize_events",
    "build_death_events",
    "event_stay_count",
    "match_icd_prefix",
    "classify_offset_minutes",
    "contains_any_token",
    "is_observation_offset_minutes",
    "is_outcome_offset_minutes",
    "is_pre_or_at_landmark",
    "normalize_text",
    "observation_time_bin",
    "ordered_columns",
    "require_columns",
    "sanitize_events",
    "series_contains_any",
    "COHORT_REQUIRED_COLUMNS",
    "EVENT_REQUIRED_COLUMNS",
    "LABEL_REQUIRED_COLUMNS",
]
