"""Feature extraction and engineering from physiological time-series.

Public API
----------
- :func:`build_feature_table` — aggregate observation-window events into a
  per-stay feature matrix (50+ features).
- :func:`extract_features` — alias for ``build_feature_table``.

Lactate-specific (:mod:`physiograph.features.lactate`)
- :func:`bin_lactate`, :func:`bin_lactate_focus`
- :func:`compute_lactate_clearance_4h_pct`, :func:`compute_lactate_slope_per_hr`
- :func:`compute_persistent_lactate_flag`, :func:`compute_lactate_threshold_flags`
- :func:`compute_lactate_scai_modifier`
- :func:`compute_lactate_acidemia_interaction`,
  :func:`compute_lactate_hypotension_interaction`,
  :func:`compute_lactate_tachycardia_interaction`
- :func:`compute_occult_hypoperfusion_flag`

Hemodynamics (:mod:`physiograph.features.hemodynamics`)
- :func:`compute_hypotension_flag`, :func:`compute_severe_hypotension_flag`
- :func:`compute_tachycardia_flag`
- :func:`compute_lactate_map_ratio`, :func:`compute_lactate_sbp_ratio`
- :func:`compute_measurement_fraction_series`
- :func:`bin_heart_rate`, :func:`bin_heart_rate_fine`
- :func:`compute_perfusion_burden_score`

Missingness (:mod:`physiograph.features.missingness`)
- :func:`classify_missingness`, :func:`compute_missingness_flags_df`
- :func:`apply_forward_fill_and_decay`, :func:`fill_clinical_normals`
"""

from physiograph.features.extraction import build_feature_table, extract_features

__all__ = [
    "build_feature_table",
    "extract_features",
]
