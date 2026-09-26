"""Prespecified analyses for the PhysioGraph research question."""

from .biomarker_benchmark import (
    assign_kapur_scai_stage,
    build_biomarker_analysis_frame,
    fit_biomarker_associations,
    fit_biomarker_prediction_benchmark,
    fit_cross_database_biomarker_transportability,
    run_biomarker_benchmark,
)

from .spo2_epidemiology import (
    assign_exposure,
    build_dose_response_tables,
    build_mantel_haenszel_tables,
    build_paired_precedence_table,
    build_specificity_matrix,
    build_stratified_risk_tables,
    run_epidemiology_analyses,
)
from .spo2_advanced_inference import run_advanced_episode_inference
from .spo2_lactate_mechanistic import (
    build_episode_lactate_controlled_summary,
    build_episode_lactate_paired_summary,
    build_episode_lactate_records,
    build_lactate_episode_measurement_weighted,
    build_lactate_episode_meta_analysis,
    build_lactate_observation_process,
    run_lactate_episode_analyses,
)
from .spo2_multiorgan_mechanistic import (
    run_locked_external_replication,
    run_multiorgan_episode_analyses,
)
from .spo2_instability import (
    build_multiscale_instability_features,
    fit_instability_associations,
    fit_instability_prediction_models,
    run_multiscale_instability_analysis,
)
from .spo2_protocol import (
    PRIMARY_ENDPOINTS,
    SECONDARY_ENDPOINTS,
    POST_LANDMARK_HORIZONS_HOURS,
    build_continuous_trajectory_associations,
    build_endpoint_completeness_audit,
    build_temporal_precedence,
    compute_early_decompensation_outcomes,
    fit_external_transportability,
    fit_grouped_incremental_models,
    fit_grouped_missingness_control,
)

__all__ = [
    "PRIMARY_ENDPOINTS",
    "SECONDARY_ENDPOINTS",
    "POST_LANDMARK_HORIZONS_HOURS",
    "assign_exposure",
    "assign_kapur_scai_stage",
    "build_biomarker_analysis_frame",
    "build_dose_response_tables",
    "build_continuous_trajectory_associations",
    "build_endpoint_completeness_audit",
    "build_episode_lactate_controlled_summary",
    "build_episode_lactate_paired_summary",
    "build_episode_lactate_records",
    "build_lactate_episode_measurement_weighted",
    "build_lactate_episode_meta_analysis",
    "build_lactate_observation_process",
    "build_mantel_haenszel_tables",
    "build_multiscale_instability_features",
    "build_paired_precedence_table",
    "build_specificity_matrix",
    "build_stratified_risk_tables",
    "build_temporal_precedence",
    "compute_early_decompensation_outcomes",
    "fit_external_transportability",
    "fit_biomarker_associations",
    "fit_biomarker_prediction_benchmark",
    "fit_cross_database_biomarker_transportability",
    "fit_grouped_incremental_models",
    "fit_grouped_missingness_control",
    "fit_instability_associations",
    "fit_instability_prediction_models",
    "run_epidemiology_analyses",
    "run_advanced_episode_inference",
    "run_biomarker_benchmark",
    "run_lactate_episode_analyses",
    "run_locked_external_replication",
    "run_multiorgan_episode_analyses",
    "run_multiscale_instability_analysis",
]
