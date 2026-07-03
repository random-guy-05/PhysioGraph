"""Project-wide constants and enumerations.

Defines column names, feature groups, outcome definitions,
and other constants used across the PhysioGraph codebase.

Constants are sourced from:
- PhysioGraph_HF_Shock.ipynb (Cells 1-3)
- PhysioGraph_External_Pipeline.ipynb (Cells 6-11)
- configs/default.yaml (canonical source for shared values)

Where a value exists in configs/default.yaml, this module imports it
via the config loader to avoid duplication.  Notebook-specific aliases
are provided for backward compatibility.
"""

from __future__ import annotations

from typing import Any

from physiograph.config import load_config

# ---------------------------------------------------------------------------
# Load canonical config (dataset=None → default.yaml only)
# ---------------------------------------------------------------------------
_CFG: dict[str, Any] = load_config()

# ═══════════════════════════════════════════════════════════════════════════
# Time Windows
# ═══════════════════════════════════════════════════════════════════════════
OBSERVATION_HOURS: float = _CFG["observation_hours"]
OUTCOME_HOURS: float = _CFG["outcome_hours"]
OUTCOME_WINDOW_END_HOURS: float = _CFG["outcome_window_end_hours"]
TIME_STEP_HOURS: float = _CFG["time_step_hours"]
TIME_STEP_MINUTES: int = _CFG["time_step_minutes"]
LANDMARK_MINUTES: int = _CFG["landmark_minutes"]
OUTCOME_WINDOW_END_MINUTES: int = _CFG["outcome_window_end_minutes"]
NUM_STEPS: int = _CFG["num_steps"]

# Notebook aliases (PhysioGraph_HF_Shock.ipynb)
INPUT_WINDOW: float = OBSERVATION_HOURS
PREDICTION_HORIZON: float = OUTCOME_HOURS
OUTCOME_WINDOW_END: float = OUTCOME_WINDOW_END_HOURS
TIME_STEP_SIZE: float = TIME_STEP_HOURS

# ═══════════════════════════════════════════════════════════════════════════
# Lactate Binning
# ═══════════════════════════════════════════════════════════════════════════
LACTATE_BIN_EDGES: list[float] = _CFG["lactate_bin_edges"]
LACTATE_BIN_LABELS: list[str] = _CFG["lactate_bin_labels"]
FOCUS_LACTATE_BIN_EDGES: list[float] = _CFG["focus_lactate_bin_edges"]
FOCUS_LACTATE_BIN_LABELS: list[str] = _CFG["focus_lactate_bin_labels"]

# ═══════════════════════════════════════════════════════════════════════════
# Heart Rate Banding
# ═══════════════════════════════════════════════════════════════════════════
HR_FINE_BAND_EDGES: list[float] = _CFG["hr_fine_band_edges"]
HR_FINE_BAND_LABELS: list[str] = _CFG["hr_fine_band_labels"]

# ═══════════════════════════════════════════════════════════════════════════
# SCAI Staging
# ═══════════════════════════════════════════════════════════════════════════
SCAI_STAGE_LABELS: list[str] = _CFG["scai_stage_labels"]

# ═══════════════════════════════════════════════════════════════════════════
# Clinical Normals (imputation defaults)
# ═══════════════════════════════════════════════════════════════════════════
CLINICAL_NORMALS: dict[str, float] = _CFG["clinical_normals"]

# ═══════════════════════════════════════════════════════════════════════════
# ICD Code Patterns (shared across datasets)
# ═══════════════════════════════════════════════════════════════════════════
HF_ICD_PATTERNS: dict[int, list[str]] = _CFG["hf_icd_patterns"]
CARDIOGENIC_SHOCK_ICD_PATTERNS: dict[int, list[str]] = _CFG["cardiogenic_shock_icd_patterns"]

# ═══════════════════════════════════════════════════════════════════════════
# MIMIC Item IDs - Portable Subset (External Pipeline)
# ═══════════════════════════════════════════════════════════════════════════
MIMIC_LAB_IDS: dict[str, list[int]] = _CFG["mimic_lab_ids"]
MIMIC_VITAL_IDS: dict[str, list[int]] = _CFG["mimic_vital_ids"]

# ═══════════════════════════════════════════════════════════════════════════
# MIMIC Item IDs - Full Set (HF Shock Notebook)
# ═══════════════════════════════════════════════════════════════════════════
LAB_IDS: dict[str, list[int]] = _CFG["mimic_lab_ids_full"]
VITAL_IDS: dict[str, list[int]] = _CFG["mimic_vital_ids_full"]

# ═══════════════════════════════════════════════════════════════════════════
# Pressor and MCS Item IDs (MIMIC)
# ═══════════════════════════════════════════════════════════════════════════
PRESSOR_ITEMIDS: list[int] = _CFG["pressor_itemids"]
MCS_PROCEDUREEVENT_IDS: dict[str, list[int]] = _CFG["mcs_procedureevent_ids"]

# ═══════════════════════════════════════════════════════════════════════════
# eICU Token Mappings
# ═══════════════════════════════════════════════════════════════════════════
EICU_LAB_NAME_MAP: dict[str, str] = _CFG["eicu_lab_name_map"]
EICU_VITAL_COLUMN_MAP: dict[str, str] = _CFG["eicu_vital_column_map"]
EICU_INFUSION_DRUG_TOKENS: list[str] = _CFG["eicu_infusion_drug_tokens"]
EICU_TREATMENT_TOKENS: list[str] = _CFG["eicu_treatment_tokens"]
EICU_DIAGNOSIS_TOKENS: dict[str, list[str]] = _CFG["eicu_diagnosis_tokens"]

# ═══════════════════════════════════════════════════════════════════════════
# Portable Context Variables (External Pipeline)
# ═══════════════════════════════════════════════════════════════════════════
PORTABLE_CONTEXT_VARIABLES: list[str] = _CFG["portable_context_variables"]

# ═══════════════════════════════════════════════════════════════════════════
# Node Names (28 entries: 18 labs + 10 vitals)
# ═══════════════════════════════════════════════════════════════════════════
NODE_NAMES: list[str] = _CFG["node_names"]

# Pretrained node sets (External Pipeline)
PRETRAINED_LAB_NAMES: set[str] = set(_CFG["pretrained_lab_names"])
PRETRAINED_VITAL_NAMES: set[str] = set(_CFG["pretrained_vital_names"])

# Legacy alias (External Pipeline notebook)
LEGACY_PRETRAINED_NODE_NAMES: list[str] = list(NODE_NAMES)

# ═══════════════════════════════════════════════════════════════════════════
# Prior Edge Index Cross-Links
# ═══════════════════════════════════════════════════════════════════════════
PRIOR_EDGE_CROSS_LINKS: list[list[str]] = _CFG["prior_edge_cross_links"]
USE_PRIOR_EDGE_INDEX: bool = _CFG["use_prior_edge_index"]

# ═══════════════════════════════════════════════════════════════════════════
# Auxiliary Targets
# ═══════════════════════════════════════════════════════════════════════════
AUX_TARGET_COLS: list[str] = _CFG["aux_target_cols"]

# ═══════════════════════════════════════════════════════════════════════════
# Static Feature Columns (HF Shock)
# ═══════════════════════════════════════════════════════════════════════════
STATIC_CONTINUOUS_COLS: list[str] = _CFG["static_continuous_cols"]
STATIC_BINARY_COLS: list[str] = _CFG["static_binary_cols"]
STATIC_COLS: list[str] = STATIC_CONTINUOUS_COLS + STATIC_BINARY_COLS

# ═══════════════════════════════════════════════════════════════════════════
# Analysis-Only Context Columns (not for model features)
# ═══════════════════════════════════════════════════════════════════════════
ANALYSIS_ONLY_CONTEXT_COLUMNS: list[str] = _CFG["analysis_only_context_columns"]

# Notebook alias (HF Shock)
ANALYSIS_ONLY_CONTEXT_COLS: list[str] = ANALYSIS_ONLY_CONTEXT_COLUMNS

# ═══════════════════════════════════════════════════════════════════════════
# Outcome Flag Columns
# ═══════════════════════════════════════════════════════════════════════════
OUTCOME_FLAG_COLUMNS: list[str] = _CFG["outcome_flag_columns"]

# ═══════════════════════════════════════════════════════════════════════════
# Primary Feature Columns (External Pipeline)
# ═══════════════════════════════════════════════════════════════════════════
PRIMARY_FEATURE_COLUMNS: list[str] = _CFG["primary_feature_columns"]

# ═══════════════════════════════════════════════════════════════════════════
# Required Column Sets (Schema Validation)
# ═══════════════════════════════════════════════════════════════════════════
COHORT_REQUIRED_COLUMNS: list[str] = _CFG["cohort_required_columns"]
LABEL_REQUIRED_COLUMNS: list[str] = _CFG["label_required_columns"]
EVENT_REQUIRED_COLUMNS: list[str] = _CFG["event_required_columns"]

# ═══════════════════════════════════════════════════════════════════════════
# eICU Column Sets
# ═══════════════════════════════════════════════════════════════════════════
EICU_COHORT_FLAG_COLUMNS: list[str] = _CFG["eicu_cohort_flag_columns"]
EICU_PATIENT_COLUMNS: list[str] = _CFG["eicu_patient_columns"]

# ═══════════════════════════════════════════════════════════════════════════
# Model Hyperparameters
# ═══════════════════════════════════════════════════════════════════════════
MODEL_CONFIG: dict[str, Any] = _CFG["model"]

# ═══════════════════════════════════════════════════════════════════════════
# Training Parameters
# ═══════════════════════════════════════════════════════════════════════════
TRAINING_CONFIG: dict[str, Any] = _CFG["training"]
SEED: int = TRAINING_CONFIG["seed"]
MAX_EPOCHS: int = TRAINING_CONFIG["max_epochs"]
EARLY_STOPPING_PATIENCE: int = TRAINING_CONFIG["early_stopping_patience"]
BATCH_SIZE: int = TRAINING_CONFIG["batch_size"]
FOCAL_GAMMA: float = TRAINING_CONFIG["focal_gamma"]
GRAD_CLIP_NORM: float = TRAINING_CONFIG["grad_clip_norm"]
DECAY_RATE: float = TRAINING_CONFIG["decay_rate"]
CHUNKSIZE: int = TRAINING_CONFIG["chunksize"]

TEST_FRACTION: float = TRAINING_CONFIG["test_fraction"]
VALIDATION_FRACTION: float = TRAINING_CONFIG["validation_fraction"]

OPTIMIZER_CONFIG: dict[str, Any] = TRAINING_CONFIG["optimizer"]
SCHEDULER_CONFIG: dict[str, Any] = TRAINING_CONFIG["scheduler"]
AUX_TASK_WEIGHTS: list[float] = TRAINING_CONFIG["aux_task_weights"]

# ═══════════════════════════════════════════════════════════════════════════
# Comparator Model Specs (External Pipeline)
# ═══════════════════════════════════════════════════════════════════════════
DEFAULT_VALIDATION_FRACTION: float = _CFG["default_validation_fraction"]
DEFAULT_REGULARIZATION_GRID: tuple[float, ...] = tuple(_CFG["default_regularization_grid"])
COMPARATOR_SPECS: dict[str, Any] = _CFG["comparator_specs"]

# ═══════════════════════════════════════════════════════════════════════════
# Context Summary Variables (HF Shock)
# ═══════════════════════════════════════════════════════════════════════════
CONTEXT_SUMMARY_VARS: list[str] = _CFG["context_summary_vars"]

# ═══════════════════════════════════════════════════════════════════════════
# Causal Coupling Strengths (Counterfactual Analysis)
# ═══════════════════════════════════════════════════════════════════════════
CAUSAL_COUPLINGS: dict[str, dict[str, float]] = _CFG["causal_couplings"]

# ═══════════════════════════════════════════════════════════════════════════
# Interpretation Parameters
# ═══════════════════════════════════════════════════════════════════════════
INTERPRETATION_CONFIG: dict[str, Any] = _CFG["interpretation"]

# ═══════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════
PATHS_CONFIG: dict[str, Any] = _CFG["paths"]
STUDY_TAG: str = PATHS_CONFIG["study_tag"]
STUDY_NAME: str = PATHS_CONFIG["study_name"]

# ═══════════════════════════════════════════════════════════════════════════
# Leakage Guard Columns
# ═══════════════════════════════════════════════════════════════════════════
FORBIDDEN_FEATURE_COLUMNS: set[str] = set(ANALYSIS_ONLY_CONTEXT_COLUMNS + OUTCOME_FLAG_COLUMNS)

# ═══════════════════════════════════════════════════════════════════════════
# SCAI Stage Thresholds (operational 4-hour proxy)
# ═══════════════════════════════════════════════════════════════════════════
SCAI_STAGE_THRESHOLDS: dict[str, dict[str, Any]] = {
    "E": {
        "description": "Severe acidemia or severe hypotension with multi-organ involvement",
        "criteria": [
            "severe_acidemia_flag == 1",
            "severe_hypotension_flag == 1 AND modifier_burden >= 2",
        ],
    },
    "D": {
        "description": "Persistent lactate elevation or lactate >= 2.0 with multi-organ involvement",
        "criteria": [
            "persistent_lactate_flag == 1",
            "baseline_lactate >= 2.0 AND modifier_burden >= 2",
        ],
    },
    "C": {
        "description": "Lactate elevation, acidemia, or end-organ hypoperfusion",
        "criteria": [
            "baseline_lactate >= 2.0",
            "acidemia_flag == 1",
            "renal_hypoperfusion_flag == 1",
            "hepatic_hypoperfusion_flag == 1",
        ],
    },
    "B": {
        "description": "Tachycardia or hypotension without lactate elevation",
        "criteria": [
            "tachycardia_flag == 1",
            "hypotension_flag == 1",
        ],
    },
    "A": {
        "description": "No hemodynamic or metabolic derangement",
        "criteria": [],
    },
}

SCAI_STAGE_NUM_MAP: dict[str, int] = {stage: idx + 1 for idx, stage in enumerate(SCAI_STAGE_LABELS)}

SCAI_STAGE_COLLAPSED_MAP: dict[str, str] = {
    "A": "A",
    "B": "B/C",
    "C": "B/C",
    "D": "D/E",
    "E": "D/E",
}

# ═══════════════════════════════════════════════════════════════════════════
# Clinical Thresholds (feature engineering)
# ═══════════════════════════════════════════════════════════════════════════
TACHYCARDIA_THRESHOLD: float = 100.0
HYPOTENSION_SBP_THRESHOLD: float = 90.0
HYPOTENSION_MAP_THRESHOLD: float = 65.0
SEVERE_HYPOTENSION_SBP_THRESHOLD: float = 80.0
SEVERE_HYPOTENSION_MAP_THRESHOLD: float = 60.0
ACIDEMIA_THRESHOLD: float = 7.25
SEVERE_ACIDEMIA_THRESHOLD: float = 7.20
LACTATE_ELEVATED_THRESHOLD: float = 2.0
LACTATE_HIGH_THRESHOLD: float = 3.1
LACTATE_CRITICAL_THRESHOLD: float = 5.0
RENAL_HYPOPERFUSION_THRESHOLD: float = 2.0
HEPATIC_HYPOPERFUSION_THRESHOLD: float = 2.0
PERSISTENT_LACTATE_MEAN_THRESHOLD: float = 2.5
PERSISTENT_LACTATE_DELTA_THRESHOLD: float = 0.3

# ═══════════════════════════════════════════════════════════════════════════
# Ratio Clip Thresholds (feature engineering)
# ═══════════════════════════════════════════════════════════════════════════
LACTATE_MAP_RATIO_MAP_CLIP: float = 35.0
LACTATE_SBP_RATIO_SBP_CLIP: float = 60.0

# ═══════════════════════════════════════════════════════════════════════════
# Decay Mechanism Constants (HF Shock tensorization)
# ═══════════════════════════════════════════════════════════════════════════
DECAY_GAP_HOURS: float = 2.0
DECAY_RATE_TENSOR: float = 0.1

# ═══════════════════════════════════════════════════════════════════════════
# Temperature Scaling Constants
# ═══════════════════════════════════════════════════════════════════════════
TEMPERATURE_SCALING_MIN: float = 1e-4

# ═══════════════════════════════════════════════════════════════════════════
# Positive Weight Bounds (class imbalance handling)
# ═══════════════════════════════════════════════════════════════════════════
POS_WEIGHT_MIN: float = 1.0
POS_WEIGHT_MAX: float = 10.0
AUX_POS_WEIGHT_MIN: float = 1.0
AUX_POS_WEIGHT_MAX: float = 12.0
AUX_RATE_CLIP_LOWER: float = 1e-4
AUX_RATE_CLIP_UPPER: float = 1.0 - 1e-4

# ═══════════════════════════════════════════════════════════════════════════
# Monotonic Stage Regularizer Constants
# ═══════════════════════════════════════════════════════════════════════════
MONOTONIC_STAGE_MARGIN: float = 0.01

# ═══════════════════════════════════════════════════════════════════════════
# Early ICU Threshold
# ═══════════════════════════════════════════════════════════════════════════
EARLY_ICU_HOURS_THRESHOLD: float = 24.0

# ═══════════════════════════════════════════════════════════════════════════
# Chunksize for CSV Reading
# ═══════════════════════════════════════════════════════════════════════════
CSV_CHUNKSIZE: int = CHUNKSIZE

# ═══════════════════════════════════════════════════════════════════════════
# Subgroup Analysis Constants
# ═══════════════════════════════════════════════════════════════════════════
SUBGROUP_MIN_SIZE: int = INTERPRETATION_CONFIG["subgroup_min_size"]
BOOTSTRAP_N: int = INTERPRETATION_CONFIG["bootstrap_n"]
BOOTSTRAP_ALPHA: float = INTERPRETATION_CONFIG["bootstrap_alpha"]

# ═══════════════════════════════════════════════════════════════════════════
# Integrated Gradients Constants
# ═══════════════════════════════════════════════════════════════════════════
IG_STEPS: int = INTERPRETATION_CONFIG["ig_steps"]
IG_MAX_BATCHES: int = INTERPRETATION_CONFIG["ig_max_batches"]
COUNTERFACTUAL_STANDARDIZED_DELTA: float = INTERPRETATION_CONFIG["counterfactual_standardized_delta"]

# ═══════════════════════════════════════════════════════════════════════════
# ECE Binning
# ═══════════════════════════════════════════════════════════════════════════
ECE_N_BINS: int = 10

# ═══════════════════════════════════════════════════════════════════════════
# Decision Curve Analysis
# ═══════════════════════════════════════════════════════════════════════════
DCA_THRESHOLD_MIN: float = 0.01
DCA_THRESHOLD_MAX: float = 0.99
DCA_N_THRESHOLDS: int = 100

# ═══════════════════════════════════════════════════════════════════════════
# Publication Figure Constants
# ═══════════════════════════════════════════════════════════════════════════
FIGURE_DPI: int = 300
FIGURE_FONT_SCALE: float = 1.2

# ═══════════════════════════════════════════════════════════════════════════
# Color Palette (publication figures)
# ═══════════════════════════════════════════════════════════════════════════
COLOR_PHYSIOGRAPH: str = "#1976D2"
COLOR_LOGISTIC_REGRESSION: str = "#FF5722"
COLOR_VANILLA_GRU: str = "#4CAF50"
COLOR_GNN_ONLY: str = "#9C27B0"

# ═══════════════════════════════════════════════════════════════════════════
# Model Architecture Constants
# ═══════════════════════════════════════════════════════════════════════════
PHYSIOGRAPH_HIDDEN_DIM: int = MODEL_CONFIG["physiograph"]["hidden_dim"]
PHYSIOGRAPH_HEADS: int = MODEL_CONFIG["physiograph"]["heads"]
PHYSIOGRAPH_DROPOUT: float = MODEL_CONFIG["physiograph"]["dropout"]
PHYSIOGRAPH_GRU_LAYERS: int = MODEL_CONFIG["physiograph"]["gru_layers"]
PHYSIOGRAPH_BIDIRECTIONAL: bool = MODEL_CONFIG["physiograph"]["bidirectional"]

VANILLA_GRU_HIDDEN_DIM: int = MODEL_CONFIG["vanilla_gru"]["hidden_dim"]
VANILLA_GRU_DROPOUT: float = MODEL_CONFIG["vanilla_gru"]["dropout"]
VANILLA_GRU_LAYERS: int = MODEL_CONFIG["vanilla_gru"]["gru_layers"]

GNN_ONLY_HIDDEN_DIM: int = MODEL_CONFIG["gnn_only"]["hidden_dim"]
GNN_ONLY_HEADS: int = MODEL_CONFIG["gnn_only"]["heads"]
GNN_ONLY_DROPOUT: float = MODEL_CONFIG["gnn_only"]["dropout"]

# ═══════════════════════════════════════════════════════════════════════════
# Static Encoder Constants
# ═══════════════════════════════════════════════════════════════════════════
STATIC_ENCODER_DIM: int = 32
READOUT_GATE_DIM: int = 32
SHARED_HEAD_DIM: int = 64

# ═══════════════════════════════════════════════════════════════════════════
# Alpha Parameter Initialization (edge learner)
# ═══════════════════════════════════════════════════════════════════════════
ALPHA_INIT: float = -3.0

# ═══════════════════════════════════════════════════════════════════════════
# Temporal Query Initialization
# ═══════════════════════════════════════════════════════════════════════════
TEMPORAL_QUERY_INIT_STD: float = 1.0

# ═══════════════════════════════════════════════════════════════════════════
# Static Dropout Factor
# ═══════════════════════════════════════════════════════════════════════════
STATIC_DROPOUT_FACTOR: float = 0.5

# ═══════════════════════════════════════════════════════════════════════════
# Temperature Scaling Constants
# ═══════════════════════════════════════════════════════════════════════════
TEMPERATURE_LR: float = 0.1
TEMPERATURE_MAX_ITER: int = 200
TEMPERATURE_LINE_SEARCH: str = "strong_wolfe"

# ═══════════════════════════════════════════════════════════════════════════
# Top-K Edges for Topology Analysis
# ═══════════════════════════════════════════════════════════════════════════
TOPOLOGY_TOP_K: int = 10

# ═══════════════════════════════════════════════════════════════════════════
# Odds Ratio Continuity Correction
# ═══════════════════════════════════════════════════════════════════════════
ODDS_RATIO_CONTINUITY_CORRECTION: float = 0.5

# ═══════════════════════════════════════════════════════════════════════════
# Confidence Interval Z-Score
# ═══════════════════════════════════════════════════════════════════════════
Z_SCORE_95_CI: float = 1.96

# ═══════════════════════════════════════════════════════════════════════════
# Youden's J Index
# ═══════════════════════════════════════════════════════════════════════════
YOUDEN_J_OPTIMAL: str = "optimal"

# ═══════════════════════════════════════════════════════════════════════════
# Export all public names
# ═══════════════════════════════════════════════════════════════════════════
__all__ = [
    "OBSERVATION_HOURS",
    "OUTCOME_HOURS",
    "OUTCOME_WINDOW_END_HOURS",
    "TIME_STEP_HOURS",
    "TIME_STEP_MINUTES",
    "LANDMARK_MINUTES",
    "OUTCOME_WINDOW_END_MINUTES",
    "NUM_STEPS",
    "INPUT_WINDOW",
    "PREDICTION_HORIZON",
    "OUTCOME_WINDOW_END",
    "TIME_STEP_SIZE",
    "LACTATE_BIN_EDGES",
    "LACTATE_BIN_LABELS",
    "FOCUS_LACTATE_BIN_EDGES",
    "FOCUS_LACTATE_BIN_LABELS",
    "HR_FINE_BAND_EDGES",
    "HR_FINE_BAND_LABELS",
    "SCAI_STAGE_LABELS",
    "SCAI_STAGE_THRESHOLDS",
    "SCAI_STAGE_NUM_MAP",
    "SCAI_STAGE_COLLAPSED_MAP",
    "CLINICAL_NORMALS",
    "HF_ICD_PATTERNS",
    "CARDIOGENIC_SHOCK_ICD_PATTERNS",
    "MIMIC_LAB_IDS",
    "MIMIC_VITAL_IDS",
    "LAB_IDS",
    "VITAL_IDS",
    "PRESSOR_ITEMIDS",
    "MCS_PROCEDUREEVENT_IDS",
    "EICU_LAB_NAME_MAP",
    "EICU_VITAL_COLUMN_MAP",
    "EICU_INFUSION_DRUG_TOKENS",
    "EICU_TREATMENT_TOKENS",
    "EICU_DIAGNOSIS_TOKENS",
    "PORTABLE_CONTEXT_VARIABLES",
    "NODE_NAMES",
    "PRETRAINED_LAB_NAMES",
    "PRETRAINED_VITAL_NAMES",
    "LEGACY_PRETRAINED_NODE_NAMES",
    "PRIOR_EDGE_CROSS_LINKS",
    "USE_PRIOR_EDGE_INDEX",
    "AUX_TARGET_COLS",
    "STATIC_CONTINUOUS_COLS",
    "STATIC_BINARY_COLS",
    "STATIC_COLS",
    "ANALYSIS_ONLY_CONTEXT_COLUMNS",
    "ANALYSIS_ONLY_CONTEXT_COLS",
    "OUTCOME_FLAG_COLUMNS",
    "PRIMARY_FEATURE_COLUMNS",
    "COHORT_REQUIRED_COLUMNS",
    "LABEL_REQUIRED_COLUMNS",
    "EVENT_REQUIRED_COLUMNS",
    "EICU_COHORT_FLAG_COLUMNS",
    "EICU_PATIENT_COLUMNS",
    "MODEL_CONFIG",
    "TRAINING_CONFIG",
    "SEED",
    "MAX_EPOCHS",
    "EARLY_STOPPING_PATIENCE",
    "BATCH_SIZE",
    "FOCAL_GAMMA",
    "GRAD_CLIP_NORM",
    "DECAY_RATE",
    "CHUNKSIZE",
    "TEST_FRACTION",
    "VALIDATION_FRACTION",
    "OPTIMIZER_CONFIG",
    "SCHEDULER_CONFIG",
    "AUX_TASK_WEIGHTS",
    "DEFAULT_VALIDATION_FRACTION",
    "DEFAULT_REGULARIZATION_GRID",
    "COMPARATOR_SPECS",
    "CONTEXT_SUMMARY_VARS",
    "CAUSAL_COUPLINGS",
    "INTERPRETATION_CONFIG",
    "PATHS_CONFIG",
    "STUDY_TAG",
    "STUDY_NAME",
    "FORBIDDEN_FEATURE_COLUMNS",
    "TACHYCARDIA_THRESHOLD",
    "HYPOTENSION_SBP_THRESHOLD",
    "HYPOTENSION_MAP_THRESHOLD",
    "SEVERE_HYPOTENSION_SBP_THRESHOLD",
    "SEVERE_HYPOTENSION_MAP_THRESHOLD",
    "ACIDEMIA_THRESHOLD",
    "SEVERE_ACIDEMIA_THRESHOLD",
    "LACTATE_ELEVATED_THRESHOLD",
    "LACTATE_HIGH_THRESHOLD",
    "LACTATE_CRITICAL_THRESHOLD",
    "RENAL_HYPOPERFUSION_THRESHOLD",
    "HEPATIC_HYPOPERFUSION_THRESHOLD",
    "PERSISTENT_LACTATE_MEAN_THRESHOLD",
    "PERSISTENT_LACTATE_DELTA_THRESHOLD",
    "LACTATE_MAP_RATIO_MAP_CLIP",
    "LACTATE_SBP_RATIO_SBP_CLIP",
    "DECAY_GAP_HOURS",
    "DECAY_RATE_TENSOR",
    "TEMPERATURE_SCALING_MIN",
    "TEMPERATURE_LR",
    "TEMPERATURE_MAX_ITER",
    "TEMPERATURE_LINE_SEARCH",
    "POS_WEIGHT_MIN",
    "POS_WEIGHT_MAX",
    "AUX_POS_WEIGHT_MIN",
    "AUX_POS_WEIGHT_MAX",
    "AUX_RATE_CLIP_LOWER",
    "AUX_RATE_CLIP_UPPER",
    "MONOTONIC_STAGE_MARGIN",
    "EARLY_ICU_HOURS_THRESHOLD",
    "CSV_CHUNKSIZE",
    "SUBGROUP_MIN_SIZE",
    "BOOTSTRAP_N",
    "BOOTSTRAP_ALPHA",
    "IG_STEPS",
    "IG_MAX_BATCHES",
    "COUNTERFACTUAL_STANDARDIZED_DELTA",
    "ECE_N_BINS",
    "DCA_THRESHOLD_MIN",
    "DCA_THRESHOLD_MAX",
    "DCA_N_THRESHOLDS",
    "FIGURE_DPI",
    "FIGURE_FONT_SCALE",
    "COLOR_PHYSIOGRAPH",
    "COLOR_LOGISTIC_REGRESSION",
    "COLOR_VANILLA_GRU",
    "COLOR_GNN_ONLY",
    "PHYSIOGRAPH_HIDDEN_DIM",
    "PHYSIOGRAPH_HEADS",
    "PHYSIOGRAPH_DROPOUT",
    "PHYSIOGRAPH_GRU_LAYERS",
    "PHYSIOGRAPH_BIDIRECTIONAL",
    "VANILLA_GRU_HIDDEN_DIM",
    "VANILLA_GRU_DROPOUT",
    "VANILLA_GRU_LAYERS",
    "GNN_ONLY_HIDDEN_DIM",
    "GNN_ONLY_HEADS",
    "GNN_ONLY_DROPOUT",
    "STATIC_ENCODER_DIM",
    "READOUT_GATE_DIM",
    "SHARED_HEAD_DIM",
    "ALPHA_INIT",
    "TEMPORAL_QUERY_INIT_STD",
    "STATIC_DROPOUT_FACTOR",
    "TOPOLOGY_TOP_K",
    "ODDS_RATIO_CONTINUITY_CORRECTION",
    "Z_SCORE_95_CI",
    "YOUDEN_J_OPTIMAL",
]
