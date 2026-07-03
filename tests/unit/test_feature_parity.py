"""Parity tests for feature engineering module.

Verifies that feature extraction functions produce outputs matching
the exact computation logic from the PhysioGraph notebooks.
Tests use synthetic DataFrames with known values — no MIMIC/eICU
data files required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from physiograph.constants import (
    CLINICAL_NORMALS,
    DECAY_GAP_HOURS,
    DECAY_RATE_TENSOR,
    LACTATE_BIN_EDGES,
    LACTATE_BIN_LABELS,
    PRIMARY_FEATURE_COLUMNS,
    SCAI_STAGE_LABELS,
)
from physiograph.features.extraction import (
    DEFAULT_PRIMARY_FEATURE_COLUMNS,
    _assign_scai_stage,
    _extract_config,
    build_feature_table,
)
from physiograph.features.hemodynamics import (
    bin_heart_rate,
    bin_heart_rate_fine,
    compute_hypotension_flag,
    compute_lactate_map_ratio,
    compute_lactate_sbp_ratio,
    compute_measurement_fraction_series,
    compute_perfusion_burden_score,
    compute_severe_hypotension_flag,
    compute_tachycardia_flag,
)
from physiograph.features.lactate import (
    bin_lactate,
    bin_lactate_focus,
    compute_lactate_acidemia_interaction,
    compute_lactate_clearance_4h_pct,
    compute_lactate_hypotension_interaction,
    compute_lactate_scai_modifier,
    compute_lactate_slope_per_hr,
    compute_lactate_tachycardia_interaction,
    compute_lactate_threshold_flags,
    compute_occult_hypoperfusion_flag,
    compute_persistent_lactate_flag,
)
from physiograph.features.missingness import (
    MISSINGNESS_INFORMATIVE,
    MISSINGNESS_RANDOM,
    MISSINGNESS_STRUCTURAL,
    apply_forward_fill_and_decay,
    classify_missingness,
    compute_missingness_flags_df,
    fill_clinical_normals,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_series(values, index=None):
    """Convenience: wrap values in a pd.Series."""
    return pd.Series(values, index=index, dtype=float)


# ---------------------------------------------------------------------------
# 1. Lactate binning
# ---------------------------------------------------------------------------


class TestLactateBinning:
    """Verify LACTATE_BIN_EDGES produce correct bin labels."""

    def test_bin_edges_match_constants(self):
        """Default bin edges in lactate.py match constants.py."""
        from physiograph.features.lactate import DEFAULT_LACTATE_BIN_EDGES
        # Constants use -inf/inf; module uses -np.inf/np.inf — compare values.
        for actual, expected in zip(DEFAULT_LACTATE_BIN_EDGES, LACTATE_BIN_EDGES):
            if np.isinf(actual) and np.isinf(expected):
                assert np.sign(actual) == np.sign(expected)
            else:
                assert actual == expected

    def test_bin_labels_match_constants(self):
        """Default bin labels in lactate.py match constants.py."""
        from physiograph.features.lactate import DEFAULT_LACTATE_BIN_LABELS
        assert DEFAULT_LACTATE_BIN_LABELS == LACTATE_BIN_LABELS

    def test_lactate_bins_known_values(self):
        """Known lactate values map to expected bin labels."""
        values = _make_series([0.5, 2.5, 4.0, 6.0])
        result = bin_lactate(values)
        assert result[0] == "<2"
        assert result[1] == "2-3.1"
        assert result[2] == "3.1-5"
        assert result[3] == ">=5"

    def test_lactate_bins_boundary_values(self):
        """Boundary values (exactly at bin edges) map correctly (left-closed)."""
        values = _make_series([2.0, 3.1, 5.0])
        result = bin_lactate(values)
        # pd.cut with right=False: bins are [a, b)
        assert result[0] == "2-3.1"   # 2.0 falls in [2.0, 3.1)
        assert result[1] == "3.1-5"   # 3.1 falls in [3.1, 5.0)
        assert result[2] == ">=5"     # 5.0 falls in [5.0, inf)

    def test_lactate_bins_nan_filled(self):
        """NaN values are filled with 'missing'."""
        values = _make_series([1.0, np.nan, 4.0])
        result = bin_lactate(values)
        assert result[1] == "missing"

    def test_focus_bins_known_values(self):
        """Focus bin edges produce correct labels."""
        values = _make_series([1.5, 2.5, 3.5, 6.0])
        result = bin_lactate_focus(values)
        assert result[0] == "<2"
        assert result[1] == "2-3"
        assert result[2] == "3-5"
        assert result[3] == ">=5"


# ---------------------------------------------------------------------------
# 2. Lactate clearance
# ---------------------------------------------------------------------------


class TestLactateClearance:
    """Verify clearance metrics compute correctly."""

    def test_clearance_decreasing_lactate(self):
        """Lactate dropping from 4 to 2 → 50% clearance."""
        first = _make_series([4.0])
        last = _make_series([2.0])
        result = compute_lactate_clearance_4h_pct(first, last)
        assert result.iloc[0] == pytest.approx(50.0)

    def test_clearance_increasing_lactate(self):
        """Lactate rising from 2 to 4 → -100% clearance (negative)."""
        first = _make_series([2.0])
        last = _make_series([4.0])
        result = compute_lactate_clearance_4h_pct(first, last)
        assert result.iloc[0] == pytest.approx(-100.0)

    def test_clearance_no_change(self):
        """Lactate unchanged → 0% clearance."""
        first = _make_series([3.0])
        last = _make_series([3.0])
        result = compute_lactate_clearance_4h_pct(first, last)
        assert result.iloc[0] == pytest.approx(0.0)

    def test_clearance_zero_first(self):
        """First lactate = 0 → NaN (division by zero → inf → NaN)."""
        first = _make_series([0.0])
        last = _make_series([2.0])
        result = compute_lactate_clearance_4h_pct(first, last)
        assert pd.isna(result.iloc[0])

    def test_slope_per_hr(self):
        """Slope = delta / span with minimum span clipping."""
        delta = _make_series([2.0])     # last - first = 2.0
        span = _make_series([1.0])       # 1 hour span
        result = compute_lactate_slope_per_hr(delta, span)
        assert result.iloc[0] == pytest.approx(2.0)

    def test_slope_minimum_span(self):
        """Span below minimum is clipped to min_time_span_hours."""
        delta = _make_series([1.0])
        span = _make_series([0.1])  # below default min of 0.25
        result = compute_lactate_slope_per_hr(delta, span, min_time_span_hours=0.25)
        assert result.iloc[0] == pytest.approx(1.0 / 0.25)


# ---------------------------------------------------------------------------
# 3. Lactate × acidemia interaction
# ---------------------------------------------------------------------------


class TestLactateAcidemiaInteraction:
    """Verify lactate × acidemia interaction term."""

    def test_interaction_with_acidemia(self):
        """Lactate 4.0 × acidemia flag 1 → 4.0."""
        lactate = _make_series([4.0])
        acidemia = pd.Series([1], dtype=int)
        result = compute_lactate_acidemia_interaction(lactate, acidemia)
        assert result.iloc[0] == pytest.approx(4.0)

    def test_interaction_without_acidemia(self):
        """Lactate 4.0 × acidemia flag 0 → 0.0."""
        lactate = _make_series([4.0])
        acidemia = pd.Series([0], dtype=int)
        result = compute_lactate_acidemia_interaction(lactate, acidemia)
        assert result.iloc[0] == pytest.approx(0.0)

    def test_interaction_nan_lactate_filled_zero(self):
        """NaN lactate is filled with 0.0 before multiplication."""
        lactate = _make_series([np.nan])
        acidemia = pd.Series([1], dtype=int)
        result = compute_lactate_acidemia_interaction(lactate, acidemia)
        assert result.iloc[0] == pytest.approx(0.0)

    def test_hypotension_interaction(self):
        """Lactate × hypotension interaction works identically."""
        lactate = _make_series([3.5])
        hypo = pd.Series([1], dtype=int)
        result = compute_lactate_hypotension_interaction(lactate, hypo)
        assert result.iloc[0] == pytest.approx(3.5)

    def test_tachycardia_interaction(self):
        """Lactate × tachycardia interaction works identically."""
        lactate = _make_series([5.0])
        tachy = pd.Series([1], dtype=int)
        result = compute_lactate_tachycardia_interaction(lactate, tachy)
        assert result.iloc[0] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 4. Hypotension flags
# ---------------------------------------------------------------------------


class TestHypotensionFlags:
    """Verify SBP<90 and MAP<65 flags."""

    def test_sbp_below_90(self):
        """SBP min < 90 → hypotension flag = 1."""
        sbp = _make_series([85.0])
        map_ = _make_series([70.0])
        result = compute_hypotension_flag(sbp, map_)
        assert result.iloc[0] == 1

    def test_map_below_65(self):
        """MAP min < 65 → hypotension flag = 1."""
        sbp = _make_series([100.0])
        map_ = _make_series([60.0])
        result = compute_hypotension_flag(sbp, map_)
        assert result.iloc[0] == 1

    def test_no_hypotension(self):
        """SBP >= 90 and MAP >= 65 → hypotension flag = 0."""
        sbp = _make_series([110.0])
        map_ = _make_series([75.0])
        result = compute_hypotension_flag(sbp, map_)
        assert result.iloc[0] == 0

    def test_severe_hypotension_sbp(self):
        """SBP < 80 → severe hypotension flag = 1."""
        sbp = _make_series([75.0])
        map_ = _make_series([65.0])
        result = compute_severe_hypotension_flag(sbp, map_)
        assert result.iloc[0] == 1

    def test_severe_hypotension_map(self):
        """MAP < 60 → severe hypotension flag = 1."""
        sbp = _make_series([90.0])
        map_ = _make_series([55.0])
        result = compute_severe_hypotension_flag(sbp, map_)
        assert result.iloc[0] == 1

    def test_no_severe_hypotension(self):
        """SBP >= 80 and MAP >= 60 → severe hypotension flag = 0."""
        sbp = _make_series([95.0])
        map_ = _make_series([70.0])
        result = compute_severe_hypotension_flag(sbp, map_)
        assert result.iloc[0] == 0

    def test_hypotension_nan_filled_false(self):
        """NaN in SBP/MAP → flag = 0 (fillna False → int 0)."""
        sbp = _make_series([np.nan])
        map_ = _make_series([70.0])
        result = compute_hypotension_flag(sbp, map_)
        assert result.iloc[0] == 0


# ---------------------------------------------------------------------------
# 5. Tachycardia flag
# ---------------------------------------------------------------------------


class TestTachycardiaFlag:
    """Verify HR > 100 flag."""

    def test_tachycardia_present(self):
        """HR max >= 100 → tachycardia flag = 1."""
        hr = _make_series([105.0])
        result = compute_tachycardia_flag(hr)
        assert result.iloc[0] == 1

    def test_no_tachycardia(self):
        """HR max < 100 → tachycardia flag = 0."""
        hr = _make_series([88.0])
        result = compute_tachycardia_flag(hr)
        assert result.iloc[0] == 0

    def test_tachycardia_exactly_100(self):
        """HR max exactly 100 → tachycardia flag = 1 (>= threshold)."""
        hr = _make_series([100.0])
        result = compute_tachycardia_flag(hr)
        assert result.iloc[0] == 1

    def test_tachycardia_nan(self):
        """NaN HR → flag = 0."""
        hr = _make_series([np.nan])
        result = compute_tachycardia_flag(hr)
        assert result.iloc[0] == 0


# ---------------------------------------------------------------------------
# 6. Lactate-hemodynamics ratios
# ---------------------------------------------------------------------------


class TestLactateHemodynamicsRatios:
    """Verify lactate/MAP and lactate/SBP ratios."""

    def test_lactate_map_ratio(self):
        """Lactate 4.0 / MAP 80.0 = 0.05."""
        lactate = _make_series([4.0])
        map_ = _make_series([80.0])
        result = compute_lactate_map_ratio(lactate, map_)
        assert result.iloc[0] == pytest.approx(4.0 / 80.0)

    def test_lactate_map_ratio_clipped(self):
        """MAP below clip threshold (35) is clipped to 35."""
        lactate = _make_series([3.5])
        map_ = _make_series([30.0])  # below default clip of 35
        result = compute_lactate_map_ratio(lactate, map_)
        assert result.iloc[0] == pytest.approx(3.5 / 35.0)

    def test_lactate_sbp_ratio(self):
        """Lactate 3.0 / SBP 90.0 = 0.0333..."""
        lactate = _make_series([3.0])
        sbp = _make_series([90.0])
        result = compute_lactate_sbp_ratio(lactate, sbp)
        assert result.iloc[0] == pytest.approx(3.0 / 90.0)

    def test_lactate_sbp_ratio_clipped(self):
        """SBP below clip threshold (60) is clipped to 60."""
        lactate = _make_series([6.0])
        sbp = _make_series([50.0])  # below default clip of 60
        result = compute_lactate_sbp_ratio(lactate, sbp)
        assert result.iloc[0] == pytest.approx(6.0 / 60.0)


# ---------------------------------------------------------------------------
# 7. Forward-fill with exponential decay
# ---------------------------------------------------------------------------


class TestForwardFillDecay:
    """Verify forward-fill with exponential decay (rate=0.1)."""

    def test_forward_fill_basic(self):
        """Missing values are forward-filled from last observation."""
        # 1 patient, 1 node, 5 time steps
        values = np.array([[[1.0, np.nan, np.nan, np.nan, np.nan]]])
        mask = np.array([[[True, False, False, False, False]]])
        normals = np.array([0.5])
        result_vals, result_deltas = apply_forward_fill_and_decay(
            values, mask, normals, step_hours=0.25, decay_start_hours=2.0, decay_rate=0.1
        )
        # Step 1: forward-filled to 1.0 (no decay yet, delta=0.25 < 2.0)
        assert result_vals[0, 0, 1] == pytest.approx(1.0)

    def test_decay_after_threshold(self):
        """After decay_start_hours, values decay toward clinical normal."""
        # 1 patient, 1 node, 9 time steps (0.25h each → 2h at step 8)
        # First observation at t=0, then missing for 8 steps
        values = np.array([[[10.0] + [np.nan] * 8]])
        mask = np.array([[[True] + [False] * 8]])
        normals = np.array([5.0])
        result_vals, result_deltas = apply_forward_fill_and_decay(
            values, mask, normals, step_hours=0.25, decay_start_hours=2.0, decay_rate=0.1
        )
        # At step 8 (delta = 2.0h), decay kicks in:
        # decayed = prev * (1 - 0.1) + normal * 0.1
        # Step 8: prev is step 7 value (forward-filled, no decay yet since delta < 2.0)
        # Actually, decay starts when delta >= decay_start_hours
        # Step 8: delta = 8 * 0.25 = 2.0 >= 2.0, so decay applies
        # Step 7: delta = 7 * 0.25 = 1.75 < 2.0, no decay, value = 10.0
        # Step 8: decayed = 10.0 * 0.9 + 5.0 * 0.1 = 9.5
        assert result_vals[0, 0, 8] == pytest.approx(9.5)

    def test_decay_rate_0_1(self):
        """Decay rate 0.1 closes 10% of gap toward normal per step."""
        # 1 patient, 1 node, 3 time steps with decay starting immediately
        values = np.array([[[10.0, np.nan, np.nan]]])
        mask = np.array([[[True, False, False]]])
        normals = np.array([5.0])
        # decay_start_hours=0 so decay starts at step 1
        result_vals, _ = apply_forward_fill_and_decay(
            values, mask, normals, step_hours=0.25, decay_start_hours=0.0, decay_rate=0.1
        )
        # Step 1: forward-fill to 10.0, then decay: 10.0*0.9 + 5.0*0.1 = 9.5
        assert result_vals[0, 0, 1] == pytest.approx(9.5)
        # Step 2: prev=9.5, decay: 9.5*0.9 + 5.0*0.1 = 9.05
        assert result_vals[0, 0, 2] == pytest.approx(9.05)

    def test_delta_tracking(self):
        """Delta tracks time since last real observation."""
        values = np.array([[[1.0, np.nan, np.nan, np.nan]]])
        mask = np.array([[[True, False, False, False]]])
        normals = np.array([0.5])
        _, deltas = apply_forward_fill_and_decay(
            values, mask, normals, step_hours=0.25, decay_start_hours=2.0, decay_rate=0.1
        )
        # Deltas: step 0 = 0, step 1 = 0.25, step 2 = 0.5, step 3 = 0.75
        assert deltas[0, 0, 0] == pytest.approx(0.0)
        assert deltas[0, 0, 1] == pytest.approx(0.25)
        assert deltas[0, 0, 2] == pytest.approx(0.5)
        assert deltas[0, 0, 3] == pytest.approx(0.75)

    def test_delta_resets_on_observation(self):
        """Delta resets to 0 when a real observation arrives."""
        values = np.array([[[1.0, np.nan, 2.0, np.nan]]])
        mask = np.array([[[True, False, True, False]]])
        normals = np.array([0.5])
        _, deltas = apply_forward_fill_and_decay(
            values, mask, normals, step_hours=0.25, decay_start_hours=2.0, decay_rate=0.1
        )
        # Step 2 has a real observation → delta resets to 0
        assert deltas[0, 0, 2] == pytest.approx(0.0)
        # Step 3: delta = 0 + 0.25 = 0.25
        assert deltas[0, 0, 3] == pytest.approx(0.25)

    def test_normals_length_mismatch_raises(self):
        """Mismatched node_normals length raises ValueError."""
        values = np.ones((1, 3, 2))
        mask = np.ones((1, 3, 2), dtype=bool)
        normals = np.array([1.0, 2.0])  # length 2, but values has 3 nodes
        with pytest.raises(ValueError, match="node_normals length mismatch"):
            apply_forward_fill_and_decay(values, mask, normals)


# ---------------------------------------------------------------------------
# 8. Clinical normals fill
# ---------------------------------------------------------------------------


class TestClinicalNormalsFill:
    """Verify missing values filled with clinical normals."""

    def test_fill_3d(self):
        """NaN values in 3D array are filled with clinical normals."""
        values = np.array([[[1.0, np.nan], [np.nan, 2.0]]])
        # 1 patient, 2 nodes, 2 time steps
        node_names = ["hr", "sbp"]
        clinical_normals = {"hr": 80.0, "sbp": 120.0}
        result = fill_clinical_normals(values, node_names, clinical_normals)
        assert result[0, 0, 1] == pytest.approx(80.0)   # hr NaN → 80
        assert result[0, 1, 0] == pytest.approx(120.0)  # sbp NaN → 120
        assert result[0, 0, 0] == pytest.approx(1.0)    # hr non-NaN preserved
        assert result[0, 1, 1] == pytest.approx(2.0)    # sbp non-NaN preserved

    def test_fill_2d(self):
        """NaN values in 2D array are filled with clinical normals."""
        # 2 rows (1 patient × 2 nodes), 2 time steps
        values = np.array([[1.0, np.nan], [np.nan, 2.0]])
        node_names = ["hr", "sbp"]
        clinical_normals = {"hr": 80.0, "sbp": 120.0}
        result = fill_clinical_normals(values, node_names, clinical_normals)
        assert result[0, 1] == pytest.approx(80.0)   # hr NaN → 80
        assert result[1, 0] == pytest.approx(120.0)  # sbp NaN → 120

    def test_fill_unknown_node_defaults_zero(self):
        """Unknown node name defaults to 0.0."""
        values = np.array([[[np.nan]]])
        node_names = ["unknown_var"]
        clinical_normals = {}  # no mapping for unknown_var
        result = fill_clinical_normals(values, node_names, clinical_normals)
        assert result[0, 0, 0] == pytest.approx(0.0)

    def test_clinical_normals_count(self):
        """CLINICAL_NORMALS dict has 28 entries (from config)."""
        assert len(CLINICAL_NORMALS) == 28


# ---------------------------------------------------------------------------
# 9. Missingness flags
# ---------------------------------------------------------------------------


class TestMissingnessFlags:
    """Verify structural/informative/random missingness taxonomy."""

    def test_structural_missingness(self):
        """Selective variable with 0 measurements → structural."""
        assert classify_missingness("bnp", 0) == MISSINGNESS_STRUCTURAL
        assert classify_missingness("troponin_t", 0) == MISSINGNESS_STRUCTURAL

    def test_informative_missingness(self):
        """Ubiquitous variable with 0 measurements → informative."""
        assert classify_missingness("lactate", 0) == MISSINGNESS_INFORMATIVE
        assert classify_missingness("hr", 0) == MISSINGNESS_INFORMATIVE

    def test_random_missingness_sparse(self):
        """Variable with 1 measurement → random."""
        assert classify_missingness("lactate", 1) == MISSINGNESS_RANDOM
        assert classify_missingness("bnp", 1) == MISSINGNESS_RANDOM

    def test_random_missingness_sufficient(self):
        """Variable with 2+ measurements → random."""
        assert classify_missingness("lactate", 5) == MISSINGNESS_RANDOM
        assert classify_missingness("bnp", 2) == MISSINGNESS_RANDOM

    def test_missingness_flags_df(self):
        """compute_missingness_flags_df returns expected column names."""
        events = pd.DataFrame({
            "stay_id": [10, 10, 10, 20],
            "concept": ["lactate", "hr", "bnp", "lactate"],
            "value_numeric": [2.0, 90.0, 100.0, 3.0],
        })
        result = compute_missingness_flags_df(events, ["lactate", "hr", "bnp"])
        expected_cols = {"stay_id", "lactate_missingness", "hr_missingness", "bnp_missingness"}
        assert set(result.columns) == expected_cols
        assert set(result["stay_id"]) == {10, 20}

    def test_missingness_flags_df_missing_variable(self):
        """Variable absent from events → column still present."""
        events = pd.DataFrame({
            "stay_id": [10],
            "concept": ["lactate"],
            "value_numeric": [2.0],
        })
        result = compute_missingness_flags_df(events, ["lactate", "bnp"])
        expected_cols = {"stay_id", "lactate_missingness", "bnp_missingness"}
        assert set(result.columns) == expected_cols


# ---------------------------------------------------------------------------
# 10. SCAI staging
# ---------------------------------------------------------------------------


class TestSCAIStaging:
    """Verify SCAI stage assignment (A/B/C/D/E) matches expected logic."""

    def _make_row(self, **kwargs):
        """Create a row dict with defaults for SCAI staging."""
        defaults = {
            "severe_acidemia_flag": 0,
            "severe_hypotension_flag": 0,
            "modifier_burden": 0,
            "persistent_lactate_flag": 0,
            "baseline_lactate": np.nan,
            "acidemia_flag": 0,
            "renal_hypoperfusion_flag": 0,
            "hepatic_hypoperfusion_flag": 0,
            "tachycardia_flag": 0,
            "hypotension_flag": 0,
        }
        defaults.update(kwargs)
        return pd.Series(defaults)

    def test_stage_a_no_derangement(self):
        """No flags → Stage A."""
        row = self._make_row()
        assert _assign_scai_stage(row, SCAI_STAGE_LABELS) == "A"

    def test_stage_b_tachycardia(self):
        """Tachycardia only → Stage B."""
        row = self._make_row(tachycardia_flag=1)
        assert _assign_scai_stage(row, SCAI_STAGE_LABELS) == "B"

    def test_stage_b_hypotension(self):
        """Hypotension only → Stage B."""
        row = self._make_row(hypotension_flag=1)
        assert _assign_scai_stage(row, SCAI_STAGE_LABELS) == "B"

    def test_stage_c_elevated_lactate(self):
        """Lactate >= 2.0 → Stage C."""
        row = self._make_row(baseline_lactate=2.5)
        assert _assign_scai_stage(row, SCAI_STAGE_LABELS) == "C"

    def test_stage_c_acidemia(self):
        """Acidemia flag → Stage C."""
        row = self._make_row(acidemia_flag=1)
        assert _assign_scai_stage(row, SCAI_STAGE_LABELS) == "C"

    def test_stage_c_renal_hypoperfusion(self):
        """Renal hypoperfusion → Stage C."""
        row = self._make_row(renal_hypoperfusion_flag=1)
        assert _assign_scai_stage(row, SCAI_STAGE_LABELS) == "C"

    def test_stage_c_hepatic_hypoperfusion(self):
        """Hepatic hypoperfusion → Stage C."""
        row = self._make_row(hepatic_hypoperfusion_flag=1)
        assert _assign_scai_stage(row, SCAI_STAGE_LABELS) == "C"

    def test_stage_d_persistent_lactate(self):
        """Persistent lactate flag → Stage D."""
        row = self._make_row(persistent_lactate_flag=1)
        assert _assign_scai_stage(row, SCAI_STAGE_LABELS) == "D"

    def test_stage_d_lactate_with_burden(self):
        """Lactate >= 2.0 AND modifier_burden >= 2 → Stage D."""
        row = self._make_row(baseline_lactate=2.5, modifier_burden=2)
        assert _assign_scai_stage(row, SCAI_STAGE_LABELS) == "D"

    def test_stage_e_severe_acidemia(self):
        """Severe acidemia → Stage E."""
        row = self._make_row(severe_acidemia_flag=1)
        assert _assign_scai_stage(row, SCAI_STAGE_LABELS) == "E"

    def test_stage_e_severe_hypo_with_burden(self):
        """Severe hypotension AND modifier_burden >= 2 → Stage E."""
        row = self._make_row(severe_hypotension_flag=1, modifier_burden=2)
        assert _assign_scai_stage(row, SCAI_STAGE_LABELS) == "E"

    def test_stage_e_priority_over_d(self):
        """Stage E takes priority over Stage D conditions."""
        row = self._make_row(
            severe_acidemia_flag=1,
            persistent_lactate_flag=1,
        )
        assert _assign_scai_stage(row, SCAI_STAGE_LABELS) == "E"

    def test_stage_d_priority_over_c(self):
        """Stage D takes priority over Stage C conditions."""
        row = self._make_row(
            persistent_lactate_flag=1,
            acidemia_flag=1,
        )
        assert _assign_scai_stage(row, SCAI_STAGE_LABELS) == "D"

    def test_scai_modifier_lactate5_acidemia(self):
        """Lactate >= 5 AND pH < 7.2 → 'lactate>=5 & pH<7.2'."""
        lactate = _make_series([6.0])
        ph_min = _make_series([7.15])
        result = compute_lactate_scai_modifier(lactate, ph_min)
        assert result[0] == "lactate>=5 & pH<7.2"

    def test_scai_modifier_lactate5_only(self):
        """Lactate >= 5 AND pH >= 7.2 → 'lactate>=5 only'."""
        lactate = _make_series([6.0])
        ph_min = _make_series([7.30])
        result = compute_lactate_scai_modifier(lactate, ph_min)
        assert result[0] == "lactate>=5 only"

    def test_scai_modifier_acidemia_only(self):
        """Lactate < 5 AND pH < 7.2 → 'pH<7.2 only'."""
        lactate = _make_series([3.0])
        ph_min = _make_series([7.15])
        result = compute_lactate_scai_modifier(lactate, ph_min)
        assert result[0] == "pH<7.2 only"

    def test_scai_modifier_neither(self):
        """Lactate < 5 AND pH >= 7.2 → 'neither'."""
        lactate = _make_series([1.5])
        ph_min = _make_series([7.40])
        result = compute_lactate_scai_modifier(lactate, ph_min)
        assert result[0] == "neither"


# ---------------------------------------------------------------------------
# 11. Feature column count
# ---------------------------------------------------------------------------


class TestFeatureColumnCount:
    """Verify feature matrix has 50 columns."""

    def test_primary_feature_columns_count(self):
        """DEFAULT_PRIMARY_FEATURE_COLUMNS has exactly 50 entries."""
        assert len(DEFAULT_PRIMARY_FEATURE_COLUMNS) == 50

    def test_constants_feature_columns_count(self):
        """PRIMARY_FEATURE_COLUMNS from constants matches 50."""
        assert len(PRIMARY_FEATURE_COLUMNS) == 50

    def test_expected_columns_present(self):
        """Key expected columns are in the feature list."""
        expected = [
            "baseline_lactate", "baseline_hr", "baseline_sbp", "baseline_map",
            "tachycardia_flag", "hypotension_flag", "severe_hypotension_flag",
            "acidemia_flag", "severe_acidemia_flag", "modifier_burden",
            "lactate_delta", "lactate_slope_per_hr", "lactate_clearance_4h_pct",
            "persistent_lactate_flag", "renal_hypoperfusion_flag",
            "hepatic_hypoperfusion_flag",
            "lactate_ge_2_flag", "lactate_ge_3_1_flag", "lactate_ge_5_flag",
            "map_below_65_fraction", "sbp_below_90_fraction",
            "hr_above_100_fraction", "ph_below_7_25_fraction",
            "lactate_map_ratio", "lactate_sbp_ratio",
            "lactate_acidemia_interaction",
            "lactate_hypotension_interaction",
            "lactate_tachycardia_interaction",
            "occult_hypoperfusion_flag", "perfusion_burden_score",
            "baseline_lactate_bin", "baseline_lactate_focus_bin",
            "hr_band", "hr_band_fine",
            "lactate_scai_modifier",
            "scai_stage", "scai_stage_num", "scai_stage_collapsed",
        ]
        for col in expected:
            assert col in DEFAULT_PRIMARY_FEATURE_COLUMNS, f"Missing: {col}"


# ---------------------------------------------------------------------------
# 12. build_feature_table smoke test
# ---------------------------------------------------------------------------


class TestBuildFeatureTableSmoke:
    """Smoke test with synthetic data."""

    @pytest.fixture
    def synthetic_data(self):
        """Create minimal synthetic events and cohort DataFrames."""
        cohort = pd.DataFrame({
            "dataset": ["mimic"] * 3,
            "stay_id": [1001, 1002, 1003],
            "age": [65, 72, 45],
            "is_male": [1, 0, 1],
            "cohort_hf_flag": [1, 0, 1],
            "shock_icd_flag": [0, 0, 1],
        })

        # Create events for 3 stays with multiple variables
        events = pd.DataFrame([
            # Stay 1001: elevated lactate, tachycardia, normal BP
            {"stay_id": 1001, "concept": "lactate", "value_numeric": 3.5, "offset_minutes": 0, "window": "observation"},
            {"stay_id": 1001, "concept": "lactate", "value_numeric": 2.8, "offset_minutes": 120, "window": "observation"},
            {"stay_id": 1001, "concept": "hr", "value_numeric": 110.0, "offset_minutes": 0, "window": "observation"},
            {"stay_id": 1001, "concept": "sbp", "value_numeric": 95.0, "offset_minutes": 0, "window": "observation"},
            {"stay_id": 1001, "concept": "map", "value_numeric": 70.0, "offset_minutes": 0, "window": "observation"},
            {"stay_id": 1001, "concept": "ph", "value_numeric": 7.30, "offset_minutes": 0, "window": "observation"},
            {"stay_id": 1001, "concept": "creatinine", "value_numeric": 1.5, "offset_minutes": 0, "window": "observation"},
            {"stay_id": 1001, "concept": "bilirubin_total", "value_numeric": 0.8, "offset_minutes": 0, "window": "observation"},
            {"stay_id": 1001, "concept": "spo2", "value_numeric": 95.0, "offset_minutes": 0, "window": "observation"},
            {"stay_id": 1001, "concept": "resp_rate", "value_numeric": 20.0, "offset_minutes": 0, "window": "observation"},
            {"stay_id": 1001, "concept": "temp", "value_numeric": 37.0, "offset_minutes": 0, "window": "observation"},
            # Stay 1002: normal lactate, hypotension
            {"stay_id": 1002, "concept": "lactate", "value_numeric": 1.5, "offset_minutes": 0, "window": "observation"},
            {"stay_id": 1002, "concept": "hr", "value_numeric": 85.0, "offset_minutes": 0, "window": "observation"},
            {"stay_id": 1002, "concept": "sbp", "value_numeric": 85.0, "offset_minutes": 0, "window": "observation"},
            {"stay_id": 1002, "concept": "map", "value_numeric": 55.0, "offset_minutes": 0, "window": "observation"},
            {"stay_id": 1002, "concept": "ph", "value_numeric": 7.35, "offset_minutes": 0, "window": "observation"},
            {"stay_id": 1002, "concept": "creatinine", "value_numeric": 1.0, "offset_minutes": 0, "window": "observation"},
            {"stay_id": 1002, "concept": "bilirubin_total", "value_numeric": 0.5, "offset_minutes": 0, "window": "observation"},
            {"stay_id": 1002, "concept": "spo2", "value_numeric": 98.0, "offset_minutes": 0, "window": "observation"},
            {"stay_id": 1002, "concept": "resp_rate", "value_numeric": 18.0, "offset_minutes": 0, "window": "observation"},
            {"stay_id": 1002, "concept": "temp", "value_numeric": 36.8, "offset_minutes": 0, "window": "observation"},
            # Stay 1003: severe acidemia, high lactate
            {"stay_id": 1003, "concept": "lactate", "value_numeric": 6.0, "offset_minutes": 0, "window": "observation"},
            {"stay_id": 1003, "concept": "lactate", "value_numeric": 5.5, "offset_minutes": 60, "window": "observation"},
            {"stay_id": 1003, "concept": "hr", "value_numeric": 120.0, "offset_minutes": 0, "window": "observation"},
            {"stay_id": 1003, "concept": "sbp", "value_numeric": 75.0, "offset_minutes": 0, "window": "observation"},
            {"stay_id": 1003, "concept": "map", "value_numeric": 50.0, "offset_minutes": 0, "window": "observation"},
            {"stay_id": 1003, "concept": "ph", "value_numeric": 7.15, "offset_minutes": 0, "window": "observation"},
            {"stay_id": 1003, "concept": "creatinine", "value_numeric": 2.5, "offset_minutes": 0, "window": "observation"},
            {"stay_id": 1003, "concept": "bilirubin_total", "value_numeric": 2.5, "offset_minutes": 0, "window": "observation"},
            {"stay_id": 1003, "concept": "spo2", "value_numeric": 88.0, "offset_minutes": 0, "window": "observation"},
            {"stay_id": 1003, "concept": "resp_rate", "value_numeric": 28.0, "offset_minutes": 0, "window": "observation"},
            {"stay_id": 1003, "concept": "temp", "value_numeric": 38.0, "offset_minutes": 0, "window": "observation"},
        ])

        return events, cohort

    def test_smoke_produces_50_columns(self, synthetic_data):
        """build_feature_table produces exactly 50 columns."""
        events, cohort = synthetic_data
        result = build_feature_table(events, cohort)
        assert result.shape[1] == 50

    def test_smoke_stay_count_matches(self, synthetic_data):
        """Output row count matches cohort row count."""
        events, cohort = synthetic_data
        result = build_feature_table(events, cohort)
        assert len(result) == len(cohort)

    def test_smoke_stay_ids_preserved(self, synthetic_data):
        """stay_id column is preserved in output."""
        events, cohort = synthetic_data
        result = build_feature_table(events, cohort)
        assert set(result["stay_id"]) == {1001, 1002, 1003}

    def test_smoke_scai_staging(self, synthetic_data):
        """SCAI stages are assigned correctly for synthetic data."""
        events, cohort = synthetic_data
        result = build_feature_table(events, cohort)
        # Stay 1003: severe acidemia (pH 7.15 < 7.20) → Stage E
        stay_1003 = result[result["stay_id"] == 1003].iloc[0]
        assert stay_1003["scai_stage"] == "E"

    def test_smoke_tachycardia_flag(self, synthetic_data):
        """Tachycardia flag is set for HR >= 100."""
        events, cohort = synthetic_data
        result = build_feature_table(events, cohort)
        # Stay 1001: HR max = 110 → tachycardia
        stay_1001 = result[result["stay_id"] == 1001].iloc[0]
        assert stay_1001["tachycardia_flag"] == 1
        # Stay 1002: HR max = 85 → no tachycardia
        stay_1002 = result[result["stay_id"] == 1002].iloc[0]
        assert stay_1002["tachycardia_flag"] == 0

    def test_smoke_hypotension_flag(self, synthetic_data):
        """Hypotension flag is set for SBP < 90 or MAP < 65."""
        events, cohort = synthetic_data
        result = build_feature_table(events, cohort)
        # Stay 1002: SBP min = 85 < 90 → hypotension
        stay_1002 = result[result["stay_id"] == 1002].iloc[0]
        assert stay_1002["hypotension_flag"] == 1
        # Stay 1001: SBP min = 95, MAP min = 70 → no hypotension
        stay_1001 = result[result["stay_id"] == 1001].iloc[0]
        assert stay_1001["hypotension_flag"] == 0

    def test_smoke_lactate_bins(self, synthetic_data):
        """Lactate bins are assigned correctly."""
        events, cohort = synthetic_data
        result = build_feature_table(events, cohort)
        # Stay 1001: baseline_lactate ≈ 2.8 (last value) → "2-3.1"
        stay_1001 = result[result["stay_id"] == 1001].iloc[0]
        assert stay_1001["baseline_lactate_bin"] == "2-3.1"
        # Stay 1002: baseline_lactate = 1.5 → "<2"
        stay_1002 = result[result["stay_id"] == 1002].iloc[0]
        assert stay_1002["baseline_lactate_bin"] == "<2"

    def test_smoke_no_nan_in_flags(self, synthetic_data):
        """Binary flag columns should not contain NaN."""
        events, cohort = synthetic_data
        result = build_feature_table(events, cohort)
        flag_cols = [
            "tachycardia_flag", "hypotension_flag", "severe_hypotension_flag",
            "acidemia_flag", "severe_acidemia_flag",
            "lactate_ge_2_flag", "lactate_ge_3_1_flag", "lactate_ge_5_flag",
            "persistent_lactate_flag",
            "renal_hypoperfusion_flag", "hepatic_hypoperfusion_flag",
            "occult_hypoperfusion_flag",
        ]
        for col in flag_cols:
            assert result[col].notna().all(), f"NaN found in {col}"


# ---------------------------------------------------------------------------
# Additional parity tests for edge cases
# ---------------------------------------------------------------------------


class TestLactateThresholdFlags:
    """Verify lactate threshold flags at boundary values."""

    def test_ge_2_flag(self):
        """Lactate >= 2.0 → flag = 1."""
        lactate = _make_series([1.9, 2.0, 2.1, np.nan])
        result = compute_lactate_threshold_flags(lactate)
        assert result["lactate_ge_2_flag"].iloc[0] == 0  # 1.9 < 2
        assert result["lactate_ge_2_flag"].iloc[1] == 1  # 2.0 >= 2
        assert result["lactate_ge_2_flag"].iloc[2] == 1  # 2.1 >= 2
        assert result["lactate_ge_2_flag"].iloc[3] == 0  # NaN → False → 0

    def test_ge_3_1_flag(self):
        """Lactate >= 3.1 → flag = 1."""
        lactate = _make_series([3.0, 3.1, 5.0])
        result = compute_lactate_threshold_flags(lactate)
        assert result["lactate_ge_3_1_flag"].iloc[0] == 0  # 3.0 < 3.1
        assert result["lactate_ge_3_1_flag"].iloc[1] == 1  # 3.1 >= 3.1
        assert result["lactate_ge_3_1_flag"].iloc[2] == 1  # 5.0 >= 3.1

    def test_ge_5_flag(self):
        """Lactate >= 5.0 → flag = 1."""
        lactate = _make_series([4.9, 5.0, 6.0])
        result = compute_lactate_threshold_flags(lactate)
        assert result["lactate_ge_5_flag"].iloc[0] == 0  # 4.9 < 5
        assert result["lactate_ge_5_flag"].iloc[1] == 1  # 5.0 >= 5
        assert result["lactate_ge_5_flag"].iloc[2] == 1  # 6.0 >= 5


class TestPersistentLactateFlag:
    """Verify persistent lactate elevation flag logic."""

    def test_persistent_condition_a(self):
        """Multiple measurements, first >= 2.0, last >= 2.0 → flag = 1."""
        count = _make_series([3.0])
        first = _make_series([2.5])
        last = _make_series([3.0])
        mean = _make_series([2.8])
        delta = _make_series([0.5])
        result = compute_persistent_lactate_flag(count, first, last, mean, delta)
        assert result.iloc[0] == 1

    def test_persistent_condition_b(self):
        """Multiple measurements, mean >= 2.5, delta > 0.3 → flag = 1."""
        count = _make_series([3.0])
        first = _make_series([1.5])   # first < 2.0
        last = _make_series([2.0])    # last >= 2.0 but condition A fails
        mean = _make_series([2.6])   # mean >= 2.5
        delta = _make_series([0.5])   # delta > 0.3
        result = compute_persistent_lactate_flag(count, first, last, mean, delta)
        assert result.iloc[0] == 1

    def test_not_persistent(self):
        """Single measurement → flag = 0."""
        count = _make_series([1.0])
        first = _make_series([3.0])
        last = _make_series([3.0])
        mean = _make_series([3.0])
        delta = _make_series([0.0])
        result = compute_persistent_lactate_flag(count, first, last, mean, delta)
        assert result.iloc[0] == 0


class TestOccultHypoperfusion:
    """Verify occult hypoperfusion flag logic."""

    def test_occult_hypoperfusion(self):
        """Lactate >= 2 AND no hypotension → occult hypoperfusion = 1."""
        lactate_ge_2 = pd.Series([1], dtype=int)
        hypo = pd.Series([0], dtype=int)
        result = compute_occult_hypoperfusion_flag(lactate_ge_2, hypo)
        assert result.iloc[0] == 1

    def test_not_occult_with_hypotension(self):
        """Lactate >= 2 AND hypotension → occult hypoperfusion = 0."""
        lactate_ge_2 = pd.Series([1], dtype=int)
        hypo = pd.Series([1], dtype=int)
        result = compute_occult_hypoperfusion_flag(lactate_ge_2, hypo)
        assert result.iloc[0] == 0

    def test_not_occult_no_lactate(self):
        """Lactate < 2 AND no hypotension → occult hypoperfusion = 0."""
        lactate_ge_2 = pd.Series([0], dtype=int)
        hypo = pd.Series([0], dtype=int)
        result = compute_occult_hypoperfusion_flag(lactate_ge_2, hypo)
        assert result.iloc[0] == 0


class TestPerfusionBurdenScore:
    """Verify perfusion burden score computation."""

    def test_burden_score(self):
        """Perfusion burden = modifier_burden + renal + hepatic + lactate_ge_3_1."""
        modifier = pd.Series([2], dtype=int)
        renal = pd.Series([1], dtype=int)
        hepatic = pd.Series([0], dtype=int)
        lactate_ge_3_1 = pd.Series([1], dtype=int)
        result = compute_perfusion_burden_score(modifier, renal, hepatic, lactate_ge_3_1)
        assert result.iloc[0] == pytest.approx(4.0)

    def test_burden_score_zero(self):
        """All zeros → burden = 0."""
        modifier = pd.Series([0], dtype=int)
        renal = pd.Series([0], dtype=int)
        hepatic = pd.Series([0], dtype=int)
        lactate_ge_3_1 = pd.Series([0], dtype=int)
        result = compute_perfusion_burden_score(modifier, renal, hepatic, lactate_ge_3_1)
        assert result.iloc[0] == pytest.approx(0.0)


class TestHeartRateBanding:
    """Verify HR banding produces correct categories."""

    def test_coarse_bands(self):
        """Coarse HR bands: <100, 100-119, >=120."""
        hr = _make_series([80.0, 105.0, 130.0])
        result = bin_heart_rate(hr)
        assert result[0] == "<100"
        assert result[1] == "100-119"
        assert result[2] == ">=120"

    def test_fine_bands(self):
        """Fine HR bands: <100, 100-109, 110-119, 120-139, >=140."""
        hr = _make_series([80.0, 105.0, 115.0, 130.0, 150.0])
        result = bin_heart_rate_fine(hr)
        assert result[0] == "<100"
        assert result[1] == "100-109"
        assert result[2] == "110-119"
        assert result[3] == "120-139"
        assert result[4] == ">=140"

    def test_hr_nan_filled(self):
        """NaN HR values are filled with 'missing'."""
        hr = _make_series([np.nan])
        result = bin_heart_rate(hr)
        assert result[0] == "missing"


class TestExtractConfig:
    """Verify _extract_config returns correct defaults."""

    def test_defaults_with_none(self):
        """Passing None returns all default values."""
        config = _extract_config(None)
        assert config["variables"] == [
            "lactate", "hr", "sbp", "map", "ph", "creatinine",
            "bilirubin_total", "spo2", "resp_rate", "temp",
        ]
        assert config["time_step_hours"] == 0.25
        assert config["scai_stage_labels"] == ["A", "B", "C", "D", "E"]

    def test_override_with_dict(self):
        """Passing a dict overrides specific keys."""
        config = _extract_config({"time_step_hours": 0.5})
        assert config["time_step_hours"] == 0.5
        # Other defaults remain
        assert config["variables"][0] == "lactate"