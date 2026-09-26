"""Parity tests for ETL pipeline shared utilities and audit logging.

Verifies that shared utility functions, constants, audit logging, and
data structures produce correct outputs with synthetic test data. No
actual MIMIC/eICU CSV files are required.
"""

from __future__ import annotations

import math
from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

from physiograph.etl.audit import AuditLogger
from physiograph.etl.shared import (
    COHORT_REQUIRED_COLUMNS,
    EVENT_REQUIRED_COLUMNS,
    LANDMARK_MINUTES,
    OUTCOME_WINDOW_END_MINUTES,
    OBSERVATION_HOURS,
    OUTCOME_HOURS,
    TIME_STEP_MINUTES,
    SourceExtraction,
    _iter_csv_chunks,
    build_event_frame,
    classify_offset_minutes,
    contains_any_token,
    match_icd_prefix,
    normalize_text,
    observation_time_bin,
    require_columns,
    sanitize_events,
    series_contains_any,
)
from physiograph.pipeline import derive_labels


def test_explicit_landmark_pressor_status_overrides_earlier_start():
    cohort = pd.DataFrame(
        {
            "dataset": ["mimic"],
            "stay_id": [1],
            "person_id": [10],
            "age": [60],
            "is_male": [1],
            "cohort_hf_flag": [1],
            "shock_icd_flag": [0],
            "death_offset_minutes": [np.nan],
            "followup_end_offset_minutes": [2000],
        }
    )
    events = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "concept": ["pressor", "pressor_active"],
            "offset_minutes": [100.0, 240.0],
            "value_numeric": [1.0, 0.0],
            "is_intervention": [1, 0],
            "event_family": ["intervention", "intervention_status"],
            "window": ["observation", "landmark"],
        }
    )
    updated, _ = derive_labels(cohort, events)
    assert updated.iloc[0]["baseline_vasoactive_flag"] == 0
    assert updated.iloc[0]["baseline_vasoactive_method"] == (
        "interval_active_at_minute_240"
    )


def test_explicit_landmark_mcs_status_overrides_ended_early_support():
    cohort = pd.DataFrame(
        {
            "dataset": ["mimic"],
            "stay_id": [1],
            "person_id": [10],
            "age": [60],
            "is_male": [1],
            "cohort_hf_flag": [1],
            "shock_icd_flag": [0],
            "death_offset_minutes": [np.nan],
            "followup_end_offset_minutes": [2000],
        }
    )
    events = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "concept": ["mcs", "mcs_active"],
            "offset_minutes": [100.0, 240.0],
            "value_numeric": [1.0, 0.0],
            "is_intervention": [1, 0],
            "event_family": ["intervention", "intervention_status"],
            "window": ["observation", "landmark"],
        }
    )
    updated, _ = derive_labels(cohort, events)
    assert updated.iloc[0]["baseline_mcs_flag"] == 0


# ──────────────────────────────────────────────────────────────────────
# 1. CHUNK_SIZE parity
# ──────────────────────────────────────────────────────────────────────


def test_chunk_size():
    """Verify CHUNK_SIZE is 250,000 in both extractors."""
    from physiograph.etl.mimic_extractor import CHUNK_SIZE as MIMIC_CHUNK
    from physiograph.etl.eicu_extractor import CHUNK_SIZE as EICU_CHUNK

    assert MIMIC_CHUNK == 250_000, f"MIMIC CHUNK_SIZE={MIMIC_CHUNK}, expected 250000"
    assert EICU_CHUNK == 250_000, f"eICU CHUNK_SIZE={EICU_CHUNK}, expected 250000"
    assert MIMIC_CHUNK == EICU_CHUNK, "MIMIC and eICU CHUNK_SIZE must match"


def test_chunk_streamer_disables_nested_low_memory_inference(monkeypatch, tmp_path):
    """Mixed clinical CSV values must not trigger pandas' sub-chunk parser bug."""
    captured = {}

    def fake_read_csv(path, **kwargs):
        captured.update(kwargs)
        return iter([pd.DataFrame({"value": ["1", "ventilator"]})])

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)
    chunks = list(
        _iter_csv_chunks(
            tmp_path / "mixed.csv",
            usecols=["value"],
            chunksize=10,
            max_chunks=None,
        )
    )
    assert captured["low_memory"] is False
    assert len(chunks) == 1


def test_mimic_lab_dictionary_discovers_new_release_itemids(tmp_path):
    from physiograph.etl.mimic_extractor import _mimic_lab_item_map

    pd.DataFrame(
        {
            "itemid": [50813, 52442, 53154, 52024, 52546, 52642, 53089, 53189, 99999],
            "label": [
                "Lactate",
                "Lactate",
                "Lactate",
                "Creatinine, Whole Blood",
                "Creatinine",
                "Troponin I",
                "Bilirubin, Total",
                "Platelet Count",
                "Creatinine, Urine",
            ],
            "fluid": [
                "Blood",
                "Blood",
                "Blood",
                "Blood",
                "Blood",
                "Blood",
                "Blood",
                "Blood",
                "Urine",
            ],
        }
    ).to_csv(tmp_path / "d_labitems.csv", index=False)
    mapping = _mimic_lab_item_map(tmp_path)
    assert mapping[52442] == "lactate"
    assert mapping[53154] == "lactate"
    assert mapping[52546] == "creatinine"
    assert mapping[52024] == "creatinine"
    assert mapping[52642] == "troponin_i"
    assert mapping[53089] == "bilirubin_total"
    assert mapping[53189] == "platelets"
    assert 99999 not in mapping


def test_mimic_dictionary_maps_only_operational_mcs_chart_context(tmp_path):
    from physiograph.etl.mimic_extractor import _mimic_chartevent_item_map

    pd.DataFrame(
        {
            "itemid": [900001, 900002, 900003, 900004, 900005, 900006],
            "label": [
                "Flow (LVAD)",
                "Flow (RVAD)",
                "Flow Rate (Impella)",
                "Impella Line Discontinued",
                "FiO2 (ECMO)",
                "Inspired O2 Fraction",
            ],
            "linksto": ["chartevents"] * 6,
            "category": [
                "Centrimag",
                "Centrimag",
                "Impella",
                "Impella",
                "ECMO",
                "Respiratory",
            ],
        }
    ).to_csv(tmp_path / "d_items.csv", index=False)
    mapping = _mimic_chartevent_item_map(tmp_path)
    assert mapping[900001] == "mcs_context_lvad"
    assert mapping[900002] == "mcs_context_rvad"
    assert mapping[900003] == "mcs_context_impella"
    assert 900004 not in mapping
    assert mapping[900005] == "mcs_context_ecmo"
    assert mapping[900006] == "fio2"


def test_mimic_operational_mcs_context_extends_through_outcome_window(tmp_path):
    from physiograph.etl.mimic_extractor import _stream_mimic_measurement_events

    pd.DataFrame(
        {
            "stay_id": [1, 1],
            "itemid": [900001, 900002],
            "charttime": ["2020-01-01 05:00:00", "2020-01-01 05:00:00"],
            "valuenum": [80.0, 4.5],
            "value": ["80", "4.5"],
            "valueuom": ["bpm", "L/min"],
            "warning": [0, 0],
        }
    ).to_csv(tmp_path / "chartevents.csv", index=False)
    events, _ = _stream_mimic_measurement_events(
        tmp_path,
        file_name="chartevents.csv",
        source_table="chartevents.csv",
        item_map={900001: "hr", 900002: "mcs_context_lvad"},
        event_family="vital",
        cohort_ids=np.array([1]),
        anchors=pd.DataFrame(
            {"stay_id": [1], "anchor_time": pd.to_datetime(["2020-01-01"])}
        ),
        id_column="stay_id",
        offset_min=0.0,
        offset_max=240.0,
        upper_inclusive=False,
        extended_context_concepts={"mcs_context_lvad"},
        chunk_size=10,
        max_chunks=None,
    )
    assert set(events["concept"]) == {"mcs_context_lvad"}
    assert events.iloc[0]["offset_minutes"] == 300


def test_mimic_pressor_restart_grace_avoids_false_reinitiation(tmp_path):
    from physiograph.etl.mimic_extractor import _extract_mimic_pressor_events

    pd.DataFrame(
        {
            "stay_id": [1, 1, 2, 2],
            "starttime": [
                "2020-01-01 00:00:00",
                "2020-01-01 00:13:00",
                "2020-01-01 00:00:00",
                "2020-01-01 00:16:00",
            ],
            "endtime": [
                "2020-01-01 00:10:00",
                "2020-01-01 00:20:00",
                "2020-01-01 00:10:00",
                "2020-01-01 00:20:00",
            ],
            "itemid": [221906, 221906, 221906, 221906],
            "rate": [0.1, 0.1, 0.1, 0.1],
            "rateuom": ["mcg/kg/min"] * 4,
            "patientweight": [80.0] * 4,
        }
    ).to_csv(tmp_path / "inputevents.csv", index=False)
    anchors = pd.DataFrame(
        {
            "stay_id": [1, 2],
            "anchor_time": pd.to_datetime(["2020-01-01", "2020-01-01"]),
        }
    )
    events, _ = _extract_mimic_pressor_events(
        tmp_path,
        np.array([1, 2]),
        anchors,
        chunk_size=10,
        max_chunks=None,
    )
    initiations = events.loc[events["concept"].eq("pressor_initiation")]
    assert initiations.loc[initiations["stay_id"].eq(1), "offset_minutes"].tolist() == [0.0]
    assert initiations.loc[initiations["stay_id"].eq(2), "offset_minutes"].tolist() == [0.0, 16.0]


def test_eicu_pressor_trade_names_are_not_silently_missed(tmp_path):
    from physiograph.etl.eicu_extractor import _stream_eicu_infusion_events

    pd.DataFrame(
        {
            "patientunitstayid": [1, 1, 1, 1],
            "infusionoffset": [100, 200, 300, 400],
            "drugname": [
                "Levophed (mcg/min)",
                "Neo-Synephrine (mcg/min)",
                "Dobutrex (mcg/kg/min)",
                "Primacore (mcg/kg/min)",
            ],
            "drugrate": [2.0, 3.0, 4.0, 0.5],
            "infusionrate": [np.nan] * 4,
        }
    ).to_csv(tmp_path / "infusionDrug.csv", index=False)
    events = _stream_eicu_infusion_events(
        tmp_path,
        [1],
        chunk_size=10,
        max_chunks=None,
    )
    pressors = events.loc[events["concept"].eq("pressor")]
    assert len(pressors) == 4
    assert set(pressors["raw_name"]) == {
        "Levophed (mcg/min)",
        "Neo-Synephrine (mcg/min)",
        "Dobutrex (mcg/kg/min)",
        "Primacore (mcg/kg/min)",
    }


def test_eicu_mounted_lab_aliases_are_extracted(tmp_path):
    from physiograph.etl.eicu_extractor import _stream_eicu_lab_events

    names = ["platelets x 1000", "pt - inr", "troponin - i", "troponin - t"]
    pd.DataFrame(
        {
            "patientunitstayid": [1] * 4,
            "labresultoffset": [100, 110, 120, 130],
            "labname": names,
            "labresult": [200.0, 1.2, 0.1, 0.02],
            "labmeasurenamesystem": [""] * 4,
            "labmeasurenameinterface": [""] * 4,
        }
    ).to_csv(tmp_path / "lab.csv", index=False)
    events = _stream_eicu_lab_events(
        tmp_path,
        [1],
        chunk_size=10,
        max_chunks=None,
    )
    assert set(events["concept"]) == {
        "platelets",
        "inr",
        "troponin_i",
        "troponin_t",
    }


def test_eicu_mcs_removal_is_not_misclassified_as_initiation(tmp_path):
    from physiograph.etl.eicu_extractor import _load_eicu_treatment_events

    pd.DataFrame(
        {
            "patientunitstayid": [1, 2],
            "treatmentoffset": [300, 300],
            "treatmentstring": [
                "cardiovascular|non-operative procedures|intraaortic balloon pump removal",
                "cardiovascular|non-operative procedures|intraaortic balloon pump",
            ],
        }
    ).to_csv(tmp_path / "treatment.csv", index=False)
    events = _load_eicu_treatment_events(
        tmp_path,
        [1, 2],
        chunk_size=10,
        max_chunks=None,
    )
    mcs = events.loc[events["concept"].eq("mcs")]
    assert mcs["stay_id"].tolist() == [2]
    assert mcs["value_text"].tolist() == ["iabp|first_documentation"]


def test_eicu_dialysis_catheter_is_not_baseline_rrt_therapy(tmp_path):
    from physiograph.etl.eicu_extractor import _load_eicu_treatment_events

    pd.DataFrame(
        {
            "patientunitstayid": [1, 2],
            "treatmentoffset": [30, 30],
            "treatmentstring": [
                "renal|dialysis|insertion of venous catheter for hemodialysis",
                "renal|dialysis|C V V H D",
            ],
        }
    ).to_csv(tmp_path / "treatment.csv", index=False)
    events = _load_eicu_treatment_events(
        tmp_path, [1, 2], chunk_size=10, max_chunks=None
    )
    rrt = events.loc[events["concept"].eq("rrt")]
    assert rrt["stay_id"].tolist() == [2]


def test_eicu_urine_excludes_occurrence_counts_and_mixed_stool(tmp_path):
    from physiograph.etl.eicu_extractor import _stream_eicu_urine_output_events

    pd.DataFrame(
        {
            "patientunitstayid": [1, 1, 1, 1],
            "intakeoutputoffset": [60, 120, 180, 240],
            "intakeoutputentryoffset": [60, 120, 180, 240],
            "celllabel": [
                "Urine",
                "Urine Count",
                "Urine Occurrence",
                "Mixed Urine/Stool Volume",
            ],
            "cellpath": ["I&O|Output (ml)"] * 4,
            "cellvaluenumeric": [100.0, 1.0, 1.0, 200.0],
        }
    ).to_csv(tmp_path / "intakeOutput.csv", index=False)
    events = _stream_eicu_urine_output_events(
        tmp_path,
        [1],
        chunk_size=10,
        max_chunks=None,
    )
    assert events["value_numeric"].tolist() == [100.0]
    assert events["raw_name"].tolist() == ["Urine"]


def test_eicu_urine_uses_observation_offset_not_entry_offset(tmp_path):
    from physiograph.etl.eicu_extractor import _stream_eicu_urine_output_events

    pd.DataFrame(
        {
            "patientunitstayid": [1],
            "intakeoutputoffset": [200],
            "intakeoutputentryoffset": [300],
            "celllabel": ["Urine"],
            "cellpath": ["I&O|Output (ml)|Urine"],
            "cellvaluenumeric": [100.0],
        }
    ).to_csv(tmp_path / "intakeOutput.csv", index=False)
    events = _stream_eicu_urine_output_events(
        tmp_path,
        [1],
        chunk_size=10,
        max_chunks=None,
    )
    assert events.iloc[0]["offset_minutes"] == 200


def test_eicu_uses_one_first_unit_anchor_per_hospital_encounter(tmp_path):
    from physiograph.etl.eicu_extractor import (
        EICU_COHORT_FLAG_COLUMNS,
        EICU_PATIENT_COLUMNS,
        _build_eicu_cohort,
    )

    base = {column: [pd.NA, pd.NA] for column in EICU_PATIENT_COLUMNS}
    base.update(
        {
            "patientunitstayid": [100, 101],
            "patienthealthsystemstayid": [50, 50],
            "uniquepid": ["p1", "p1"],
            "gender": ["Female", "Female"],
            "age": [70, 70],
            "unitvisitnumber": [1, 2],
            "hospitaladmitoffset": [-60, -180],
            "unitdischargeoffset": [2000, 1500],
            "hospitaldischargeoffset": [4000, 4000],
            "hospitaldischargeyear": [2015, 2015],
            "admissionweight": [70, 70],
        }
    )
    patients = pd.DataFrame(base)
    flags = pd.DataFrame(
        {
            "stay_id": [100, 101],
            "cohort_hf_flag": [0, 1],
            "shock_icd_flag": [0, 0],
            "cardiomyopathy_flag": [0, 0],
            "acute_mi_flag": [0, 0],
        }
    )
    cohort, ids = _build_eicu_cohort(
        tmp_path,
        patients,
        flags,
        AuditLogger(dataset="eicu"),
        max_stays=None,
    )
    assert ids == [100]
    assert cohort.iloc[0]["cohort_hf_flag"] == 1
    assert cohort.iloc[0]["hospital_encounter_id"] == 50
    assert set(EICU_COHORT_FLAG_COLUMNS).issuperset(
        {"cohort_hf_flag", "shock_icd_flag"}
    )


def test_eicu_shock_cardiomyopathy_or_mi_never_substitutes_for_hf(tmp_path):
    from physiograph.etl.eicu_extractor import (
        EICU_PATIENT_COLUMNS,
        _build_eicu_cohort,
    )

    base = {column: [pd.NA] * 3 for column in EICU_PATIENT_COLUMNS}
    base.update(
        {
            "patientunitstayid": [100, 200, 300],
            "patienthealthsystemstayid": [10, 20, 30],
            "uniquepid": ["p1", "p2", "p3"],
            "gender": ["Female", "Male", "Female"],
            "age": [70, 70, 70],
            "unitvisitnumber": [1, 1, 1],
            "hospitaladmitoffset": [-60, -60, -60],
            "unitdischargeoffset": [2000, 2000, 2000],
            "hospitaldischargeoffset": [4000, 4000, 4000],
            "hospitaldischargeyear": [2015, 2015, 2015],
        }
    )
    flags = pd.DataFrame(
        {
            "stay_id": [100, 200, 300],
            "cohort_hf_flag": [1, 0, 0],
            "shock_icd_flag": [0, 1, 0],
            "cardiomyopathy_flag": [0, 0, 1],
            "acute_mi_flag": [0, 1, 1],
        }
    )
    cohort, ids = _build_eicu_cohort(
        tmp_path,
        pd.DataFrame(base),
        flags,
        AuditLogger(dataset="eicu"),
        max_stays=None,
    )
    assert ids == [100]
    assert cohort["cohort_hf_flag"].eq(1).all()


def test_eicu_treatment_table_counts_as_respiratory_context_source(tmp_path):
    from physiograph.etl.eicu_extractor import (
        EICU_PATIENT_COLUMNS,
        _build_eicu_cohort,
    )

    (tmp_path / "treatment.csv").touch()
    base = {column: [pd.NA] for column in EICU_PATIENT_COLUMNS}
    base.update(
        {
            "patientunitstayid": [100],
            "patienthealthsystemstayid": [50],
            "uniquepid": ["p1"],
            "gender": ["Female"],
            "age": [70],
            "unitvisitnumber": [1],
            "hospitaladmitoffset": [-60],
            "unitdischargeoffset": [2000],
            "hospitaldischargeoffset": [4000],
            "hospitaldischargeyear": [2015],
        }
    )
    flags = pd.DataFrame(
        {
            "stay_id": [100],
            "cohort_hf_flag": [1],
            "shock_icd_flag": [0],
            "cardiomyopathy_flag": [0],
            "acute_mi_flag": [0],
        }
    )
    cohort, _ = _build_eicu_cohort(
        tmp_path,
        pd.DataFrame(base),
        flags,
        AuditLogger(dataset="eicu"),
        max_stays=None,
    )
    assert cohort.iloc[0]["respiratory_source_available"] == 1


# ──────────────────────────────────────────────────────────────────────
# 2. normalize_text
# ──────────────────────────────────────────────────────────────────────


class TestNormalizeText:
    """Tests for normalize_text utility."""

    def test_lowercase(self):
        assert normalize_text("Heart Failure") == "heart failure"

    def test_strip_whitespace(self):
        assert normalize_text("  lactate  ") == "lactate"

    def test_special_characters_removed(self):
        assert normalize_text("BUN/Creatinine") == "bun creatinine"

    def test_none_returns_empty(self):
        assert normalize_text(None) == ""

    def test_nan_returns_empty(self):
        assert normalize_text(float("nan")) == ""

    def test_multiple_spaces_collapsed(self):
        assert normalize_text("a   b   c") == "a b c"

    def test_numbers_preserved(self):
        assert normalize_text("ICD 9 Code") == "icd 9 code"


# ──────────────────────────────────────────────────────────────────────
# 3. match_icd_prefix
# ──────────────────────────────────────────────────────────────────────


class TestMatchIcdPrefix:
    """Tests for ICD-9/10 prefix matching."""

    def test_icd9_match(self):
        codes = pd.Series(["4280", "4289", "I509", "39891"])
        versions = pd.Series([9, 9, 10, 9])
        pattern_map = {9: [r"^428"], 10: [r"^I50"]}
        result = match_icd_prefix(codes, versions, pattern_map)
        assert result.tolist() == [True, True, True, False]

    def test_icd10_match(self):
        codes = pd.Series(["I509", "I500", "4280"])
        versions = pd.Series([10, 10, 9])
        pattern_map = {9: [r"^428"], 10: [r"^I50"]}
        result = match_icd_prefix(codes, versions, pattern_map)
        assert result.tolist() == [True, True, True]

    def test_no_match(self):
        codes = pd.Series(["A00", "B99"])
        versions = pd.Series([10, 10])
        pattern_map = {9: [r"^428"], 10: [r"^I50"]}
        result = match_icd_prefix(codes, versions, pattern_map)
        assert result.tolist() == [False, False]

    def test_nan_codes(self):
        codes = pd.Series([None, "4280"])
        versions = pd.Series([9, 9])
        pattern_map = {9: [r"^428"]}
        result = match_icd_prefix(codes, versions, pattern_map)
        assert result.tolist() == [False, True]

    def test_multiple_patterns_per_version(self):
        codes = pd.Series(["4280", "78551", "I509", "R570"])
        versions = pd.Series([9, 9, 10, 10])
        pattern_map = {9: [r"^428", r"^78551"], 10: [r"^I50", r"^R570"]}
        result = match_icd_prefix(codes, versions, pattern_map)
        assert result.tolist() == [True, True, True, True]


# ──────────────────────────────────────────────────────────────────────
# 4. classify_offset_minutes
# ──────────────────────────────────────────────────────────────────────


class TestClassifyOffsetMinutes:
    """Tests for time window classification."""

    def test_observation_window(self):
        assert classify_offset_minutes(0) == "observation"
        assert classify_offset_minutes(120) == "observation"
        assert classify_offset_minutes(239) == "observation"

    def test_landmark(self):
        assert classify_offset_minutes(240) == "landmark"

    def test_outcome_window(self):
        assert classify_offset_minutes(241) == "outcome"
        assert classify_offset_minutes(1000) == "outcome"
        assert classify_offset_minutes(1440) == "outcome"

    def test_outside_window(self):
        assert classify_offset_minutes(-1) == "outside"
        assert classify_offset_minutes(1681) == "outside"
        assert classify_offset_minutes(None) == "outside"

    def test_boundary_values(self):
        # Exactly 0 is observation
        assert classify_offset_minutes(0) == "observation"
        # Exactly 240 is landmark (not observation)
        assert classify_offset_minutes(240) == "landmark"
        # Exactly 1680 is outcome (inclusive end of window)
        assert classify_offset_minutes(1680) == "outcome"


# ──────────────────────────────────────────────────────────────────────
# 5. observation_time_bin
# ──────────────────────────────────────────────────────────────────────


class TestObservationTimeBin:
    """Tests for 15-minute time binning within observation window."""

    def test_bin_zero(self):
        assert observation_time_bin(0) == 0

    def test_bin_mid_window(self):
        # 120 minutes / 15 = bin 8
        assert observation_time_bin(120) == 8

    def test_bin_near_landmark(self):
        # 225 minutes / 15 = bin 15
        assert observation_time_bin(225) == 15

    def test_outside_observation_returns_none(self):
        assert observation_time_bin(240) is None
        assert observation_time_bin(300) is None
        assert observation_time_bin(-1) is None
        assert observation_time_bin(None) is None

    def test_max_bin_clamp(self):
        # 239 minutes / 15 = 15.93 → floor = 15, but max_bin = 15
        assert observation_time_bin(239) == 15

    def test_bin_count(self):
        # 4 hours = 240 minutes, 15-min steps → 16 bins (0..15)
        assert observation_time_bin(0) == 0
        assert observation_time_bin(239) == 15


# ──────────────────────────────────────────────────────────────────────
# 6. build_event_frame
# ──────────────────────────────────────────────────────────────────────


class TestBuildEventFrame:
    """Tests for event frame construction."""

    def test_basic_construction(self):
        df = build_event_frame(
            dataset="mimic",
            stay_id=pd.Series([1001, 1002]),
            event_family="vital",
            concept="hr",
            source_table="chartevents.csv",
            raw_name="Heart Rate",
            offset_minutes=pd.Series([10.0, 20.0]),
            value_numeric=pd.Series([80.0, 90.0]),
            value_text=pd.NA,
            unit="bpm",
            is_intervention=0,
        )
        assert len(df) == 2
        assert set(df.columns) == set(EVENT_REQUIRED_COLUMNS)
        assert df["dataset"].tolist() == ["mimic", "mimic"]
        assert df["event_family"].tolist() == ["vital", "vital"]
        assert df["concept"].tolist() == ["hr", "hr"]
        assert df["is_intervention"].tolist() == [0, 0]
        assert df["is_outcome_event"].tolist() == [0, 0]

    def test_string_concept(self):
        df = build_event_frame(
            dataset="eicu",
            stay_id=pd.Series([2001]),
            event_family="death",
            concept="death",
            source_table="patient.csv",
            raw_name="discharge_status",
            offset_minutes=pd.Series([500.0]),
            value_numeric=pd.NA,
            value_text="death",
            unit=pd.NA,
            is_intervention=0,
        )
        assert len(df) == 1
        assert df["concept"].iloc[0] == "death"

    def test_stay_id_numeric_coercion(self):
        df = build_event_frame(
            dataset="mimic",
            stay_id=pd.Series(["1001", "1002"]),
            event_family="lab",
            concept="lactate",
            source_table="labevents.csv",
            raw_name="Lactate",
            offset_minutes=pd.Series([5.0, 10.0]),
        )
        assert df["stay_id"].dtype == "Int64"
        assert df["stay_id"].iloc[0] == 1001


# ──────────────────────────────────────────────────────────────────────
# 7. AuditLogger
# ──────────────────────────────────────────────────────────────────────


class TestAuditLogger:
    """Tests for audit logging format and structure."""

    def test_entry_format(self):
        logger = AuditLogger(dataset="mimic")
        logger.log("step_one", row_count=100, stay_count=50)
        assert len(logger.entries) == 1
        entry = logger.entries[0]
        assert "timestamp_utc" in entry
        assert entry["dataset"] == "mimic"
        assert entry["step"] == "step_one"
        assert entry["row_count"] == 100
        assert entry["stay_count"] == 50

    def test_optional_fields(self):
        logger = AuditLogger(dataset="eicu")
        logger.log("step_two", row_count=200)
        entry = logger.entries[0]
        assert "stay_count" not in entry
        assert entry["row_count"] == 200

    def test_details_field(self):
        logger = AuditLogger(dataset="mimic")
        logger.log("step_three", row_count=50, details={"max_stays": 10})
        entry = logger.entries[0]
        assert entry["details"] == {"max_stays": 10}

    def test_multiple_entries(self):
        logger = AuditLogger(dataset="mimic")
        logger.log("step_a", row_count=10)
        logger.log("step_b", row_count=20)
        assert len(logger.entries) == 2
        assert logger.entries[0]["step"] == "step_a"
        assert logger.entries[1]["step"] == "step_b"

    def test_timestamp_is_iso_format(self):
        logger = AuditLogger(dataset="mimic")
        logger.log("step_ts", row_count=1)
        ts = logger.entries[0]["timestamp_utc"]
        # ISO format should contain 'T' separator
        assert "T" in ts


# ──────────────────────────────────────────────────────────────────────
# 8. require_columns
# ──────────────────────────────────────────────────────────────────────


class TestRequireColumns:
    """Tests for column requirement assertion."""

    def test_passes_when_all_present(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        # Should not raise
        require_columns(df, ["a", "b"], "test_df")

    def test_raises_on_missing(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="missing required columns"):
            require_columns(df, ["a", "b", "c"], "test_df")

    def test_error_message_contains_name(self):
        df = pd.DataFrame({"x": [1]})
        with pytest.raises(ValueError, match="my_table"):
            require_columns(df, ["missing_col"], "my_table")


# ──────────────────────────────────────────────────────────────────────
# 9. sanitize_events (Fahrenheit → Celsius)
# ──────────────────────────────────────────────────────────────────────


class TestSanitizeEvents:
    """Tests for Fahrenheit-to-Celsius temperature conversion."""

    def _make_events_df(self, concepts, values):
        return pd.DataFrame(
            {
                "dataset": "eicu",
                "stay_id": [1] * len(concepts),
                "event_family": "vital",
                "concept": concepts,
                "source_table": "vitalPeriodic.csv",
                "raw_name": "temperature",
                "offset_minutes": [10.0] * len(concepts),
                "window": "",
                "time_bin": pd.NA,
                "value_numeric": values,
                "value_text": pd.NA,
                "unit": pd.NA,
                "is_intervention": 0,
                "is_outcome_event": 0,
            }
        )

    def test_fahrenheit_to_celsius(self):
        # 98.6°F = 37.0°C
        df = self._make_events_df(["temp"], [98.6])
        result = sanitize_events(df)
        assert math.isclose(result["value_numeric"].iloc[0], 37.0, abs_tol=0.1)

    def test_celsius_preserved(self):
        # 37.0°C should stay 37.0 (≤ 50 threshold)
        df = self._make_events_df(["temp"], [37.0])
        result = sanitize_events(df)
        assert math.isclose(result["value_numeric"].iloc[0], 37.0, abs_tol=0.01)

    def test_non_temp_unaffected(self):
        # HR values should never be converted
        df = self._make_events_df(["hr"], [80.0])
        result = sanitize_events(df)
        assert result["value_numeric"].iloc[0] == 80.0

    def test_mixed_concepts(self):
        df = self._make_events_df(
            ["temp", "hr", "temp"],
            [98.6, 80.0, 104.0],
        )
        result = sanitize_events(df)
        # 98.6°F → ~37°C, 80.0 HR unchanged, 104°F → ~40°C
        assert math.isclose(result["value_numeric"].iloc[0], 37.0, abs_tol=0.1)
        assert result["value_numeric"].iloc[1] == 80.0
        assert math.isclose(result["value_numeric"].iloc[2], 40.0, abs_tol=0.1)

    def test_impossible_adjustment_values_are_missing_not_model_inputs(self):
        df = self._make_events_df(
            ["spo2", "hr", "ph", "creatinine", "resp_rate"],
            [150.0, 999.0, 2.0, -1.0, 0.0],
        )
        result = sanitize_events(df)
        assert result["value_numeric"].isna().all()


# ──────────────────────────────────────────────────────────────────────
# 10. contains_any_token
# ──────────────────────────────────────────────────────────────────────


class TestContainsAnyToken:
    """Tests for token matching in eICU diagnosis strings."""

    def test_single_token_match(self):
        assert contains_any_token("congestive heart failure", ["heart failure"]) is True

    def test_no_token_match(self):
        assert contains_any_token("pneumonia", ["heart failure"]) is False

    def test_multiple_tokens_any_match(self):
        assert (
            contains_any_token(
                "acute myocardial infarction",
                ["heart failure", "myocardial infarction"],
            )
            is True
        )

    def test_normalized_matching(self):
        # normalize_text lowercases and strips special chars
        assert contains_any_token("Cardiogenic Shock!", ["cardiogenic shock"]) is True

    def test_empty_value(self):
        assert contains_any_token("", ["heart failure"]) is False

    def test_empty_tokens(self):
        assert contains_any_token("heart failure", []) is False

    def test_series_contains_any(self):
        series = pd.Series(
            ["congestive heart failure", "pneumonia", "cardiogenic shock"]
        )
        result = series_contains_any(series, ["heart failure", "cardiogenic shock"])
        assert result.tolist() == [True, False, True]


# ──────────────────────────────────────────────────────────────────────
# 11. SourceExtraction dataclass
# ──────────────────────────────────────────────────────────────────────


class TestSourceExtraction:
    """Tests for SourceExtraction dataclass fields."""

    def test_has_expected_fields(self):
        field_names = {f.name for f in fields(SourceExtraction)}
        assert field_names == {"cohort_df", "events_df"}

    def test_instantiation(self):
        cohort = pd.DataFrame({"stay_id": [1, 2]})
        events = pd.DataFrame({"stay_id": [1], "concept": ["hr"]})
        extraction = SourceExtraction(cohort_df=cohort, events_df=events)
        assert len(extraction.cohort_df) == 2
        assert len(extraction.events_df) == 1


# ──────────────────────────────────────────────────────────────────────
# 12. COHORT_REQUIRED_COLUMNS
# ──────────────────────────────────────────────────────────────────────


class TestCohortRequiredColumns:
    """Tests for COHORT_REQUIRED_COLUMNS list."""

    def test_contains_required_fields(self):
        required = [
            "dataset",
            "stay_id",
            "person_id",
            "admit_time",
            "admit_year",
            "age",
            "is_male",
            "cohort_hf_flag",
            "shock_icd_flag",
            "early_icu_flag",
            "death_offset_minutes",
            "excluded_before_landmark_flag",
            "exclusion_reason",
        ]
        for col in required:
            assert col in COHORT_REQUIRED_COLUMNS, f"Missing: {col}"

    def test_length(self):
        assert len(COHORT_REQUIRED_COLUMNS) == 13

    def test_event_required_columns(self):
        assert "stay_id" in EVENT_REQUIRED_COLUMNS
        assert "offset_minutes" in EVENT_REQUIRED_COLUMNS
        assert "concept" in EVENT_REQUIRED_COLUMNS
        assert "window" in EVENT_REQUIRED_COLUMNS
        assert "time_bin" in EVENT_REQUIRED_COLUMNS


# ──────────────────────────────────────────────────────────────────────
# Time window constants parity
# ──────────────────────────────────────────────────────────────────────


class TestTimeWindowConstants:
    """Verify time window constants match specification."""

    def test_observation_hours(self):
        assert OBSERVATION_HOURS == 4.0

    def test_outcome_hours(self):
        assert OUTCOME_HOURS == 24.0

    def test_landmark_minutes(self):
        assert LANDMARK_MINUTES == 240

    def test_outcome_window_end_minutes(self):
        assert OUTCOME_WINDOW_END_MINUTES == 1680

    def test_time_step_minutes(self):
        assert TIME_STEP_MINUTES == 15
