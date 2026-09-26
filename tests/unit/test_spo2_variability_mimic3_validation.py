from physiograph.analysis.spo2_variability_mimic3_validation import (
    MIMIC3_MAPPING,
    evaluate_frozen_feasibility,
)


def test_mimic3_mapping_excludes_pigtail_drain() -> None:
    assert 40086 not in MIMIC3_MAPPING["urine_output"]["itemids"]
    assert MIMIC3_MAPPING["vitals"]["spo2"] == [646]


def test_event_gate_remains_binding_despite_user_waivers() -> None:
    result = evaluate_frozen_feasibility(
        eligible_stays=100,
        valid_exposures=25,
        modeled_primary_events=49,
        exact_composite_observable=True,
    )
    assert result["association_authorized"] is False
    assert "minimum_50_modeled_events_not_demonstrable" in result["failures"]


def test_fifty_events_authorizes_single_site_replication() -> None:
    result = evaluate_frozen_feasibility(
        eligible_stays=100,
        valid_exposures=25,
        modeled_primary_events=50,
        exact_composite_observable=True,
    )
    assert result["association_authorized"] is True
    assert result["gates"]["coverage_observed"] is False
    assert result["gates"]["coverage_effective_after_waiver"] is True
