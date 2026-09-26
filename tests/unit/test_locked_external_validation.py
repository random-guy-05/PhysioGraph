from __future__ import annotations

import pandas as pd

from physiograph.cohort.harmonization import harmonize_hf_cohort
from physiograph.analysis.spo2_multiorgan_mechanistic import (
    _clean_eicu_ventilation_proxy,
)
from physiograph.etl.mimic_extractor import (
    extract_mimic_respiratory_procedure_events,
)


def test_harmonized_hf_rule_is_identical_and_fail_closed_across_databases() -> None:
    cohort = pd.DataFrame(
        [
            {
                "dataset": "mimic",
                "stay_id": 1,
                "hospital_encounter_id": "m1",
                "age": 60,
                "cohort_hf_flag": 1,
                "early_icu_flag": 1,
                "shock_icd_flag": 0,
                "excluded_before_landmark_flag": 0,
            },
            {
                "dataset": "mimic",
                "stay_id": 2,
                "hospital_encounter_id": "m2",
                "age": 60,
                "cohort_hf_flag": 0,
                "early_icu_flag": 1,
                "shock_icd_flag": 1,
                "excluded_before_landmark_flag": 0,
            },
            {
                "dataset": "eicu",
                "stay_id": 10,
                "hospital_encounter_id": "e1",
                "unitvisitnumber": 1,
                "age": 70,
                "cohort_hf_flag": 0,
                "early_icu_flag": 1,
                "shock_icd_flag": 0,
                "excluded_before_landmark_flag": 0,
            },
            {
                "dataset": "eicu",
                "stay_id": 11,
                "hospital_encounter_id": "e1",
                "unitvisitnumber": 2,
                "age": 70,
                "cohort_hf_flag": 1,
                "early_icu_flag": 0,
                "shock_icd_flag": 0,
                "excluded_before_landmark_flag": 0,
            },
            {
                "dataset": "eicu",
                "stay_id": 12,
                "hospital_encounter_id": "e2",
                "unitvisitnumber": 1,
                "age": 65,
                "cohort_hf_flag": 1,
                "early_icu_flag": 0,
                "shock_icd_flag": 0,
                "excluded_before_landmark_flag": 0,
            },
            {
                "dataset": "eicu",
                "stay_id": 13,
                "hospital_encounter_id": "e3",
                "unitvisitnumber": 1,
                "age": 65,
                "cohort_hf_flag": 1,
                "early_icu_flag": 0,
                "shock_icd_flag": 1,
                "excluded_before_landmark_flag": 0,
            },
            {
                "dataset": "eicu",
                "stay_id": 14,
                "hospital_encounter_id": "e4",
                "unitvisitnumber": 1,
                "age": 17,
                "cohort_hf_flag": 1,
                "early_icu_flag": 1,
                "shock_icd_flag": 0,
                "excluded_before_landmark_flag": 0,
            },
        ]
    )
    harmonized, audit = harmonize_hf_cohort(cohort)
    assert set(zip(harmonized["dataset"], harmonized["stay_id"])) == {
        ("mimic", 1),
        ("eicu", 10),
        ("eicu", 13),
    }
    # HF documented on a later transfer is propagated to the encounter, but
    # the first ICU unit remains the sole analysis anchor.
    eicu_first = harmonized.loc[harmonized["stay_id"].eq(10)].iloc[0]
    assert eicu_first["cohort_hf_flag"] == 1
    eicu_audit = audit.set_index("dataset").loc["eicu"]
    assert eicu_audit["transfer_rows_removed"] == 1
    assert eicu_audit["included_encounters"] == 2


def test_targeted_mimic_procedure_scan_recovers_direct_respiratory_starts(
    tmp_path,
) -> None:
    pd.DataFrame(
        {
            "itemid": [225792, 224385, 999999],
            "label": ["Invasive Ventilation", "Intubation", "Other Procedure"],
            "linksto": ["procedureevents", "procedureevents", "procedureevents"],
            "category": ["2-Ventilation", "1-Intubation/Extubation", "Other"],
        }
    ).to_csv(tmp_path / "d_items.csv", index=False)
    pd.DataFrame(
        {
            "stay_id": [1, 1, 2, 1],
            "starttime": [
                "2020-01-01 08:00:00",
                "2020-01-01 07:55:00",
                "2020-01-01 08:00:00",
                "2020-01-01 09:00:00",
            ],
            "endtime": [
                "2020-01-02 08:00:00",
                "2020-01-01 07:55:00",
                "2020-01-02 08:00:00",
                "2020-01-01 09:00:00",
            ],
            "itemid": [225792, 224385, 225792, 999999],
        }
    ).to_csv(tmp_path / "procedureevents.csv", index=False)
    cohort = pd.DataFrame(
        {
            "dataset": ["mimic"],
            "stay_id": [1],
            "admit_time": ["2020-01-01 00:00:00"],
        }
    )
    events, counts = extract_mimic_respiratory_procedure_events(tmp_path, cohort)
    starts = events.loc[
        events["concept"].isin(["mechanical_ventilation", "intubation"])
    ].set_index("concept")
    assert starts.loc["mechanical_ventilation", "offset_minutes"] == 480.0
    assert starts.loc["intubation", "offset_minutes"] == 475.0
    assert starts.loc["mechanical_ventilation", "source_table"] == (
        "procedureevents.csv"
    )
    assert counts == {"mechanical_ventilation_rows": 1, "intubation_rows": 1}
    active = events.loc[
        events["concept"].eq("mechanical_ventilation_active")
    ].set_index("offset_minutes")
    assert active.loc[0.0, "value_numeric"] == 0
    assert active.loc[240.0, "value_numeric"] == 0


def test_eicu_ventilation_proxy_rejects_known_false_text_matches() -> None:
    raw_names = [
        "pulmonary|ventilation and oxygenation|mechanical ventilation",
        "pulmonary|ventilation and oxygenation|mechanical ventilation|assist controlled",
        "pulmonary|ventilation and oxygenation|mechanical ventilation|non-invasive ventilation",
        "pulmonary|ventilation and oxygenation|ventilator weaning",
        "cardiovascular|drug|simvastatin",
        "pulmonary|tracheostomy|performed for ventilatory support",
    ]
    events = pd.DataFrame(
        {
            "dataset": ["eicu"] * len(raw_names),
            "stay_id": range(1, len(raw_names) + 1),
            "concept": ["mechanical_ventilation"] * len(raw_names),
            "source_table": ["treatment.csv"] * len(raw_names),
            "raw_name": raw_names,
            "offset_minutes": [300.0] * len(raw_names),
            "value_numeric": [1.0] * len(raw_names),
        }
    )
    cleaned, audit = _clean_eicu_ventilation_proxy(events)
    assert cleaned["stay_id"].tolist() == [1, 2]
    row = audit.iloc[0]
    assert row["input_documentation_rows"] == 6
    assert row["retained_literal_mechanical_ventilation_rows"] == 2
    assert row["removed_noninvasive_rows"] == 1
    assert row["removed_weaning_rows"] == 1
    assert row["removed_nonliteral_false_matches"] == 2
