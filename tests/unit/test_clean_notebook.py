import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[2] / "PhysioGraph_Final_Clean.ipynb"


def test_clean_notebook_is_valid_json_and_all_code_cells_compile():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    for cell in payload["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        compile(source, f"{NOTEBOOK.name}:{cell.get('id', 'unknown')}", "exec")


def test_clean_notebook_defaults_to_analysis_only_resume():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "ANALYSIS_ONLY = True" in source
    assert "BUILD_NEW = False" in source
    assert "analysis_only=ANALYSIS_ONLY" in source
    assert "spo2_lactate_episode_controlled_summary" in source
    assert "spo2_lactate_episode_measurement_weighted" in source
    assert "spo2_multiorgan_episode_effects" in source
    assert "spo2_multiorgan_episode_measurement_weighted" in source
    assert "spo2_multiorgan_episode_key_results" in source
