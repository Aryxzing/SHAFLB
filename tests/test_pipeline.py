from src import config
from src.pipeline import run_pipeline


def test_pipeline_end_to_end_without_touching_artifacts(raw_frame, tmp_path):
    csv = tmp_path / "diabetic_data.csv"
    raw_frame.to_csv(csv, index=False)
    before = {p: p.stat().st_mtime for p in config.ARTIFACTS_DIR.glob("*")}

    results = run_pipeline(data_path=csv, save=False, progress=lambda *_: None)

    assert {p: p.stat().st_mtime for p in config.ARTIFACTS_DIR.glob("*")} == before
    assert set(results["models"]) == {"centralized_lr", "centralized_rf", *config.FL_VARIANTS}
    assert len(results["hospitals"]) == len(config.HOSPITAL_IDS)
    assert len(results["local_only"]) == len(config.HOSPITAL_IDS)
    assert results["ledger"]["verified"] and results["ledger"]["blocks"] == config.FLConfig().rounds + 1
    for m in results["models"].values():
        assert 0.0 <= m["roc_auc"] <= 1.0
