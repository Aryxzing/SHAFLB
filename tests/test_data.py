import numpy as np
import pandas as pd

from src.data import (
    apply_cohort_selection,
    categorize_icd9,
    dataset_statistics,
    load_id_labels,
    prepare_train_test_data,
)
from src.models import compute_metrics


def test_categorize_icd9():
    cases = {
        "250.01": "Diabetes", "250": "Diabetes", "410.1": "Circulatory", "459.9": "Circulatory",
        "785.0": "Circulatory", "486": "Respiratory", "540.9": "Digestive", "599.0": "Genitourinary",
        "174.9": "Neoplasms", "715.9": "Musculoskeletal", "820.8": "Injury", "V45.81": "Other",
        "E878": "Other", "?": "Missing",
    }
    for code, group in cases.items():
        assert categorize_icd9(code) == group, code
    assert categorize_icd9(np.nan) == "Missing"


def test_cohort_excludes_hospice_and_keeps_first_encounter():
    df = pd.DataFrame(
        {
            "encounter_id": [2, 1, 3, 4, 5],
            "patient_nbr": [101, 101, 102, 103, 104],
            "discharge_disposition_id": [1, 1, 11, 1, 1],
            "gender": ["Female", "Female", "Male", "Unknown/Invalid", "Female"],
            "diag_1": ["250.01", "410", "486", "540", "174"],
            "readmitted": ["NO", "<30", "NO", "NO", ">30"],
        }
    )
    out = apply_cohort_selection(df)
    # 102 is hospice, 103 has invalid gender, 101 keeps encounter 1 (the earliest)
    assert len(out) == 2
    assert out["target"].tolist() == [1, 0]
    assert out["diag_1_group"].tolist() == ["Circulatory", "Neoplasms"]


def test_missing_lab_results_become_none(raw_frame):
    out = apply_cohort_selection(raw_frame)
    assert "nan" not in set(out["max_glu_serum"])
    assert "None" in set(out["A1Cresult"])
    assert "?" not in set(out["race"])


def test_statistics():
    stats = dataset_statistics(pd.DataFrame({"a": [1, 2, 3, 4], "target": [1, 0, 0, 0]}))
    assert stats == {
        "total_records": 4, "positive_samples": 1, "negative_samples": 3,
        "readmission_rate_percent": 25.0, "total_features": 1,
    }


def test_preprocessing_shapes_and_no_nans(raw_frame):
    cohort = apply_cohort_selection(raw_frame)
    X_train, X_test, y_train, y_test, pre = prepare_train_test_data(cohort, test_size=0.25)
    assert X_train.shape[0] + X_test.shape[0] == len(cohort)
    assert X_train.shape[1] == X_test.shape[1] == len(pre.get_feature_names_out())
    assert np.isfinite(X_train).all() and np.isfinite(X_test).all()
    # scaler statistics come from the training split only
    assert np.allclose(X_train[:, 0].mean(), 0, atol=1e-9)


def test_id_labels_parse(tmp_path):
    p = tmp_path / "ids.csv"
    p.write_text('admission_type_id,description\n1,Emergency\n2,Urgent\n,\n'
                 'discharge_disposition_id,description\n1,Discharged to home\n19,"Expired, hospice"\n')
    labels = load_id_labels(p)
    assert labels["admission_type_id"] == {"1": "Emergency", "2": "Urgent"}
    assert labels["discharge_disposition_id"]["19"] == "Expired, hospice"


def test_metrics_perfect_and_threshold():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.6, 0.7, 0.9])
    m = compute_metrics(y, p)
    assert m["roc_auc"] == 1.0 and m["recall"] == 1.0
    assert m["confusion_matrix"] == {"tn": 2, "fp": 1, "fn": 0, "tp": 2}
    assert compute_metrics(y, p, threshold=0.65)["precision"] == 1.0
