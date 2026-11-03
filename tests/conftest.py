import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_frame():
    """Small synthetic frame shaped like diabetic_data.csv."""
    rng = np.random.RandomState(0)
    n = 400
    return pd.DataFrame(
        {
            "encounter_id": np.arange(n),
            "patient_nbr": np.arange(1000, 1000 + n),
            "race": rng.choice(["Caucasian", "AfricanAmerican", "?"], n),
            "gender": rng.choice(["Female", "Male"], n),
            "age": rng.choice(["[50-60)", "[60-70)", "[70-80)"], n),
            "admission_type_id": rng.choice([1, 2, 3], n),
            "discharge_disposition_id": rng.choice([1, 3, 6], n),
            "admission_source_id": rng.choice([1, 7], n),
            "time_in_hospital": rng.randint(1, 14, n),
            "num_lab_procedures": rng.randint(1, 100, n),
            "num_procedures": rng.randint(0, 6, n),
            "num_medications": rng.randint(1, 60, n),
            "number_outpatient": rng.poisson(0.3, n),
            "number_emergency": rng.poisson(0.2, n),
            "number_inpatient": rng.poisson(0.6, n),
            "diag_1": rng.choice(["250.01", "428", "486", "V45"], n),
            "diag_2": rng.choice(["250", "401", "?"], n),
            "diag_3": rng.choice(["599", "715", "820"], n),
            "number_diagnoses": rng.randint(1, 16, n),
            "max_glu_serum": rng.choice(np.array([None, ">200", "Norm"], dtype=object), n),
            "A1Cresult": rng.choice(np.array([None, ">8", "Norm"], dtype=object), n),
            "insulin": rng.choice(["No", "Steady", "Up", "Down"], n),
            "change": rng.choice(["No", "Ch"], n),
            "diabetesMed": rng.choice(["No", "Yes"], n),
            "readmitted": rng.choice(["<30", ">30", "NO"], n, p=[0.15, 0.35, 0.5]),
        }
    )
