"""Paths, constants and hyperparameters for SHAFLB."""

from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_FILE = RAW_DATA_DIR / "diabetic_data.csv"
IDS_MAPPING_FILE = RAW_DATA_DIR / "IDS_mapping.csv"
DATASET_URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_FILE = ARTIFACTS_DIR / "results.json"
LEDGER_FILE = ARTIFACTS_DIR / "audit_chain.json"
MODEL_BUNDLE_FILE = ARTIFACTS_DIR / "model_bundle.joblib"
ARRAYS_FILE = ARTIFACTS_DIR / "arrays.npz"

RANDOM_SEED = 42
TEST_SIZE = 0.20

# Cohort definition
EXCLUDED_DISCHARGE_IDS = [11, 13, 14, 19, 20, 21]  # expired / hospice: readmission impossible
TARGET_COLUMN = "readmitted"
POSITIVE_TARGET_LABEL = "<30"

NUMERICAL_COLS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

CATEGORICAL_COLS = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "max_glu_serum",
    "A1Cresult",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "glipizide",
    "glyburide",
    "pioglitazone",
    "rosiglitazone",
    "insulin",
    "change",
    "diabetesMed",
    "diag_1_group",
    "diag_2_group",
    "diag_3_group",
]

# Simulated hospital network. Shares control how positive / negative training
# samples are split, which yields non-IID cohort sizes and readmission rates.
HOSPITALS = {
    "Hospital_A": {"profile": "Urban academic medical center", "pos_share": 0.38, "neg_share": 0.30},
    "Hospital_B": {"profile": "Suburban community hospital", "pos_share": 0.24, "neg_share": 0.27},
    "Hospital_C": {"profile": "Rural regional clinic", "pos_share": 0.08, "neg_share": 0.16},
    "Hospital_D": {"profile": "Specialist diabetes institute", "pos_share": 0.22, "neg_share": 0.16},
    "Hospital_E": {"profile": "Veterans medical center", "pos_share": 0.08, "neg_share": 0.11},
}
HOSPITAL_IDS = list(HOSPITALS)


@dataclass
class FLConfig:
    """Federated training settings. ``algorithm`` is "fedavg" or "fedprox"."""

    rounds: int = 8
    local_epochs: int = 3
    learning_rate: float = 0.05
    batch_size: int = 64
    l2_reg: float = 1e-4
    algorithm: str = "fedavg"
    prox_mu: float = 0.0
    clip_norm: float = 0.0          # 0 disables update clipping
    noise_multiplier: float = 0.0   # Gaussian noise (x clip_norm) added to the aggregate
    seed: int = RANDOM_SEED

    @property
    def label(self) -> str:
        name = "FedProx" if self.algorithm == "fedprox" else "FedAvg"
        return f"DP-{name}" if self.noise_multiplier > 0 else name

    def to_dict(self) -> dict:
        return asdict(self)


# Variants trained by the pipeline for the algorithm comparison.
FL_VARIANTS = {
    "fedavg": FLConfig(),
    "fedprox": FLConfig(algorithm="fedprox", prox_mu=0.05),
    "dp_fedavg": FLConfig(clip_norm=1.0, noise_multiplier=0.1),
}
