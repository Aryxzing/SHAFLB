"""Dataset download, cohort selection, ICD-9 grouping and feature preprocessing."""

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_COLS,
    DATASET_URL,
    EXCLUDED_DISCHARGE_IDS,
    IDS_MAPPING_FILE,
    NUMERICAL_COLS,
    POSITIVE_TARGET_LABEL,
    RANDOM_SEED,
    RAW_DATA_DIR,
    RAW_DATA_FILE,
    TARGET_COLUMN,
    TEST_SIZE,
)

ICD9_GROUPS = [
    "Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury",
    "Musculoskeletal", "Genitourinary", "Neoplasms", "Other", "Missing",
]


def download_dataset(dest_dir: Path = RAW_DATA_DIR) -> Path:
    """Fetch the UCI archive and extract diabetic_data.csv and IDS_mapping.csv."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATASET_URL, timeout=60) as resp:
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))
    for member in archive.namelist():
        name = Path(member).name
        if name in ("diabetic_data.csv", "IDS_mapping.csv"):
            (dest_dir / name).write_bytes(archive.read(member))
    return dest_dir / "diabetic_data.csv"


def load_raw_dataset(path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python run_experiment.py --download` to fetch it."
        )
    return pd.read_csv(path, low_memory=False)


def load_id_labels(path: Path = IDS_MAPPING_FILE) -> dict:
    """Parse IDS_mapping.csv (three stacked tables) into {column: {id: description}}."""
    labels, current = {}, None
    if not Path(path).exists():
        return labels
    for line in Path(path).read_text().splitlines():
        key, _, desc = line.partition(",")
        if not key.strip():
            current = None
        elif current is None:
            current = key.strip()
            labels[current] = {}
        else:
            labels[current][key.strip()] = desc.strip().strip('"')
    return labels


def categorize_icd9(code) -> str:
    """Map an ICD-9 code to one of nine clinical groups (Strack et al., 2014)."""
    if pd.isna(code) or str(code).strip() in ("", "?"):
        return "Missing"
    code = str(code).strip()
    if code[0] in "VE":
        return "Other"
    try:
        num = float(code)
    except ValueError:
        return "Other"
    if int(num) == 250:
        return "Diabetes"
    if 390 <= num < 460 or int(num) == 785:
        return "Circulatory"
    if 460 <= num < 520 or int(num) == 786:
        return "Respiratory"
    if 520 <= num < 580 or int(num) == 787:
        return "Digestive"
    if 580 <= num < 630 or int(num) == 788:
        return "Genitourinary"
    if 140 <= num < 240:
        return "Neoplasms"
    if 710 <= num < 740:
        return "Musculoskeletal"
    if 800 <= num < 1000:
        return "Injury"
    return "Other"


def apply_cohort_selection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the modelling cohort:
      1. drop expired / hospice discharges (readmission is impossible),
      2. keep each patient's first encounter (independent samples, no cross-split leakage),
      3. group ICD-9 diagnoses, 4. binarise the target (<30 days = 1),
      5. keep only modelling columns, with categoricals as strings.
    """
    df = df[~df["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_IDS)]
    if "encounter_id" in df.columns:
        df = df.sort_values("encounter_id")
    if "patient_nbr" in df.columns:
        df = df.drop_duplicates(subset="patient_nbr", keep="first")
    if "gender" in df.columns:
        df = df[df["gender"].isin(["Female", "Male"])]

    out = pd.DataFrame(index=df.index)
    for col in NUMERICAL_COLS:
        if col in df.columns:
            out[col] = df[col]
    for col in CATEGORICAL_COLS:
        if col.endswith("_group"):
            src = col.removesuffix("_group")
            if src in df.columns:
                out[col] = df[src].map(categorize_icd9)
        elif col in df.columns:
            # pandas parses the literal "None" (test not performed) as NaN; restore it.
            out[col] = df[col].fillna("None").astype(str).replace("?", "Missing")
    out["target"] = (df[TARGET_COLUMN] == POSITIVE_TARGET_LABEL).astype(int)
    return out.reset_index(drop=True)


def dataset_statistics(df: pd.DataFrame) -> dict:
    n, pos = len(df), int(df["target"].sum())
    return {
        "total_records": n,
        "positive_samples": pos,
        "negative_samples": n - pos,
        "readmission_rate_percent": round(100 * pos / n, 2) if n else 0.0,
        "total_features": df.shape[1] - 1,
    }


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Standardise numeric columns and one-hot encode categoricals (unknown levels ignored)."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), [c for c in NUMERICAL_COLS if c in df.columns]),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                [c for c in CATEGORICAL_COLS if c in df.columns],
            ),
        ],
        verbose_feature_names_out=False,
    )


def prepare_train_test_data(df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = RANDOM_SEED):
    """Stratified split, then fit the preprocessor on the training split only (no leakage)."""
    X, y = df.drop(columns="target"), df["target"].to_numpy(dtype=int)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    preprocessor = build_preprocessor(X_train_df)
    X_train = preprocessor.fit_transform(X_train_df).astype(np.float64)
    X_test = preprocessor.transform(X_test_df).astype(np.float64)
    return X_train, X_test, y_train, y_test, preprocessor
