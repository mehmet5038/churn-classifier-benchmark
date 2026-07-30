"""Download, load, and apply basic cleaning to the Telco Customer Churn dataset.

Cleaning is intentionally minimal here (drop id, fix TotalCharges dtype, map the
target); imputation/encoding/scaling are left to ``preprocessing.py`` so they run
inside cross-validation and cannot leak test information.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = REPO_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"

TARGET = "Churn"
ID_COL = "customerID"


def download_data(dest: Path = RAW_PATH, force: bool = False) -> Path:
    """Download the raw CSV to ``dest`` if it is not already present."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if force or not dest.exists():
        print(f"Downloading dataset -> {dest}")
        urlretrieve(DATA_URL, dest)
    return dest


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw CSV, downloading it first if missing."""
    path = Path(path)
    if not path.exists():
        download_data(path)
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy: drop ``customerID``, fix ``TotalCharges``, map target."""
    df = df.copy()

    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    if "TotalCharges" in df.columns:
        # Blank strings for tenure-0 customers become NaN (imputed downstream).
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    if TARGET in df.columns and not pd.api.types.is_numeric_dtype(df[TARGET]):
        df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1}).astype("int64")

    return df


def load_clean(path: Path = RAW_PATH) -> pd.DataFrame:
    """Download (if needed) -> load -> clean."""
    return clean(load_raw(path))


if __name__ == "__main__":
    frame = load_clean()
    print("Shape:", frame.shape)
    print("Churn rate: {:.3f}".format(frame[TARGET].mean()))
    print("Missing TotalCharges:", int(frame["TotalCharges"].isna().sum()))
