"""Feature typing, train/test split, and the shared preprocessing transformer.

All learned preprocessing lives inside a ColumnTransformer that is wrapped in
each model's pipeline, so it is re-fit per training fold during CV (no leakage).
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_loader import TARGET

RANDOM_STATE = 42


def split_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return ``(numeric_cols, categorical_cols)``, excluding the target."""
    features = df.drop(columns=[TARGET])
    numeric = features.select_dtypes(include=["number"]).columns.tolist()
    categorical = features.select_dtypes(exclude=["number"]).columns.tolist()
    return numeric, categorical


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Numeric: median impute + scale. Categorical: mode impute + one-hot encode."""
    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ]
    )


def make_split(df: pd.DataFrame, test_size: float = 0.2):
    """Stratified train/test split preserving the churn rate."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
