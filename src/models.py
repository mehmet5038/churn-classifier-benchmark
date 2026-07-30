"""The five-model zoo and hyperparameter grids for the churn comparison.

Grid keys are prefixed ``clf__`` because the estimator sits under the ``clf`` step
of the pipeline built in :func:`build_pipeline`. Grids are kept compact for a fast
CPU run.
"""

from __future__ import annotations

from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42


def get_model_zoo() -> dict[str, dict]:
    """Return ``{name: {"estimator": ..., "param_grid": {...}}}`` for all 5 models."""
    return {
        "Decision Tree": {
            "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "clf__max_depth": [4, 6, 8, None],
                "clf__min_samples_leaf": [1, 20, 50],
            },
        },
        "Naive Bayes": {
            "estimator": GaussianNB(),
            "param_grid": {
                "clf__var_smoothing": [1e-9, 1e-7, 1e-5],
            },
        },
        "k-NN": {
            "estimator": KNeighborsClassifier(),
            "param_grid": {
                "clf__n_neighbors": [11, 21, 31, 51],
                "clf__weights": ["uniform", "distance"],
            },
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            "param_grid": {
                "clf__n_estimators": [300],
                "clf__max_depth": [8, 12, None],
                "clf__min_samples_leaf": [1, 5, 20],
            },
        },
        "XGBoost": {
            "estimator": XGBClassifier(
                random_state=RANDOM_STATE,
                n_estimators=400,
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
            ),
            "param_grid": {
                "clf__max_depth": [3, 5, 7],
                "clf__learning_rate": [0.05, 0.1],
                "clf__subsample": [0.8, 1.0],
            },
        },
    }


def build_pipeline(preprocessor: ColumnTransformer, estimator) -> Pipeline:
    """Wrap preprocessor + estimator into one leakage-safe pipeline."""
    return Pipeline(steps=[("prep", preprocessor), ("clf", estimator)])


def build_imbalanced_pipeline(preprocessor: ColumnTransformer, estimator, sampler):
    """Pipeline with a resampler (e.g. SMOTE) applied to training folds only."""
    return ImbPipeline(
        steps=[("prep", preprocessor), ("sample", sampler), ("clf", estimator)]
    )
