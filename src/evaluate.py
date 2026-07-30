"""Evaluation metrics and report-ready plotting helpers.

Figures are saved to ``figures/`` and metric tables to ``results/`` for the report.
Functions that draw also return the underlying numbers for inline display.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "figures"
RESULTS_DIR = REPO_ROOT / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    """Standard classification metrics; ``y_proba`` is the positive-class probability."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
    }


def metrics_table(results: dict[str, dict[str, float]], save_as: str | None = "metrics.csv") -> pd.DataFrame:
    """Turn ``{model: metrics}`` into a ROC-AUC-sorted DataFrame and save it."""
    table = pd.DataFrame(results).T.sort_values("ROC-AUC", ascending=False).round(4)
    if save_as:
        table.to_csv(RESULTS_DIR / save_as)
    return table


def mcnemar_test(y_true, pred_a, pred_b) -> dict[str, float]:
    """Exact McNemar test comparing two classifiers' predictions on the same set.

    Counts the discordant pairs (one model right where the other is wrong) and
    returns an exact p-value from the binomial test on those pairs. A small
    p-value means the two models' error patterns differ significantly.
    """
    y_true = np.asarray(y_true)
    a_correct = np.asarray(pred_a) == y_true
    b_correct = np.asarray(pred_b) == y_true
    b = int(np.sum(a_correct & ~b_correct))  # A right, B wrong
    c = int(np.sum(~a_correct & b_correct))  # A wrong, B right
    n = b + c
    p_value = binomtest(min(b, c), n, 0.5).pvalue if n > 0 else 1.0
    chi2_cc = (abs(b - c) - 1) ** 2 / n if n > 0 else 0.0
    return {"b (A_right_B_wrong)": b, "c (A_wrong_B_right)": c,
            "chi2_cc": chi2_cc, "p_value": p_value}


def plot_confusion_matrix(y_true, y_pred, model_name: str, save: bool = True):
    """Plot a labeled confusion matrix for one model."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No churn", "Churn"],
        yticklabels=["No churn", "Churn"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    if save:
        slug = model_name.lower().replace(" ", "_").replace("-", "")
        fig.savefig(FIG_DIR / f"confusion_{slug}.png", dpi=150)
    return fig


def plot_roc_curves(fitted_models: dict, X_test, y_test, save: bool = True):
    """Overlay ROC curves for every fitted model."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in fitted_models.items():
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(FIG_DIR / "roc_curves.png", dpi=150)
    return fig


def plot_pr_curves(fitted_models: dict, X_test, y_test, save: bool = True):
    """Overlay precision-recall curves (more informative under imbalance)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    baseline = float(np.mean(y_test))
    for name, model in fitted_models.items():
        proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, proba)
        ax.plot(recall, precision, label=f"{name} (AP-AUC={auc(recall, precision):.3f})")
    ax.axhline(baseline, ls="--", color="k", alpha=0.5, label=f"Baseline ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Model Comparison")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(FIG_DIR / "pr_curves.png", dpi=150)
    return fig


def get_feature_names(preprocessor) -> np.ndarray:
    """Recover output feature names from a fitted ColumnTransformer."""
    return preprocessor.get_feature_names_out()


def plot_feature_importance(fitted_pipeline, model_name: str, top_n: int = 15, save: bool = True):
    """Bar chart of the top-N feature importances for a tree/ensemble pipeline."""
    prep = fitted_pipeline.named_steps["prep"]
    clf = fitted_pipeline.named_steps["clf"]
    names = get_feature_names(prep)
    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=importances[order], y=names[order], ax=ax, color="#3b7dd8")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}")
    fig.tight_layout()
    if save:
        slug = model_name.lower().replace(" ", "_").replace("-", "")
        fig.savefig(FIG_DIR / f"importance_{slug}.png", dpi=150)
    return fig, pd.Series(importances[order], index=names[order])
