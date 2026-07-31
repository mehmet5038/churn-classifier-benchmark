# churn-classifier-benchmark

A comparative data-mining study on the Telco Customer Churn dataset. It trains and
tunes five classifiers (Decision Tree, Gaussian Naïve Bayes, k-Nearest Neighbors,
Random Forest, and XGBoost) and compares how well they predict whether a telecom
customer will churn. It also looks at how the choice of imbalance handling (SMOTE,
class weighting, or nothing) shifts the precision/recall trade-off.

## Course

BIL 476 (Data Mining), Summer 2026, TOBB University of Economics and Technology.
Individual term project. Topic: classification (binary). Written in Python.

## Dataset

| | |
|---|---|
| Source | IBM Telco Customer Churn (public mirror) |
| Records | 7,043 |
| Attributes | 20 features + 1 target |
| Types | Mixed: nominal, numeric |
| Target | `Churn` (Yes/No → 1/0), ~26.5% positive (imbalanced) |

The raw CSV is committed at `data/raw/Telco-Customer-Churn.csv`. If missing, it
is downloaded automatically from the mirror defined in `src/data_loader.py`.

## Project structure

```
churn-classifier-benchmark/
├── data/
│   ├── raw/                 # original dataset (committed)
│   └── processed/           # regenerated intermediates (gitignored)
├── src/
│   ├── data_loader.py       # download + load + basic cleaning
│   ├── preprocessing.py     # feature typing, split, ColumnTransformer
│   ├── models.py            # 5-model zoo + hyperparameter grids
│   └── evaluate.py          # metrics tables + report figures
├── notebooks/
│   └── churn_analysis.ipynb # the full narrative: EDA → modeling → evaluation
├── figures/                 # exported plots used in the report
├── results/                 # exported metric tables (CSV)
├── environment.yml
├── requirements.txt
└── README.md
```

## Setup

Using conda (recommended):

```bash
conda env create -f environment.yml
conda activate churn
python -m ipykernel install --user --name churn --display-name "Python (churn)"
```

Or with pip in an existing Python 3.11 environment:

```bash
pip install -r requirements.txt
```

## Reproduce the results

1. (Optional) fetch the dataset explicitly; otherwise the notebook does it for you:
   ```bash
   python -m src.data_loader
   ```
   Expected output: `Shape: (7043, 21)`.
2. Launch Jupyter and run the analysis notebook top to bottom:
   ```bash
   jupyter lab notebooks/churn_analysis.ipynb
   ```
   Use **Kernel → Restart & Run All**. This regenerates every figure in
   `figures/` and the metric tables in `results/`.

All randomness is seeded with `random_state = 42` for reproducibility.

## Methods overview

- **Preprocessing** (leakage-safe, inside cross-validation): median imputation +
  standardization for numeric features; most-frequent imputation + one-hot
  encoding for categoricals.
- **Models**: Decision Tree, Naïve Bayes, k-NN, Random Forest, and XGBoost, each
  tuned with `GridSearchCV` over 5-fold stratified CV and scored on ROC-AUC.
- **Evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC; confusion matrices;
  combined ROC and precision-recall curves; feature-importance and SHAP analysis.
- **Stability and significance**: 5-fold CV ROC-AUC mean and std per model, plus a
  McNemar test between the two top models.
- **Imbalance study**: baseline vs. SMOTE vs. class-weighting on the best model.

## License / attribution

Dataset © IBM, distributed publicly for educational use. Code written for the
course project. See the report for full references and the AI-assistance
declaration.
