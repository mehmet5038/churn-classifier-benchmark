"""Churn classifier benchmark package.

Reusable building blocks for the Telco Customer Churn classification study:

- :mod:`src.data_loader`   -- download, load and basic cleaning of the raw dataset.
- :mod:`src.preprocessing` -- feature typing, train/test split, and the sklearn
  ColumnTransformer used by every model pipeline.
- :mod:`src.models`        -- the model zoo and hyperparameter grids.
- :mod:`src.evaluate`      -- metric tables and plotting helpers for the report.
"""

__all__ = ["data_loader", "preprocessing", "models", "evaluate"]
