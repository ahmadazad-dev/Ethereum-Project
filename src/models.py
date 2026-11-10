"""Model factories and mandatory baseline estimators (AGENTS.md §5)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier


def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """Compute negative/positive ratio strictly on a training fold."""
    y_arr = np.asarray(y_train)
    n_pos = int(np.sum(y_arr == 1))
    n_neg = int(np.sum(y_arr == 0))
    if n_pos == 0:
        raise ValueError("Cannot compute scale_pos_weight with zero positive samples.")
    if n_neg + n_pos != len(y_arr):
        raise ValueError("Target array must be binary {0, 1}.")
    return float(n_neg / n_pos)


def build_majority_baseline() -> DummyClassifier:
    """Mandatory majority-class baseline (predicts most frequent training label)."""
    return DummyClassifier(strategy="most_frequent")


class HeuristicRuleBaseline(BaseEstimator, ClassifierMixin):
    """Simple interpretable rule baseline on two domain columns (AGENTS.md §5).

    Predicts positive fraud (1) if ``primary_col >= primary_threshold`` AND
    ``secondary_col <= secondary_threshold`` (or ``>=`` when configured).
    Thresholds can be passed explicitly or left explicit in config.
    """

    def __init__(
        self,
        primary_col: str,
        primary_threshold: float,
        secondary_col: str | None = None,
        secondary_threshold: float | None = None,
        secondary_op: str = "<=",
    ) -> None:
        self.primary_col = primary_col
        self.primary_threshold = primary_threshold
        self.secondary_col = secondary_col
        self.secondary_threshold = secondary_threshold
        self.secondary_op = secondary_op

    def fit(self, X: Any, y: Any) -> "HeuristicRuleBaseline":
        self.classes_ = np.array([0, 1], dtype=int)
        if hasattr(X, "columns"):
            if self.primary_col not in X.columns:
                raise ValueError(f"Missing heuristic column '{self.primary_col}'")
            if self.secondary_col is not None and self.secondary_col not in X.columns:
                raise ValueError(f"Missing heuristic column '{self.secondary_col}'")
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if not hasattr(X, "columns"):
            raise TypeError("HeuristicRuleBaseline requires a pandas DataFrame input.")
        cond = X[self.primary_col].fillna(0.0).to_numpy() >= self.primary_threshold
        if self.secondary_col is not None:
            if self.secondary_threshold is None:
                raise ValueError("secondary_threshold must be set when secondary_col is provided")
            sec_vals = X[self.secondary_col].fillna(0.0).to_numpy()
            if self.secondary_op == "<=":
                cond = cond & (sec_vals <= self.secondary_threshold)
            elif self.secondary_op == ">=":
                cond = cond & (sec_vals >= self.secondary_threshold)
            else:
                raise ValueError(f"Unsupported secondary_op '{self.secondary_op}'")
        pos = cond.astype(float)
        return np.column_stack([1.0 - pos, pos])

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_random_forest(
    *,
    random_state: int,
    params: Mapping[str, Any] | None = None,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    """Build deterministic RandomForestClassifier with class_weight='balanced' default."""
    cfg: dict[str, Any] = {
        "n_estimators": 300,
        "class_weight": "balanced",
        "random_state": random_state,
        "n_jobs": n_jobs,
    }
    if params is not None:
        cfg.update(params)
        cfg["random_state"] = random_state
    return RandomForestClassifier(**cfg)


def build_xgboost(
    *,
    random_state: int,
    scale_pos_weight: float = 1.0,
    params: Mapping[str, Any] | None = None,
    n_jobs: int = -1,
) -> Any:
    """Build deterministic XGBClassifier (scikit-learn API)."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for XGBClassifier. Install via `pip install -r requirements.txt`."
        ) from exc

    cfg: dict[str, Any] = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "logloss",
        "random_state": random_state,
        "n_jobs": n_jobs,
    }
    if params is not None:
        cfg.update(params)
        cfg["random_state"] = random_state
        cfg["scale_pos_weight"] = scale_pos_weight
    return XGBClassifier(**cfg)

