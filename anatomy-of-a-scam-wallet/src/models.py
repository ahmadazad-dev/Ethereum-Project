#!/usr/bin/env python3
"""
Phase 4 — Model Builders & Feature-Subset Pipeline Wrapper
Project: Anatomy of a Scam Wallet (Ethereum Fraud Detection with Random Forest & XGBoost)
"""

from __future__ import annotations

from typing import Any, Dict, List

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import SEED


def build_rf(**params: Any) -> RandomForestClassifier:
    """
    Build a Random Forest classifier.
    Objective: Constructs an ensemble of decorrelated CART trees via bootstrap
    aggregation (bagging) and random feature subspace selection (`max_features`),
    minimizing Gini impurity at each split and aggregating predictions by majority
    vote (averaged class probabilities).
    """
    cfg: Dict[str, Any] = {"random_state": SEED, "n_jobs": -1}
    cfg.update(params)
    return RandomForestClassifier(**cfg)


def build_xgb(**params: Any) -> XGBClassifier:
    """
    Build an eXtreme Gradient Boosting (XGBoost) classifier.
    Objective: Sequentially fits additive regression trees using second-order
    gradient and Hessian approximations of the regularized binary cross-entropy
    (log-loss) objective, with explicit L1/L2 leaf-weight regularization (`reg_alpha`,
    `reg_lambda`) and cost-sensitive positive-class weighting (`scale_pos_weight`).
    """
    cfg: Dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": SEED,
        "n_jobs": -1,
    }
    cfg.update(params)
    return XGBClassifier(**cfg)


def make_pipeline(
    model_name: str,
    params: Dict[str, Any],
    features: List[str],
) -> Pipeline:
    """Wrap an estimator in a Pipeline with a ColumnTransformer projecting `features`."""
    if not features:
        raise ValueError("`features` must be a non-empty list of column names.")

    selector = ColumnTransformer(
        transformers=[("select", "passthrough", list(features))],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    name_lower = model_name.strip().lower()
    if name_lower == "rf":
        estimator = build_rf(**params)
    elif name_lower == "xgb":
        estimator = build_xgb(**params)
    else:
        raise ValueError(f"Unsupported model_name `{model_name}`. Choose 'rf' or 'xgb'.")

    return Pipeline(steps=[("features", selector), ("clf", estimator)])