#!/usr/bin/env python3
"""
Phase 4 — Shared Configuration, Feature-Group Registry, & Hyperparameter Search Spaces
Project: Anatomy of a Scam Wallet (Ethereum Fraud Detection with Random Forest & XGBoost)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

SEED = 42
TARGET = "is_scam"
ID_COL = "wallet"
LEAKAGE_COLS = {"is_scam", "wallet", "src_is_scam", "dst_is_scam", "is_contract"}

ROOT_DIR = Path(".")
FEATURES_DIR = ROOT_DIR / "data" / "features"
TRAIN_SCALED_PATH = FEATURES_DIR / "train_scaled.parquet"
TEST_SCALED_PATH = FEATURES_DIR / "test_scaled.parquet"
FEATURE_GROUPS_PATH = FEATURES_DIR / "feature_groups.json"

TUNING_OUTPUT_DIR = ROOT_DIR / "outputs" / "tuning"
MODELS_DIR = ROOT_DIR / "models"


def _load_feature_groups(path: Path = FEATURE_GROUPS_PATH) -> Dict[str, List[str]]:
    """Load and validate the 28-feature behavioral group dictionary from JSON."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature_groups.json at {path}. Run Phase 3 (03_engineer_features.py) first."
        )
    with open(path, "r", encoding="utf-8") as f:
        groups: Dict[str, List[str]] = json.load(f)

    all_cols: List[str] = []
    for grp_name, cols in groups.items():
        if not isinstance(cols, list) or not cols:
            raise ValueError(f"Feature group `{grp_name}` must be a non-empty list.")
        all_cols.extend(cols)

    assert len(all_cols) == 28, f"Expected 28 features in feature_groups.json, found {len(all_cols)}."
    assert len(set(all_cols)) == len(all_cols), "Duplicate feature names found across groups!"
    overlap_leak = set(all_cols) & LEAKAGE_COLS
    assert not overlap_leak, f"Leakage/ID columns found in FEATURE_GROUPS: {sorted(overlap_leak)}"
    return groups


FEATURE_GROUPS: Dict[str, List[str]] = _load_feature_groups()
ALL_FEATURES: List[str] = [col for cols in FEATURE_GROUPS.values() for col in cols]


def get_features(
    exclude_groups: Optional[List[str]] = None,
    only_groups: Optional[List[str]] = None,
) -> List[str]:
    """Return an ordered list of feature columns for full-model or ablation experiments."""
    if exclude_groups is not None and only_groups is not None:
        raise ValueError("Pass either `exclude_groups` or `only_groups`, not both.")

    valid_groups = set(FEATURE_GROUPS.keys())
    if only_groups is not None:
        unknown = set(only_groups) - valid_groups
        if unknown:
            raise ValueError(f"Unknown group(s) in only_groups: {sorted(unknown)}")
        selected = [g for g in FEATURE_GROUPS if g in only_groups]
    elif exclude_groups is not None:
        unknown = set(exclude_groups) - valid_groups
        if unknown:
            raise ValueError(f"Unknown group(s) in exclude_groups: {sorted(unknown)}")
        selected = [g for g in FEATURE_GROUPS if g not in exclude_groups]
    else:
        selected = list(FEATURE_GROUPS.keys())

    features = [col for grp in selected for col in FEATURE_GROUPS[grp]]
    if not features:
        raise ValueError("Feature selection resulted in an empty feature list.")
    return features


RF_SEARCH_SPACE: Dict[str, List[Any]] = {
    "n_estimators": [100, 300, 500],
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "class_weight": ["balanced", None],
}

# Note on scale_pos_weight:
# Because the training set is balanced (1,008 normal / 1,009 fraud, ratio ~ 0.999),
# `n_neg / n_pos` is ~ 1.0. We therefore include 0.5 and 2.0 alongside 1.0 (and the
# exact train ratio) as an explicit precision-recall sensitivity test for Fraud-Class F1.
XGB_SEARCH_SPACE: Dict[str, List[Any]] = {
    "n_estimators": [100, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 6, 10],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "reg_lambda": [1, 5, 10],
    "scale_pos_weight": [1.0, 0.5, 2.0],
}


def get_xgb_search_space(n_neg: int, n_pos: int) -> Dict[str, List[Any]]:
    """Return XGB_SEARCH_SPACE with the exact train-set `n_neg / n_pos` ratio included."""
    ratio = round(float(n_neg) / max(float(n_pos), 1.0), 4)
    spw_vals = [1.0]
    if abs(ratio - 1.0) > 1e-3:
        spw_vals.append(ratio)
    spw_vals.extend([0.5, 2.0])
    space = dict(XGB_SEARCH_SPACE)
    space["scale_pos_weight"] = spw_vals
    return space