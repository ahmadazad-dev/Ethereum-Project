"""Leak-free cross-validation, threshold selection, statistical tests, and registry logging."""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

REGISTRY_COLUMNS: tuple[str, ...] = (
    "run_id",
    "timestamp_utc",
    "experiment_id",
    "git_commit",
    "config_path",
    "config_hash",
    "data_version",
    "data_hash",
    "split_type",
    "n_splits",
    "seeds",
    "model",
    "feature_group_config",
    "n_features",
    "threshold_mode",
    "threshold_value",
    "f1_mean",
    "f1_sd",
    "precision_mean",
    "precision_sd",
    "recall_mean",
    "recall_sd",
    "pr_auc_mean",
    "pr_auc_sd",
    "roc_auc_mean",
    "roc_auc_sd",
    "runtime_sec",
    "library_versions",
    "status",
    "supersedes_run_id",
    "notes",
)


def make_leak_free_pipeline(estimator: Any, imputer_strategy: str = "median") -> Pipeline:
    """Wrap an estimator with an in-fold imputer inside a scikit-learn Pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=imputer_strategy)),
            ("model", clone(estimator)),
        ]
    )


def generate_cv_splits(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    seeds: Sequence[int],
    n_splits: int = 5,
    groups: Sequence[Any] | np.ndarray | None = None,
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """Pre-compute identical (seed, fold_idx, train_idx, val_idx) splits for paired ablations."""
    if len(seeds) == 0:
        raise ValueError("seeds must be non-empty.")

    splits: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for seed in seeds:
        if groups is not None:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            fold_iter = splitter.split(X, y, groups)
        else:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            fold_iter = splitter.split(X, y)

        for fold_idx, (train_idx, val_idx) in enumerate(fold_iter):
            splits.append((int(seed), int(fold_idx), train_idx, val_idx))

    return splits


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.50,
) -> dict[str, float]:
    """Compute fraud-class F1 (pos_label=1) and mandatory companion metrics."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob_arr >= threshold).astype(int)

    roc_auc = float("nan")
    pr_auc = float("nan")
    if len(np.unique(y_true_arr)) == 2:
        roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
        pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))

    return {
        "f1": float(f1_score(y_true_arr, y_pred, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y_true_arr, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, pos_label=1, zero_division=0)),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    }


def select_optimal_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Select probability threshold maximizing binary fraud-class F1 on training-fold OOF predictions."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    candidates = np.unique(np.clip(y_prob_arr, 0.0, 1.0))
    if len(candidates) == 0:
        return 0.50

    best_thr = 0.50
    best_f1 = -1.0
    for thr in candidates:
        score = float(f1_score(y_true_arr, (y_prob_arr >= thr).astype(int), pos_label=1, zero_division=0))
        if score > best_f1 or (math.isclose(score, best_f1) and abs(thr - 0.50) < abs(best_thr - 0.50)):
            best_f1 = score
            best_thr = float(thr)
    return best_thr


def evaluate_cv_paired(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    feature_cols: Sequence[str],
    splits: Sequence[tuple[int, int, np.ndarray, np.ndarray]],
    pipeline_factory: Callable[[int, np.ndarray], Pipeline],
    *,
    inner_val_splits: int = 3,
    groups: Sequence[Any] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Run paired leak-free CV over precomputed splits with 0.50 and train-fold-tuned thresholds.

    All preprocessing and threshold selection happen strictly inside each training fold.
    """
    if len(feature_cols) == 0:
        raise ValueError("feature_cols must not be empty.")
    missing_cols = [c for c in feature_cols if c not in X.columns]
    if missing_cols:
        raise ValueError(f"Feature columns missing from DataFrame: {missing_cols}")

    y_arr = np.asarray(y, dtype=int)
    groups_arr = np.asarray(groups) if groups is not None else None

    rows_fixed: list[dict[str, float]] = []
    rows_tuned: list[dict[str, float]] = []
    tuned_thresholds: list[float] = []

    for seed, fold_idx, train_idx, val_idx in splits:
        X_train = X.iloc[train_idx][list(feature_cols)]
        y_train = y_arr[train_idx]
        X_val = X.iloc[val_idx][list(feature_cols)]
        y_val = y_arr[val_idx]

        # Inner CV strictly within X_train to choose the F1 threshold without touching X_val.
        inner_probs = np.zeros(len(train_idx), dtype=float)
        if groups_arr is not None:
            inner_cv = StratifiedGroupKFold(
                n_splits=inner_val_splits, shuffle=True, random_state=seed + fold_idx
            )
            inner_iter = inner_cv.split(X_train, y_train, groups_arr[train_idx])
        else:
            inner_cv = StratifiedKFold(
                n_splits=inner_val_splits, shuffle=True, random_state=seed + fold_idx
            )
            inner_iter = inner_cv.split(X_train, y_train)

        for in_tr, in_va in inner_iter:
            inner_pipe = pipeline_factory(seed, y_train[in_tr])
            inner_pipe.fit(X_train.iloc[in_tr], y_train[in_tr])
            inner_probs[in_va] = inner_pipe.predict_proba(X_train.iloc[in_va])[:, 1]

        opt_thr = select_optimal_f1_threshold(y_train, inner_probs)
        tuned_thresholds.append(opt_thr)

        outer_pipe = pipeline_factory(seed, y_train)
        outer_pipe.fit(X_train, y_train)
        val_probs = outer_pipe.predict_proba(X_val)[:, 1]

        fixed_m = compute_binary_metrics(y_val, val_probs, threshold=0.50)
        tuned_m = compute_binary_metrics(y_val, val_probs, threshold=opt_thr)
        fixed_m["seed"] = float(seed)
        fixed_m["fold"] = float(fold_idx)
        tuned_m["seed"] = float(seed)
        tuned_m["fold"] = float(fold_idx)
        tuned_m["threshold"] = float(opt_thr)

        rows_fixed.append(fixed_m)
        rows_tuned.append(tuned_m)

    df_fixed = pd.DataFrame(rows_fixed)
    df_tuned = pd.DataFrame(rows_tuned)

    def _summarize(df: pd.DataFrame) -> dict[str, float]:
        out: dict[str, float] = {}
        for metric in ("f1", "precision", "recall", "pr_auc", "roc_auc"):
            vals = df[metric].to_numpy(dtype=float)
            out[f"{metric}_mean"] = float(np.nanmean(vals))
            out[f"{metric}_sd"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        return out

    return {
        "fixed_050": _summarize(df_fixed),
        "train_tuned": _summarize(df_tuned),
        "tuned_threshold_mean": float(np.mean(tuned_thresholds)),
        "tuned_threshold_sd": float(np.std(tuned_thresholds, ddof=1))
        if len(tuned_thresholds) > 1
        else 0.0,
        "per_fold_fixed": df_fixed,
        "per_fold_tuned": df_tuned,
    }


def nadeau_bengio_corrected_ttest(
    diffs: Sequence[float],
    *,
    n_train: int,
    n_test: int,
) -> dict[str, float]:
    """Corrected resampled t-test (Nadeau & Bengio, 2003) for overlapping CV folds.

    Corrects variance inflation from shared training data across CV splits via
    factor (1 / k + n_test / n_train).
    """
    arr = np.asarray(diffs, dtype=float)
    k = len(arr)
    if k < 2:
        raise ValueError("Need at least 2 paired fold differences for t-test.")
    if n_train <= 0 or n_test <= 0:
        raise ValueError("n_train and n_test must be positive.")

    mean_diff = float(np.mean(arr))
    sample_var = float(np.var(arr, ddof=1))
    corrected_se = math.sqrt((1.0 / k + n_test / n_train) * sample_var)

    if corrected_se == 0.0:
        t_stat = 0.0
        p_val = 1.0
    else:
        t_stat = mean_diff / corrected_se
        p_val = float(2.0 * stats.t.sf(abs(t_stat), df=k - 1))

    return {
        "mean_diff": mean_diff,
        "corrected_se": corrected_se,
        "t_stat": t_stat,
        "p_value": p_val,
        "df": float(k - 1),
    }


def append_registry_row(registry_path: str | Path, row: Mapping[str, Any]) -> None:
    """Append a single validated row to results/registry.csv without overwriting."""
    path = Path(registry_path)
    if not path.is_file():
        raise FileNotFoundError(f"Registry file does not exist: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)

    if header is None or tuple(header) != REGISTRY_COLUMNS:
        raise ValueError(
            f"Registry header mismatch in {path}. Expected {REGISTRY_COLUMNS}, got {header}"
        )

    missing_keys = [col for col in REGISTRY_COLUMNS if col not in row]
    extra_keys = [k for k in row if k not in REGISTRY_COLUMNS]
    if missing_keys or extra_keys:
        raise ValueError(
            f"Row schema mismatch. Missing={missing_keys}, unexpected={extra_keys}"
        )

    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(REGISTRY_COLUMNS))
        writer.writerow({col: row[col] for col in REGISTRY_COLUMNS})

