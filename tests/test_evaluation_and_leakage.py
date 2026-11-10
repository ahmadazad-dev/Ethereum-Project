"""Tests for hand-computed metrics, leak-free pipelines, Nadeau-Bengio test, and registry."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    REGISTRY_COLUMNS,
    append_registry_row,
    compute_binary_metrics,
    evaluate_cv_paired,
    generate_cv_splits,
    make_leak_free_pipeline,
    nadeau_bengio_corrected_ttest,
)
from src.models import (
    HeuristicRuleBaseline,
    build_majority_baseline,
    build_random_forest,
    build_xgboost,
    compute_scale_pos_weight,
)


def test_hand_computed_toy_metrics() -> None:
    # 8-row toy example computed by hand:
    # Positives (3): 0.9, 0.6, 0.2 ; Negatives (5): 0.8, 0.4, 0.3, 0.1, 0.05
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.6, 0.2, 0.8, 0.4, 0.3, 0.1, 0.05])

    metrics = compute_binary_metrics(y_true, y_prob, threshold=0.50)
    assert metrics["precision"] == pytest.approx(2.0 / 3.0)
    assert metrics["recall"] == pytest.approx(2.0 / 3.0)
    assert metrics["f1"] == pytest.approx(2.0 / 3.0)
    assert metrics["roc_auc"] == pytest.approx(11.0 / 15.0)
    assert metrics["pr_auc"] == pytest.approx(13.0 / 18.0)
    assert compute_scale_pos_weight(y_true) == pytest.approx(5.0 / 3.0)


def test_imputer_fits_on_train_fold_only_no_leakage() -> None:
    X_train = pd.DataFrame({"x": [1.0, 2.0, 3.0, np.nan]})
    y_train = np.array([0, 0, 1, 1])

    X_val_clean = pd.DataFrame({"x": [2.0, 4.0]})
    X_val_extreme = pd.DataFrame({"x": [1e12, 1e12]})

    pipe_a = make_leak_free_pipeline(build_random_forest(random_state=42, n_jobs=1), "median")
    pipe_b = make_leak_free_pipeline(build_random_forest(random_state=42, n_jobs=1), "median")

    pipe_a.fit(X_train, y_train)
    pipe_b.fit(X_train, y_train)

    # Train median of [1, 2, 3] is 2.0 regardless of what is in validation
    assert pipe_a.named_steps["imputer"].statistics_[0] == pytest.approx(2.0)
    assert pipe_b.named_steps["imputer"].statistics_[0] == pytest.approx(2.0)

    # Predicting on clean vs extreme val leaves train imputer untouched
    _ = pipe_a.predict_proba(X_val_clean)
    _ = pipe_b.predict_proba(X_val_extreme)
    assert pipe_a.named_steps["imputer"].statistics_[0] == pytest.approx(2.0)
    assert pipe_b.named_steps["imputer"].statistics_[0] == pytest.approx(2.0)


def test_nadeau_bengio_corrected_ttest_matches_hand_calc() -> None:
    diffs = [0.02, 0.04, 0.03, 0.05]
    res = nadeau_bengio_corrected_ttest(diffs, n_train=80, n_test=20)
    expected_se = math.sqrt((1.0 / 4.0 + 20.0 / 80.0) * (0.0005 / 3.0))
    expected_t = 0.035 / expected_se
    assert res["mean_diff"] == pytest.approx(0.035)
    assert res["corrected_se"] == pytest.approx(expected_se)
    assert res["t_stat"] == pytest.approx(expected_t)


def test_evaluate_cv_paired_on_toy_dataframe() -> None:
    df = pd.DataFrame(
        {
            "fan_in_out": [5.0, 4.0, 6.0, 3.5, 4.5, 3.8, 0.5, 0.8, 1.0, 0.2, 0.4, 0.9],
            "lifespan": [10.0, 15.0, 5.0, 20.0, 12.0, 8.0, 200.0, 150.0, 300.0, 180.0, 220.0, 250.0],
        }
    )
    y = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    splits = generate_cv_splits(df, y, seeds=[42], n_splits=3)

    res_rf = evaluate_cv_paired(
        df,
        y,
        ["fan_in_out", "lifespan"],
        splits,
        lambda seed, _y_tr: make_leak_free_pipeline(
            build_random_forest(random_state=seed, params={"n_estimators": 20}, n_jobs=1)
        ),
        inner_val_splits=2,
    )
    assert 0.0 <= res_rf["fixed_050"]["f1_mean"] <= 1.0
    assert len(res_rf["per_fold_fixed"]) == 3

    res_xgb = evaluate_cv_paired(
        df,
        y,
        ["fan_in_out", "lifespan"],
        splits,
        lambda seed, y_tr: make_leak_free_pipeline(
            build_xgboost(
                random_state=seed,
                scale_pos_weight=compute_scale_pos_weight(y_tr),
                params={"n_estimators": 10, "max_depth": 2},
                n_jobs=1,
            )
        ),
        inner_val_splits=2,
    )
    assert 0.0 <= res_xgb["fixed_050"]["f1_mean"] <= 1.0
    assert len(res_xgb["per_fold_fixed"]) == 3

    maj = build_majority_baseline()
    maj.fit(df, y)
    assert maj.predict(df).shape == y.shape

    rule = HeuristicRuleBaseline(
        primary_col="fan_in_out",
        primary_threshold=2.0,
        secondary_col="lifespan",
        secondary_threshold=50.0,
        secondary_op="<=",
    )
    rule.fit(df, y)
    assert np.array_equal(rule.predict(df), y)


def test_real_registry_header_matches_schema() -> None:
    reg = Path("results/registry.csv")
    assert reg.is_file()
    with reg.open("r", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    assert tuple(header) == REGISTRY_COLUMNS


def test_registry_append_only_and_schema_guard(tmp_path: Path) -> None:
    reg = tmp_path / "registry.csv"
    with reg.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(REGISTRY_COLUMNS)

    row_template = {col: "0" for col in REGISTRY_COLUMNS}
    row_template["run_id"] = "RUN-0001"
    row_template["status"] = "valid"
    append_registry_row(reg, row_template)

    row2 = dict(row_template)
    row2["run_id"] = "RUN-0002"
    append_registry_row(reg, row2)

    with reg.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["run_id"] for r in rows] == ["RUN-0001", "RUN-0002"]

    bad_row = dict(row_template)
    del bad_row["status"]
    with pytest.raises(ValueError, match="Row schema mismatch"):
        append_registry_row(reg, bad_row)
