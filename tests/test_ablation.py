"""Tests for ablation protocols and paired split determinism."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ablation import build_ablation_suite
from src.evaluation import generate_cv_splits


def test_ablation_suite_counts_and_subsets() -> None:
    toy_groups = {
        "G1": ["g1_a", "g1_b"],
        "G2": ["g2_a"],
        "G3": ["g3_a"],
        "G4": ["g4_a"],
        "G5": ["g5_a"],
    }
    suite = build_ablation_suite(
        toy_groups,
        add_one_ordering=["G4", "G1", "G2", "G5", "G3"],
    )
    # 1 full + 5 group_only + 5 drop_one + 5 add_one = 16
    assert len(suite) == 16

    by_name = {cfg.name: cfg for cfg in suite}
    assert by_name["ALL_G1_TO_G5"].feature_cols == (
        "g1_a",
        "g1_b",
        "g2_a",
        "g3_a",
        "g4_a",
        "g5_a",
    )
    assert by_name["ONLY_G1"].feature_cols == ("g1_a", "g1_b")
    assert "g1_a" not in by_name["DROP_G1"].feature_cols
    assert len(by_name["DROP_G1"].feature_cols) == 4
    assert by_name["ADD_G4_G1"].feature_cols == ("g4_a", "g1_a", "g1_b")


def test_ablation_rejects_invalid_ordering() -> None:
    toy_groups = {f"G{i}": [f"f{i}"] for i in range(1, 6)}
    with pytest.raises(ValueError, match="permutation"):
        build_ablation_suite(toy_groups, add_one_ordering=["G1", "G2", "G3", "G4"])


def test_identical_folds_across_ablation_configs_and_grouped_duplicates() -> None:
    df = pd.DataFrame(
        {
            "g1_a": np.arange(20, dtype=float),
            "g4_a": np.arange(20, 0, -1, dtype=float),
        }
    )
    y = np.array([0, 1] * 10)
    # Pair near-duplicate wallets (0,1), (2,3), etc. into shared groups
    groups = np.repeat(np.arange(10), 2)

    splits_full = generate_cv_splits(df[["g1_a", "g4_a"]], y, seeds=[42, 100], n_splits=5, groups=groups)
    splits_g1 = generate_cv_splits(df[["g1_a"]], y, seeds=[42, 100], n_splits=5, groups=groups)

    assert len(splits_full) == 10
    for (s1, f1, tr1, va1), (s2, f2, tr2, va2) in zip(splits_full, splits_g1, strict=True):
        assert (s1, f1) == (s2, f2)
        assert np.array_equal(tr1, tr2)
        assert np.array_equal(va1, va2)
        # No wallet group may straddle train and val
        assert set(groups[tr1]).isdisjoint(set(groups[va1]))

