"""Tests for configuration loading, hashing, and seed contract validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import compute_dict_sha256, load_seeds_config


def test_load_seeds_config_contract() -> None:
    cfg = load_seeds_config(Path("configs/seeds.yaml"))
    assert cfg["holdout"]["test_size"] == pytest.approx(0.15)
    assert cfg["holdout"]["random_state"] == 42
    assert cfg["cv"]["n_splits"] == 5
    assert cfg["cv"]["seeds"] == [42, 100, 2026, 7, 999]


def test_missing_config_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_seeds_config(tmp_path / "does_not_exist.yaml")


def test_compute_dict_sha256_is_order_independent() -> None:
    h1 = compute_dict_sha256({"b": 2, "a": [1, 3]})
    h2 = compute_dict_sha256({"a": [1, 3], "b": 2})
    assert h1 == h2
    assert len(h1) == 64

