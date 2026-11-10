"""Tests for behavioral feature-group partitioning and identifier leakage guards."""

from __future__ import annotations

import pytest

from src.features import (
    CANONICAL_GROUP_IDS,
    FEATURE_GROUPS,
    assert_no_forbidden_columns,
    validate_feature_groups,
)


def test_canonical_feature_groups_structure() -> None:
    assert tuple(FEATURE_GROUPS.keys()) == CANONICAL_GROUP_IDS
    validate_feature_groups(FEATURE_GROUPS, allow_empty=True)


def test_every_feature_appears_in_exactly_one_group() -> None:
    toy_groups = {
        "G1": ["active_lifespan_mins", "tx_velocity"],
        "G2": ["total_eth_sent", "total_eth_received"],
        "G3": ["contracts_created", "eth_to_contract_ratio"],
        "G4": ["unique_in_peers", "unique_out_peers"],
        "G5": ["erc20_tx_count", "erc20_unique_tokens"],
    }
    ordered = validate_feature_groups(toy_groups, allow_empty=False)
    assert len(ordered) == 10
    assert len(set(ordered)) == 10


def test_duplicate_feature_across_groups_raises() -> None:
    bad_groups = {
        "G1": ["f1"],
        "G2": ["f1"],
        "G3": ["f3"],
        "G4": ["f4"],
        "G5": ["f5"],
    }
    with pytest.raises(ValueError, match="assigned multiple times"):
        validate_feature_groups(bad_groups, allow_empty=False)


def test_forbidden_identifier_columns_blocked() -> None:
    bad_groups = {
        "G1": ["Address"],
        "G2": ["f2"],
        "G3": ["f3"],
        "G4": ["f4"],
        "G5": ["f5"],
    }
    with pytest.raises(ValueError, match="Forbidden identifier"):
        validate_feature_groups(bad_groups, allow_empty=False)

    with pytest.raises(ValueError, match="Forbidden identifier"):
        assert_no_forbidden_columns(["f1", "Unnamed: 0", "FLAG"])

