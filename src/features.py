"""Single source of truth for on-chain behavioral feature groups (G1-G5).

Changing membership in FEATURE_GROUPS invalidates downstream ablation results
and requires an ADR in docs/DECISIONS.md plus a full rerun (AGENTS.md §1).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

CANONICAL_GROUP_IDS: tuple[str, ...] = ("G1", "G2", "G3", "G4", "G5")

GROUP_DESCRIPTIONS: dict[str, str] = {
    "G1": "Temporal and cadence",
    "G2": "Native ETH value flow",
    "G3": "Contract and internal-trace activity",
    "G4": "Counterparty fan-in/fan-out topology",
    "G5": "ERC-20 token behavior",
}

# Columns that must never enter model feature matrices (AGENTS.md §6).
FORBIDDEN_IDENTIFIER_COLUMNS: frozenset[str] = frozenset(
    {
        "Unnamed: 0",
        "Index",
        "index",
        "Address",
        "address",
        "FLAG",
        "flag",
        "label",
        "target",
    }
)

# Single source of truth for feature-to-group assignments.
# Populated and frozen via ADR once the dataset schema is audited and registered.
FEATURE_GROUPS: dict[str, list[str]] = {
    "G1": [],
    "G2": [],
    "G3": [],
    "G4": [],
    "G5": [],
}


def validate_feature_groups(
    feature_groups: Mapping[str, Sequence[str]],
    *,
    allow_empty: bool = False,
) -> list[str]:
    """Validate that feature_groups partitions features cleanly into G1..G5.

    Args:
        feature_groups: Mapping from group ID ('G1'..'G5') to feature names.
        allow_empty: Whether empty group lists are permitted (useful before schema lock).

    Returns:
        Ordered list of all unique feature names across G1..G5.

    Raises:
        ValueError: On missing/extra group keys, empty groups (if disallowed),
            forbidden identifier columns, or duplicate features within/across groups.
    """
    keys = tuple(feature_groups.keys())
    if set(keys) != set(CANONICAL_GROUP_IDS):
        raise ValueError(
            f"Group keys must exactly match {CANONICAL_GROUP_IDS}, got {keys}"
        )

    seen: dict[str, str] = {}
    ordered_features: list[str] = []

    for gid in CANONICAL_GROUP_IDS:
        cols = list(feature_groups[gid])
        if not allow_empty and len(cols) == 0:
            raise ValueError(f"Feature group '{gid}' must not be empty")

        for col in cols:
            if not isinstance(col, str) or not col.strip():
                raise ValueError(f"Invalid feature name {col!r} in group '{gid}'")
            if col in FORBIDDEN_IDENTIFIER_COLUMNS:
                raise ValueError(
                    f"Forbidden identifier/label column '{col}' found in group '{gid}'"
                )
            if col in seen:
                prev = seen[col]
                raise ValueError(
                    f"Feature '{col}' assigned multiple times (in '{prev}' and '{gid}')"
                )
            seen[col] = gid
            ordered_features.append(col)

    return ordered_features


def assert_no_forbidden_columns(
    columns: Iterable[str],
    forbidden: Iterable[str] = FORBIDDEN_IDENTIFIER_COLUMNS,
) -> None:
    """Fail loudly if any identifier, index, or target column appears in a feature matrix."""
    forbidden_set = set(forbidden)
    leaked = sorted(col for col in columns if col in forbidden_set)
    if leaked:
        raise ValueError(f"Forbidden identifier/label columns present in features: {leaked}")

