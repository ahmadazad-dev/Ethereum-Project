"""Ablation configuration builders for Full, Group-Only, Drop-One, and Add-One protocols."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.features import CANONICAL_GROUP_IDS, validate_feature_groups


@dataclass(frozen=True)
class AblationConfig:
    """Immutable specification of a single ablation feature subset."""

    protocol: str
    name: str
    included_groups: tuple[str, ...]
    feature_cols: tuple[str, ...]


def build_ablation_suite(
    feature_groups: Mapping[str, Sequence[str]],
    *,
    add_one_ordering: Sequence[str] | None = None,
) -> list[AblationConfig]:
    """Generate the full ablation matrix specified in AGENTS.md §5.

    Protocols generated:
        1. full: G1+G2+G3+G4+G5
        2. group_only: {G1}, {G2}, {G3}, {G4}, {G5}
        3. drop_one: All \\ Gi for each i in 1..5
        4. add_one (if add_one_ordering is supplied): forward cumulative subsets
    """
    validate_feature_groups(feature_groups, allow_empty=False)

    configs: list[AblationConfig] = []

    def _cols_for(groups: Sequence[str]) -> tuple[str, ...]:
        cols: list[str] = []
        for g in groups:
            cols.extend(feature_groups[g])
        return tuple(cols)

    # 1. Full baseline
    configs.append(
        AblationConfig(
            protocol="full",
            name="ALL_G1_TO_G5",
            included_groups=CANONICAL_GROUP_IDS,
            feature_cols=_cols_for(CANONICAL_GROUP_IDS),
        )
    )

    # 2. Group-Only (RQ1 sufficiency)
    for gid in CANONICAL_GROUP_IDS:
        configs.append(
            AblationConfig(
                protocol="group_only",
                name=f"ONLY_{gid}",
                included_groups=(gid,),
                feature_cols=_cols_for((gid,)),
            )
        )

    # 3. Drop-One-Group (RQ2 necessity)
    for gid in CANONICAL_GROUP_IDS:
        remaining = tuple(g for g in CANONICAL_GROUP_IDS if g != gid)
        configs.append(
            AblationConfig(
                protocol="drop_one",
                name=f"DROP_{gid}",
                included_groups=remaining,
                feature_cols=_cols_for(remaining),
            )
        )

    # 4. Add-One-Group (RQ3 parsimony; pre-registered dev Group-Only ranking)
    if add_one_ordering is not None:
        order = list(add_one_ordering)
        if len(order) != len(CANONICAL_GROUP_IDS) or set(order) != set(CANONICAL_GROUP_IDS):
            raise ValueError(
                f"add_one_ordering must be a permutation of {CANONICAL_GROUP_IDS}, got {order}"
            )
        cumulative: list[str] = []
        for gid in order:
            cumulative.append(gid)
            configs.append(
                AblationConfig(
                    protocol="add_one",
                    name="ADD_" + "_".join(cumulative),
                    included_groups=tuple(cumulative),
                    feature_cols=_cols_for(cumulative),
                )
            )

    return configs

