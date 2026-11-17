#!/usr/bin/env python3
"""
Phase 3 — Exploratory Data Analysis & Multicollinearity Visualizations
Project: Anatomy of a Scam Wallet (Ethereum Fraud Detection with Random Forest & XGBoost)

Inputs (from data/features/):
  - train_scaled.parquet
  - test_scaled.parquet
  - feature_groups.json
Outputs (in report/figures/):
  - class_distribution.{pdf,png}: Train & test wallet counts per class
  - corr_heatmap_all.{pdf,png}: Spearman correlation of all features on training set, ordered by group
  - corr_heatmap_by_group.{pdf,png}: 2x2 panel of within-group Spearman correlations
  - high_corr_pairs.csv: Pairs with |rho| > 0.9 motivating grouped ablation
  - top10_features_by_class.{pdf,png}: Top-10 features by |mean_fraud - mean_normal| grouped by color
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.0,
    "axes.titlesize": 10.0,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 8.0,
    "figure.dpi": 300,
})

PALETTE_CLASS = {"normal": "#1f4e78", "fraud": "#c0392b"}
GROUP_COLORS = {
    "G_temp": "#1f4e78",
    "G_val": "#2e7d32",
    "G_net": "#6a1b9a",
    "G_tok": "#c0392b",
}
GROUP_ORDER = ["G_temp", "G_val", "G_net", "G_tok"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("04_eda_visualizations")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3: EDA Visualizations & Multicollinearity Audit")
    parser.add_argument("--features-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--fig-dir", type=Path, default=Path("report/figures"))
    return parser.parse_args()


def plot_class_distribution(train_df: pd.DataFrame, test_df: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.30})
    labels = ["Normal (0)", "Fraud (1)"]
    colors = [PALETTE_CLASS["normal"], PALETTE_CLASS["fraud"]]

    for ax, df_split, title in [
        (axes[0], train_df, "(a) Training Split (80%)"),
        (axes[1], test_df, "(b) Test Split (20%)"),
    ]:
        counts = [int((df_split["is_scam"] == 0).sum()), int((df_split["is_scam"] == 1).sum())]
        bars = ax.bar(labels, counts, color=colors, width=0.50, edgecolor="black", linewidth=0.6)
        ax.set_title(title, fontweight="bold", pad=8)
        ax.set_ylabel("Unique Wallets")
        ax.set_ylim(0, max(counts) * 1.25)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        total = max(sum(counts), 1)
        for b in bars:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                h + max(counts) * 0.03,
                f"{int(h):,}\n({100.0 * h / total:.1f}%)",
                ha="center", va="bottom", fontsize=8.0, fontweight="bold",
            )

    fig.savefig(fig_dir / "class_distribution.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "class_distribution.png", bbox_inches="tight")
    plt.close(fig)


def plot_corr_heatmaps_and_pairs(
    train_df: pd.DataFrame,
    feature_groups: Dict[str, List[str]],
    fig_dir: Path,
) -> None:
    ordered_cols: List[str] = []
    boundaries: List[int] = []
    for grp in GROUP_ORDER:
        cols = feature_groups.get(grp, [])
        ordered_cols.extend(cols)
        boundaries.append(len(ordered_cols))

    corr_mat = train_df[ordered_cols].corr(method="spearman").fillna(0.0)
    short_names = [c.replace("G_temp_", "T:").replace("G_val_", "V:").replace("G_net_", "N:").replace("G_tok_", "K:") for c in ordered_cols]

    # 1. Full Spearman correlation heatmap with group boundaries
    n = len(ordered_cols)
    fig, ax = plt.subplots(figsize=(max(8.0, n * 0.34), max(7.0, n * 0.30)))
    im = ax.imshow(corr_mat.values, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(short_names, rotation=65, ha="right", fontsize=7.0)
    ax.set_yticklabels(short_names, fontsize=7.0)
    ax.set_title("Spearman Correlation Across Behavioral Feature Groups (Train Set)", fontweight="bold", pad=10)

    for b in boundaries[:-1]:
        ax.axhline(b - 0.5, color="black", linewidth=1.4)
        ax.axvline(b - 0.5, color="black", linewidth=1.4)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"Spearman $\rho$", fontsize=8.5)
    plt.tight_layout()
    fig.savefig(fig_dir / "corr_heatmap_all.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "corr_heatmap_all.png", bbox_inches="tight")
    plt.close(fig)

    # 2. 2x2 Panel by Group
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.2), gridspec_kw={"wspace": 0.45, "hspace": 0.48})
    for ax, grp in zip(axes.flat, GROUP_ORDER):
        cols = feature_groups.get(grp, [])
        sub_corr = corr_mat.loc[cols, cols]
        sub_labels = [c.replace(f"{grp}_", "") for c in cols]
        im_sub = ax.imshow(sub_corr.values, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
        ax.set_xticks(np.arange(len(cols)))
        ax.set_yticks(np.arange(len(cols)))
        ax.set_xticklabels(sub_labels, rotation=45, ha="right", fontsize=7.0)
        ax.set_yticklabels(sub_labels, fontsize=7.0)
        ax.set_title(f"Group: {grp} ({len(cols)} features)", fontweight="bold", pad=6)
        for i in range(len(cols)):
            for j in range(len(cols)):
                val = sub_corr.values[i, j]
                txt_color = "white" if abs(val) > 0.55 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=txt_color, fontsize=6.0)

    fig.colorbar(im_sub, ax=axes.ravel().tolist(), fraction=0.025, pad=0.03, label=r"Spearman $\rho$")
    fig.savefig(fig_dir / "corr_heatmap_by_group.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "corr_heatmap_by_group.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Export high_corr_pairs.csv (|rho| > 0.9)
    col_to_group = {c: grp for grp in GROUP_ORDER for c in feature_groups.get(grp, [])}
    high_pairs = []
    vals = corr_mat.values
    for i in range(n):
        for j in range(i + 1, n):
            rho = float(vals[i, j])
            if abs(rho) > 0.90:
                high_pairs.append({
                    "feature_1": ordered_cols[i],
                    "group_1": col_to_group[ordered_cols[i]],
                    "feature_2": ordered_cols[j],
                    "group_2": col_to_group[ordered_cols[j]],
                    "spearman_rho": round(rho, 4),
                    "abs_rho": round(abs(rho), 4),
                })

    pairs_df = pd.DataFrame(
        high_pairs,
        columns=["feature_1", "group_1", "feature_2", "group_2", "spearman_rho", "abs_rho"],
    ).sort_values("abs_rho", ascending=False)
    pairs_path = fig_dir / "high_corr_pairs.csv"
    pairs_df.to_csv(pairs_path, index=False)
    logger.info("Saved %d high-correlation pairs (|rho| > 0.90) to %s", len(pairs_df), pairs_path)


def plot_top10_features(
    train_df: pd.DataFrame,
    feature_groups: Dict[str, List[str]],
    fig_dir: Path,
) -> None:
    col_to_group = {c: grp for grp in GROUP_ORDER for c in feature_groups.get(grp, [])}
    all_cols = [c for grp in GROUP_ORDER for c in feature_groups.get(grp, [])]

    fraud_means = train_df.loc[train_df["is_scam"] == 1, all_cols].mean()
    normal_means = train_df.loc[train_df["is_scam"] == 0, all_cols].mean()
    mean_diff = (fraud_means - normal_means).abs().sort_values(ascending=False)
    top10_cols = list(mean_diff.head(10).index)[::-1]  # reverse for horizontal bar/box order

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    y_pos = np.arange(len(top10_cols))
    diffs = [float(mean_diff[c]) for c in top10_cols]
    colors = [GROUP_COLORS[col_to_group[c]] for c in top10_cols]
    signed_dir = [
        "+" if fraud_means[c] >= normal_means[c] else "-" for c in top10_cols
    ]

    bars = ax.barh(y_pos, diffs, color=colors, edgecolor="black", linewidth=0.5, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top10_cols, fontsize=8.0)
    ax.set_xlabel(r"Standardized Mean Difference $|\mu_{\mathrm{fraud}} - \mu_{\mathrm{normal}}|$ (Train Set)")
    ax.set_title("Top-10 Discriminative Features by Class Mean Separation (Colored by Group)", fontweight="bold", pad=8)
    ax.set_xlim(0, max(diffs) * 1.18)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for b, sgn in zip(bars, signed_dir):
        w = b.get_width()
        ax.text(w + max(diffs) * 0.02, b.get_y() + b.get_height() / 2.0, f"{w:.3f} ({sgn})", va="center", fontsize=7.5)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g], label=g)
        for g in GROUP_ORDER
    ]
    ax.legend(handles=handles, title="Behavior Group", loc="lower right", frameon=True)

    fig.savefig(fig_dir / "top10_features_by_class.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "top10_features_by_class.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    train_path = args.features_dir / "train_scaled.parquet"
    test_path = args.features_dir / "test_scaled.parquet"
    groups_path = args.features_dir / "feature_groups.json"

    for p in (train_path, test_path, groups_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run 03_engineer_features.py first.")

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    with open(groups_path, "r", encoding="utf-8") as f:
        feature_groups = json.load(f)

    plot_class_distribution(train_df, test_df, args.fig_dir)
    plot_corr_heatmaps_and_pairs(train_df, feature_groups, args.fig_dir)
    plot_top10_features(train_df, feature_groups, args.fig_dir)
    logger.info("All Phase 3 EDA visualizations saved to %s", args.fig_dir)


if __name__ == "__main__":
    main()