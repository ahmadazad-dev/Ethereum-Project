#!/usr/bin/env python3
"""
Phase 2 — Deliverable 1 (File 3): Diagnostic Figures for LaTeX Report
Project: Anatomy of a Scam Wallet (Ethereum Fraud Detection with Random Forest & XGBoost)

Inputs:
  - data/raw/sampled_wallets.csv
  - data/clean/clean_transactions.parquet
  - data/clean/cleaning_report.json
Outputs (saved as both PDF and PNG in report/figures/):
  1. class_balance.{pdf,png}: Wallets per class and transactions touching fraud vs. normal wallets
  2. contract_mix_by_class.{pdf,png}: Confound-control verification showing identical `is_contract`
     mix across Fraud (1) and Normal (0) sampled wallets
  3. value_distribution.{pdf,png}: Before vs. After 99.9th percentile winsorization histogram
     of `log1p(value_eth)`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.5,
    "axes.titlesize": 10.0,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
})


def make_figures(
    wallets_path: Path,
    clean_tx_path: Path,
    report_path: Path,
    fig_dir: Path,
) -> None:
    if not wallets_path.exists():
        raise FileNotFoundError(f"Missing {wallets_path}. Run 01_collect_data.py first.")
    if not clean_tx_path.exists():
        raise FileNotFoundError(f"Missing {clean_tx_path}. Run 02_clean_data.py first.")
    if not report_path.exists():
        raise FileNotFoundError(f"Missing {report_path}. Run 02_clean_data.py first.")

    fig_dir.mkdir(parents=True, exist_ok=True)
    wallets_df = pd.read_csv(wallets_path)
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    # Read only lightweight columns needed for histograms and endpoint counts
    tx_df = pq.read_table(
        clean_tx_path,
        columns=[
            "src_is_scam", "dst_is_scam",
            "value_eth_uncapped", "log1p_value_eth_uncapped", "log1p_value_eth",
        ],
    ).to_pandas()

    # ------------------------------------------------------------------------
    # Figure 1: Class Balance (Wallets per class & Transactions touching each class)
    # ------------------------------------------------------------------------
    n_normal_w = int((wallets_df["is_scam"] == 0).sum())
    n_fraud_w = int((wallets_df["is_scam"] == 1).sum())

    tx_touch_normal = int(((tx_df["src_is_scam"] == 0) | (tx_df["dst_is_scam"] == 0)).fillna(False).sum())
    tx_touch_fraud = int(((tx_df["src_is_scam"] == 1) | (tx_df["dst_is_scam"] == 1)).fillna(False).sum())

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), gridspec_kw={"wspace": 0.32})
    palette = ["#1f4e78", "#c0392b"]
    labels = ["Normal (0)", "Fraud (1)"]

    w_vals = [n_normal_w, n_fraud_w]
    bars1 = axes[0].bar(labels, w_vals, color=palette, width=0.50, edgecolor="black", linewidth=0.6)
    axes[0].set_title("(a) Sampled Wallets per Class", fontweight="bold", pad=8)
    axes[0].set_ylabel("Unique Wallets")
    axes[0].set_ylim(0, max(w_vals) * 1.25)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    for b in bars1:
        h = b.get_height()
        axes[0].text(
            b.get_x() + b.get_width() / 2.0,
            h + max(w_vals) * 0.03,
            f"{int(h):,}\n({100.0 * h / max(sum(w_vals), 1):.1f}%)",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold",
        )

    t_vals = [tx_touch_normal, tx_touch_fraud]
    bars2 = axes[1].bar(labels, t_vals, color=palette, width=0.50, edgecolor="black", linewidth=0.6)
    axes[1].set_title("(b) Transactions Touching Each Class", fontweight="bold", pad=8)
    axes[1].set_ylabel("Transaction Count")
    axes[1].set_ylim(0, max(t_vals) * 1.25)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    for b in bars2:
        h = b.get_height()
        axes[1].text(
            b.get_x() + b.get_width() / 2.0,
            h + max(t_vals) * 0.03,
            f"{int(h):,}",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold",
        )

    fig.savefig(fig_dir / "class_balance.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "class_balance.png", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------------
    # Figure 2: is_contract Mix by Class (Confound Control Verification)
    # ------------------------------------------------------------------------
    ct = pd.crosstab(wallets_df["is_scam"], wallets_df["is_contract"]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    eoa_counts = ct[0].to_numpy()
    contract_counts = ct[1].to_numpy()
    x = np.arange(2)
    width = 0.34

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    b_eoa = ax.bar(x - width / 2, eoa_counts, width, label="EOA (is_contract = 0)", color="#2c7bb6", edgecolor="black", linewidth=0.6)
    b_con = ax.bar(x + width / 2, contract_counts, width, label="Smart Contract (is_contract = 1)", color="#fdae61", edgecolor="black", linewidth=0.6)

    ax.set_title("Confound Control: Entity Type Mix by Class", fontweight="bold", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Normal Wallets (0)", "Fraud Wallets (1)"])
    ax.set_ylabel("Number of Sampled Wallets")
    max_bar = max(int(eoa_counts.max()), int(contract_counts.max()), 1)
    ax.set_ylim(0, max_bar * 1.25)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bars in (b_eoa, b_con):
        for b in bars:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                h + max_bar * 0.03,
                f"{int(h):,}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
            )

    fig.savefig(fig_dir / "contract_mix_by_class.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "contract_mix_by_class.png", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------------
    # Figure 3: Before vs. After Winsorization Histogram of log1p(value_eth)
    # ------------------------------------------------------------------------
    step6_info = next(s for s in report["step_by_step_log"] if s["step"] == 6)
    val_cap = float(step6_info["value_eth"]["p99_9_threshold_eth"])
    log_cap = float(np.log1p(val_cap))

    non_zero = tx_df[tx_df["value_eth_uncapped"] > 0]
    plot_series_before = non_zero["log1p_value_eth_uncapped"] if len(non_zero) > 0 else tx_df["log1p_value_eth_uncapped"]
    plot_series_after = non_zero["log1p_value_eth"] if len(non_zero) > 0 else tx_df["log1p_value_eth"]

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=True, gridspec_kw={"wspace": 0.18})
    axes[0].hist(plot_series_before, bins=45, color="#34495e", edgecolor="white", linewidth=0.4, alpha=0.9)
    axes[0].axvline(log_cap, color="#c0392b", linestyle="--", linewidth=1.2, label=f"p99.9 cap ({val_cap:.2f} ETH)")
    axes[0].set_title("(a) Before Winsorization", fontweight="bold", pad=8)
    axes[0].set_xlabel(r"$\log(1 + \mathrm{value\_eth})$")
    axes[0].set_ylabel("Transaction Frequency")
    axes[0].legend(loc="upper right", frameon=True)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].hist(plot_series_after, bins=45, color="#1f4e78", edgecolor="white", linewidth=0.4, alpha=0.9)
    axes[1].axvline(log_cap, color="#c0392b", linestyle="--", linewidth=1.2, label=f"Capped at {val_cap:.2f} ETH")
    axes[1].set_title("(b) After 99.9th Pct. Winsorization", fontweight="bold", pad=8)
    axes[1].set_xlabel(r"$\log(1 + \mathrm{value\_eth}_{\mathrm{capped}})$")
    axes[1].legend(loc="upper right", frameon=True)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.savefig(fig_dir / "value_distribution.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "value_distribution.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 Step 3: Generate Diagnostic Figures")
    parser.add_argument("--wallets", type=Path, default=Path("data/raw/sampled_wallets.csv"))
    parser.add_argument("--clean-tx", type=Path, default=Path("data/clean/clean_transactions.parquet"))
    parser.add_argument("--report", type=Path, default=Path("data/clean/cleaning_report.json"))
    parser.add_argument("--fig-dir", type=Path, default=Path("report/figures"))
    args = parser.parse_args()

    make_figures(
        wallets_path=args.wallets,
        clean_tx_path=args.clean_tx,
        report_path=args.report,
        fig_dir=args.fig_dir,
    )


if __name__ == "__main__":
    main()