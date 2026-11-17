#!/usr/bin/env python3
"""
Phase 3 — Feature Engineering Pipeline (Vectorized, Leakage-Free, Grouped)
Project: Anatomy of a Scam Wallet (Ethereum Fraud Detection with Random Forest & XGBoost)

Inputs:
  - data/clean/clean_transactions.parquet
  - data/raw/sampled_wallets.csv
Outputs (in data/features/):
  - features_unscaled.parquet (wallet, is_scam, and raw engineered features)
  - train_scaled.parquet (80% stratified split, log1p-transformed & StandardScaler fit on train)
  - test_scaled.parquet (20% stratified split, transformed with train-fitted log1p list & StandardScaler)
  - split_ids.json
  - feature_groups.json ({"G_temp": [...], "G_val": [...], "G_net": [...], "G_tok": [...]})
  - feature_report.json (counts per group, skew list, unavailable columns, NaN audit, runtime)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
SKEW_THRESHOLD = 2.0
TEST_SIZE = 0.20

LEAKAGE_COLUMNS = {"is_scam", "src_is_scam", "dst_is_scam", "is_contract"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("03_engineer_features")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3: Vectorized Account-Level Behavioral Feature Engineering"
    )
    parser.add_argument(
        "--clean-tx",
        type=Path,
        default=Path("data/clean/clean_transactions.parquet"),
        help="Path to cleaned transactions Parquet file",
    )
    parser.add_argument(
        "--wallets",
        type=Path,
        default=Path("data/raw/sampled_wallets.csv"),
        help="Path to sampled wallets CSV file",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/features"),
        help="Output directory for feature matrices and metadata",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (default: 42)")
    return parser.parse_args()


def engineer_wallet_features(
    clean_tx_path: Path,
    wallets_path: Path,
    out_dir: Path,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Build grouped account-level behavioral features via vectorized pandas groupby."""
    t_start = time.perf_counter()

    if not clean_tx_path.exists():
        raise FileNotFoundError(
            f"Missing input file: {clean_tx_path}. Run Phase 2 (02_clean_data.py) first."
        )
    if not wallets_path.exists():
        raise FileNotFoundError(
            f"Missing input file: {wallets_path}. Run Phase 2 (01_collect_data.py) first."
        )

    wallets_df = pd.read_csv(wallets_path)
    required_w_cols = {"node_id", "address", "is_scam"}
    missing_w = required_w_cols - set(wallets_df.columns)
    if missing_w:
        raise RuntimeError(f"sampled_wallets.csv missing required columns: {sorted(missing_w)}")

    wallets_df = wallets_df[["node_id", "address", "is_scam"]].drop_duplicates(subset=["node_id"]).copy()
    wallets_df["node_id"] = wallets_df["node_id"].astype("uint64")
    wallets_df["wallet"] = wallets_df["address"].astype(str).str.strip().str.lower()
    wallets_df["is_scam"] = wallets_df["is_scam"].astype("int8")

    # Inspect clean_transactions.parquet schema for column projection & optional telemetry
    pf = pq.ParquetFile(clean_tx_path)
    avail_cols = {f.name for f in pf.schema_arrow}

    required_tx_cols = {"src_id", "dst_id", "ts", "value_eth", "is_contract_creation"}
    missing_tx = required_tx_cols - avail_cols
    if missing_tx:
        raise RuntimeError(
            f"clean_transactions.parquet missing required columns: {sorted(missing_tx)}. "
            f"Available: {sorted(avail_cols)}"
        )

    # Check optional G_tok columns without fabricating unavailable fields
    has_tx_fee = "tx_fee_eth" in avail_cols
    has_gas_price = "gas_price_gwei" in avail_cols or "gas_price" in avail_cols
    gas_price_col = "gas_price_gwei" if "gas_price_gwei" in avail_cols else (
        "gas_price" if "gas_price" in avail_cols else None
    )
    has_is_error = "is_error" in avail_cols
    erc20_candidates = [c for c in avail_cols if "erc20" in c.lower() or "token" in c.lower()]
    erc20_col = erc20_candidates[0] if erc20_candidates else None

    read_cols = ["src_id", "dst_id", "ts", "value_eth", "is_contract_creation"]
    if has_tx_fee:
        read_cols.append("tx_fee_eth")
    if gas_price_col:
        read_cols.append(gas_price_col)
    if has_is_error:
        read_cols.append("is_error")
    if erc20_col:
        read_cols.append(erc20_col)

    tx_df = pq.read_table(clean_tx_path, columns=read_cols).to_pandas()
    tx_df["src_id"] = tx_df["src_id"].astype("uint64")
    tx_df["dst_id"] = tx_df["dst_id"].astype("uint64")
    tx_df["ts"] = tx_df["ts"].astype("int64")
    tx_df["value_eth"] = tx_df["value_eth"].astype("float64")
    tx_df["is_contract_creation"] = tx_df["is_contract_creation"].astype(bool)

    sampled_ids = set(wallets_df["node_id"].tolist())
    sent_df = tx_df[tx_df["src_id"].isin(sampled_ids)].copy()
    recv_df = tx_df[tx_df["dst_id"].isin(sampled_ids)].copy()

    # Base index aligned to sampled_wallets.csv
    feat_df = pd.DataFrame(index=pd.Index(wallets_df["node_id"], name="node_id"))

    # ========================================================================
    # GROUP 1: G_temp (Temporal & Frequency Dynamics)
    # ========================================================================
    sent_df = sent_df.sort_values(["src_id", "ts"])
    recv_df = recv_df.sort_values(["dst_id", "ts"])

    sent_df["inter_tx_sec"] = sent_df.groupby("src_id")["ts"].diff()
    recv_df["inter_tx_sec"] = recv_df.groupby("dst_id")["ts"].diff()

    sent_temp = sent_df.groupby("src_id").agg(
        G_temp_sent_tx_count=("ts", "count"),
        G_temp_mean_inter_tx_sent_sec=("inter_tx_sec", "mean"),
    )
    recv_temp = recv_df.groupby("dst_id").agg(
        G_temp_recv_tx_count=("ts", "count"),
        G_temp_mean_inter_tx_recv_sec=("inter_tx_sec", "mean"),
    )

    # Combined endpoint timestamps for lifespan & 1-hour burstiness
    all_ep = pd.concat(
        [
            sent_df[["src_id", "ts"]].rename(columns={"src_id": "node_id"}),
            recv_df[["dst_id", "ts"]].rename(columns={"dst_id": "node_id"}),
        ],
        ignore_index=True,
    )
    lifespan_agg = all_ep.groupby("node_id").agg(
        _min_ts=("ts", "min"),
        _max_ts=("ts", "max"),
    )
    lifespan_agg["G_temp_lifespan_days"] = (
        lifespan_agg["_max_ts"] - lifespan_agg["_min_ts"]
    ) / 86400.0

    all_ep["hour_bucket"] = all_ep["ts"] // 3600
    burst_agg = (
        all_ep.groupby(["node_id", "hour_bucket"])
        .size()
        .groupby("node_id")
        .max()
        .rename("G_temp_burstiness_max_1h")
    )

    feat_df = feat_df.join(lifespan_agg[["G_temp_lifespan_days"]], how="left")
    feat_df = feat_df.join(sent_temp, how="left")
    feat_df = feat_df.join(recv_temp, how="left")
    feat_df = feat_df.join(burst_agg, how="left")

    # ========================================================================
    # GROUP 2: G_val (Value & Volume Flows)
    # ========================================================================
    sent_val = sent_df.groupby("src_id")["value_eth"].agg(
        G_val_total_eth_sent="sum",
        G_val_sent_min_eth="min",
        G_val_sent_max_eth="max",
        G_val_sent_mean_eth="mean",
        G_val_sent_std_eth="std",
    )
    recv_val = recv_df.groupby("dst_id")["value_eth"].agg(
        G_val_total_eth_recv="sum",
        G_val_recv_min_eth="min",
        G_val_recv_max_eth="max",
        G_val_recv_mean_eth="mean",
        G_val_recv_std_eth="std",
    )
    feat_df = feat_df.join(sent_val, how="left").join(recv_val, how="left")
    feat_df = feat_df.fillna(0.0)

    feat_df["G_val_net_eth"] = feat_df["G_val_total_eth_recv"] - feat_df["G_val_total_eth_sent"]
    feat_df["G_val_in_out_ratio"] = (feat_df["G_val_total_eth_recv"] + 1e-6) / (
        feat_df["G_val_total_eth_sent"] + 1e-6
    )

    # ========================================================================
    # GROUP 3: G_net (Counterparty Network Activity)
    # ========================================================================
    unique_sent_to = sent_df.groupby("src_id")["dst_id"].nunique().rename("G_net_unique_sent_to")
    unique_recv_from = recv_df.groupby("dst_id")["src_id"].nunique().rename("G_net_unique_recv_from")

    cp_union = pd.concat(
        [
            sent_df[["src_id", "dst_id"]].rename(columns={"src_id": "node_id", "dst_id": "cp_id"}),
            recv_df[["dst_id", "src_id"]].rename(columns={"dst_id": "node_id", "src_id": "cp_id"}),
        ],
        ignore_index=True,
    )
    unique_cp_total = cp_union.groupby("node_id")["cp_id"].nunique().rename("_unique_cp_total")

    feat_df = feat_df.join(unique_sent_to, how="left").join(unique_recv_from, how="left").join(unique_cp_total, how="left")
    feat_df = feat_df.fillna(0.0)

    total_txs = feat_df["G_temp_sent_tx_count"] + feat_df["G_temp_recv_tx_count"]
    feat_df["G_net_unique_cp_per_tx"] = np.where(
        total_txs > 0, feat_df["_unique_cp_total"] / total_txs, 0.0
    )
    feat_df["G_net_fan_in_out_ratio"] = (feat_df["G_net_unique_recv_from"] + 1.0) / (
        feat_df["G_net_unique_sent_to"] + 1.0
    )
    feat_df = feat_df.drop(columns=["_unique_cp_total"])

    # ========================================================================
    # GROUP 4: G_tok (Smart Contract, Gas/Fee, & Token Interactions)
    # ========================================================================
    unavailable_tok_fields: List[str] = []

    if gas_price_col:
        gp_agg = sent_df.groupby("src_id")[gas_price_col].agg(
            G_tok_mean_gas_price="mean",
            G_tok_max_gas_price="max",
        )
        feat_df = feat_df.join(gp_agg, how="left")
    else:
        unavailable_tok_fields.append(
            "gas_price (not present in fesevu/ethereum_fraud_dataset_by_activity; using tx_fee_eth)"
        )

    if has_tx_fee:
        fee_agg = sent_df.groupby("src_id")["tx_fee_eth"].agg(
            G_tok_mean_tx_fee_eth="mean",
            G_tok_max_tx_fee_eth="max",
        )
        feat_df = feat_df.join(fee_agg, how="left")
    else:
        unavailable_tok_fields.append("tx_fee_eth")

    if has_is_error:
        err_agg = sent_df.groupby("src_id")["is_error"].mean().rename("G_tok_failed_tx_ratio")
        feat_df = feat_df.join(err_agg, how="left")
    else:
        unavailable_tok_fields.append("is_error (not present in fesevu/ethereum_fraud_dataset_by_activity)")

    # Contract interaction proxies: contract creations + zero-value ETH calls (token approvals/calls)
    sent_df["_zero_val_call"] = (sent_df["value_eth"] == 0.0).astype("int8")
    all_ep_tok = pd.concat(
        [
            sent_df[["src_id", "is_contract_creation", "_zero_val_call"]].rename(columns={"src_id": "node_id"}),
            recv_df[["dst_id", "is_contract_creation"]].assign(_zero_val_call=0).rename(columns={"dst_id": "node_id"}),
        ],
        ignore_index=True,
    )
    tok_proxy = all_ep_tok.groupby("node_id").agg(
        G_tok_contract_creation_count=("is_contract_creation", "sum"),
        G_tok_zero_val_call_count=("_zero_val_call", "sum"),
    )
    feat_df = feat_df.join(tok_proxy, how="left").fillna(0.0)
    feat_df["G_tok_contract_creation_ratio"] = np.where(
        total_txs > 0, feat_df["G_tok_contract_creation_count"] / total_txs, 0.0
    )
    feat_df["G_tok_zero_val_call_ratio"] = np.where(
        feat_df["G_temp_sent_tx_count"] > 0,
        feat_df["G_tok_zero_val_call_count"] / feat_df["G_temp_sent_tx_count"],
        0.0,
    )

    if erc20_col:
        erc_agg = all_ep_tok.groupby("node_id").size().rename("G_tok_erc20_tx_count")
        feat_df = feat_df.join(erc_agg, how="left").fillna(0.0)
    else:
        unavailable_tok_fields.append("erc20_token_transfer_count (no ERC-20 column in clean_transactions)")

    # Final NaN fill for any undefined directional stats (0 when no tx in that direction)
    feat_df = feat_df.fillna(0.0)

    # Assemble feature_groups mapping & enforce zero leakage
    feature_cols = list(feat_df.columns)
    assert not (set(feature_cols) & LEAKAGE_COLUMNS), "LEAKAGE DETECTED in feature columns!"

    feature_groups = {
        "G_temp": [c for c in feature_cols if c.startswith("G_temp_")],
        "G_val": [c for c in feature_cols if c.startswith("G_val_")],
        "G_net": [c for c in feature_cols if c.startswith("G_net_")],
        "G_tok": [c for c in feature_cols if c.startswith("G_tok_")],
    }

    # Attach wallet & target for features_unscaled.parquet
    unscaled_df = wallets_df[["wallet", "is_scam"]].reset_index(drop=True).join(
        feat_df.reset_index(drop=True)
    )
    assert int(unscaled_df.isna().sum().sum()) == 0, "NaNs detected in features_unscaled!"

    out_dir.mkdir(parents=True, exist_ok=True)
    unscaled_path = out_dir / "features_unscaled.parquet"
    unscaled_df.to_parquet(unscaled_path, index=False)

    # ========================================================================
    # Stratified 80/20 Split, Skewness log1p Transform, & Train-Only StandardScaler
    # ========================================================================
    train_idx, test_idx = train_test_split(
        unscaled_df.index.to_numpy(),
        test_size=TEST_SIZE,
        stratify=unscaled_df["is_scam"].to_numpy(),
        random_state=seed,
    )
    train_df = unscaled_df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = unscaled_df.iloc[test_idx].copy().reset_index(drop=True)

    # Identify right-skewed non-negative features on TRAIN ONLY
    skew_transformed_cols: List[str] = []
    train_skews: Dict[str, float] = {}
    for col in feature_cols:
        s_val = float(train_df[col].skew())
        train_skews[col] = round(s_val, 4)
        if (train_df[col] >= 0.0).all() and (test_df[col] >= 0.0).all() and abs(s_val) > SKEW_THRESHOLD:
            train_df[col] = np.log1p(train_df[col])
            test_df[col] = np.log1p(test_df[col])
            skew_transformed_cols.append(col)

    # Fit StandardScaler on TRAIN ONLY
    scaler = StandardScaler()
    train_scaled_vals = scaler.fit_transform(train_df[feature_cols])
    test_scaled_vals = scaler.transform(test_df[feature_cols])

    train_scaled_df = pd.concat(
        [
            train_df[["wallet", "is_scam"]].reset_index(drop=True),
            pd.DataFrame(train_scaled_vals, columns=feature_cols),
        ],
        axis=1,
    )
    test_scaled_df = pd.concat(
        [
            test_df[["wallet", "is_scam"]].reset_index(drop=True),
            pd.DataFrame(test_scaled_vals, columns=feature_cols),
        ],
        axis=1,
    )

    assert int(train_scaled_df.isna().sum().sum()) == 0, "NaNs detected in train_scaled!"
    assert int(test_scaled_df.isna().sum().sum()) == 0, "NaNs detected in test_scaled!"

    train_scaled_df.to_parquet(out_dir / "train_scaled.parquet", index=False)
    test_scaled_df.to_parquet(out_dir / "test_scaled.parquet", index=False)

    with open(out_dir / "split_ids.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": seed,
                "test_size": TEST_SIZE,
                "train_count": int(len(train_scaled_df)),
                "test_count": int(len(test_scaled_df)),
                "train_wallets": train_scaled_df["wallet"].tolist(),
                "test_wallets": test_scaled_df["wallet"].tolist(),
            },
            f,
            indent=2,
        )

    with open(out_dir / "feature_groups.json", "w", encoding="utf-8") as f:
        json.dump(feature_groups, f, indent=2)

    runtime_s = round(time.perf_counter() - t_start, 2)
    report = {
        "seed": seed,
        "total_wallets": int(len(unscaled_df)),
        "total_features": len(feature_cols),
        "counts_per_group": {g: len(cols) for g, cols in feature_groups.items()},
        "is_contract_dropped_justification": (
            "Dropped `is_contract` from features because Phase 2 matched both classes identically "
            "on entity type (EOA vs. contract) to remove confounding."
        ),
        "unavailable_g_tok_fields": unavailable_tok_fields,
        "log1p_skew_threshold": SKEW_THRESHOLD,
        "log1p_transformed_features": skew_transformed_cols,
        "train_feature_skewness": train_skews,
        "nan_audit": {
            "unscaled_nans": int(unscaled_df.isna().sum().sum()),
            "train_scaled_nans": int(train_scaled_df.isna().sum().sum()),
            "test_scaled_nans": int(test_scaled_df.isna().sum().sum()),
        },
        "split_class_balance": {
            "train": {
                "normal_0": int((train_scaled_df["is_scam"] == 0).sum()),
                "fraud_1": int((train_scaled_df["is_scam"] == 1).sum()),
            },
            "test": {
                "normal_0": int((test_scaled_df["is_scam"] == 0).sum()),
                "fraud_1": int((test_scaled_df["is_scam"] == 1).sum()),
            },
        },
        "runtime_seconds": runtime_s,
    }
    with open(out_dir / "feature_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print concise summary table
    print("\n=== Phase 3 Feature Engineering Summary ===")
    print(f"Total Wallets : {report['total_wallets']:,} (Train: {len(train_scaled_df):,} | Test: {len(test_scaled_df):,})")
    print(f"Total Features: {report['total_features']} | Zero NaNs: True | Runtime: {runtime_s:.2f}s")
    print(f"{'Group':<10} | {'Count':>5} | Features")
    print("-" * 72)
    for grp, cols in feature_groups.items():
        print(f"{grp:<10} | {len(cols):>5} | {', '.join(cols)}")
    print(f"\nlog1p-transformed ({len(skew_transformed_cols)}): {', '.join(skew_transformed_cols)}")
    if unavailable_tok_fields:
        print(f"Unavailable G_tok fields (omitted): {'; '.join(unavailable_tok_fields)}")

    return report


def main() -> None:
    args = parse_args()
    engineer_wallet_features(
        clean_tx_path=args.clean_tx,
        wallets_path=args.wallets,
        out_dir=args.out_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()