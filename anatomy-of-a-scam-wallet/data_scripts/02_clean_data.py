#!/usr/bin/env python3
"""
Phase 2 — Deliverable 1 (File 2): Deterministic 8-Step Ethereum Data Cleaning Pipeline
Project: Anatomy of a Scam Wallet (Ethereum Fraud Detection with Random Forest & XGBoost)

Inputs:
  - data/raw/raw_transactions.parquet
  - data/raw/sampled_wallets.csv
Outputs:
  - data/clean/clean_transactions.parquet
  - data/clean/cleaning_report.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import duckdb
except ImportError as exc:
    raise ImportError("CRITICAL: `duckdb` is required. Install via: pip install duckdb") from exc

REQUIRED_RAW_COLS = {
    "tx_hash", "src_id", "dst_id", "src_address", "dst_address",
    "ts", "value_wei", "tx_fee_wei", "block_number", "contract_creation", "source_month",
}
REQUIRED_WALLET_COLS = {"node_id", "address", "is_scam", "is_contract"}

CONTRACT_CREATION_PLACEHOLDER = "CONTRACT_CREATION"
ETH_HEX_REGEX = re.compile(r"^0x[0-9a-f]{40}$")
WINSOR_Q = 0.999
MIN_VALID_UNIX_TS = 1_438_214_400
MAX_VALID_UNIX_TS = 1_798_761_599

FIELD_MAPPING_TABLE = [
    {
        "requested_ethereum_field": "tx_hash",
        "status": "AVAILABLE",
        "dataset_column": "tx_hash (large_string)",
        "handling_rule": "Lowercased, deduplicated (keep first), non-null enforced.",
    },
    {
        "requested_ethereum_field": "from",
        "status": "AVAILABLE",
        "dataset_column": "src_id (uint64) + src_address (nullable string)",
        "handling_rule": "Every row has src_id; src_address populated when endpoint is in the 114,739 labeled wallets.",
    },
    {
        "requested_ethereum_field": "to",
        "status": "AVAILABLE",
        "dataset_column": "dst_id (uint64) + dst_address (nullable string)",
        "handling_rule": "Every row has dst_id; contract creations flagged via is_contract_creation and placeholder.",
    },
    {
        "requested_ethereum_field": "value",
        "status": "AVAILABLE",
        "dataset_column": "value_wei (decimal string with 9 trailing zeros)",
        "handling_rule": "Parsed and divided by 1e18 -> value_eth (float64), winsorized at p99.9 + log1p_value_eth.",
    },
    {
        "requested_ethereum_field": "timestamp",
        "status": "AVAILABLE",
        "dataset_column": "ts (int64 UNIX seconds)",
        "handling_rule": "Converted to UTC timestamp (timestamp_utc); out-of-range/unparseable rows dropped.",
    },
    {
        "requested_ethereum_field": "tx_fee (gas_used * gas_price)",
        "status": "AVAILABLE",
        "dataset_column": "tx_fee_wei (decimal string with 9 trailing zeros)",
        "handling_rule": "Parsed and divided by 1e18 -> tx_fee_eth (float64), winsorized at p99.9 + log1p_tx_fee_eth.",
    },
    {
        "requested_ethereum_field": "gas_used",
        "status": "UNAVAILABLE",
        "dataset_column": "None (not present in edges_by_month)",
        "handling_rule": "Documented as limitation; NEVER fabricated or imputed.",
    },
    {
        "requested_ethereum_field": "gas_price",
        "status": "UNAVAILABLE",
        "dataset_column": "None (not present in edges_by_month)",
        "handling_rule": "Documented as limitation; NEVER fabricated or imputed.",
    },
    {
        "requested_ethereum_field": "is_error",
        "status": "UNAVAILABLE",
        "dataset_column": "None (not present in edges_by_month)",
        "handling_rule": "Documented as limitation; NEVER fabricated or imputed.",
    },
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("02_clean_data")


def compute_series_stats(s: pd.Series) -> Dict[str, float]:
    """Return summary statistics for a numeric pandas Series."""
    clean_s = pd.to_numeric(s, errors="coerce").dropna()
    if clean_s.empty:
        return {}
    return {
        "count": int(len(clean_s)),
        "min": float(clean_s.min()),
        "p25": float(clean_s.quantile(0.25)),
        "p50_median": float(clean_s.median()),
        "mean": float(clean_s.mean()),
        "std": float(clean_s.std()),
        "p75": float(clean_s.quantile(0.75)),
        "p99": float(clean_s.quantile(0.99)),
        "p99_9": float(clean_s.quantile(0.999)),
        "max": float(clean_s.max()),
    }


def run_cleaning_pipeline(
    raw_tx_path: Path,
    sampled_wallets_path: Path,
    clean_tx_path: Path,
    report_path: Path,
) -> Dict[str, Any]:
    """Run the 8-step deterministic cleaning pipeline and write outputs."""
    if not raw_tx_path.exists():
        raise FileNotFoundError(f"Missing raw transactions file: {raw_tx_path}. Run 01_collect_data.py first.")
    if not sampled_wallets_path.exists():
        raise FileNotFoundError(f"Missing sampled wallets file: {sampled_wallets_path}. Run 01_collect_data.py first.")

    pf = pq.ParquetFile(raw_tx_path)
    actual_raw_cols = {f.name for f in pf.schema_arrow}
    missing_raw = REQUIRED_RAW_COLS - actual_raw_cols
    if missing_raw:
        raise RuntimeError(
            f"SCHEMA ERROR in {raw_tx_path}: missing required columns {sorted(missing_raw)}. "
            f"Found: {sorted(actual_raw_cols)}"
        )

    wallets_df = pd.read_csv(sampled_wallets_path)
    missing_w = REQUIRED_WALLET_COLS - set(wallets_df.columns)
    if missing_w:
        raise RuntimeError(
            f"SCHEMA ERROR in {sampled_wallets_path}: missing columns {sorted(missing_w)}."
        )

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA memory_limit='2.5GB';")
    df = con.execute(f"""
        SELECT
            tx_hash,
            CAST(src_id AS UBIGINT) AS src_id,
            CAST(dst_id AS UBIGINT) AS dst_id,
            src_address,
            dst_address,
            ts,
            TRY_CAST(value_wei AS DOUBLE) AS value_wei_num,
            TRY_CAST(tx_fee_wei AS DOUBLE) AS tx_fee_wei_num,
            block_number,
            contract_creation,
            source_month
        FROM read_parquet('{raw_tx_path.as_posix()}')
    """).df()
    con.close()

    initial_rows = int(len(df))
    step_logs: List[Dict[str, Any]] = []
    logger.info("Loaded %d raw transactions from %s", initial_rows, raw_tx_path)

    # STEP 1: Timestamps (ts UNIX seconds -> UTC datetime; drop unparseable)
    rows_before = int(len(df))
    ts_num = pd.to_numeric(df["ts"], errors="coerce")
    valid_ts_mask = ts_num.notna() & (ts_num >= MIN_VALID_UNIX_TS) & (ts_num <= MAX_VALID_UNIX_TS)
    dropped_ts = int((~valid_ts_mask).sum())
    df = df.loc[valid_ts_mask].copy()
    df["ts"] = ts_num.loc[valid_ts_mask].astype("int64")
    df["timestamp_utc"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    rows_after = int(len(df))

    step_logs.append({
        "step": 1,
        "name": "Standardize ts (UNIX seconds) to UTC datetime & drop unparseable/out-of-range",
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": dropped_ts,
    })

    # STEP 2: Wei to Ether (value_eth = value_wei / 1e18, tx_fee_eth = tx_fee_wei / 1e18)
    rows_before = int(len(df))
    df["value_eth"] = df["value_wei_num"].astype("float64") / 1e18
    df["tx_fee_eth"] = df["tx_fee_wei_num"].astype("float64") / 1e18
    df = df.drop(columns=["value_wei_num", "tx_fee_wei_num"])

    before_stats = {
        "value_eth": compute_series_stats(df["value_eth"]),
        "tx_fee_eth": compute_series_stats(df["tx_fee_eth"]),
    }
    rows_after = int(len(df))
    step_logs.append({
        "step": 2,
        "name": "Convert value_wei and tx_fee_wei decimal strings to value_eth and tx_fee_eth (float64)",
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": 0,
        "precision_note": (
            "Parsed from decimal strings (with 9 trailing fractional zeros) into IEEE-754 float64 "
            "divided by 1e18 (~15-17 significant decimal digits)."
        ),
    })

    # STEP 3: Missing values & explicit field-availability rules (no gas imputation)
    rows_before = int(len(df))
    df["is_contract_creation"] = df["contract_creation"].fillna(False).astype(bool) | df["dst_id"].isna()
    cc_count = int(df["is_contract_creation"].sum())

    dst_addr_null = df["dst_address"].isna() | (df["dst_address"].astype(str).str.strip() == "")
    df.loc[df["is_contract_creation"] & dst_addr_null, "dst_address"] = CONTRACT_CREATION_PLACEHOLDER

    tx_hash_null = df["tx_hash"].isna() | (df["tx_hash"].astype(str).str.strip() == "")
    src_id_null = df["src_id"].isna()
    dst_id_unflagged_null = df["dst_id"].isna() & (~df["is_contract_creation"])
    val_invalid = df["value_eth"].isna() | (df["value_eth"] < 0.0)
    fee_invalid = df["tx_fee_eth"].isna() | (df["tx_fee_eth"] < 0.0)

    drop_mask_s3 = tx_hash_null | src_id_null | dst_id_unflagged_null | val_invalid | fee_invalid
    dropped_s3 = int(drop_mask_s3.sum())
    if dropped_s3 > 0:
        df = df.loc[~drop_mask_s3].reset_index(drop=True)
    rows_after = int(len(df))

    step_logs.append({
        "step": 3,
        "name": "Missing values handling (is_contract_creation flag, drop null critical fields, zero gas imputation)",
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": dropped_s3,
        "contract_creation_rows_flagged": cc_count,
        "dropped_breakdown": {
            "null_tx_hash": int(tx_hash_null.sum()),
            "null_src_id": int(src_id_null.sum()),
            "null_dst_id_non_contract": int(dst_id_unflagged_null.sum()),
            "null_or_negative_value_eth": int(val_invalid.sum()),
            "null_or_negative_tx_fee_eth": int(fee_invalid.sum()),
        },
    })

    # STEP 4: Drop duplicate tx_hash (keep first; log count even if 0)
    rows_before = int(len(df))
    df["tx_hash"] = df["tx_hash"].astype(str).str.strip().str.lower()
    df = df.drop_duplicates(subset=["tx_hash"], keep="first").reset_index(drop=True)
    rows_after = int(len(df))
    dups_removed = rows_before - rows_after
    step_logs.append({
        "step": 4,
        "name": "Drop duplicate tx_hash (keep first)",
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": dups_removed,
    })

    # STEP 5: Normalize & validate known hexadecimal addresses (leave counterparties as IDs)
    rows_before = int(len(df))
    has_src_addr = df["src_address"].notna()
    df.loc[has_src_addr, "src_address"] = (
        df.loc[has_src_addr, "src_address"].astype(str).str.strip().str.lower()
    )
    bad_src_addr = has_src_addr & (~df["src_address"].str.match(ETH_HEX_REGEX, na=False))

    has_dst_addr = df["dst_address"].notna() & (df["dst_address"] != CONTRACT_CREATION_PLACEHOLDER)
    df.loc[has_dst_addr, "dst_address"] = (
        df.loc[has_dst_addr, "dst_address"].astype(str).str.strip().str.lower()
    )
    bad_dst_addr = has_dst_addr & (~df["dst_address"].str.match(ETH_HEX_REGEX, na=False))

    bad_addr_mask = bad_src_addr | bad_dst_addr
    dropped_bad_addr = int(bad_addr_mask.sum())
    if dropped_bad_addr > 0:
        df = df.loc[~bad_addr_mask].reset_index(drop=True)
    rows_after = int(len(df))

    step_logs.append({
        "step": 5,
        "name": "Lowercase & validate known src_address/dst_address (0x + 40 hex); preserve unlabeled IDs",
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": dropped_bad_addr,
        "malformed_src_address_dropped": int(bad_src_addr.sum()),
        "malformed_dst_address_dropped": int(bad_dst_addr.sum()),
    })

    # STEP 6: Outliers — Winsorize value_eth & tx_fee_eth at p99.9 + log1p columns
    rows_before = int(len(df))
    df["value_eth_uncapped"] = df["value_eth"]
    df["tx_fee_eth_uncapped"] = df["tx_fee_eth"]

    val_p999 = float(df["value_eth"].quantile(WINSOR_Q))
    fee_p999 = float(df["tx_fee_eth"].quantile(WINSOR_Q))

    val_capped_rows = int((df["value_eth"] > val_p999).sum())
    fee_capped_rows = int((df["tx_fee_eth"] > fee_p999).sum())

    df["value_eth"] = np.minimum(df["value_eth"], val_p999)
    df["tx_fee_eth"] = np.minimum(df["tx_fee_eth"], fee_p999)

    df["log1p_value_eth_uncapped"] = np.log1p(df["value_eth_uncapped"])
    df["log1p_value_eth"] = np.log1p(df["value_eth"])
    df["log1p_tx_fee_eth"] = np.log1p(df["tx_fee_eth"])
    rows_after = int(len(df))

    step_logs.append({
        "step": 6,
        "name": "Winsorize value_eth and tx_fee_eth at 99.9th percentile, keep uncapped copies, add log1p",
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": 0,
        "winsorization_quantile": WINSOR_Q,
        "value_eth": {
            "p99_9_threshold_eth": val_p999,
            "rows_affected": val_capped_rows,
            "pct_affected": round(100.0 * val_capped_rows / max(rows_after, 1), 4),
        },
        "tx_fee_eth": {
            "p99_9_threshold_eth": fee_p999,
            "rows_affected": fee_capped_rows,
            "pct_affected": round(100.0 * fee_capped_rows / max(rows_after, 1), 4),
        },
    })

    # STEP 7: Attach endpoint labels (src_is_scam, dst_is_scam, src_is_sampled, dst_is_sampled)
    rows_before = int(len(df))
    w_map = dict(
        zip(
            wallets_df["node_id"].astype("uint64"),
            wallets_df["is_scam"].astype("int8"),
        )
    )
    sampled_node_set = set(w_map.keys())

    df["src_is_sampled"] = df["src_id"].isin(sampled_node_set)
    df["dst_is_sampled"] = df["dst_id"].isin(sampled_node_set)

    df["src_is_scam"] = df["src_id"].map(w_map).astype("Int8")
    df["dst_is_scam"] = df["dst_id"].map(w_map).astype("Int8")

    touches_sampled = df["src_is_sampled"] | df["dst_is_sampled"]
    dropped_unattached = int((~touches_sampled).sum())
    if dropped_unattached > 0:
        df = df.loc[touches_sampled].reset_index(drop=True)
    rows_after = int(len(df))

    tx_touching_fraud = int(((df["src_is_scam"] == 1) | (df["dst_is_scam"] == 1)).fillna(False).sum())
    tx_touching_normal = int(((df["src_is_scam"] == 0) | (df["dst_is_scam"] == 0)).fillna(False).sum())

    step_logs.append({
        "step": 7,
        "name": "Attach endpoint labels (src_is_scam, dst_is_scam, src_is_sampled, dst_is_sampled)",
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": dropped_unattached,
        "transactions_touching_sampled_fraud": tx_touching_fraud,
        "transactions_touching_sampled_normal": tx_touching_normal,
    })

    # STEP 8: Final Integrity Assertions
    assert not df["tx_hash"].duplicated().any(), "ASSERTION FAILED: Duplicate tx_hash present!"
    assert (df["value_eth"] >= 0.0).all(), "ASSERTION FAILED: Negative value_eth present!"
    assert (df["tx_fee_eth"] >= 0.0).all(), "ASSERTION FAILED: Negative tx_fee_eth present!"
    assert (df["log1p_value_eth"] >= 0.0).all(), "ASSERTION FAILED: Negative log1p_value_eth present!"
    assert (df["log1p_tx_fee_eth"] >= 0.0).all(), "ASSERTION FAILED: Negative log1p_tx_fee_eth present!"

    key_non_null_cols = [
        "tx_hash", "src_id", "ts", "timestamp_utc", "value_eth",
        "tx_fee_eth", "block_number", "is_contract_creation",
        "src_is_sampled", "dst_is_sampled", "source_month",
    ]
    for kc in key_non_null_cols:
        assert df[kc].notna().all(), f"ASSERTION FAILED: Nulls found in key column `{kc}`!"

    assert (df["src_is_sampled"] | df["dst_is_sampled"]).all(), (
        "ASSERTION FAILED: Every row must have at least one sampled endpoint!"
    )
    observed_classes = set(df["src_is_scam"].dropna().unique()) | set(df["dst_is_scam"].dropna().unique())
    assert observed_classes == {0, 1}, (
        f"ASSERTION FAILED: Both classes {{0, 1}} must be present, found {observed_classes}!"
    )

    after_stats = {
        "value_eth": compute_series_stats(df["value_eth"]),
        "tx_fee_eth": compute_series_stats(df["tx_fee_eth"]),
        "log1p_value_eth": compute_series_stats(df["log1p_value_eth"]),
        "log1p_tx_fee_eth": compute_series_stats(df["log1p_tx_fee_eth"]),
    }

    step_logs.append({
        "step": 8,
        "name": "Final assertions (no duplicate tx_hash, no negatives, no nulls in key cols, both classes present)",
        "rows_before": rows_after,
        "rows_after": rows_after,
        "rows_removed": 0,
        "assertions_passed": True,
    })

    clean_tx_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False),
        clean_tx_path,
        compression="snappy",
    )

    report = {
        "cleaned_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_raw_parquet": str(raw_tx_path),
        "input_sampled_wallets": str(sampled_wallets_path),
        "output_clean_parquet": str(clean_tx_path),
        "initial_row_count": initial_rows,
        "final_row_count": rows_after,
        "total_rows_removed": initial_rows - rows_after,
        "field_mapping_table": FIELD_MAPPING_TABLE,
        "step_by_step_log": step_logs,
        "summary_statistics_before_winsorization": before_stats,
        "summary_statistics_after_winsorization": after_stats,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved %d clean rows to %s and report to %s", rows_after, clean_tx_path, report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 Step 2: Deterministic 8-Step Ethereum Data Cleaning")
    parser.add_argument("--raw-tx", type=Path, default=Path("data/raw/raw_transactions.parquet"))
    parser.add_argument("--wallets", type=Path, default=Path("data/raw/sampled_wallets.csv"))
    parser.add_argument("--clean-tx", type=Path, default=Path("data/clean/clean_transactions.parquet"))
    parser.add_argument("--report", type=Path, default=Path("data/clean/cleaning_report.json"))
    args = parser.parse_args()

    run_cleaning_pipeline(
        raw_tx_path=args.raw_tx,
        sampled_wallets_path=args.wallets,
        clean_tx_path=args.clean_tx,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()