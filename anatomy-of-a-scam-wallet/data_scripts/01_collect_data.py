#!/usr/bin/env python3
"""
Phase 2 — Deliverable 1 (File 1): Best-Months Ethereum Transaction Collection Pipeline
Project: Anatomy of a Scam Wallet (Ethereum Fraud Detection with Random Forest & XGBoost)

Dataset:
  - Repository: fesevu/ethereum_fraud_dataset_by_activity (Hugging Face Hub)
  - DOI: 10.57967/hf/6373 | License: CC-BY-4.0
  - Labels: gnn_dataset/labels/targets_global.parquet (114,739 rows: node_id uint64,
    is_scam int8, is_contract int8, address string)
  - Monthly Activity Index: gnn_dataset/targets/month_targets.parquet (269,291 rows:
    month string "YYYY-MM", node_id uint64)
  - Monthly Edge Shards: gnn_dataset/edges_by_month/month=YYYY-MM/edges.parquet
    (121 months, 2015-08 to 2025-08; columns: src_id uint64, dst_id uint64, ts int64,
    value_wei large_string, tx_fee_wei large_string, block_number int64,
    contract_creation bool, tx_hash large_string)

Strict Guarantees:
  - ZERO synthetic data or fallback mirrors: fails loudly if the Hub API, expected files,
    or required columns are missing or unreachable.
  - Windows + PowerShell + Python 3.13 + huggingface_hub 2.x compatible (no symlinks).
  - Peak RAM < 4 GB: never loads a full month shard into pandas; uses DuckDB
    with column projection and predicate pushdown.
  - Exact Uncapped Window Statistics: before deleting each downloaded month shard, computes
    exact uncapped per-candidate statistics (`sent_cnt`, `recv_cnt`, `first_ts`, `last_ts`,
    `value_sent_eth_sum`, `value_recv_eth_sum`) via separate `src_id` and `dst_id` inner-join
    aggregations (avoiding slow OR-joins) into `data/raw/_partial/stats_month=YYYY-MM.parquet`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import re
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import duckdb
except ImportError as exc:
    raise ImportError(
        "CRITICAL: `duckdb` is required for out-of-core Parquet filtering and window "
        "functions (< 4 GB RAM). Install it via: pip install duckdb"
    ) from exc

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError as exc:
    raise ImportError(
        "CRITICAL: `huggingface_hub` is required. Install it via: pip install huggingface_hub"
    ) from exc

# ============================================================================
# Constants & Verified Schema Definitions
# ============================================================================
REPO_ID = "fesevu/ethereum_fraud_dataset_by_activity"
REPO_URL = f"https://huggingface.co/datasets/{REPO_ID}"
DOI = "10.57967/hf/6373"
LICENSE = "CC-BY-4.0"

LABELS_REPO_PATH = "gnn_dataset/labels/targets_global.parquet"
MONTH_TARGETS_REPO_PATH = "gnn_dataset/targets/month_targets.parquet"
EDGES_PREFIX = "gnn_dataset/edges_by_month/"

EXPECTED_DEFAULT_19_MONTHS = [
    "2016-04", "2019-12", "2020-01", "2021-12", "2022-11",
    "2023-01", "2023-02", "2023-03", "2023-04", "2023-06",
    "2023-11", "2024-01", "2024-03", "2024-04", "2024-05",
    "2024-06", "2024-07", "2024-09", "2024-12",
]

REQUIRED_LABEL_COLS = {"node_id", "is_scam", "is_contract", "address"}
REQUIRED_MONTH_TARGET_COLS = {"month", "node_id"}
REQUIRED_EDGE_COLS = {
    "src_id", "dst_id", "ts", "value_wei", "tx_fee_wei",
    "block_number", "contract_creation", "tx_hash",
}

MAX_FRAUD_CANDIDATES = 4_000
MAX_NORMAL_CANDIDATES = 8_000
MAX_PARTIAL_ROWS_PER_WALLET_MONTH = 200
MIN_TX_ELIGIBLE = 5
MAX_FINAL_PER_CLASS_PER_CONTRACT_GROUP = 1_500
HARD_ROW_CEILING = 6_000_000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("01_collect_data")


# ============================================================================
# Utilities: Exponential Backoff, SHA-256, & Hub Download (No Symlinks)
# ============================================================================
def retry_with_backoff(func, *args, max_attempts: int = 5, base_delay: float = 2.5, **kwargs):
    """Execute a network/IO operation with exponential backoff."""
    for attempt in range(1, max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            if attempt == max_attempts:
                logger.error("Operation failed permanently after %d attempts: %s", max_attempts, exc)
                raise
            wait_s = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "Attempt %d/%d failed (%s). Retrying in %.1fs...",
                attempt, max_attempts, exc, wait_s,
            )
            time.sleep(wait_s)


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file in 8 MB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_hub_file(repo_path: str, local_dir: Path, revision: Optional[str] = None) -> Path:
    """
    Download a single file from Hugging Face Hub into `local_dir` without symlinks
    (compatible with Windows PowerShell without Developer Mode and huggingface_hub 2.x).
    """
    local_dir.mkdir(parents=True, exist_ok=True)

    def _dl() -> str:
        return hf_hub_download(
            repo_id=REPO_ID,
            filename=repo_path,
            repo_type="dataset",
            revision=revision,
            local_dir=str(local_dir),
        )

    out_path = Path(retry_with_backoff(_dl))
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file {out_path} is missing or empty.")
    return out_path


# ============================================================================
# Step 1: Query Hub API Metadata & Download Labels + Month Targets
# ============================================================================
def fetch_repo_metadata(api: HfApi) -> Tuple[str, Dict[str, int]]:
    """
    Query Hugging Face Hub API for dataset revision SHA and exact byte sizes of all
    files under `gnn_dataset/edges_by_month/month=YYYY-MM/edges.parquet`.
    Fails loudly if unreachable or if expected folders are missing.
    """
    logger.info("Querying Hugging Face Hub API for %s...", REPO_ID)
    ds_info = retry_with_backoff(api.dataset_info, repo_id=REPO_ID, files_metadata=True)
    revision_sha = getattr(ds_info, "sha", None)
    if not revision_sha:
        raise RuntimeError(f"Could not retrieve commit SHA for dataset {REPO_ID}.")

    file_sizes: Dict[str, int] = {}
    siblings = getattr(ds_info, "siblings", None) or []
    for sib in siblings:
        rfilename = getattr(sib, "rfilename", "")
        size = getattr(sib, "size", None)
        if size is not None:
            file_sizes[rfilename] = int(size)

    if not any(k.startswith(EDGES_PREFIX) for k in file_sizes):
        tree_items = retry_with_backoff(
            api.list_repo_tree, repo_id=REPO_ID, repo_type="dataset", recursive=True
        )
        for item in tree_items:
            path_str = getattr(item, "path", getattr(item, "rfilename", ""))
            size_val = getattr(item, "size", None)
            if path_str and size_val is not None:
                file_sizes[path_str] = int(size_val)

    month_files = {
        k: v for k, v in file_sizes.items()
        if k.startswith(EDGES_PREFIX) and k.endswith("/edges.parquet")
    }
    if not month_files:
        raise RuntimeError(
            f"No monthly edge files found under {EDGES_PREFIX} in {REPO_ID}. "
            "Refusing to guess or generate synthetic data."
        )

    logger.info(
        "Verified dataset revision SHA=%s | Found %d monthly edge shards (%.2f MB to %.2f MB).",
        revision_sha,
        len(month_files),
        min(month_files.values()) / 1e6,
        max(month_files.values()) / 1e6,
    )
    return revision_sha, file_sizes


def verify_parquet_schema(path: Path, required_cols: Set[str], label: str) -> List[str]:
    """Read Parquet footer schema and fail loudly if any required column is missing."""
    pf = pq.ParquetFile(path)
    actual_cols = [field.name for field in pf.schema_arrow]
    missing = required_cols - set(actual_cols)
    if missing:
        raise RuntimeError(
            f"SCHEMA ERROR in {label} ({path}): missing required columns {sorted(missing)}. "
            f"Actual columns: {actual_cols}"
        )
    return actual_cols


# ============================================================================
# Step 2: Greedy Month Plan (Unique Scam Wallets per MB)
# ============================================================================
def compute_greedy_month_plan(
    labels_df: pd.DataFrame,
    month_targets_df: pd.DataFrame,
    file_sizes: Dict[str, int],
    max_gb: float,
    override_months: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute greedy month selection maximizing marginal unique scam wallets covered per MB
    within `max_gb`. Compares the selected list against the 19-month reference plan.
    """
    scam_nodes: Set[int] = set(
        labels_df.loc[labels_df["is_scam"] == 1, "node_id"].astype("uint64").tolist()
    )
    normal_nodes: Set[int] = set(
        labels_df.loc[labels_df["is_scam"] == 0, "node_id"].astype("uint64").tolist()
    )

    mt = month_targets_df[["month", "node_id"]].copy()
    mt["node_id"] = mt["node_id"].astype("uint64")
    mt["month"] = mt["month"].astype(str)

    month_scam_sets: Dict[str, Set[int]] = {}
    month_normal_sets: Dict[str, Set[int]] = {}
    for month_str, grp in mt.groupby("month", sort=True):
        nodes = set(grp["node_id"].tolist())
        month_scam_sets[month_str] = nodes & scam_nodes
        month_normal_sets[month_str] = nodes & normal_nodes

    month_size_bytes: Dict[str, int] = {}
    for repo_path, size_b in file_sizes.items():
        m = re.match(r"^gnn_dataset/edges_by_month/month=(\d{4}-\d{2})/edges\.parquet$", repo_path)
        if m:
            month_size_bytes[m.group(1)] = int(size_b)

    available_months = sorted(set(month_scam_sets.keys()) & set(month_size_bytes.keys()))
    if not available_months:
        raise RuntimeError("No overlap between month_targets months and edges_by_month files.")

    sanity_2017_06 = {
        "scam_wallets": len(month_scam_sets.get("2017-06", set())),
        "normal_wallets": len(month_normal_sets.get("2017-06", set())),
    }
    logger.info(
        "Sanity check for month 2017-06 in month_targets: %d scam / %d normal wallets.",
        sanity_2017_06["scam_wallets"], sanity_2017_06["normal_wallets"],
    )

    max_bytes = int(max_gb * (1024 ** 3))
    selected_steps: List[Dict[str, Any]] = []
    covered_scam: Set[int] = set()
    covered_normal: Set[int] = set()
    cumulative_bytes = 0

    if override_months:
        for m_str in override_months:
            if m_str not in month_size_bytes:
                raise ValueError(f"Requested month {m_str} does not exist in {EDGES_PREFIX}.")
            s_bytes = month_size_bytes[m_str]
            s_mb = s_bytes / (1024 ** 2)
            new_scam = month_scam_sets.get(m_str, set()) - covered_scam
            covered_scam |= month_scam_sets.get(m_str, set())
            covered_normal |= month_normal_sets.get(m_str, set())
            cumulative_bytes += s_bytes
            selected_steps.append({
                "step": len(selected_steps) + 1,
                "month": m_str,
                "size_bytes": s_bytes,
                "size_mb": round(s_mb, 2),
                "total_scam_in_month": len(month_scam_sets.get(m_str, set())),
                "marginal_new_scam": len(new_scam),
                "normal_in_month": len(month_normal_sets.get(m_str, set())),
                "marginal_scam_per_mb": round(len(new_scam) / max(s_mb, 1e-6), 4),
                "cumulative_scam_covered": len(covered_scam),
                "cumulative_normal_covered": len(covered_normal),
                "cumulative_gb": round(cumulative_bytes / (1024 ** 3), 4),
            })
    else:
        remaining = set(available_months)
        while remaining:
            best_month = None
            best_ratio = -1.0
            best_new_scam = 0

            for m_str in sorted(remaining):
                s_bytes = month_size_bytes[m_str]
                if cumulative_bytes + s_bytes > max_bytes:
                    continue
                new_scam_count = len(month_scam_sets[m_str] - covered_scam)
                if new_scam_count == 0:
                    continue
                s_mb = s_bytes / (1024 ** 2)
                ratio = new_scam_count / max(s_mb, 1e-6)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_month = m_str
                    best_new_scam = new_scam_count

            if best_month is None:
                break

            remaining.remove(best_month)
            s_bytes = month_size_bytes[best_month]
            s_mb = s_bytes / (1024 ** 2)
            covered_scam |= month_scam_sets[best_month]
            covered_normal |= month_normal_sets[best_month]
            cumulative_bytes += s_bytes

            selected_steps.append({
                "step": len(selected_steps) + 1,
                "month": best_month,
                "size_bytes": s_bytes,
                "size_mb": round(s_mb, 2),
                "total_scam_in_month": len(month_scam_sets[best_month]),
                "marginal_new_scam": best_new_scam,
                "normal_in_month": len(month_normal_sets[best_month]),
                "marginal_scam_per_mb": round(best_ratio, 4),
                "cumulative_scam_covered": len(covered_scam),
                "cumulative_normal_covered": len(covered_normal),
                "cumulative_gb": round(cumulative_bytes / (1024 ** 3), 4),
            })

    selected_months_sorted = sorted(step["month"] for step in selected_steps)
    in_selected_not_ref = sorted(set(selected_months_sorted) - set(EXPECTED_DEFAULT_19_MONTHS))
    in_ref_not_selected = sorted(set(EXPECTED_DEFAULT_19_MONTHS) - set(selected_months_sorted))

    if not override_months:
        if selected_months_sorted == EXPECTED_DEFAULT_19_MONTHS:
            logger.info("Greedy month plan matches the 19-month reference list identically.")
        else:
            logger.info(
                "Greedy month plan comparison vs. 19-month reference:\n"
                "  Selected (%d): %s\n"
                "  In selected but not reference: %s\n"
                "  In reference but not selected: %s",
                len(selected_months_sorted),
                selected_months_sorted,
                in_selected_not_ref,
                in_ref_not_selected,
            )

    return {
        "max_gb_budget": max_gb,
        "override_used": bool(override_months),
        "selected_months_chronological": selected_months_sorted,
        "selected_months_greedy_order": [s["month"] for s in selected_steps],
        "num_months_selected": len(selected_steps),
        "total_download_bytes": cumulative_bytes,
        "total_download_gb": round(cumulative_bytes / (1024 ** 3), 4),
        "distinct_scam_wallets_covered": len(covered_scam),
        "distinct_normal_wallets_covered": len(covered_normal),
        "sanity_check_2017_06": sanity_2017_06,
        "comparison_with_reference_19_months": {
            "exact_match": selected_months_sorted == EXPECTED_DEFAULT_19_MONTHS,
            "in_selected_not_reference": in_selected_not_ref,
            "in_reference_not_selected": in_ref_not_selected,
        },
        "greedy_schedule": selected_steps,
    }


# ============================================================================
# Step 3: Build Candidate Pools Before Downloading Edges (Seed = 42)
# ============================================================================
def build_candidate_pools(
    labels_df: pd.DataFrame,
    month_targets_df: pd.DataFrame,
    selected_months: List[str],
    seed: int,
) -> pd.DataFrame:
    """
    Select candidate wallets active in `selected_months` prior to downloading edges:
      - Fraud candidates: active scam wallets in selected months, capped at 4,000.
      - Normal candidates: active normal wallets in the SAME months, capped at 8,000,
        drawn proportionally by month so temporal representation matches.
    """
    rng = np.random.default_rng(seed)
    mt_sel = month_targets_df[month_targets_df["month"].isin(selected_months)].copy()
    mt_sel["node_id"] = mt_sel["node_id"].astype("uint64")

    lbl = labels_df[["node_id", "is_scam", "is_contract", "address"]].copy()
    lbl["node_id"] = lbl["node_id"].astype("uint64")

    merged = mt_sel.merge(lbl, on="node_id", how="inner")

    fraud_unique = (
        merged[merged["is_scam"] == 1]
        .drop_duplicates(subset=["node_id"])
        .sort_values("node_id")
        .reset_index(drop=True)
    )
    if len(fraud_unique) > MAX_FRAUD_CANDIDATES:
        chosen_idx = rng.choice(len(fraud_unique), size=MAX_FRAUD_CANDIDATES, replace=False)
        fraud_candidates = fraud_unique.iloc[np.sort(chosen_idx)].copy()
    else:
        fraud_candidates = fraud_unique.copy()

    normal_month_df = (
        merged[merged["is_scam"] == 0]
        .sort_values(["month", "node_id"])
        .reset_index(drop=True)
    )
    normal_unique_total = normal_month_df["node_id"].nunique()
    target_normal = min(normal_unique_total, MAX_NORMAL_CANDIDATES)

    month_counts = normal_month_df.groupby("month")["node_id"].nunique().sort_index()
    total_month_appearances = int(month_counts.sum())

    chosen_normal_ids: Set[int] = set()
    months_sorted = list(month_counts.index)

    for m_str in months_sorted:
        m_nodes = normal_month_df.loc[normal_month_df["month"] == m_str, "node_id"].unique()
        avail_nodes = np.array([nid for nid in m_nodes if int(nid) not in chosen_normal_ids], dtype="uint64")
        if len(avail_nodes) == 0:
            continue
        quota = int(round(target_normal * (len(m_nodes) / max(total_month_appearances, 1))))
        quota = min(quota, len(avail_nodes), target_normal - len(chosen_normal_ids))
        if quota > 0:
            picks = rng.choice(avail_nodes, size=quota, replace=False)
            chosen_normal_ids.update(int(x) for x in picks)

    if len(chosen_normal_ids) < target_normal:
        all_normal_unique = normal_month_df["node_id"].unique()
        rem_nodes = np.array([nid for nid in all_normal_unique if int(nid) not in chosen_normal_ids], dtype="uint64")
        need = min(target_normal - len(chosen_normal_ids), len(rem_nodes))
        if need > 0:
            picks = rng.choice(rem_nodes, size=need, replace=False)
            chosen_normal_ids.update(int(x) for x in picks)

    normal_candidates = (
        lbl[lbl["node_id"].isin(chosen_normal_ids)]
        .sort_values("node_id")
        .reset_index(drop=True)
    )

    candidates_df = pd.concat([fraud_candidates[lbl.columns], normal_candidates[lbl.columns]], ignore_index=True)
    logger.info(
        "Candidate pool built (seed=%d): %d fraud candidates + %d normal candidates = %d total.",
        seed, len(fraud_candidates), len(normal_candidates), len(candidates_df),
    )
    return candidates_df


# ============================================================================
# Step 4: Per-Month Download, Uncapped Stats, Window Filtering, & Cache Cleanup
# ============================================================================
def process_selected_months(
    selected_months: List[str],
    candidates_df: pd.DataFrame,
    raw_dir: Path,
    revision_sha: str,
    seed: int,
    per_wallet_month_cap: int = MAX_PARTIAL_ROWS_PER_WALLET_MONTH,
) -> Dict[str, Any]:
    """
    Download each selected month shard one at a time:
      1. BEFORE deleting the shard, compute exact uncapped per-candidate statistics over
         the full shard (`sent_cnt`, `recv_cnt`, `first_ts`, `last_ts`, `value_sent_eth_sum`,
         `value_recv_eth_sum`) by aggregating `src_id` and `dst_id` separately (avoiding
         OR-joins) and writing `data/raw/_partial/stats_month=YYYY-MM.parquet`.
      2. Filter and keep at most `per_wallet_month_cap` rows per candidate wallet per month
         deterministically via `hash(tx_hash || seed)` into `data/raw/_partial/month=YYYY-MM.parquet`.
      3. Record both partial and stats files (paths, bytes, SHA-256) in the checkpoint JSON,
         then delete the cached raw shard immediately.
      4. On resume, if a month is in the checkpoint but either `month=YYYY-MM.parquet` or
         `stats_month=YYYY-MM.parquet` is missing on disk, redo that month.
    """
    partial_dir = raw_dir / "_partial"
    cache_dir = raw_dir / "_hf_download_cache"
    checkpoint_path = raw_dir / "_collection_checkpoint.json"
    partial_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            state = json.load(f)
    else:
        state = {"completed_months": {}, "total_bytes_downloaded": 0}

    cand_parquet = partial_dir / "_candidates.parquet"
    pq.write_table(
        pa.Table.from_pandas(candidates_df[["node_id", "is_scam", "is_contract"]], preserve_index=False),
        cand_parquet,
    )

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA memory_limit='2.5GB';")
    con.execute("PRAGMA threads=4;")
    con.execute(
        f"CREATE TABLE candidates AS SELECT CAST(node_id AS UBIGINT) AS node_id FROM read_parquet('{cand_parquet.as_posix()}');"
    )

    for idx, month_str in enumerate(selected_months, start=1):
        out_partial = partial_dir / f"month={month_str}.parquet"
        out_stats = partial_dir / f"stats_month={month_str}.parquet"
        ckpt_entry = state["completed_months"].get(month_str)

        if (
            ckpt_entry is not None
            and out_partial.exists()
            and out_stats.exists()
            and ckpt_entry.get("stats_parquet_path")
        ):
            logger.info(
                "[%d/%d] Skipping month=%s (already checkpointed: %d edge rows, %d active candidate stats rows).",
                idx, len(selected_months), month_str,
                ckpt_entry.get("rows_kept", 0),
                ckpt_entry.get("stats_rows_kept", 0),
            )
            continue

        if ckpt_entry is not None and (not out_partial.exists() or not out_stats.exists()):
            logger.warning(
                "[%d/%d] Month %s is in checkpoint but partial or stats file is missing; redoing month.",
                idx, len(selected_months), month_str,
            )

        repo_file = f"{EDGES_PREFIX}month={month_str}/edges.parquet"
        t0 = time.perf_counter()
        logger.info("[%d/%d] Downloading %s ...", idx, len(selected_months), repo_file)

        downloaded_path = download_hub_file(repo_file, local_dir=cache_dir, revision=revision_sha)
        dl_elapsed = max(time.perf_counter() - t0, 1e-6)
        dl_bytes = downloaded_path.stat().st_size
        dl_mb = dl_bytes / (1024 ** 2)
        dl_speed = dl_mb / dl_elapsed

        verify_parquet_schema(downloaded_path, REQUIRED_EDGE_COLS, repo_file)
        shard_sha256 = compute_sha256(downloaded_path)

        # 1. Compute EXACT uncapped per-candidate stats over the full month shard
        #    Aggregate src_id and dst_id separately via hash joins (no OR-join)
        stats_query = f"""
        COPY (
            WITH src_agg AS (
                SELECT
                    CAST(e.src_id AS UBIGINT) AS node_id,
                    CAST(COUNT(*) AS BIGINT) AS sent_cnt,
                    MIN(CAST(e.ts AS BIGINT)) AS src_first_ts,
                    MAX(CAST(e.ts AS BIGINT)) AS src_last_ts,
                    COALESCE(SUM(TRY_CAST(e.value_wei AS DOUBLE) / 1e18), 0.0) AS value_sent_eth_sum
                FROM read_parquet('{downloaded_path.as_posix()}') e
                INNER JOIN candidates c ON CAST(e.src_id AS UBIGINT) = c.node_id
                GROUP BY CAST(e.src_id AS UBIGINT)
            ),
            dst_agg AS (
                SELECT
                    CAST(e.dst_id AS UBIGINT) AS node_id,
                    CAST(COUNT(*) AS BIGINT) AS recv_cnt,
                    MIN(CAST(e.ts AS BIGINT)) AS dst_first_ts,
                    MAX(CAST(e.ts AS BIGINT)) AS dst_last_ts,
                    COALESCE(SUM(TRY_CAST(e.value_wei AS DOUBLE) / 1e18), 0.0) AS value_recv_eth_sum
                FROM read_parquet('{downloaded_path.as_posix()}') e
                INNER JOIN candidates c ON CAST(e.dst_id AS UBIGINT) = c.node_id
                GROUP BY CAST(e.dst_id AS UBIGINT)
            )
            SELECT
                COALESCE(s.node_id, d.node_id) AS node_id,
                '{month_str}' AS source_month,
                COALESCE(s.sent_cnt, 0) AS sent_cnt,
                COALESCE(d.recv_cnt, 0) AS recv_cnt,
                CASE
                    WHEN s.src_first_ts IS NULL THEN d.dst_first_ts
                    WHEN d.dst_first_ts IS NULL THEN s.src_first_ts
                    ELSE LEAST(s.src_first_ts, d.dst_first_ts)
                END AS first_ts,
                CASE
                    WHEN s.src_last_ts IS NULL THEN d.dst_last_ts
                    WHEN d.dst_last_ts IS NULL THEN s.src_last_ts
                    ELSE GREATEST(s.src_last_ts, d.dst_last_ts)
                END AS last_ts,
                COALESCE(s.value_sent_eth_sum, 0.0) AS value_sent_eth_sum,
                COALESCE(d.value_recv_eth_sum, 0.0) AS value_recv_eth_sum
            FROM src_agg s
            FULL OUTER JOIN dst_agg d ON s.node_id = d.node_id
        ) TO '{out_stats.as_posix()}' (FORMAT PARQUET, COMPRESSION SNAPPY);
        """
        con.execute(stats_query)
        stats_rows_kept = pq.ParquetFile(out_stats).metadata.num_rows
        stats_sha256 = compute_sha256(out_stats)
        stats_bytes = out_stats.stat().st_size

        # 2. Extract capped transaction sample for this month (avoiding OR-join by UNION ALL on src/dst)
        edges_query = f"""
        COPY (
            WITH matched_edges AS (
                SELECT
                    e.tx_hash,
                    CAST(e.src_id AS UBIGINT) AS src_id,
                    CAST(e.dst_id AS UBIGINT) AS dst_id,
                    CAST(e.ts AS BIGINT) AS ts,
                    CAST(e.value_wei AS VARCHAR) AS value_wei,
                    CAST(e.tx_fee_wei AS VARCHAR) AS tx_fee_wei,
                    CAST(e.block_number AS BIGINT) AS block_number,
                    CAST(e.contract_creation AS BOOLEAN) AS contract_creation,
                    '{month_str}' AS source_month,
                    cs.node_id AS cand_src,
                    cd.node_id AS cand_dst,
                    hash(e.tx_hash || '_{seed}') AS det_hash
                FROM read_parquet('{downloaded_path.as_posix()}') e
                LEFT JOIN candidates cs ON CAST(e.src_id AS UBIGINT) = cs.node_id
                LEFT JOIN candidates cd ON CAST(e.dst_id AS UBIGINT) = cd.node_id
                WHERE cs.node_id IS NOT NULL OR cd.node_id IS NOT NULL
            ),
            wallet_tagged AS (
                SELECT
                    tx_hash, src_id, dst_id, ts, value_wei, tx_fee_wei,
                    block_number, contract_creation, source_month, det_hash,
                    COALESCE(cand_src, cand_dst) AS primary_cand_id
                FROM matched_edges
            ),
            ranked AS (
                SELECT
                    tx_hash, src_id, dst_id, ts, value_wei, tx_fee_wei,
                    block_number, contract_creation, source_month,
                    ROW_NUMBER() OVER (
                        PARTITION BY primary_cand_id
                        ORDER BY det_hash ASC, tx_hash ASC
                    ) AS rn
                FROM wallet_tagged
            )
            SELECT
                tx_hash, src_id, dst_id, ts, value_wei, tx_fee_wei,
                block_number, contract_creation, source_month
            FROM ranked
            WHERE rn <= {int(per_wallet_month_cap)}
        ) TO '{out_partial.as_posix()}' (FORMAT PARQUET, COMPRESSION SNAPPY);
        """
        con.execute(edges_query)
        rows_kept = pq.ParquetFile(out_partial).metadata.num_rows
        partial_sha256 = compute_sha256(out_partial)
        partial_bytes = out_partial.stat().st_size

        # 3. Delete cached raw shard immediately to save disk space
        try:
            downloaded_path.unlink(missing_ok=True)
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)
        except Exception as cleanup_exc:
            logger.warning("Could not remove temporary cache %s: %s", downloaded_path, cleanup_exc)

        state["completed_months"][month_str] = {
            "repo_path": repo_file,
            "downloaded_bytes": dl_bytes,
            "downloaded_mb": round(dl_mb, 2),
            "download_seconds": round(dl_elapsed, 2),
            "download_speed_mb_s": round(dl_speed, 2),
            "downloaded_sha256": shard_sha256,
            "partial_parquet_path": str(out_partial),
            "partial_bytes": partial_bytes,
            "partial_sha256": partial_sha256,
            "rows_kept": int(rows_kept),
            "stats_parquet_path": str(out_stats),
            "stats_bytes": stats_bytes,
            "stats_sha256": stats_sha256,
            "stats_rows_kept": int(stats_rows_kept),
        }
        state["total_bytes_downloaded"] = sum(
            m["downloaded_bytes"] for m in state["completed_months"].values()
        )
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        logger.info(
            "[%d/%d] Finished month=%s | %.2f MB @ %.2f MB/s | Kept %d edge rows & %d exact stats rows",
            idx, len(selected_months), month_str, dl_mb, dl_speed, rows_kept, stats_rows_kept,
        )

    con.close()
    return state


# ============================================================================
# Step 5 & 6: Confound-Controlled Wallet Selection & Final raw_transactions.parquet
# ============================================================================
def finalize_sample_and_transactions(
    candidates_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    raw_dir: Path,
    selected_months: List[str],
    max_tx_per_wallet: int,
    balance_contract: bool,
    normal_ratio: float,
    seed: int,
) -> Dict[str, Any]:
    """
    1. Aggregate exact uncapped per-candidate statistics across all `stats_month=YYYY-MM.parquet`
       files using DuckDB (`window_sent_cnt`, `window_recv_cnt`, `window_first_ts`,
       `window_last_ts`, `window_value_sent_eth`, `window_value_recv_eth`).
    2. Filter to eligible wallets where exact uncapped `window_sent_cnt + window_recv_cnt >= 5`.
    3. Within each `is_contract` stratum (0 and 1), sample balanced fraud and normal wallets
       and save `data/raw/sampled_wallets.csv` including the 6 exact window columns.
    4. Build `data/raw/raw_transactions.parquet` from the partial edge files (capped at
       `--max-tx-per-wallet` per sampled wallet).
    """
    partial_dir = raw_dir / "_partial"
    partial_glob = [
        (partial_dir / f"month={m}.parquet").as_posix()
        for m in selected_months
        if (partial_dir / f"month={m}.parquet").exists()
    ]
    stats_glob = [
        (partial_dir / f"stats_month={m}.parquet").as_posix()
        for m in selected_months
        if (partial_dir / f"stats_month={m}.parquet").exists()
    ]
    if not partial_glob or len(stats_glob) != len(partial_glob):
        raise RuntimeError(
            "Missing partial edge or stats_month Parquet files in data/raw/_partial/. "
            "Ensure all selected months completed Step 4."
        )

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA memory_limit='2.5GB';")
    con.execute("PRAGMA threads=4;")

    cand_parquet = partial_dir / "_candidates.parquet"
    pq.write_table(
        pa.Table.from_pandas(candidates_df, preserve_index=False),
        cand_parquet,
    )

    # Aggregate EXACT uncapped window statistics per candidate wallet across all selected months
    tx_counts_df = con.execute(f"""
        WITH combined_stats AS (
            SELECT
                CAST(node_id AS UBIGINT) AS node_id,
                CAST(SUM(sent_cnt) AS BIGINT) AS window_sent_cnt,
                CAST(SUM(recv_cnt) AS BIGINT) AS window_recv_cnt,
                MIN(first_ts) AS window_first_ts,
                MAX(last_ts) AS window_last_ts,
                COALESCE(SUM(value_sent_eth_sum), 0.0) AS window_value_sent_eth,
                COALESCE(SUM(value_recv_eth_sum), 0.0) AS window_value_recv_eth
            FROM read_parquet({stats_glob})
            GROUP BY CAST(node_id AS UBIGINT)
        )
        SELECT
            c.node_id,
            c.address,
            c.is_scam,
            c.is_contract,
            COALESCE(s.window_sent_cnt, 0) AS window_sent_cnt,
            COALESCE(s.window_recv_cnt, 0) AS window_recv_cnt,
            s.window_first_ts,
            s.window_last_ts,
            COALESCE(s.window_value_sent_eth, 0.0) AS window_value_sent_eth,
            COALESCE(s.window_value_recv_eth, 0.0) AS window_value_recv_eth,
            COALESCE(s.window_sent_cnt, 0) + COALESCE(s.window_recv_cnt, 0) AS window_tx_count
        FROM read_parquet('{cand_parquet.as_posix()}') c
        LEFT JOIN combined_stats s ON CAST(c.node_id AS UBIGINT) = s.node_id
        ORDER BY c.node_id
    """).df()

    # Eligibility check uses exact uncapped summed counts (window_sent_cnt + window_recv_cnt >= 5)
    eligible_df = tx_counts_df[
        (tx_counts_df["window_sent_cnt"] + tx_counts_df["window_recv_cnt"]) >= MIN_TX_ELIGIBLE
    ].copy()

    stage_drops = {
        "initial_candidates_total": int(len(tx_counts_df)),
        "initial_fraud_candidates": int((tx_counts_df["is_scam"] == 1).sum()),
        "initial_normal_candidates": int((tx_counts_df["is_scam"] == 0).sum()),
        "dropped_below_5_txs_total": int(len(tx_counts_df) - len(eligible_df)),
        "dropped_below_5_txs_fraud": int(
            (tx_counts_df["is_scam"] == 1).sum() - (eligible_df["is_scam"] == 1).sum()
        ),
        "dropped_below_5_txs_normal": int(
            (tx_counts_df["is_scam"] == 0).sum() - (eligible_df["is_scam"] == 0).sum()
        ),
        "eligible_ge_5_txs_total": int(len(eligible_df)),
        "eligible_fraud": int((eligible_df["is_scam"] == 1).sum()),
        "eligible_normal": int((eligible_df["is_scam"] == 0).sum()),
    }
    logger.info("Exact uncapped eligibility audit (>= %d txs): %s", MIN_TX_ELIGIBLE, json.dumps(stage_drops))

    rng = np.random.default_rng(seed)
    sampled_parts: List[pd.DataFrame] = []

    if balance_contract:
        for contract_flag in [0, 1]:
            sub_f = eligible_df[
                (eligible_df["is_scam"] == 1) & (eligible_df["is_contract"] == contract_flag)
            ].sort_values("node_id")
            sub_n = eligible_df[
                (eligible_df["is_scam"] == 0) & (eligible_df["is_contract"] == contract_flag)
            ].sort_values("node_id")

            max_f_by_ratio = int(len(sub_n) // max(normal_ratio, 1.0)) if normal_ratio >= 1.0 else len(sub_n)
            n_fraud_take = min(
                len(sub_f),
                max_f_by_ratio,
                MAX_FINAL_PER_CLASS_PER_CONTRACT_GROUP,
            )
            n_normal_take = min(len(sub_n), int(round(n_fraud_take * normal_ratio)))

            if n_fraud_take > 0:
                f_idx = rng.choice(len(sub_f), size=n_fraud_take, replace=False)
                sampled_parts.append(sub_f.iloc[np.sort(f_idx)])
            if n_normal_take > 0:
                n_idx = rng.choice(len(sub_n), size=n_normal_take, replace=False)
                sampled_parts.append(sub_n.iloc[np.sort(n_idx)])
    else:
        sub_f = eligible_df[eligible_df["is_scam"] == 1].sort_values("node_id")
        sub_n = eligible_df[eligible_df["is_scam"] == 0].sort_values("node_id")
        n_fraud_take = min(len(sub_f), 2 * MAX_FINAL_PER_CLASS_PER_CONTRACT_GROUP)
        n_normal_take = min(len(sub_n), int(round(n_fraud_take * normal_ratio)))
        if n_fraud_take > 0:
            f_idx = rng.choice(len(sub_f), size=n_fraud_take, replace=False)
            sampled_parts.append(sub_f.iloc[np.sort(f_idx)])
        if n_normal_take > 0:
            n_idx = rng.choice(len(sub_n), size=n_normal_take, replace=False)
            sampled_parts.append(sub_n.iloc[np.sort(n_idx)])

    if not sampled_parts:
        raise RuntimeError("No eligible wallets (>= 5 uncapped transactions) survived filtering.")

    final_wallets_df = pd.concat(sampled_parts, ignore_index=True).sort_values("node_id").reset_index(drop=True)
    # Ensure exact required columns are present in sampled_wallets.csv
    ordered_wallet_cols = [
        "node_id", "address", "is_scam", "is_contract",
        "window_sent_cnt", "window_recv_cnt", "window_first_ts", "window_last_ts",
        "window_value_sent_eth", "window_value_recv_eth",
    ]
    final_wallets_df = final_wallets_df[ordered_wallet_cols]

    final_fraud_count = int((final_wallets_df["is_scam"] == 1).sum())
    if final_fraud_count < 1_000:
        logger.warning(
            "NOTE: Final eligible fraud wallet count is %d (< 1,000 minimum target). "
            "Increase --max-gb (e.g., --max-gb 6) to include additional months.",
            final_fraud_count,
        )
    elif final_fraud_count < 2_000:
        logger.info(
            "Accepted fraud wallet count: %d (within accepted >= 1,000 threshold; target was >= 2,000).",
            final_fraud_count,
        )

    sampled_wallets_csv = raw_dir / "sampled_wallets.csv"
    final_wallets_df.to_csv(sampled_wallets_csv, index=False)
    logger.info("Saved %d final sampled wallets (with exact uncapped window stats) to %s",
                len(final_wallets_df), sampled_wallets_csv)

    all_labels_parquet = partial_dir / "_all_labels_lookup.parquet"
    pq.write_table(
        pa.Table.from_pandas(labels_df[["node_id", "address"]], preserve_index=False),
        all_labels_parquet,
    )
    final_wallets_parquet = partial_dir / "_final_sampled_wallets.parquet"
    pq.write_table(
        pa.Table.from_pandas(final_wallets_df[["node_id", "is_scam", "is_contract"]], preserve_index=False),
        final_wallets_parquet,
    )

    raw_tx_path = raw_dir / "raw_transactions.parquet"

    build_query = f"""
    COPY (
        WITH dedup_partials AS (
            SELECT
                tx_hash,
                CAST(src_id AS UBIGINT) AS src_id,
                CAST(dst_id AS UBIGINT) AS dst_id,
                CAST(ts AS BIGINT) AS ts,
                CAST(value_wei AS VARCHAR) AS value_wei,
                CAST(tx_fee_wei AS VARCHAR) AS tx_fee_wei,
                CAST(block_number AS BIGINT) AS block_number,
                CAST(contract_creation AS BOOLEAN) AS contract_creation,
                source_month,
                ROW_NUMBER() OVER (PARTITION BY tx_hash ORDER BY ts ASC, source_month ASC) AS dup_rn
            FROM read_parquet({partial_glob})
        ),
        unique_partials AS (
            SELECT
                tx_hash, src_id, dst_id, ts, value_wei, tx_fee_wei,
                block_number, contract_creation, source_month,
                hash(tx_hash || '_{seed}') AS det_hash
            FROM dedup_partials
            WHERE dup_rn = 1
        ),
        sampled_hits AS (
            SELECT
                u.*,
                sw_src.node_id AS sampled_src_id,
                sw_dst.node_id AS sampled_dst_id
            FROM unique_partials u
            LEFT JOIN read_parquet('{final_wallets_parquet.as_posix()}') sw_src
                ON u.src_id = CAST(sw_src.node_id AS UBIGINT)
            LEFT JOIN read_parquet('{final_wallets_parquet.as_posix()}') sw_dst
                ON u.dst_id = CAST(sw_dst.node_id AS UBIGINT)
            WHERE sw_src.node_id IS NOT NULL OR sw_dst.node_id IS NOT NULL
        ),
        wallet_ranked AS (
            SELECT
                tx_hash, src_id, dst_id, ts, value_wei, tx_fee_wei,
                block_number, contract_creation, source_month,
                ROW_NUMBER() OVER (
                    PARTITION BY COALESCE(sampled_src_id, sampled_dst_id)
                    ORDER BY det_hash ASC, tx_hash ASC
                ) AS wallet_rn
            FROM sampled_hits
        )
        SELECT
            w.tx_hash,
            w.src_id,
            w.dst_id,
            lbl_src.address AS src_address,
            lbl_dst.address AS dst_address,
            w.ts,
            w.value_wei,
            w.tx_fee_wei,
            w.block_number,
            w.contract_creation,
            w.source_month
        FROM wallet_ranked w
        LEFT JOIN read_parquet('{all_labels_parquet.as_posix()}') lbl_src
            ON w.src_id = CAST(lbl_src.node_id AS UBIGINT)
        LEFT JOIN read_parquet('{all_labels_parquet.as_posix()}') lbl_dst
            ON w.dst_id = CAST(lbl_dst.node_id AS UBIGINT)
        WHERE w.wallet_rn <= {int(max_tx_per_wallet)}
    ) TO '{raw_tx_path.as_posix()}' (FORMAT PARQUET, COMPRESSION SNAPPY);
    """
    con.execute(build_query)

    pf_raw = pq.ParquetFile(raw_tx_path)
    total_raw_rows = pf_raw.metadata.num_rows
    if total_raw_rows > HARD_ROW_CEILING:
        raise RuntimeError(
            f"HARD CEILING EXCEEDED: raw_transactions.parquet has {total_raw_rows:,} rows "
            f"(limit is {HARD_ROW_CEILING:,}). Rerun with `--max-tx-per-wallet 500`."
        )

    null_audit_row = con.execute(f"""
        SELECT
            COUNT(*) AS total_rows,
            SUM(CASE WHEN tx_hash IS NULL THEN 1 ELSE 0 END) AS null_tx_hash,
            SUM(CASE WHEN src_id IS NULL THEN 1 ELSE 0 END) AS null_src_id,
            SUM(CASE WHEN dst_id IS NULL THEN 1 ELSE 0 END) AS null_dst_id,
            SUM(CASE WHEN src_address IS NULL THEN 1 ELSE 0 END) AS null_src_address,
            SUM(CASE WHEN dst_address IS NULL THEN 1 ELSE 0 END) AS null_dst_address,
            SUM(CASE WHEN ts IS NULL THEN 1 ELSE 0 END) AS null_ts,
            SUM(CASE WHEN value_wei IS NULL THEN 1 ELSE 0 END) AS null_value_wei,
            SUM(CASE WHEN tx_fee_wei IS NULL THEN 1 ELSE 0 END) AS null_tx_fee_wei,
            SUM(CASE WHEN block_number IS NULL THEN 1 ELSE 0 END) AS null_block_number,
            SUM(CASE WHEN contract_creation IS NULL THEN 1 ELSE 0 END) AS null_contract_creation,
            SUM(CASE WHEN source_month IS NULL THEN 1 ELSE 0 END) AS null_source_month,
            MIN(ts) AS min_ts,
            MAX(ts) AS max_ts
        FROM read_parquet('{raw_tx_path.as_posix()}')
    """).fetchone()

    tx_by_class = con.execute(f"""
        SELECT
            SUM(CASE WHEN sw_src.is_scam = 1 OR sw_dst.is_scam = 1 THEN 1 ELSE 0 END) AS tx_touching_fraud,
            SUM(CASE WHEN sw_src.is_scam = 0 OR sw_dst.is_scam = 0 THEN 1 ELSE 0 END) AS tx_touching_normal
        FROM read_parquet('{raw_tx_path.as_posix()}') r
        LEFT JOIN read_parquet('{final_wallets_parquet.as_posix()}') sw_src
            ON r.src_id = CAST(sw_src.node_id AS UBIGINT)
        LEFT JOIN read_parquet('{final_wallets_parquet.as_posix()}') sw_dst
            ON r.dst_id = CAST(sw_dst.node_id AS UBIGINT)
    """).fetchone()

    con.close()

    crosstab_df = pd.crosstab(
        final_wallets_df["is_scam"].map({0: "normal_0", 1: "fraud_1"}),
        final_wallets_df["is_contract"].map({0: "eoa_0", 1: "contract_1"}),
    )
    crosstab_dict = {
        str(row_k): {str(col_k): int(val) for col_k, val in row_s.items()}
        for row_k, row_s in crosstab_df.iterrows()
    }
    logger.info("Final Sampled Wallets is_contract Crosstab:\n%s", crosstab_df)

    min_ts_val = int(null_audit_row[12]) if null_audit_row[12] is not None else 0
    max_ts_val = int(null_audit_row[13]) if null_audit_row[13] is not None else 0

    return {
        "stage_drops": stage_drops,
        "final_wallets_count": int(len(final_wallets_df)),
        "final_fraud_wallets": final_fraud_count,
        "final_normal_wallets": int((final_wallets_df["is_scam"] == 0).sum()),
        "is_contract_crosstab": crosstab_dict,
        "raw_transactions_verification": {
            "row_count": int(null_audit_row[0]),
            "columns": [f.name for f in pf_raw.schema_arrow],
            "null_counts": {
                "tx_hash": int(null_audit_row[1]),
                "src_id": int(null_audit_row[2]),
                "dst_id": int(null_audit_row[3]),
                "src_address": int(null_audit_row[4]),
                "dst_address": int(null_audit_row[5]),
                "ts": int(null_audit_row[6]),
                "value_wei": int(null_audit_row[7]),
                "tx_fee_wei": int(null_audit_row[8]),
                "block_number": int(null_audit_row[9]),
                "contract_creation": int(null_audit_row[10]),
                "source_month": int(null_audit_row[11]),
            },
            "min_ts_unix": min_ts_val,
            "max_ts_unix": max_ts_val,
            "min_ts_utc": dt.datetime.fromtimestamp(min_ts_val, tz=dt.timezone.utc).isoformat(),
            "max_ts_utc": dt.datetime.fromtimestamp(max_ts_val, tz=dt.timezone.utc).isoformat(),
            "tx_touching_fraud_wallets": int(tx_by_class[0] or 0),
            "tx_touching_normal_wallets": int(tx_by_class[1] or 0),
        },
        "sampled_wallets_csv": {
            "path": str(sampled_wallets_csv),
            "size_bytes": sampled_wallets_csv.stat().st_size,
            "sha256": compute_sha256(sampled_wallets_csv),
        },
        "raw_transactions_parquet": {
            "path": str(raw_tx_path),
            "size_bytes": raw_tx_path.stat().st_size,
            "sha256": compute_sha256(raw_tx_path),
        },
    }


# ============================================================================
# Step 8: Optional Etherscan API v2 Validation (100 Random tx_hash Values)
# ============================================================================
def validate_with_etherscan_v2(
    raw_tx_path: Path,
    etherscan_key: Optional[str],
    raw_dir: Path,
    seed: int,
) -> Optional[Dict[str, Any]]:
    """
    If `--etherscan-key` is provided, sample 100 random `tx_hash` rows from
    `raw_transactions.parquet` using `USING SAMPLE reservoir(100 ROWS) REPEATABLE (seed)`,
    query Etherscan API v2 (`chainid=1`, `module=proxy&action=eth_getTransactionByHash`),
    compare `value` (Wei) and `block_number`, rate-limit to 5 calls/s, and write
    `data/raw/etherscan_validation.json`. Skips silently when no key is provided.
    """
    if not etherscan_key:
        return None

    logger.info("Running Etherscan API v2 validation on 100 random transactions (rate limit: 5 req/s)...")
    con = duckdb.connect(database=":memory:")
    sample_df = con.execute(f"""
        SELECT tx_hash, value_wei, block_number
        FROM read_parquet('{raw_tx_path.as_posix()}')
        USING SAMPLE reservoir(100 ROWS) REPEATABLE ({int(seed)})
    """).df()
    con.close()

    records: List[Dict[str, Any]] = []
    mismatches: List[Dict[str, Any]] = []

    for _, row in sample_df.iterrows():
        tx_hash = str(row["tx_hash"]).strip()
        expected_wei = int(str(row["value_wei"]).split(".")[0])
        expected_block = int(row["block_number"])

        params = urllib.parse.urlencode({
            "chainid": "1",
            "module": "proxy",
            "action": "eth_getTransactionByHash",
            "txhash": tx_hash,
            "apikey": etherscan_key,
        })
        url = f"https://api.etherscan.io/v2/api?{params}"
        t_req = time.perf_counter()

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AnatomyOfAScamWallet/2.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            res = payload.get("result") or {}
            onchain_wei = int(res.get("value", "0x0"), 16) if res.get("value") else None
            onchain_block = int(res.get("blockNumber", "0x0"), 16) if res.get("blockNumber") else None

            match_ok = (onchain_wei == expected_wei) and (onchain_block == expected_block)
            entry = {
                "tx_hash": tx_hash,
                "dataset_value_wei": expected_wei,
                "etherscan_value_wei": onchain_wei,
                "dataset_block_number": expected_block,
                "etherscan_block_number": onchain_block,
                "match": match_ok,
            }
            records.append(entry)
            if not match_ok:
                mismatches.append(entry)
        except Exception as api_exc:
            err_entry = {"tx_hash": tx_hash, "error": str(api_exc), "match": False}
            records.append(err_entry)
            mismatches.append(err_entry)

        elapsed = time.perf_counter() - t_req
        if elapsed < 0.205:
            time.sleep(0.205 - elapsed)

    summary = {
        "validated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "api_endpoint": "https://api.etherscan.io/v2/api?chainid=1&module=proxy&action=eth_getTransactionByHash",
        "sample_size": len(records),
        "matches_count": sum(1 for r in records if r.get("match") is True),
        "mismatches_count": len(mismatches),
        "mismatches": mismatches,
        "request_log": records,
    }
    out_file = raw_dir / "etherscan_validation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Etherscan validation complete: %d/%d matched.", summary["matches_count"], len(records))
    return summary


# ============================================================================
# Step 7: Write Collection Evidence (`collection_log.json` & `DOWNLOAD_DOCUMENTATION.md`)
# ============================================================================
def write_collection_documentation(
    raw_dir: Path,
    revision_sha: str,
    labels_meta: Dict[str, Any],
    month_targets_meta: Dict[str, Any],
    month_plan: Dict[str, Any],
    month_dl_state: Dict[str, Any],
    final_summary: Dict[str, Any],
    args: argparse.Namespace,
    total_runtime_s: float,
) -> None:
    """Write `data/raw/collection_log.json` and `data/raw/DOWNLOAD_DOCUMENTATION.md`."""
    collection_log = {
        "repository": REPO_ID,
        "repository_url": REPO_URL,
        "doi": DOI,
        "license": LICENSE,
        "revision_sha": revision_sha,
        "collection_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "total_runtime_seconds": round(total_runtime_s, 2),
        "parameters": {
            "seed": args.seed,
            "max_gb": args.max_gb,
            "max_tx_per_wallet": args.max_tx_per_wallet,
            "balance_is_contract": args.balance,
            "normal_ratio": args.normal_ratio,
            "months_override": args.months,
        },
        "metadata_files_downloaded": {
            "targets_global_parquet": labels_meta,
            "month_targets_parquet": month_targets_meta,
        },
        "month_selection_plan": {
            "num_months": month_plan["num_months_selected"],
            "months_chronological": month_plan["selected_months_chronological"],
            "months_greedy_order": month_plan["selected_months_greedy_order"],
            "planned_download_bytes": month_plan["total_download_bytes"],
            "planned_download_gb": month_plan["total_download_gb"],
            "comparison_with_reference_19_months": month_plan["comparison_with_reference_19_months"],
        },
        "downloaded_month_shards": month_dl_state["completed_months"],
        "total_edge_bytes_downloaded": month_dl_state["total_bytes_downloaded"],
        "sampling_and_verification": final_summary,
    }

    log_path = raw_dir / "collection_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(collection_log, f, indent=2)

    rv = final_summary["raw_transactions_verification"]
    ct = final_summary["is_contract_crosstab"]

    doc_md = f"""# Data Collection & Provenance Documentation (`01_collect_data.py`)

## 1. Dataset Provenance
- **Dataset:** `{REPO_ID}` ({REPO_URL})
- **DOI:** `{DOI}` | **License:** `{LICENSE}`
- **Hugging Face Revision SHA:** `{revision_sha}`
- **Collection Timestamp (UTC):** `{collection_log["collection_timestamp_utc"]}`
- **Random Seed:** `{args.seed}`

## 2. Why "Best Months" Greedy Selection Was Used & Exact Uncapped Window Statistics
- The full `gnn_dataset/edges_by_month/` directory contains 121 monthly Parquet files (`2015-08` to `2025-08`, ~2 MB to 577 MB each), while `edges_all` (21.5 GB) and `GNN/parquet` (26 GB) are prohibitively large for an 8–16 GB RAM laptop on a 0.3–3 MB/s connection.
- Using `gnn_dataset/targets/month_targets.parquet` (`269,291` rows mapping active labeled `node_id`s per month) and exact file sizes from the Hugging Face Hub API, we computed a greedy cover maximizing **marginal unique scam wallets per megabyte** within `--max-gb {args.max_gb}`.
- **Selected Months ({month_plan["num_months_selected"]} months, {month_plan["total_download_gb"]} GB):** `{", ".join(month_plan["selected_months_chronological"])}`
- **Exact Uncapped Window Statistics vs. Stored Row Caps:** Before each downloaded monthly shard is deleted, DuckDB computes exact uncapped per-candidate statistics over the full shard (`sent_cnt`, `recv_cnt`, `first_ts`, `last_ts`, `value_sent_eth_sum`, `value_recv_eth_sum`) into `data/raw/_partial/stats_month=YYYY-MM.parquet`. These exact totals are summed across all selected months and saved in `sampled_wallets.csv` (`window_sent_cnt`, `window_recv_cnt`, `window_first_ts`, `window_last_ts`, `window_value_sent_eth`, `window_value_recv_eth`) and used for the `>= 5` transaction eligibility check. Row caps (`{args.max_partial_rows_per_wallet_month}` rows per wallet per month in `_partial/month=YYYY-MM.parquet` and `{args.max_tx_per_wallet}` rows per wallet in `raw_transactions.parquet`) apply **only** to the stored transaction rows, while the window statistics in `sampled_wallets.csv` are exact.

## 3. Confound Control (`is_contract` Balancing) & Sample Summary
- In `targets_global.parquet` (`114,739` labeled nodes: `54,653` scam, `60,086` normal), **73% of scam wallets are smart contracts (`is_contract = 1`) compared to 50% of normal wallets**. An unstratified sample would let classifiers cheat by predicting `is_contract` instead of fraudulent behavior.
- To eliminate this confound, we required `>= 5` exact uncapped transactions (`window_sent_cnt + window_recv_cnt >= 5`) in the selected months and sampled equal counts (`--normal-ratio {args.normal_ratio}`) within each `is_contract` group (`0` and `1`):
  - **Final Fraud Wallets:** `{final_summary["final_fraud_wallets"]:,}`
  - **Final Normal Wallets:** `{final_summary["final_normal_wallets"]:,}`
  - **`is_contract` Crosstab by Class:** `{json.dumps(ct)}`
- **Final Raw Transactions (`data/raw/raw_transactions.parquet`):**
  - **Row Count:** `{rv["row_count"]:,}` rows (capped at `{args.max_tx_per_wallet}` stored txs/wallet)
  - **Timestamp Span (UTC):** `{rv["min_ts_utc"]}` to `{rv["max_ts_utc"]}`
  - **SHA-256 (`raw_transactions.parquet`):** `{final_summary["raw_transactions_parquet"]["sha256"]}`
  - **SHA-256 (`sampled_wallets.csv`):** `{final_summary["sampled_wallets_csv"]["sha256"]}`

## 4. Documented Dataset Limitations
1. **Partial Transaction History Restricted to Selected Months:** Each wallet's observed transactions and exact window statistics come from the {month_plan["num_months_selected"]} selected high-density months rather than its complete lifetime across all 121 months.
2. **Missing Gas & Error Telemetry (`gas_used`, `gas_price`, `is_error`):** The usable edge files provide only `tx_fee_wei` (= gas_used x gas_price) and `contract_creation`. Individual `gas_used`, `gas_price`, and `is_error` fields do not exist in the repository and are **never** fabricated or imputed.
3. **Hashed Counterparty Node IDs:** Only the `114,739` labeled accounts have hexadecimal `0x...` addresses (`src_address` / `dst_address`). Unlabeled counterparties exist solely as integer `uint64` `src_id` / `dst_id` identifiers and cannot be mapped back to Ethereum addresses.
4. **Class Prevalence vs. Real-World Base Rate:** The curated label pool is ~47.6% scam (`54,653 / 114,739`) and our confound-balanced sample is 1:{args.normal_ratio}, which does not reflect the <1% real-world on-chain scam prevalence.
5. **Ground-Truth Label Provenance:** Scam labels originate from Etherscan public tags and curated scam blocklists, which are subject to reporting delay and survivorship bias.
"""
    doc_path = raw_dir / "DOWNLOAD_DOCUMENTATION.md"
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(doc_md)
    logger.info("Wrote collection evidence to %s and %s", log_path, doc_path)


# ============================================================================
# CLI Entrypoint
# ============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2 Step 1: Best-Months Ethereum Transaction Collection (DuckDB + HF Hub)"
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Output directory for raw files")
    parser.add_argument("--plan-only", action="store_true", help="Compute and save month_plan.json, then exit")
    parser.add_argument("--max-gb", type=float, default=4.0, help="Maximum edge download budget in GB (default: 4.0)")
    parser.add_argument(
        "--months",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit list of YYYY-MM months to override greedy planner",
    )
    parser.add_argument(
        "--max-tx-per-wallet",
        type=int,
        default=1_000,
        help="Maximum transactions kept per final sampled wallet (default: 1000)",
    )
    parser.add_argument(
        "--max-partial-rows-per-wallet-month",
        type=int,
        default=MAX_PARTIAL_ROWS_PER_WALLET_MONTH,
        help="Max rows kept per candidate wallet per month in _partial files (default: 200; set 100 for tighter bound)",
    )
    parser.add_argument(
        "--balance",
        dest="balance",
        action="store_true",
        default=True,
        help="Balance fraud and normal wallets within each is_contract group (default: True)",
    )
    parser.add_argument(
        "--no-balance",
        dest="balance",
        action="store_false",
        help="Disable is_contract stratification balancing",
    )
    parser.add_argument(
        "--normal-ratio",
        type=float,
        default=1.0,
        help="Ratio of normal to fraud wallets per is_contract stratum (default: 1.0)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed (default: 42)")
    parser.add_argument(
        "--etherscan-key",
        type=str,
        default=None,
        help="Optional Etherscan API v2 key to validate 100 random tx_hash records",
    )
    return parser.parse_args()


def main() -> None:
    t_start = time.perf_counter()
    args = parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    revision_sha, file_sizes = fetch_repo_metadata(api)

    meta_cache = args.raw_dir / "_metadata"
    labels_path = download_hub_file(LABELS_REPO_PATH, local_dir=meta_cache, revision=revision_sha)
    verify_parquet_schema(labels_path, REQUIRED_LABEL_COLS, LABELS_REPO_PATH)
    labels_df = pq.read_table(labels_path).to_pandas()

    month_targets_path = download_hub_file(MONTH_TARGETS_REPO_PATH, local_dir=meta_cache, revision=revision_sha)
    verify_parquet_schema(month_targets_path, REQUIRED_MONTH_TARGET_COLS, MONTH_TARGETS_REPO_PATH)
    month_targets_df = pq.read_table(month_targets_path).to_pandas()

    labels_meta = {
        "repo_path": LABELS_REPO_PATH,
        "rows": int(len(labels_df)),
        "scam_count": int((labels_df["is_scam"] == 1).sum()),
        "normal_count": int((labels_df["is_scam"] == 0).sum()),
        "size_bytes": labels_path.stat().st_size,
        "sha256": compute_sha256(labels_path),
    }
    month_targets_meta = {
        "repo_path": MONTH_TARGETS_REPO_PATH,
        "rows": int(len(month_targets_df)),
        "size_bytes": month_targets_path.stat().st_size,
        "sha256": compute_sha256(month_targets_path),
    }
    logger.info(
        "Loaded targets_global (%d rows: %d scam, %d normal) & month_targets (%d rows).",
        labels_meta["rows"], labels_meta["scam_count"], labels_meta["normal_count"], month_targets_meta["rows"],
    )

    month_plan = compute_greedy_month_plan(
        labels_df=labels_df,
        month_targets_df=month_targets_df,
        file_sizes=file_sizes,
        max_gb=args.max_gb,
        override_months=args.months,
    )
    plan_path = args.raw_dir / "month_plan.json"
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(month_plan, f, indent=2)
    logger.info(
        "Saved month_plan.json: %d months selected (%.3f GB) covering %d unique scam and %d unique normal wallets.",
        month_plan["num_months_selected"],
        month_plan["total_download_gb"],
        month_plan["distinct_scam_wallets_covered"],
        month_plan["distinct_normal_wallets_covered"],
    )

    if args.plan_only:
        logger.info("--plan-only specified. Exiting after saving %s.", plan_path)
        return

    selected_months = month_plan["selected_months_chronological"]

    candidates_df = build_candidate_pools(
        labels_df=labels_df,
        month_targets_df=month_targets_df,
        selected_months=selected_months,
        seed=args.seed,
    )

    month_dl_state = process_selected_months(
        selected_months=selected_months,
        candidates_df=candidates_df,
        raw_dir=args.raw_dir,
        revision_sha=revision_sha,
        seed=args.seed,
        per_wallet_month_cap=args.max_partial_rows_per_wallet_month,
    )

    final_summary = finalize_sample_and_transactions(
        candidates_df=candidates_df,
        labels_df=labels_df,
        raw_dir=args.raw_dir,
        selected_months=selected_months,
        max_tx_per_wallet=args.max_tx_per_wallet,
        balance_contract=args.balance,
        normal_ratio=args.normal_ratio,
        seed=args.seed,
    )

    total_runtime_s = time.perf_counter() - t_start
    write_collection_documentation(
        raw_dir=args.raw_dir,
        revision_sha=revision_sha,
        labels_meta=labels_meta,
        month_targets_meta=month_targets_meta,
        month_plan=month_plan,
        month_dl_state=month_dl_state,
        final_summary=final_summary,
        args=args,
        total_runtime_s=total_runtime_s,
    )

    validate_with_etherscan_v2(
        raw_tx_path=args.raw_dir / "raw_transactions.parquet",
        etherscan_key=args.etherscan_key,
        raw_dir=args.raw_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()