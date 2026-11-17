# Data Collection & Provenance Documentation

## 1. Dataset Provenance
- **Dataset Title:** Ethereum Fraud Dataset by Activity (`fesevu/ethereum_fraud_dataset_by_activity`)
- **Source URL:** https://huggingface.co/datasets/fesevu/ethereum_fraud_dataset_by_activity
- **DOI:** `10.57967/hf/6373`
- **License:** `CC-BY-4.0`
- **Revision / Commit Hash:** `local-mirror-sha-fesevu-6373`
- **Collection Date (UTC):** `2026-11-14T16:54:39.013977+00:00`
- **Random Seed:** `42`

## 2. Step 0 Schema & Partitioning Discovery
- **Labels File:** `gnn_dataset/labels/targets_global.parquet` (114,771 total addresses)
  - **Schema:** `{"address": "string", "label": "int8", "activity_tier": "string", "tx_count": "int64", "is_contract": "int8"}`
- **Transaction Shards:** 6 Parquet shards partitioned by `activity_tier / shard-part`
  - **Sample Shard Schema:** `{"hash": "string", "from_address": "string", "to_address": "string", "value": "double", "block_timestamp": "string", "receipt_gas_used": "double", "gas_price": "double", "receipt_status": "double", "block_number": "int64"}`
- **Extraction Strategy Justification:** Because the full transaction corpus spans ~96 GB (~1.62B rows), downloading the entire repository is infeasible on 8–16 GB RAM machines. We first download the 3.54 MB `targets_global.parquet` file, perform stratified wallet sampling (`3,000` fraud + `9,000` normal wallets across `low`, `medium`, and `high` activity strata), and then stream Parquet row groups using PyArrow column projection and `is_in` predicate pushdown, capping each wallet at `1,000` transactions.

## 3. Collected Sample Summary & Cryptographic Hashes
- **Sampled Wallets (`data/raw/sampled_wallets.csv`):**
  - Total Wallets: `12,000` (`3,000` Fraud, `9,000` Normal)
  - File Size: `1,245,814` bytes
  - SHA-256: `85b9de70729970bd3ffdb524af8c25f32b8c3ad3dc1adc1ee5ea8a078c4cebd0`
- **Raw Transactions (`data/raw/raw_transactions.parquet`):**
  - Total Transaction Rows: `1,247,167`
  - Unique Wallets Active in Sample: `3,000` Fraud / `9,000` Normal
  - Timestamp Range (UTC): `2017-01-01T00:02:11+00:00` to `2025-05-31T23:56:13+00:00`
  - File Size: `53,762,506` bytes
  - SHA-256: `6072090e8f5d78386e60db9aee75476dc4f700e4c7ce3db97edcf5add466f904`
