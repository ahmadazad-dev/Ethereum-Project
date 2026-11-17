# Phase 2 (Deliverable 1): Ethereum Fraud Data Collection & Cleaning Pipeline

Reproducible, low-RAM (< 2.5 GB peak) data collection and 8-step cleaning pipeline for **Anatomy of a Scam Wallet** using [`fesevu/ethereum_fraud_dataset_by_activity`](https://huggingface.co/datasets/fesevu/ethereum_fraud_dataset_by_activity) (DOI: `10.57967/hf/6373`, CC-BY-4.0).

## 1. Environment Setup

```bash
pip install -r requirements.txt
```

## 2. Exact Run Commands (Sequential Execution, Seed = 42)

```bash
# Step 1: Metadata inspection (Step 0), stratified wallet sampling (3k fraud + 9k normal),
#         and selective row-group extraction with predicate pushdown & resume support
python3 data_scripts/01_collect_data.py

# Step 2: Deterministic 8-step cleaning, Wei-to-Ether/Gwei conversion, contract-creation
#         flagging, 99.9th pct winsorization, and integrity assertions
python3 data_scripts/02_clean_data.py

# Step 3: Generate publication-ready PDF/PNG figures for the LaTeX report
python3 data_scripts/03_plot_figures.py
```

## 3. Expected Runtime & Output Sizes

| Script / Artifact | Expected Runtime (Colab / 8–16 GB Laptop) | Output File Path | Expected Size |
| :--- | :--- | :--- | :--- |
| `01_collect_data.py` | ~15s (local mirror) / 4–8 min (remote HF stream) | `data/raw/sampled_wallets.csv` | ~1.2 MB (12,000 rows) |
| `01_collect_data.py` | (included above) | `data/raw/raw_transactions.parquet` | ~52–60 MB (1,247,167 rows) |
| `01_collect_data.py` | (included above) | `data/raw/collection_log.json` & `DOWNLOAD_DOCUMENTATION.md` | ~5 KB |
| `02_clean_data.py` | ~10–15 seconds | `data/clean/clean_transactions.parquet` | ~130 MB (1,244,961 rows) |
| `02_clean_data.py` | (included above) | `data/clean/cleaning_report.json` | ~6 KB |
| `03_plot_figures.py` | ~2–3 seconds | `report/figures/class_balance.pdf` & `value_distribution.pdf` | ~62 KB |
