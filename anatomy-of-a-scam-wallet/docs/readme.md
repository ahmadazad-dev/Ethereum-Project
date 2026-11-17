# Anatomy of a Scam Wallet: Which Behavioral Feature Groups Drive Ethereum Fraud Detection?

**Author:** Ahmad Azad (BS Data Science, FAST NUCES Lahore — `TODO: add author email`)  
**Dataset:** [`fesevu/ethereum_fraud_dataset_by_activity`](https://huggingface.co/datasets/fesevu/ethereum_fraud_dataset_by_activity) (DOI: `10.57967/hf/6373`, CC-BY-4.0)

This repository implements a reproducible, low-RAM (< 4 GB peak) end-to-end pipeline and grouped behavioral ablation study comparing **Random Forest** and **XGBoost** on **28 engineered on-chain features** across four behavioral groups:
- **$G_{\text{temp}}$ (6 features):** Temporal & activity cadence (lifespan, inter-transaction intervals, burstiness)
- **$G_{\text{val}}$ (12 features):** Value flow & asymmetry (sent/received ETH sums, ratios, concentration)
- **$G_{\text{net}}$ (4 features):** Counterparty network topology (unique senders/recipients, degree ratios)
- **$G_{\text{tok}}$ (6 features):** Fee & smart contract interaction (gas fee stats, contract-creation & zero-value call ratios; *note: `gas_price`, `is_error`, and ERC-20 token transfers are absent in the underlying dataset*)

## Headline Results

| Model | 5x5 Repeated CV Fraud-F1 | Held-Out Test Fraud-F1 ($n=505$) | Short-Lifespan (<7d) FPR |
| :--- | :---: | :---: | :---: |
| **Random Forest** | 0.8445 | 0.8172 | 0.70 |
| **XGBoost** | 0.8554 | 0.8175 | 0.85 |

- **Model Parity ($H_2$):** Not supported — held-out test Fraud-F1 difference between XGBoost and Random Forest is negligible ($\Delta F_1 = +0.0002$, paired bootstrap $p = 0.956$).
- **Top Feature:** `G_temp_lifespan_days` is the strongest single Random Forest predictor (importance `0.149`).

## Repository Structure

```text
.
├── data_scripts/
│   ├── 01_collect_data.py          # Low-RAM DuckDB/HF shard collector (matched 1,261 fraud / 1,261 normal)
│   ├── 02_clean_data.py            # Deterministic 8-step cleaning, Wei->ETH/Gwei, p99.9 winsorization
│   ├── 02b_plot_figures.py         # Phase 2 diagnostic plots (class balance, value distribution)
│   ├── 03_engineer_features.py     # 28 features (4 groups), 80/20 stratified split, train-only RobustScaler
│   └── 04_eda_visualizations.py    # EDA correlation heatmaps & top-10 feature distributions
├── src/
│   ├── config.py                   # SEED=42, FEATURE_GROUPS loader, get_features(), search spaces
│   ├── models.py                   # RandomForest & XGBoost ColumnTransformer pipelines
│   ├── tune_hyperparameters.py     # RandomizedSearchCV + 1-SE selection on train CV only
│   ├── train_evaluate.py           # 5x5 CV, held-out test evaluation, H2 bootstrap, prevalence table
│   ├── error_bias_analysis.py      # Subgroup fairness/FPR audit & FP/FN error cases
│   ├── ablation_study.py           # LOGO, single-group-only, random controls, H1/H3 Holm-Bonferroni, SHAP
│   ├── plot_results.py             # Publication figures (ablation bars, ROC/PR, SHAP/permutation)
│   ├── evasion_analysis.py         # False-negative evasion archetype breakdown
│   └── contribution_table.py       # Git commit contribution table helper
├── tests/
│   └── test_smoke.py               # Fast pytest smoke suite (config, leakage, splits, ablation, seed)
├── tools/
│   └── check_numbers.py            # Verifies abstract word count & headline metrics in report/main.tex
├── docs/
│   ├── viva_prep.md                # Defense talk tracks, 10-slide outline, 40-question Q&A, glossary
│   └── viva_demo_script.md         # Live terminal demo walkthrough & recovery plan
├── report/
│   ├── main.tex                    # IEEEtran 2-column final paper
│   ├── main.pdf                    # Compiled 10-page PDF
│   ├── references.bib              # Verified BibTeX bibliography
│   └── figures/                    # PDF/PNG figures included in main.tex
├── run_all.ps1                     # Windows PowerShell end-to-end runner (-Quick supported)
├── run_all.sh                      # Linux/macOS bash end-to-end runner (--quick supported)
├── requirements.txt
└── CITATION.cff
```

## Dataset Warning (96 GB Upstream)

The full Hugging Face repository (`fesevu/ethereum_fraud_dataset_by_activity`) is **~96 GB (~1.62B rows)**. **Do NOT download the entire dataset.** `data_scripts/01_collect_data.py` downloads `targets_global.parquet` (3.54 MB), selects a matched sample of **2,522 wallets** (1,261 fraud + 1,261 normal; 1,053 EOA + 208 contract per class), and streams monthly shards sequentially under a `--max-gb 4.0` budget using DuckDB predicate pushdown, deleting each raw shard after computing uncapped per-wallet stats. Peak RAM stays under **4 GB**.

## Environment Setup

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS (Bash)
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run Everything

### Quick Smoke Run (~2–5 minutes, skips data download if `data/features/train_scaled.parquet` exists)
- **Windows:** `.\run_all.ps1 -Quick`
- **Linux/macOS:** `./run_all.sh --quick`

### Full Reproducible Pipeline (Seed = 42)
- **Windows:** `.\run_all.ps1`
- **Linux/macOS:** `./run_all.sh`

Or run individual stages manually:
```bash
python data_scripts/01_collect_data.py --max-gb 4.0
python data_scripts/02_clean_data.py
python data_scripts/02b_plot_figures.py
python data_scripts/03_engineer_features.py
python data_scripts/04_eda_visualizations.py

python -m src.tune_hyperparameters --fit-final
python -m src.train_evaluate
python -m src.error_bias_analysis
python -m src.ablation_study
python -m src.plot_results
python -m src.evasion_analysis

python tools/check_numbers.py
pytest -q
```

## Model Storage

Trained `.joblib` model artifacts in `models/` are excluded from Git via `.gitignore` to keep the repository lightweight. Running `python -m src.tune_hyperparameters --fit-final` regenerates `models/rf_tuned.joblib` and `models/xgb_tuned.joblib` deterministically (`SEED = 42`).

## Limitations & Ethics

1. **Matched 50/50 Design vs. Real-World Prevalence:** The 2,522-wallet dataset is class-balanced and contract-matched; precision and false-positive volume under realistic Ethereum scam base rates (e.g., 0.1%–1%) are projected analytically in `outputs/eval/prevalence_table.csv` rather than measured on an unstratified stream.
2. **Missing Schema Fields:** `gas_price`, `is_error`, and ERC-20 token transfer logs are unavailable in the source dataset, constraining $G_{\text{tok}}$ to transaction-level fee and contract-interaction proxies.
3. **Short-Lifespan Bias:** Newly created normal wallets (`lifespan < 7 days`) exhibit high false-positive rates (`0.70` RF, `0.85` XGB); automated enforcement must not rely solely on a single threshold without human review.

## License

- **Dataset:** CC-BY-4.0 (`fesevu/ethereum_fraud_dataset_by_activity`, DOI `10.57967/hf/6373`)
- **Code:** `TODO: add LICENSE file (MIT License recommended)`
