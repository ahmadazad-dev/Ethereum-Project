# Anatomy of a Scam Wallet

Project repository for Ethereum scam wallet detection and behavioral analysis across Phases 2–5.

## Repository Structure

```text
anatomy-of-a-scam-wallet/
├── .gitignore                        # Prevents uploading raw datasets & >100MB weights
├── README.md                         # Project documentation & external model weight links
├── requirements.txt                  # Python library dependencies for reproducibility
├── data_scripts/                     # Required in Deliverable 1 (Phase 2 scripts)
│   ├── 01_collect_data.py
│   ├── 02_clean_data.py
│   ├── 03_engineer_features.py
│   └── 04_eda_visualizations.py
├── src/                              # Required in Deliverable 2 & 3 (Phase 3–5 code)
│   ├── config.py
│   ├── models.py
│   ├── tune_hyperparameters.py
│   ├── train_evaluate.py
│   ├── error_bias_analysis.py
│   ├── ablation_study.py
│   └── plot_results.py
└── report/                           # LaTeX source files for Deliverables 1, 2, and 3
    ├── main.tex
    ├── references.bib
    └── figures/                      # Saved high-resolution charts (.png / .pdf)
```

## Setup

```bash
pip install -r requirements.txt
```

## External Model Weights

Large trained model weights (>100 MB) are excluded from Git tracking via `.gitignore`. Add external download links (e.g., Hugging Face / Google Drive / Zenodo) and checksums here once trained:

- *Model checkpoint links pending training.*
