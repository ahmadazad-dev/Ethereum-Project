# Architecture and Methodology Decision Records (ADRs)

All major architectural, methodological, feature-group, split, and modeling decisions are recorded here in ADR format per `AGENTS.md` §11.

---

## ADR-0001: Repository Layout and Data-Scripts Location

- **Date:** 2026-11-10
- **Status:** Accepted
- **Context:** `AGENTS.md` §10 specifies importable modules under `src/` (`src/data_scripts/`, `src/features.py`, `src/models.py`, `src/evaluation.py`, `src/ablation.py`), whereas `docs/PDF Deliverable 1.pdf` (§3.1) mentions a `/data scripts` folder for Phase 2 ingestion and cleaning code.
- **Options:**
  1. Keep all Python modules under `src/`, placing Phase 2 ingestion/cleaning scripts in `src/data_scripts/`.
  2. Create a separate top-level `data_scripts/` directory outside `src/`.
- **Choice:** Option 1 — place all Phase 2 ingestion, cleaning, and provenance-logging scripts under `src/data_scripts/` as importable modules runnable via `python -m src.data_scripts.<module>`, and document the mapping explicitly in `README.md`.
- **Consequences:** Preserves a single clean Python package root (`src/`) without duplicating scripts across directories; satisfies `AGENTS.md` §10 and remains trivial to reference in Deliverable 1/2 documentation.

---

## ADR-0002: Experiment Registry Schema and Split/Seed Contract

- **Date:** 2026-11-10
- **Status:** Accepted
- **Context:** `AGENTS.md` §3–§5 requires append-only run logging in `results/registry.csv`, pairing of identical folds and seeds across all ablation configurations, and reporting of fraud-class F1 (`pos_label=1`) alongside precision, recall, PR-AUC, and ROC-AUC under both default (`0.50`) and training-fold-tuned thresholds.
- **Options:**
  1. Minimal registry with only model name and F1 score.
  2. Full provenance registry tracking `run_id`, `timestamp_utc`, `experiment_id`, `git_commit`, `config_path`, `config_hash`, `data_version`, `data_hash`, `split_type`, `n_splits`, `seeds`, `model`, `feature_group_config`, `n_features`, `threshold_mode`, `threshold_value`, mean/SD for all five required metrics (`f1`, `precision`, `recall`, `pr_auc`, `roc_auc`), `runtime_sec`, `library_versions`, `status`, `supersedes_run_id`, and `notes`.
- **Choice:** Option 2 — enforce the full provenance schema in `results/registry.csv` and declare the fixed split/CV contract (`test_size=0.15`, `holdout_seed=42`, `5` folds × `[42, 100, 2026, 7, 999]`) in `configs/seeds.yaml`.
- **Consequences:** Every reported table and plot can be regenerated and audited from `results/registry.csv` and git history; invalid or superseded runs remain append-only via `status` and `supersedes_run_id`.

