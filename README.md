# Ethereum-Project

**Working Title:** *Anatomy of a Scam Wallet: an ablation study of which on-chain behavior groups best detect fraudulent Ethereum accounts, using features engineered from raw data and compared across XGBoost and Random Forest by fraud-class F1.*

## Repository Layout

- `AGENTS.md` — Research operating charter, integrity rules, and experimental design contract.
- `configs/` — Declarative experiment configurations (`seeds.yaml`).
- `src/` — Importable pipeline modules:
  - `src/data_scripts/` — Phase 2 ingestion, cleaning, and provenance logging (per ADR-0001).
  - `src/config.py` — Config loading, SHA-256 hashing, and runtime provenance helpers.
  - `src/features.py` — Canonical behavioral group partition (`FEATURE_GROUPS` for `G1`–`G5`) and identifier-leakage guards.
  - `src/models.py` — Baseline estimators (Majority, Heuristic Rule) and deterministic Random Forest / XGBoost factories.
  - `src/evaluation.py` — Leak-free paired CV harness, threshold optimization, Nadeau-Bengio test, and append-only registry logger.
  - `src/ablation.py` — Full, Group-Only, Drop-One-Group, and Add-One-Group subset generators.
- `data/` — Provenance manifest (`data/MANIFEST.md`); raw and derived datasets remain gitignored under `data/raw/` and `data/derived/`.
- `results/registry.csv` — Append-only experiment run ledger.
- `docs/` — Research log, ADRs, open questions, feature rationale, domain notes, threats to validity, literature tracker, and course deliverables.
- `tests/` — Unit and leakage-verification test suite (`pytest`).

## Quickstart

```bash
pip install -r requirements.txt
python -m pytest -v
```
