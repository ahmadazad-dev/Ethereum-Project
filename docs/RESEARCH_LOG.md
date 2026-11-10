# Research Log

Chronological, append-only lab notebook following `AGENTS.md` §4 and §11. Every claim distinguishes `[observed]`, `[inferred]`, `[assumed]`, and `[from literature]`.

---

## 2026-11-10, Antigravity (assisted)

- **Goal:** Inspect the repository state, link the local workspace to the GitHub remote (`https://github.com/ahmadazad-dev/Ethereum-Project`), and scaffold the missing Part 1 project structure per `AGENTS.md` §4, §10, §11, and §12 (serves pipeline readiness for RQ1–RQ5).
- **Done:**
  - Inspected `d:\Repositories\Ethereum Project` recursively and verified that only `AGENTS.md` and five course documents in `docs/` (`PDF Project Description (1).pdf`, `PDF Deliverable 1.pdf`, `Deliverable 2.pdf`, `PDF Deliverable 3.pdf`, `Research_Guide_Anatomy_of_a_Scam_Wallet.docx`) were present, with no `.git` directory or Part 1 scaffolding.
  - Checked Python and Git tooling on the host (`git version 2.53.0.windows.1`, `Python 3.14.0` at `D:\Python\python.exe`, `Python 3.12.10` at `C:\Users\ahmad\AppData\Local\Programs\Python\Python312\python.exe`).
  - Initialized Git (`main`), linked remote `origin` to `https://github.com/ahmadazad-dev/Ethereum-Project.git`, fetched `origin/main` (`0b9ced3 Initial commit`, containing `README.md`), checked out `main` tracking `origin/main` without modifying existing files, and created feature branch `feat/part1-scaffolding`.
  - Created Part 1 foundational files: `.gitignore`, `configs/seeds.yaml`, `results/registry.csv`, `data/MANIFEST.md`, `data/raw/.gitkeep`, `data/derived/.gitkeep`, `reports/.gitkeep`, `notebooks/.gitkeep`, `requirements.txt`, `README.md` update, `docs/DECISIONS.md` (ADR-0001, ADR-0002), `docs/OPEN_QUESTIONS.md`, `docs/FEATURES.md`, `docs/THREATS_TO_VALIDITY.md`, `docs/NOTES_DOMAIN.md`, `docs/lit/CANDIDATES.md`, and `docs/RESEARCH_LOG.md`.
  - Implemented core pipeline modules (`src/__init__.py`, `src/data_scripts/__init__.py`, `src/config.py`, `src/features.py`, `src/models.py`, `src/evaluation.py`, `src/ablation.py`) and test suite (`tests/test_config.py`, `tests/test_features.py`, `tests/test_ablation.py`, `tests/test_evaluation_and_leakage.py`).
  - Installed pinned missing dependencies (`xgboost==3.4.1`, `pytest==9.2.0`, `PyYAML==6.0.3`, `shap==0.54.1`) from `requirements.txt`.
- **`[observed]`:**
  - `origin/main` at `0b9ced3` contained only `README.md` (`# Ethereum-Project`).
  - `D:\Python\python.exe` (`Python 3.14.0`) now has the full pinned stack from `requirements.txt` (`numpy==2.2.6`, `pandas==2.3.3`, `scikit-learn==1.7.2`, `scipy==1.16.2`, `xgboost==3.4.1`, `matplotlib==3.10.7`, `seaborn==0.13.2`, `shap==0.54.1`, `PyYAML==6.0.3`, `pytest==9.2.0`, `joblib==1.5.2`, `requests==2.32.5`).
  - `python -m pytest -v` passed all 16 tests (`tests/test_config.py`, `tests/test_features.py`, `tests/test_ablation.py`, `tests/test_evaluation_and_leakage.py`), including hand-computed toy metric checks, in-fold imputer leakage isolation, Nadeau-Bengio corrected t-test verification, grouped duplicate CV fold parity, and `results/registry.csv` schema validation.
  - Standard sandbox mode blocks outbound network DNS (`getaddrinfo failed` on `github.com` and PyPI); `git ls-remote`, `git fetch`, and `pip install` succeeded outside the sandbox.
- **`[inferred]`:**
  - Part 1 setup had not yet been committed locally or pushed to `origin/main`, so scaffolding the `AGENTS.md`-compliant structure on `feat/part1-scaffolding` cleanly establishes the baseline without touching `AGENTS.md` or existing `docs/` guides.
- **Surprises / anomalies:**
  - The local OS clock reads `2026-11-10`, matching the Deliverable 2 deadline in `docs/Deliverable 2.pdf` (`10th November, 2026`).
  - `mypy src tests` flagged 13 missing third-party stub errors (`import-untyped` for `yaml`, `pandas`, `scipy`, `sklearn`) with zero application type errors.
- **Risks to validity found:**
  - All candidate citations and benchmark dataset claims in `docs/Research_Guide_Anatomy_of_a_Scam_Wallet.docx` remain unverified against primary sources (`AGENTS.md` §16); logged as `UNVERIFIED` in `docs/lit/CANDIDATES.md` and `docs/OPEN_QUESTIONS.md` (`OQ-003`, `OQ-006`).
- **Unresolved:**
  - `OQ-001`: Dataset and label source selection (awaiting user decision per `AGENTS.md` §15).
- **Next most valuable step:**
  - Obtain user decision on dataset provenance (`OQ-001`), register raw data and SHA-256 checksum in `data/MANIFEST.md`, run Phase 2 schema/cleaning audit (`src/data_scripts/`), and freeze `FEATURE_GROUPS` in `src/features.py` via ADR.

