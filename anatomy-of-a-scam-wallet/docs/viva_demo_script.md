# Live Viva Demo Script: `src.ablation_study` Walkthrough

**Goal:** Demonstrate deterministic, leakage-free grouped ablation live in < 60 seconds, explain the code architecture with exact `file:line` pointers, and inspect real error cases.

---

## 1. Pre-Demo Checklist (Run 5 Minutes Before Viva)
1. Open terminal in project root and activate virtual environment:
   - **Windows PowerShell:** `.\.venv\Scripts\Activate.ps1`
   - **Linux/macOS:** `source .venv/bin/activate`
2. Verify smoke tests pass: `pytest -q`
3. Verify saved outputs exist in `outputs/ablation/` and `outputs/eval/` (so you have a clean fallback if needed).

---

## 2. Live Command (Fast Smoke Run, ~20–45s)
```bash
python -m src.ablation_study --repeats 2 --folds 3 --n-random 3 --n-boot 100 --skip-shap
```
*(Full paper run uses defaults: `--repeats 5 --folds 5 --n-random 10 --n-boot 1000` with SHAP enabled.)*

### What to Narrate as Output Prints
1. **Configuration Loading:** Point out `SEED = 42` and the 4 groups (`G_temp`: 6, `G_val`: 12, `G_net`: 4, `G_tok`: 6) totaling **28 features** with `is_contract` excluded.
2. **Model Configurations:** Explain that `build_configurations` generates **Full (28)**, **4 LOGO** subsets (22, 16, 24, 22), **4 Single-Group-Only** subsets (6, 12, 4, 6), and **Random Controls** matched by feature count $k$.
3. **Evaluation & Statistical Testing:** Show that `evaluate_config_with_cache` runs identical `StratifiedRepeatedKFold` CV on `train_scaled.parquet` (2,017) and evaluates on `test_scaled.parquet` (505), followed by `test_h1_and_h3` applying paired bootstrap + Holm-Bonferroni correction.

---

## 3. Exact Code Pointers (`file:line`) to Open in IDE

| Component | File & Line | What to Highlight to Examiners |
| :--- | :--- | :--- |
| Global Seed & Leakage Guard | `src/config.py:13-16` | `SEED = 42`, `LEAKAGE_COLS = {"is_scam", "wallet", "src_is_scam", "dst_is_scam", "is_contract"}` |
| 28-Feature Validation | `src/config.py:29-48` | `_load_feature_groups()` asserts exact count `28`, zero duplicates, zero leakage overlap |
| Ablation Subset Selector | `src/config.py:55-85` | `get_features(exclude_groups, only_groups)` cleanly switches Full / LOGO / ONLY |
| Fair Pipeline Wrapper | `src/models.py:54-81` | `make_pipeline()` uses `ColumnTransformer(remainder="drop")` so all ablations share identical estimator & CV logic |
| Ablation Config Builder | `src/ablation_study.py:89-143` | `build_configurations()` constructs Full, LOGO, ONLY, and random-k controls |
| Cached CV + Test Evaluator | `src/ablation_study.py:145-228` | `evaluate_config_with_cache()` computes repeated CV F1 and held-out test F1 |
| Hypothesis Tests ($H_1, H_3$) | `src/ablation_study.py:273-353` | `test_h1_and_h3()` runs paired bootstrap $\Delta F_1$ + `holm_bonferroni` (`src/ablation_study.py:75`) |
| Importance & SHAP | `src/ablation_study.py:355-440` | `compute_grouped_perm_and_shap()` computes grouped permutation F1 drop + TreeExplainer SHAP |

---

## 4. Live Error-Case Inspection (`outputs/eval/errors_*.csv`)
Run this one-liner to display top False Positives and False Negatives for examiner discussion:
```bash
python -c "import pandas as pd; df=pd.read_csv('outputs/eval/errors_xgb.csv'); print(df[['wallet','y_true','y_pred','prob_fraud','error_type','G_temp_lifespan_days']].head(8).to_string(index=False))"
```

### Error-Case Discussion Table (Populate from Local `outputs/eval/errors_*.csv`)

| Error Type | Expected Pattern | Example Feature Driver | Interpretation for Examiners |
| :--- | :--- | :--- | :--- |
| **False Positive (FP)** | Normal wallet (`y_true=0`) flagged as scam (`y_pred=1`) | Low `G_temp_lifespan_days` (< 7 days) | Short-lived normal accounts mimic hit-and-run drainers (FPR `0.70` RF / `0.85` XGB). |
| **False Negative (FN)** | Scam wallet (`y_true=1`) missed (`y_pred=0`) | Longer `G_temp_lifespan_days`, low burstiness | "Sleeper" or aged scam wallets blend into normal transaction cadences. |

---

## 5. Recovery Plan (If Anything Fails Live)
1. **If `--skip-shap` or full run is slow:** Kill (`Ctrl+C`) and immediately show pre-computed `outputs/ablation/ablation_results.csv` and `report/figures/ablation_logo_only.pdf`.
2. **If `data/features/` path error occurs:** Verify current working directory is the repo root (`pwd` / `Get-Location`), since `src/config.py:19` uses `ROOT_DIR = Path(".")`.
3. **If Python environment issue occurs:** Fall back to `pytest -q` or `python tools/check_numbers.py` and walk through `src/ablation_study.py` lines `89–353` directly in the editor.
