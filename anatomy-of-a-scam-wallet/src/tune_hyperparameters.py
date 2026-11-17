#!/usr/bin/env python3
"""
Phase 4 — Leakage-Free Hyperparameter Tuning (5-Fold Stratified CV on Train Only)
Project: Anatomy of a Scam Wallet (Ethereum Fraud Detection with Random Forest & XGBoost)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score

from src.config import (
    ALL_FEATURES,
    MODELS_DIR,
    RF_SEARCH_SPACE,
    SEED,
    TARGET,
    TRAIN_SCALED_PATH,
    TUNING_OUTPUT_DIR,
    get_xgb_search_space,
)
from src.models import build_rf, make_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tune_hyperparameters")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4: Hyperparameter Tuning on Train Set Only (Fraud-Class F1)"
    )
    parser.add_argument("--model", choices=["rf", "xgb", "both"], default="both")
    parser.add_argument("--n-iter", type=int, default=None)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--select", choices=["1se", "best"], default="1se")
    parser.add_argument("--fit-final", action="store_true")
    return parser.parse_args()


def escape_latex(text: Any) -> str:
    return str(text).replace("_", r"\_")


def run_leakage_sanity_check(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    n_jobs: int,
) -> float:
    scorer = make_scorer(f1_score, pos_label=1)
    rf_default = build_rf(n_jobs=n_jobs)
    scores = cross_val_score(rf_default, X_train[ALL_FEATURES], y_train, cv=cv, scoring=scorer, n_jobs=n_jobs)
    mean_f1 = float(np.mean(scores))
    std_f1 = float(np.std(scores))

    rf_default.fit(X_train[ALL_FEATURES], y_train)
    imp_s = pd.Series(rf_default.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)

    print("\n=== Pre-Tuning Sanity & Leakage Check (Default RF, 5-Fold Train CV) ===")
    print(f"Default RF Fraud-Class F1: {mean_f1:.4f} +/- {std_f1:.4f}")
    if mean_f1 > 0.98:
        logger.warning(
            "WARNING: Default RF CV F1 = %.4f (> 0.98)! Inspect features for potential leakage or sampling artifacts.",
            mean_f1,
        )
    print("Top-10 Feature Importances (Gini):")
    for fname, val in imp_s.head(10).items():
        print(f"  {fname:<34} {val:.4f}")
    return mean_f1


def select_1se_index(cv_df: pd.DataFrame, model_name: str, cv_folds: int) -> Tuple[int, int]:
    best_idx = int(cv_df["mean_test_score"].idxmax())
    best_mean = float(cv_df.loc[best_idx, "mean_test_score"])
    best_se = float(cv_df.loc[best_idx, "std_test_score"]) / np.sqrt(cv_folds)
    threshold = best_mean - best_se

    eligible = cv_df[cv_df["mean_test_score"] >= threshold].copy()
    if model_name == "rf":
        eligible["_depth_sort"] = eligible["param_clf__max_depth"].apply(
            lambda d: 9999 if pd.isna(d) or d is None else int(d)
        )
        eligible = eligible.sort_values(
            by=["_depth_sort", "param_clf__n_estimators", "param_clf__min_samples_leaf"],
            ascending=[True, True, False],
        )
    else:
        eligible = eligible.sort_values(
            by=["param_clf__max_depth", "param_clf__n_estimators", "param_clf__reg_lambda"],
            ascending=[True, True, False],
        )

    chosen_idx = int(eligible.index[0])
    return best_idx, chosen_idx


def extract_clean_params(row: pd.Series) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for col in row.index:
        if col.startswith("param_clf__"):
            key = col.replace("param_clf__", "")
            val = row[col]
            if pd.isna(val):
                val = None
            elif isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = float(val)
            params[key] = val
    return params


def write_tuning_latex_table(
    cv_df: pd.DataFrame,
    chosen_idx: int,
    out_path: Path,
    model_label: str,
    model_key: str,
) -> None:
    top10 = cv_df.sort_values("mean_test_score", ascending=False).head(10).copy()
    if chosen_idx not in top10.index:
        top10 = pd.concat([top10.iloc[:9], cv_df.loc[[chosen_idx]]])

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        rf"\caption{{Top-10 {escape_latex(model_label)} Hyperparameter Configurations (5-Fold Train CV)}}",
        rf"\label{{tab:{model_key}_tuning}}",
        r"\footnotesize",
        r"\begin{tabularx}{\linewidth}{r X c c c}",
        r"\toprule",
        r"\textbf{Rank} & \textbf{Key Parameters} & \textbf{Val $F_1$} & \textbf{Train $F_1$} & \textbf{Gap} \\",
        r"\midrule",
    ]

    for _, r in top10.iterrows():
        is_chosen = int(r.name) == chosen_idx
        rank_str = str(int(r["rank_test_score"]))
        if model_key == "rf":
            p_str = (
                f"n={r['param_clf__n_estimators']}, d={r['param_clf__max_depth']}, "
                f"mf={r['param_clf__max_features']}, cw={r['param_clf__class_weight']}"
            )
        else:
            p_str = (
                f"n={r['param_clf__n_estimators']}, lr={r['param_clf__learning_rate']}, "
                f"d={r['param_clf__max_depth']}, sub={r['param_clf__subsample']}, "
                f"spw={r['param_clf__scale_pos_weight']}"
            )
        val_str = f"{r['mean_test_score']:.4f} $\\pm$ {r['std_test_score']:.4f}"
        trn_str = f"{r['mean_train_score']:.4f}"
        gap_str = f"{r['train_val_gap']:.4f}"

        p_esc = escape_latex(p_str)
        if is_chosen:
            row_tex = (
                rf"\textbf{{{rank_str}}} & \textbf{{{p_esc} (Chosen)}} & "
                rf"\textbf{{{val_str}}} & \textbf{{{trn_str}}} & \textbf{{{gap_str}}} \\"
            )
        else:
            row_tex = rf"{rank_str} & {p_esc} & {val_str} & {trn_str} & {gap_str} \\"
        lines.append(row_tex)

    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_search_space_latex_table(xgb_space: Dict[str, List[Any]], out_path: Path) -> None:
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Hyperparameter Search Spaces for Random Forest and XGBoost}",
        r"\label{tab:search_space}",
        r"\footnotesize",
        r"\begin{tabularx}{\linewidth}{l l X}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Hyperparameter} & \textbf{Candidate Values} \\",
        r"\midrule",
    ]
    for param, vals in RF_SEARCH_SPACE.items():
        vals_str = ", ".join(str(v) for v in vals)
        lines.append(rf"RF & \texttt{{{escape_latex(param)}}} & {escape_latex(vals_str)} \\")
    lines.append(r"\midrule")
    for param, vals in xgb_space.items():
        vals_str = ", ".join(str(v) for v in vals)
        lines.append(rf"XGBoost & \texttt{{{escape_latex(param)}}} & {escape_latex(vals_str)} \\")
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def tune_single_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    search_space: Dict[str, List[Any]],
    n_iter: int,
    cv: StratifiedKFold,
    n_jobs: int,
    select_rule: str,
    out_dir: Path,
) -> Dict[str, Any]:
    est_min = 0.5 if model_name == "rf" else 0.8
    logger.info(
        "Starting %s tuning (%d iterations x %d folds = %d fits; est. ~%.1f-%.1f min on CPU)...",
        model_name.upper(), n_iter, cv.n_splits, n_iter * cv.n_splits, est_min, est_min * 4,
    )
    t0 = time.perf_counter()

    base_pipe = make_pipeline(model_name, params={}, features=ALL_FEATURES)
    prefixed_space = {f"clf__{k}": v for k, v in search_space.items()}
    scorer = make_scorer(f1_score, pos_label=1)

    search = RandomizedSearchCV(
        estimator=base_pipe,
        param_distributions=prefixed_space,
        n_iter=n_iter,
        scoring=scorer,
        cv=cv,
        refit=True,
        return_train_score=True,
        random_state=SEED,
        n_jobs=n_jobs,
        verbose=1,
    )
    search.fit(X_train[ALL_FEATURES], y_train)
    elapsed_s = round(time.perf_counter() - t0, 2)

    cv_df = pd.DataFrame(search.cv_results_).copy()
    cv_df["train_val_gap"] = cv_df["mean_train_score"] - cv_df["mean_test_score"]

    best_idx, one_se_idx = select_1se_index(cv_df, model_name=model_name, cv_folds=cv.n_splits)
    chosen_idx = one_se_idx if select_rule == "1se" else best_idx

    best_params_rank1 = extract_clean_params(cv_df.loc[best_idx])
    one_se_params = extract_clean_params(cv_df.loc[one_se_idx])
    chosen_params = one_se_params if select_rule == "1se" else best_params_rank1

    print(
        f"\n[{model_name.upper()}] Selection Rule Justification ({select_rule.upper()}): "
        f"Rank-1 F1 = {cv_df.loc[best_idx, 'mean_test_score']:.4f} +/- {cv_df.loc[best_idx, 'std_test_score']:.4f}; "
        f"1-SE F1 = {cv_df.loc[one_se_idx, 'mean_test_score']:.4f} +/- {cv_df.loc[one_se_idx, 'std_test_score']:.4f} "
        "(1-SE prefers lower complexity within 1 SE to reduce variance on unseen wallets)."
    )

    cv_df.sort_values("rank_test_score").to_csv(out_dir / f"{model_name}_cv_results.csv", index=False)

    label_str = "Random Forest" if model_name == "rf" else "XGBoost"
    write_tuning_latex_table(
        cv_df, chosen_idx=chosen_idx, out_path=out_dir / f"{model_name}_tuning.tex",
        model_label=label_str, model_key=model_name,
    )

    return {
        "model": model_name,
        "selection_rule_used": select_rule,
        "chosen_params": chosen_params,
        "rank1_params": best_params_rank1,
        "one_se_params": one_se_params,
        "cv_f1_mean": round(float(cv_df.loc[chosen_idx, "mean_test_score"]), 4),
        "cv_f1_std": round(float(cv_df.loc[chosen_idx, "std_test_score"]), 4),
        "train_f1_mean": round(float(cv_df.loc[chosen_idx, "mean_train_score"]), 4),
        "train_val_gap": round(float(cv_df.loc[chosen_idx, "train_val_gap"]), 4),
        "rank1_cv_f1_mean": round(float(cv_df.loc[best_idx, "mean_test_score"]), 4),
        "tuning_seconds": elapsed_s,
    }


def main() -> None:
    args = parse_args()
    if not TRAIN_SCALED_PATH.exists():
        raise FileNotFoundError(
            f"Missing training file: {TRAIN_SCALED_PATH}. Run Phase 3 (03_engineer_features.py) first."
        )

    train_df = pd.read_parquet(TRAIN_SCALED_PATH)
    X_train = train_df[ALL_FEATURES].copy()
    y_train = train_df[TARGET].astype(int).copy()

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    logger.info("Loaded train_scaled.parquet: %d wallets (normal=%d, fraud=%d).", len(train_df), n_neg, n_pos)

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=SEED)
    run_leakage_sanity_check(X_train, y_train, cv=cv, n_jobs=args.n_jobs)

    TUNING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    xgb_space = get_xgb_search_space(n_neg=n_neg, n_pos=n_pos)
    write_search_space_latex_table(xgb_space, TUNING_OUTPUT_DIR / "search_space.tex")

    results_summary: Dict[str, Any] = {}
    models_to_run = ["rf", "xgb"] if args.model == "both" else [args.model]

    for m_name in models_to_run:
        n_iter = args.n_iter if args.n_iter is not None else (60 if m_name == "rf" else 80)
        space = RF_SEARCH_SPACE if m_name == "rf" else xgb_space
        results_summary[m_name] = tune_single_model(
            model_name=m_name,
            X_train=X_train,
            y_train=y_train,
            search_space=space,
            n_iter=n_iter,
            cv=cv,
            n_jobs=args.n_jobs,
            select_rule=args.select,
            out_dir=TUNING_OUTPUT_DIR,
        )

    with open(TUNING_OUTPUT_DIR / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    print("\n=== Final Hyperparameter Tuning Summary (Train CV Only) ===")
    print(f"{'Model':<8} | {'CV Fraud-F1 (Mean+/-Std)':<24} | {'Train-Val Gap':<13} | Chosen Params")
    print("-" * 92)
    for m_name, info in results_summary.items():
        f1_str = f"{info['cv_f1_mean']:.4f} +/- {info['cv_f1_std']:.4f}"
        print(f"{m_name.upper():<8} | {f1_str:<24} | {info['train_val_gap']:<13.4f} | {info['chosen_params']}")

    if args.fit_final:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        oversized = False
        for m_name, info in results_summary.items():
            pipe = make_pipeline(m_name, params=info["chosen_params"], features=ALL_FEATURES)
            pipe.fit(X_train, y_train)
            model_path = MODELS_DIR / f"{m_name}_final.joblib"
            joblib.dump(pipe, model_path, compress=3)
            size_mb = model_path.stat().st_size / (1024 * 1024)
            logger.info("Saved final fitted %s model to %s (%.2f MB).", m_name.upper(), model_path, size_mb)
            if size_mb > 100.0:
                oversized = True
                logger.warning("WARNING: %s exceeds 100 MB (%.2f MB)! Do NOT commit to Git.", model_path, size_mb)

        if oversized:
            gitignore_path = Path(".gitignore")
            existing = gitignore_path.read_text(encoding="utf-8") if gitignore_path.exists() else ""
            if "models/" not in existing:
                with open(gitignore_path, "a", encoding="utf-8") as gf:
                    gf.write("\nmodels/\n")
            print(
                "\n[README SNIPPET — Large Model Storage]\n"
                "Model checkpoints exceed GitHub's 100 MB limit (`models/` added to `.gitignore`).\n"
                "Upload `models/rf_final.joblib` and `models/xgb_final.joblib` to Google Drive or Hugging Face Hub."
            )


if __name__ == "__main__":
    main()