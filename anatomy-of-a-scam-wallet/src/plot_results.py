#!/usr/bin/env python3
"""
Phase 6 (Step 6B) — Ablation, Learning-Curve, Importance, & Prediction Breakdown Figures
Project: Anatomy of a Scam Wallet (Ethereum Fraud Detection with Random Forest & XGBoost)

Generates (PDF + PNG in report/figures/, serif, 300 dpi, #1f4e78 RF / #c0392b XGB):
  1. ablation_logo_bars.{pdf,png}
  2. ablation_only_bars.{pdf,png}
  3. group_importance.{pdf,png}
  4. feature_importance_top15.{pdf,png}
  5. roc_pr_full.{pdf,png}
  6. training_history.{pdf,png}
  7. prediction_breakdown.{pdf,png}
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, learning_curve, train_test_split

from src.config import (
    ALL_FEATURES,
    FEATURE_GROUPS,
    ID_COL,
    ROOT_DIR,
    SEED,
    TARGET,
    TEST_SCALED_PATH,
    TRAIN_SCALED_PATH,
    TUNING_OUTPUT_DIR,
)
from src.models import make_pipeline

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.0,
    "axes.titlesize": 10.0,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0,
    "figure.dpi": 300,
})

COLOR_RF = "#1f4e78"
COLOR_XGB = "#c0392b"
GROUP_ORDER = ["G_temp", "G_val", "G_net", "G_tok"]
GROUP_COLORS = {
    "G_temp": "#1f4e78",
    "G_val": "#2e7d32",
    "G_net": "#6a1b9a",
    "G_tok": "#c0392b",
}

ABLATION_DIR = ROOT_DIR / "outputs" / "ablation"
FIG_DIR = ROOT_DIR / "report" / "figures"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("plot_results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 6: Generate Ablation & Deep-Dive Figures")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def save_fig(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def plot_ablation_bars(res_df: pd.DataFrame, ctrl_df: pd.DataFrame) -> None:
    x = np.arange(len(GROUP_ORDER))
    w = 0.35

    # 1. LOGO Delta F1
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for offset, model, color, label in [
        (-w / 2, "rf", COLOR_RF, "Random Forest"),
        (+w / 2, "xgb", COLOR_XGB, "XGBoost"),
    ]:
        sub = res_df[(res_df["model"] == model) & (res_df["family"] == "LOGO")].set_index("group").reindex(GROUP_ORDER)
        vals = sub["delta_f1_vs_full"].to_numpy()
        err_lo = np.maximum(0.0, vals - sub["delta_ci95_low"].to_numpy())
        err_hi = np.maximum(0.0, sub["delta_ci95_high"].to_numpy() - vals)
        ax.bar(x + offset, vals, width=w, yerr=[err_lo, err_hi], color=color, alpha=0.9, label=label, capsize=3, edgecolor="black", linewidth=0.5)

    # Shaded random leave-out band (k=6 representative)
    ctrl_lo = ctrl_df[ctrl_df["family"] == "RAND_LEAVE_k6"]
    if not ctrl_lo.empty:
        full_rf = float(res_df[(res_df["model"] == "rf") & (res_df["config"] == "FULL")].iloc[0]["cv_f1_mean"])
        band_lo = float(ctrl_lo["rand_f1_p025"].min()) - full_rf
        band_hi = float(ctrl_lo["rand_f1_p975"].max()) - full_rf
        ax.axhspan(band_lo, band_hi, color="gray", alpha=0.18, label="Random Control Band (k=6)")

    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_ORDER)
    ax.set_ylabel(r"$\Delta F_1$ vs. FULL Model")
    ax.set_title("Leave-One-Group-Out (LOGO) Ablation Impact on Fraud-Class $F_1$", fontweight="bold")
    ax.legend(loc="lower left", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_fig(fig, "ablation_logo_bars")

    # 2. ONLY F1
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for offset, model, color, label in [
        (-w / 2, "rf", COLOR_RF, "Random Forest"),
        (+w / 2, "xgb", COLOR_XGB, "XGBoost"),
    ]:
        sub = res_df[(res_df["model"] == model) & (res_df["family"] == "ONLY")].set_index("group").reindex(GROUP_ORDER)
        vals = sub["cv_f1_mean"].to_numpy()
        err_lo = np.maximum(0.0, vals - sub["oof_f1_ci95_low"].to_numpy())
        err_hi = np.maximum(0.0, sub["oof_f1_ci95_high"].to_numpy() - vals)
        ax.bar(x + offset, vals, width=w, yerr=[err_lo, err_hi], color=color, alpha=0.9, label=label, capsize=3, edgecolor="black", linewidth=0.5)

    full_rf = float(res_df[(res_df["model"] == "rf") & (res_df["config"] == "FULL")].iloc[0]["cv_f1_mean"])
    full_xgb = float(res_df[(res_df["model"] == "xgb") & (res_df["config"] == "FULL")].iloc[0]["cv_f1_mean"])
    ax.axhline(full_rf, color=COLOR_RF, linestyle="--", lw=1.2, label=f"RF FULL ({full_rf:.3f})")
    ax.axhline(full_xgb, color=COLOR_XGB, linestyle=":", lw=1.4, label=f"XGB FULL ({full_xgb:.3f})")

    ctrl_only = ctrl_df[ctrl_df["family"] == "RAND_ONLY_k6"]
    if not ctrl_only.empty:
        ax.axhspan(float(ctrl_only["rand_f1_p025"].min()), float(ctrl_only["rand_f1_p975"].max()), color="gray", alpha=0.18, label="Random Control Band (k=6)")

    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_ORDER)
    ax.set_ylabel("Fraud-Class $F_1$ (5x5 CV)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Single-Group-Only Performance vs. FULL Model", fontweight="bold")
    ax.legend(loc="lower right", frameon=True, ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_fig(fig, "ablation_only_bars")


def plot_importance_figures(grp_imp_df: pd.DataFrame, feat_imp_df: pd.DataFrame) -> None:
    x = np.arange(len(GROUP_ORDER))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), gridspec_kw={"wspace": 0.25})
    for offset, model, color, label in [(-w / 2, "rf", COLOR_RF, "Random Forest"), (+w / 2, "xgb", COLOR_XGB, "XGBoost")]:
        sub = grp_imp_df[grp_imp_df["model"] == model].set_index("group").reindex(GROUP_ORDER)
        axes[0].bar(x + offset, sub["perm_f1_drop_mean"], width=w, yerr=sub["perm_f1_drop_std"], color=color, label=label, capsize=2, edgecolor="black", linewidth=0.5)
        if sub["sum_mean_abs_shap"].notna().any():
            axes[1].bar(x + offset, sub["sum_mean_abs_shap"].fillna(0.0), width=w, color=color, label=label, edgecolor="black", linewidth=0.5)

    axes[0].set_title("(a) Grouped Permutation Importance (Test)", fontweight="bold")
    axes[0].set_ylabel(r"Fraud-$F_1$ Drop ($\Delta F_1$)")
    axes[1].set_title("(b) Summed Mean $|\\mathrm{SHAP}|$ by Group (Test)", fontweight="bold")
    axes[1].set_ylabel(r"$\sum \mathrm{mean}(|\mathrm{SHAP}|)$")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(GROUP_ORDER)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=True)
    save_fig(fig, "group_importance")

    # Top-15 Per-Feature Importance colored by group
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), gridspec_kw={"wspace": 0.55})
    for ax, model, title in [(axes[0], "rf", "(a) Random Forest Top-15 Features"), (axes[1], "xgb", "(b) XGBoost Top-15 Features")]:
        sub = feat_imp_df[feat_imp_df["model"] == model].sort_values("native_importance", ascending=False).head(15).iloc[::-1]
        colors = [GROUP_COLORS[g] for g in sub["group"]]
        ax.barh(np.arange(len(sub)), sub["native_importance"], color=colors, edgecolor="black", linewidth=0.4)
        ax.set_yticks(np.arange(len(sub)))
        ax.set_yticklabels(sub["feature"], fontsize=7.5)
        ax.set_xlabel("Native Feature Importance")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g], label=g) for g in GROUP_ORDER]
    axes[1].legend(handles=handles, title="Group", loc="lower right", frameon=True)
    save_fig(fig, "feature_importance_top15")


def plot_roc_pr_and_training(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    res_df: pd.DataFrame,
    best_params: Dict[str, Dict],
    seed: int,
) -> None:
    X_tr, y_tr = train_df[ALL_FEATURES], train_df[TARGET].astype(int).to_numpy()
    X_te, y_te = test_df[ALL_FEATURES], test_df[TARGET].astype(int).to_numpy()

    best_only_xgb = res_df[(res_df["model"] == "xgb") & (res_df["family"] == "ONLY")].sort_values("cv_f1_mean", ascending=False).iloc[0]["group"]
    specs = [
        ("XGB FULL", "xgb", ALL_FEATURES, COLOR_XGB, "-"),
        ("RF FULL", "rf", ALL_FEATURES, COLOR_RF, "-"),
        (f"XGB ONLY_{best_only_xgb}", "xgb", FEATURE_GROUPS[best_only_xgb], "#e67e22", "--"),
        ("XGB NO_LIFESPAN", "xgb", [f for f in ALL_FEATURES if f != "G_temp_lifespan_days"], "#6a1b9a", "-."),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), gridspec_kw={"wspace": 0.25})
    for label, m_name, feats, col, ls in specs:
        pipe = make_pipeline(m_name, params=best_params[m_name]["chosen_params"], features=feats)
        pipe.fit(X_tr, y_tr)
        p = pipe.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, p)
        prec, rec, _ = precision_recall_curve(y_te, p)
        auc_v = roc_auc_score(y_te, p)
        axes[0].plot(fpr, tpr, color=col, linestyle=ls, lw=1.6, label=f"{label} ({auc_v:.3f})")
        axes[1].plot(rec, prec, color=col, linestyle=ls, lw=1.6, label=label)

    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    axes[0].set_title("(a) ROC Curves (Test Set)", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right", fontsize=7.5)
    axes[1].set_title("(b) Precision-Recall Curves (Test Set)", fontweight="bold")
    axes[1].set_xlabel("Recall (Fraud Class)")
    axes[1].set_ylabel("Precision (Fraud Class)")
    axes[1].legend(loc="lower left", fontsize=7.5)
    for ax in axes:
        ax.grid(alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    save_fig(fig, "roc_pr_full")

    # Training History: XGB val log-loss, RF OOB score vs n_estimators, Learning Curve
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), gridspec_kw={"wspace": 0.30})

    X_t_sub, X_v_sub, y_t_sub, y_v_sub = train_test_split(X_tr, y_tr, test_size=0.20, stratify=y_tr, random_state=seed)
    xgb_p = dict(best_params["xgb"]["chosen_params"])
    xgb_est = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist", random_state=seed, n_jobs=-1, **xgb_p)
    xgb_est.fit(X_t_sub, y_t_sub, eval_set=[(X_t_sub, y_t_sub), (X_v_sub, y_v_sub)], verbose=False)
    ev = xgb_est.evals_result()
    axes[0].plot(ev["validation_0"]["logloss"], color="#34495e", lw=1.4, label="Train")
    axes[0].plot(ev["validation_1"]["logloss"], color=COLOR_XGB, lw=1.6, label="Val (20% Train)")
    axes[0].set_title("(a) XGBoost Log-Loss vs. Trees", fontweight="bold")
    axes[0].set_xlabel("Boosting Round")
    axes[0].set_ylabel("Binary Log-Loss")
    axes[0].legend(frameon=True)

    rf_p = dict(best_params["rf"]["chosen_params"])
    n_grid = [25, 50, 100, 150, 200, 300, 400, 500]
    oob_scores = []
    for nt in n_grid:
        p_tmp = {**rf_p, "n_estimators": nt}
        rf_est = RandomForestClassifier(oob_score=True, random_state=seed, n_jobs=-1, **p_tmp)
        rf_est.fit(X_tr, y_tr)
        oob_scores.append(rf_est.oob_score_)
    axes[1].plot(n_grid, oob_scores, "o-", color=COLOR_RF, lw=1.6)
    axes[1].set_title("(b) RF OOB Accuracy vs. Trees", fontweight="bold")
    axes[1].set_xlabel("n_estimators")
    axes[1].set_ylabel("OOB Accuracy")

    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fracs = np.linspace(0.2, 1.0, 5)
    for m_name, col, lbl in [("rf", COLOR_RF, "RF"), ("xgb", COLOR_XGB, "XGB")]:
        pipe = make_pipeline(m_name, params=best_params[m_name]["chosen_params"], features=ALL_FEATURES)
        sizes, _, val_sc = learning_curve(pipe, X_tr, y_tr, cv=cv5, train_sizes=fracs, scoring="f1", n_jobs=-1)
        axes[2].plot(sizes, val_sc.mean(axis=1), "o-", color=col, lw=1.6, label=lbl)
    axes[2].set_title("(c) Learning Curve (5-Fold Train CV)", fontweight="bold")
    axes[2].set_xlabel("Training Wallets")
    axes[2].set_ylabel("Fraud-Class $F_1$")
    axes[2].legend(loc="lower right", frameon=True)

    for ax in axes:
        ax.grid(alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    save_fig(fig, "training_history")


def plot_prediction_breakdown(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    best_params: Dict[str, Dict],
    seed: int,
) -> None:
    X_tr, y_tr = train_df[ALL_FEATURES], train_df[TARGET].astype(int).to_numpy()
    X_te, y_te = test_df[ALL_FEATURES], test_df[TARGET].astype(int).to_numpy()

    pipe_rf = make_pipeline("rf", params=best_params["rf"]["chosen_params"], features=ALL_FEATURES).fit(X_tr, y_tr)
    pipe_xgb = make_pipeline("xgb", params=best_params["xgb"]["chosen_params"], features=ALL_FEATURES).fit(X_tr, y_tr)

    p_rf = pipe_rf.predict_proba(X_te)[:, 1]
    p_xgb = pipe_xgb.predict_proba(X_te)[:, 1]
    pred_xgb = (p_xgb >= 0.5).astype(int)

    booster = pipe_xgb.named_steps["clf"].get_booster()
    dmat = xgb.DMatrix(X_te[ALL_FEATURES].to_numpy(), feature_names=ALL_FEATURES)
    contribs = booster.predict(dmat, pred_contribs=True)[:, :-1]

    rng = np.random.default_rng(seed)
    buckets = {
        "TP": np.where((y_te == 1) & (pred_xgb == 1))[0],
        "TN": np.where((y_te == 0) & (pred_xgb == 0))[0],
        "FP": np.where((y_te == 0) & (pred_xgb == 1))[0],
        "FN": np.where((y_te == 1) & (pred_xgb == 0))[0],
    }
    chosen_idx: List[tuple] = []
    for tag, count in [("TP", 2), ("TN", 2), ("FP", 1), ("FN", 1)]:
        pool = buckets[tag]
        if len(pool) > 0:
            take = rng.choice(pool, size=min(count, len(pool)), replace=False)
            for idx in take:
                chosen_idx.append((tag, int(idx)))

    if not chosen_idx:
        return

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2), gridspec_kw={"wspace": 0.55, "hspace": 0.45})
    for ax, (tag, idx) in zip(axes.flat, chosen_idx):
        sv = contribs[idx]
        top5 = np.argsort(np.abs(sv))[-5:]
        f_names = [ALL_FEATURES[j].replace("G_temp_", "T:").replace("G_val_", "V:").replace("G_net_", "N:").replace("G_tok_", "K:") for j in top5]
        vals = sv[top5]
        cols = [COLOR_XGB if v > 0 else COLOR_RF for v in vals]
        ax.barh(np.arange(len(top5)), vals, color=cols, edgecolor="black", linewidth=0.4)
        ax.axvline(0.0, color="black", lw=0.8)
        ax.set_yticks(np.arange(len(top5)))
        ax.set_yticklabels(f_names, fontsize=7.0)
        w_short = str(test_df.iloc[idx][ID_COL])[:10] + "..."
        ax.set_title(f"{tag} ({w_short})\nRF={p_rf[idx]:.2f} | XGB={p_xgb[idx]:.2f}", fontsize=8.0, fontweight="bold")
        ax.set_xlabel("XGB SHAP (log-odds)", fontsize=7.5)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    save_fig(fig, "prediction_breakdown")


def main() -> None:
    args = parse_args()
    res_path = ABLATION_DIR / "ablation_results.csv"
    ctrl_path = ABLATION_DIR / "controls_table.csv"
    grp_imp_path = ABLATION_DIR / "group_importance.csv"
    feat_imp_path = ABLATION_DIR / "feature_importance.csv"
    best_params_path = TUNING_OUTPUT_DIR / "best_params.json"

    if res_path.exists() and ctrl_path.exists():
        plot_ablation_bars(pd.read_csv(res_path), pd.read_csv(ctrl_path))
    else:
        logger.warning("Skipping ablation bar charts: missing ablation_results.csv or controls_table.csv.")

    if grp_imp_path.exists() and feat_imp_path.exists():
        plot_importance_figures(pd.read_csv(grp_imp_path), pd.read_csv(feat_imp_path))
    else:
        logger.warning("Skipping importance figures: missing group_importance.csv or feature_importance.csv.")

    if TRAIN_SCALED_PATH.exists() and TEST_SCALED_PATH.exists() and best_params_path.exists() and res_path.exists():
        train_df = pd.read_parquet(TRAIN_SCALED_PATH)
        test_df = pd.read_parquet(TEST_SCALED_PATH)
        with open(best_params_path, "r", encoding="utf-8") as f:
            best_params = json.load(f)
        plot_roc_pr_and_training(train_df, test_df, pd.read_csv(res_path), best_params, seed=args.seed)
        plot_prediction_breakdown(train_df, test_df, best_params, seed=args.seed)
    else:
        logger.warning("Skipping ROC/PR, training history, and prediction breakdown: missing train/test/best_params.")

    logger.info("Finished generating Phase 6 figures in %s.", FIG_DIR)


if __name__ == "__main__":
    main()