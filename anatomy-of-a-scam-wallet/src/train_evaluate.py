#!/usr/bin/env python3
"""
Phase 5 (Step 5A) — Held-Out Test Evaluation, H2 Hypothesis Test, 5x5 Repeated CV,
                     and Prevalence-Adjusted Accuracy Analysis
Project: Anatomy of a Scam Wallet (Ethereum Fraud Detection with Random Forest & XGBoost)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold

from src.config import (
    ALL_FEATURES,
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
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
})

COLOR_RF = "#1f4e78"
COLOR_XGB = "#c0392b"

EVAL_DIR = ROOT_DIR / "outputs" / "eval"
FIG_DIR = ROOT_DIR / "report" / "figures"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5: Held-Out Evaluation, H2 Test, 5x5 Repeated CV, & Prevalence Analysis"
    )
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def compute_point_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "fpr": float(fp / max(fp + tn, 1)),
        "fnr": float(fn / max(fn + tp, 1)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def bootstrap_metrics_and_h2(
    y_true: np.ndarray,
    proba_rf: np.ndarray,
    proba_xgb: np.ndarray,
    threshold: float,
    n_boot: int,
    seed: int,
) -> Tuple[Dict[str, Dict[str, Tuple[float, float]]], Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    keys = ["precision", "recall", "f1", "auc_roc", "pr_auc", "accuracy", "mcc", "fpr", "fnr"]
    boot_rf: Dict[str, List[float]] = {k: [] for k in keys}
    boot_xgb: Dict[str, List[float]] = {k: [] for k in keys}
    delta_f1_list: List[float] = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        if len(np.unique(yb)) < 2:
            continue
        m_rf = compute_point_metrics(yb, proba_rf[idx], threshold=threshold)
        m_xgb = compute_point_metrics(yb, proba_xgb[idx], threshold=threshold)
        for k in keys:
            boot_rf[k].append(m_rf[k])
            boot_xgb[k].append(m_xgb[k])
        delta_f1_list.append(m_xgb["f1"] - m_rf["f1"])

    cis_rf = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in boot_rf.items()}
    cis_xgb = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in boot_xgb.items()}

    delta_arr = np.asarray(delta_f1_list, dtype=float)
    p_two_sided = min(float(2.0 * min(np.mean(delta_arr <= 0.0), np.mean(delta_arr >= 0.0))), 1.0)

    pred_rf = (proba_rf >= threshold).astype(int)
    pred_xgb = (proba_xgb >= threshold).astype(int)
    rf_ok = pred_rf == y_true
    xgb_ok = pred_xgb == y_true
    b_rf_only = int(np.sum(rf_ok & (~xgb_ok)))
    c_xgb_only = int(np.sum((~rf_ok) & xgb_ok))
    n_disc = b_rf_only + c_xgb_only
    mcnemar_p = float(binomtest(min(b_rf_only, c_xgb_only), n_disc, 0.5).pvalue) if n_disc > 0 else 1.0

    delta_mean = float(np.mean(delta_arr))
    h2_supported = (delta_mean >= 0.015) and (p_two_sided < 0.05)

    h2_result = {
        "delta_f1_point": float(
            compute_point_metrics(y_true, proba_xgb, threshold)["f1"]
            - compute_point_metrics(y_true, proba_rf, threshold)["f1"]
        ),
        "delta_f1_boot_mean": delta_mean,
        "delta_f1_ci95_low": float(np.percentile(delta_arr, 2.5)),
        "delta_f1_ci95_high": float(np.percentile(delta_arr, 97.5)),
        "paired_bootstrap_p_two_sided": p_two_sided,
        "mcnemar_rf_only_correct": b_rf_only,
        "mcnemar_xgb_only_correct": c_xgb_only,
        "mcnemar_exact_p": mcnemar_p,
        "h2_threshold_delta_f1": 0.015,
        "h2_supported": bool(h2_supported),
    }
    return {"rf": cis_rf, "xgb": cis_xgb}, h2_result


def run_repeated_cv(
    full_df: pd.DataFrame,
    rf_params: Dict[str, Any],
    xgb_params: Dict[str, Any],
    n_splits: int,
    n_repeats: int,
    threshold: float,
    seed: int,
) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    X_all = full_df[ALL_FEATURES]
    y_all = full_df[TARGET].astype(int).to_numpy()
    n_samples = len(full_df)

    oof_sum_rf = np.zeros(n_samples, dtype=float)
    oof_sum_xgb = np.zeros(n_samples, dtype=float)
    oof_counts = np.zeros(n_samples, dtype=int)

    fold_metrics: Dict[str, Dict[str, List[float]]] = {
        "rf": {"f1": [], "precision": [], "recall": [], "auc_roc": []},
        "xgb": {"f1": [], "precision": [], "recall": [], "auc_roc": []},
    }

    for train_idx, val_idx in rskf.split(X_all, y_all):
        X_tr, y_tr = X_all.iloc[train_idx], y_all[train_idx]
        X_va, y_va = X_all.iloc[val_idx], y_all[val_idx]

        for name, params, oof_sum in [("rf", rf_params, oof_sum_rf), ("xgb", xgb_params, oof_sum_xgb)]:
            pipe = make_pipeline(name, params=params, features=ALL_FEATURES)
            pipe.fit(X_tr, y_tr)
            p_va = pipe.predict_proba(X_va)[:, 1]
            oof_sum[val_idx] += p_va

            pred_va = (p_va >= threshold).astype(int)
            fold_metrics[name]["f1"].append(float(f1_score(y_va, pred_va, pos_label=1, zero_division=0)))
            fold_metrics[name]["precision"].append(float(precision_score(y_va, pred_va, pos_label=1, zero_division=0)))
            fold_metrics[name]["recall"].append(float(recall_score(y_va, pred_va, pos_label=1, zero_division=0)))
            fold_metrics[name]["auc_roc"].append(float(roc_auc_score(y_va, p_va)))

        oof_counts[val_idx] += 1

    assert np.all(oof_counts == n_repeats)
    oof_df = pd.DataFrame({
        ID_COL: full_df[ID_COL].values,
        TARGET: y_all,
        "proba_rf": oof_sum_rf / oof_counts,
        "proba_xgb": oof_sum_xgb / oof_counts,
    })

    summary: Dict[str, Dict[str, float]] = {}
    for name in ["rf", "xgb"]:
        summary[name] = {}
        for metric, vals in fold_metrics[name].items():
            summary[name][f"{metric}_mean"] = float(np.mean(vals))
            summary[name][f"{metric}_std"] = float(np.std(vals))
    return summary, oof_df


def build_prevalence_table(
    y_test: np.ndarray,
    m_rf: Dict[str, Any],
    m_xgb: Dict[str, Any],
    seed: int,
    out_dir: Path,
) -> pd.DataFrame:
    dummy_norm = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y_test), 1)), y_test)
    dummy_strat = DummyClassifier(strategy="stratified", random_state=seed).fit(np.zeros((len(y_test), 1)), y_test)

    p_norm = dummy_norm.predict(np.zeros((len(y_test), 1)))
    p_strat = dummy_strat.predict(np.zeros((len(y_test), 1)))
    logger.info(
        "Test Dummy Baselines | Always-Normal Acc=%.4f, F1=%.4f | Stratified Random Acc=%.4f, F1=%.4f",
        accuracy_score(y_test, p_norm), f1_score(y_test, p_norm, pos_label=1, zero_division=0),
        accuracy_score(y_test, p_strat), f1_score(y_test, p_strat, pos_label=1, zero_division=0),
    )

    rows = []
    for pi in [0.50, 0.10, 0.05, 0.01]:
        for model_label, tpr, fpr in [
            ("Always-Normal Dummy", 0.0, 0.0),
            ("Random Forest", m_rf["recall"], m_rf["fpr"]),
            ("XGBoost", m_xgb["recall"], m_xgb["fpr"]),
        ]:
            denom = pi * tpr + (1.0 - pi) * fpr
            prec = (pi * tpr / denom) if denom > 0 else 0.0
            f1_adj = (2.0 * prec * tpr / (prec + tpr)) if (prec + tpr) > 0 else 0.0
            acc_adj = pi * tpr + (1.0 - pi) * (1.0 - fpr)
            rows.append({
                "prevalence_pct": int(round(pi * 100)),
                "model": model_label,
                "tpr_recall": round(tpr, 4),
                "fpr": round(fpr, 4),
                "precision": round(prec, 4),
                "f1": round(f1_adj, 4),
                "accuracy": round(acc_adj, 4),
            })

    prev_df = pd.DataFrame(rows)
    prev_df.to_csv(out_dir / "prevalence_table.csv", index=False)

    tex_lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Prevalence-Adjusted Precision, Fraud $F_1$, and Accuracy Across Real-World Base Rates}",
        r"\label{tab:prevalence}",
        r"\footnotesize",
        r"\begin{tabularx}{\linewidth}{r l c c c}",
        r"\toprule",
        r"\textbf{Fraud $\pi$} & \textbf{Model} & \textbf{Precision} & \textbf{Fraud $F_1$} & \textbf{Accuracy} \\",
        r"\midrule",
    ]
    for _, r in prev_df.iterrows():
        tex_lines.append(
            f"{r['prevalence_pct']}\\% & {r['model']} & {r['precision']:.4f} & {r['f1']:.4f} & {r['accuracy']:.4f} \\\\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    (out_dir / "prevalence_table.tex").write_text("\n".join(tex_lines), encoding="utf-8")
    return prev_df


def write_metrics_latex_table(
    m_rf: Dict[str, Any],
    m_xgb: Dict[str, Any],
    cis: Dict[str, Dict[str, Tuple[float, float]]],
    out_path: Path,
) -> None:
    metric_labels = [
        ("f1", "Fraud-Class $F_1$"),
        ("precision", "Precision"),
        ("recall", "Recall (TPR)"),
        ("auc_roc", "AUC-ROC"),
        ("pr_auc", "PR-AUC"),
        ("accuracy", "Accuracy"),
        ("mcc", "MCC"),
        ("fpr", "False Positive Rate"),
        ("fnr", "False Negative Rate"),
    ]
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Held-Out Test Set Performance ($N=505$, Mean and 95\% Bootstrap CI)}",
        r"\label{tab:test_metrics}",
        r"\footnotesize",
        r"\begin{tabularx}{\linewidth}{X c c}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Random Forest} & \textbf{XGBoost} \\",
        r"\midrule",
    ]
    for key, label in metric_labels:
        rf_str = f"{m_rf[key]:.4f} [{cis['rf'][key][0]:.4f}, {cis['rf'][key][1]:.4f}]"
        xgb_str = f"{m_xgb[key]:.4f} [{cis['xgb'][key][0]:.4f}, {cis['xgb'][key][1]:.4f}]"
        lines.append(f"{label} & {rf_str} & {xgb_str} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def generate_eval_figures(
    y_test: np.ndarray,
    proba_rf: np.ndarray,
    proba_xgb: np.ndarray,
    m_rf: Dict[str, Any],
    m_xgb: Dict[str, Any],
    threshold: float,
    fig_dir: Path,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ROC Curves
    fpr_rf, tpr_rf, _ = roc_curve(y_test, proba_rf)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, proba_xgb)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.plot(fpr_rf, tpr_rf, color=COLOR_RF, lw=1.8, label=f"Random Forest (AUC = {m_rf['auc_roc']:.3f})")
    ax.plot(fpr_xgb, tpr_xgb, color=COLOR_XGB, lw=1.8, label=f"XGBoost (AUC = {m_xgb['auc_roc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.6, label="Random Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Held-Out Test ROC Curves", fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(fig_dir / "roc_curves.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "roc_curves.png", bbox_inches="tight")
    plt.close(fig)

    # PR Curves
    p_rf, r_rf, _ = precision_recall_curve(y_test, proba_rf)
    p_xgb, r_xgb, _ = precision_recall_curve(y_test, proba_xgb)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.plot(r_rf, p_rf, color=COLOR_RF, lw=1.8, label=f"Random Forest (PR-AUC = {m_rf['pr_auc']:.3f})")
    ax.plot(r_xgb, p_xgb, color=COLOR_XGB, lw=1.8, label=f"XGBoost (PR-AUC = {m_xgb['pr_auc']:.3f})")
    ax.axhline(float(np.mean(y_test)), color="black", linestyle="--", lw=1.0, alpha=0.6, label="Baseline (0.50)")
    ax.set_xlabel("Recall (Fraud Class)")
    ax.set_ylabel("Precision (Fraud Class)")
    ax.set_title("Held-Out Test Precision-Recall Curves", fontweight="bold")
    ax.legend(loc="lower left", frameon=True)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(fig_dir / "pr_curves.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "pr_curves.png", bbox_inches="tight")
    plt.close(fig)

    # Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"wspace": 0.28})
    for ax, proba, title, cmap in [
        (axes[0], proba_rf, "Random Forest", "Blues"),
        (axes[1], proba_xgb, "XGBoost", "Reds"),
    ]:
        cm = confusion_matrix(y_test, (proba >= threshold).astype(int), labels=[0, 1])
        ax.imshow(cm, cmap=cmap, aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Normal (0)", "Pred Fraud (1)"])
        ax.set_yticklabels(["True Normal (0)", "True Fraud (1)"])
        ax.set_title(title, fontweight="bold")
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, f"{cm[i, j]:,}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10.5, fontweight="bold",
                )
    fig.savefig(fig_dir / "confusion_matrices.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)

    # Calibration Curve
    prob_true_rf, prob_pred_rf = calibration_curve(y_test, proba_rf, n_bins=10, strategy="uniform")
    prob_true_xgb, prob_pred_xgb = calibration_curve(y_test, proba_xgb, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.plot(prob_pred_rf, prob_true_rf, "s-", color=COLOR_RF, lw=1.6, label="Random Forest")
    ax.plot(prob_pred_xgb, prob_true_xgb, "o-", color=COLOR_XGB, lw=1.6, label="XGBoost")
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.6, label="Perfectly Calibrated")
    ax.set_xlabel("Mean Predicted Fraud Probability")
    ax.set_ylabel("Observed Fraction of Fraud Wallets")
    ax.set_title("Probability Calibration Curve (10 Bins)", fontweight="bold")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(fig_dir / "calibration_curve.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "calibration_curve.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    best_params_path = TUNING_OUTPUT_DIR / "best_params.json"
    for p in (TRAIN_SCALED_PATH, TEST_SCALED_PATH, best_params_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(TRAIN_SCALED_PATH)
    test_df = pd.read_parquet(TEST_SCALED_PATH)
    with open(best_params_path, "r", encoding="utf-8") as f:
        best_params = json.load(f)

    rf_params = best_params["rf"]["chosen_params"]
    xgb_params = best_params["xgb"]["chosen_params"]

    X_train, y_train = train_df[ALL_FEATURES], train_df[TARGET].astype(int).to_numpy()
    X_test, y_test = test_df[ALL_FEATURES], test_df[TARGET].astype(int).to_numpy()

    pipe_rf = make_pipeline("rf", params=rf_params, features=ALL_FEATURES)
    pipe_xgb = make_pipeline("xgb", params=xgb_params, features=ALL_FEATURES)
    pipe_rf.fit(X_train, y_train)
    pipe_xgb.fit(X_train, y_train)

    proba_rf = pipe_rf.predict_proba(X_test)[:, 1]
    proba_xgb = pipe_xgb.predict_proba(X_test)[:, 1]
    pred_rf = (proba_rf >= args.threshold).astype(int)
    pred_xgb = (proba_xgb >= args.threshold).astype(int)

    m_rf = compute_point_metrics(y_test, proba_rf, threshold=args.threshold)
    m_xgb = compute_point_metrics(y_test, proba_xgb, threshold=args.threshold)
    cis, h2_res = bootstrap_metrics_and_h2(
        y_test, proba_rf, proba_xgb, threshold=args.threshold, n_boot=args.n_boot, seed=args.seed
    )

    print("\n=== Held-Out Test Evaluation (N=505, Threshold=0.50) ===")
    for name, m, ci in [("RF", m_rf, cis["rf"]), ("XGB", m_xgb, cis["xgb"])]:
        print(
            f"{name:<4} | F1={m['f1']:.4f} [{ci['f1'][0]:.4f}, {ci['f1'][1]:.4f}] | "
            f"Prec={m['precision']:.4f} | Rec={m['recall']:.4f} | "
            f"AUC={m['auc_roc']:.4f} | PR-AUC={m['pr_auc']:.4f} | Acc={m['accuracy']:.4f} | "
            f"MCC={m['mcc']:.4f} | CM(TN={m['tn']}, FP={m['fp']}, FN={m['fn']}, TP={m['tp']})"
        )

    print(
        f"\nH2 Hypothesis Test (Delta F1 = F1_XGB - F1_RF): "
        f"point={h2_res['delta_f1_point']:+.4f}, boot_mean={h2_res['delta_f1_boot_mean']:+.4f} "
        f"[{h2_res['delta_f1_ci95_low']:+.4f}, {h2_res['delta_f1_ci95_high']:+.4f}], "
        f"paired bootstrap p={h2_res['paired_bootstrap_p_two_sided']:.4f}, "
        f"McNemar exact p={h2_res['mcnemar_exact_p']:.4f}. "
        f"Pre-registered H2 (Delta F1 >= 0.015, p < 0.05) supported: {h2_res['h2_supported']}."
    )

    full_df = pd.concat([train_df, test_df], ignore_index=True)
    cv_summary, oof_df = run_repeated_cv(
        full_df, rf_params=rf_params, xgb_params=xgb_params,
        n_splits=args.cv_splits, n_repeats=args.cv_repeats,
        threshold=args.threshold, seed=args.seed,
    )
    oof_df.to_csv(EVAL_DIR / "oof_predictions.csv", index=False)
    print("\n=== Repeated 5x5 Stratified CV (All 2,522 Wallets, Fixed Tuned Params) ===")
    for name in ["rf", "xgb"]:
        c = cv_summary[name]
        print(
            f"{name.upper():<4} | F1={c['f1_mean']:.4f}+/-{c['f1_std']:.4f} | "
            f"Prec={c['precision_mean']:.4f}+/-{c['precision_std']:.4f} | "
            f"Rec={c['recall_mean']:.4f}+/-{c['recall_std']:.4f} | "
            f"AUC={c['auc_roc_mean']:.4f}+/-{c['auc_roc_std']:.4f}"
        )

    build_prevalence_table(y_test, m_rf, m_xgb, seed=args.seed, out_dir=EVAL_DIR)

    pd.DataFrame({
        ID_COL: test_df[ID_COL].values,
        "y_true": y_test,
        "proba_rf": proba_rf,
        "proba_xgb": proba_xgb,
        "pred_rf": pred_rf,
        "pred_xgb": pred_xgb,
    }).to_csv(EVAL_DIR / "test_predictions.csv", index=False)

    metrics_rows = []
    for m_name, m_dict, ci_dict in [("rf", m_rf, cis["rf"]), ("xgb", m_xgb, cis["xgb"])]:
        row = {"model": m_name}
        for k, v in m_dict.items():
            row[k] = v
            if k in ci_dict:
                row[f"{k}_ci95_low"] = ci_dict[k][0]
                row[f"{k}_ci95_high"] = ci_dict[k][1]
        metrics_rows.append(row)
    pd.DataFrame(metrics_rows).to_csv(EVAL_DIR / "metrics_test.csv", index=False)

    with open(EVAL_DIR / "metrics_test.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_metrics": {"rf": m_rf, "xgb": m_xgb},
                "bootstrap_ci95": cis,
                "h2_hypothesis_test": h2_res,
                "repeated_5x5_cv": cv_summary,
            },
            f,
            indent=2,
        )

    write_metrics_latex_table(m_rf, m_xgb, cis, EVAL_DIR / "metrics_table.tex")
    generate_eval_figures(y_test, proba_rf, proba_xgb, m_rf, m_xgb, args.threshold, FIG_DIR)
    logger.info("Saved all evaluation outputs to %s and figures to %s.", EVAL_DIR, FIG_DIR)


if __name__ == "__main__":
    main()