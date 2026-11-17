#!/usr/bin/env python3
"""
Phase 5 (Step 5B) — Error Analysis & Algorithmic Bias / Subgroup Audit
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
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.metrics import f1_score, precision_score, recall_score

from src.config import FEATURES_DIR, ID_COL, ROOT_DIR, SEED, TARGET

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

EVAL_DIR = ROOT_DIR / "outputs" / "eval"
FIG_DIR = ROOT_DIR / "report" / "figures"
SAMPLED_WALLETS_PATH = ROOT_DIR / "data" / "raw" / "sampled_wallets.csv"

KEY_INSPECT_FEATURES = [
    "G_temp_lifespan_days",
    "G_temp_sent_tx_count",
    "G_temp_recv_tx_count",
    "G_temp_burstiness_max_1h",
    "G_val_total_eth_sent",
    "G_val_total_eth_recv",
    "G_net_unique_sent_to",
    "G_net_unique_recv_from",
    "G_tok_mean_tx_fee_eth",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("error_bias_analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5: Error Analysis & Subgroup Bias Audit")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def holm_bonferroni(p_values: List[float]) -> List[float]:
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [1.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = (m - rank) * p
        running_max = max(running_max, adj)
        corrected[orig_idx] = min(running_max, 1.0)
    return corrected


def analyze_errors_and_overlap(
    oof_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feat_unscaled: pd.DataFrame,
    threshold: float,
) -> Dict[str, Any]:
    oof_merged = oof_df.merge(feat_unscaled, on=[ID_COL, TARGET], how="inner")
    test_merged = test_df.rename(columns={"y_true": TARGET}).merge(feat_unscaled, on=[ID_COL, TARGET], how="inner")

    inspect_cols = [c for c in KEY_INSPECT_FEATURES if c in feat_unscaled.columns]
    overlap_summary: Dict[str, Any] = {}
    mw_results: Dict[str, Any] = {}

    for model in ["rf", "xgb"]:
        p_col = f"proba_{model}"
        oof_pred = (oof_merged[p_col] >= threshold).astype(int)
        test_pred = (test_merged[p_col] >= threshold).astype(int)

        oof_merged[f"pred_{model}"] = oof_pred
        test_merged[f"pred_{model}"] = test_pred

        for err_type, y_val, pred_val in [("fp", 0, 1), ("fn", 1, 0)]:
            sub_oof = oof_merged[(oof_merged[TARGET] == y_val) & (oof_pred == pred_val)].copy()
            sub_oof["split"] = "oof"
            sub_test = test_merged[(test_merged[TARGET] == y_val) & (test_pred == pred_val)].copy()
            sub_test["split"] = "test"
            out_err = pd.concat([sub_test, sub_oof], ignore_index=True)[
                ["split", ID_COL, TARGET, p_col] + inspect_cols
            ]
            out_err.to_csv(EVAL_DIR / f"errors_{err_type}_{model}.csv", index=False)

        mw_results[model] = {"fn_vs_tp": {}, "fp_vs_tn": {}}
        fn_df = oof_merged[(oof_merged[TARGET] == 1) & (oof_pred == 0)]
        tp_df = oof_merged[(oof_merged[TARGET] == 1) & (oof_pred == 1)]
        fp_df = oof_merged[(oof_merged[TARGET] == 0) & (oof_pred == 1)]
        tn_df = oof_merged[(oof_merged[TARGET] == 0) & (oof_pred == 0)]

        for comp_name, grp_a, grp_b in [("fn_vs_tp", fn_df, tp_df), ("fp_vs_tn", fp_df, tn_df)]:
            for col in inspect_cols:
                a = grp_a[col].to_numpy()
                b = grp_b[col].to_numpy()
                if len(a) >= 2 and len(b) >= 2:
                    u_stat, p_val = mannwhitneyu(a, b, alternative="two-sided")
                    r_eff = 1.0 - (2.0 * float(u_stat)) / float(len(a) * len(b))
                    mw_results[model][comp_name][col] = {
                        "n_a": int(len(a)),
                        "n_b": int(len(b)),
                        "mean_a": float(np.mean(a)),
                        "median_a": float(np.median(a)),
                        "mean_b": float(np.mean(b)),
                        "median_b": float(np.median(b)),
                        "rank_biserial_r": round(r_eff, 4),
                        "p_value": float(p_val),
                    }

    for err_type, y_val, pred_val in [("fp", 0, 1), ("fn", 1, 0)]:
        s_rf = set(oof_merged.loc[(oof_merged[TARGET] == y_val) & (oof_merged["pred_rf"] == pred_val), ID_COL])
        s_xgb = set(oof_merged.loc[(oof_merged[TARGET] == y_val) & (oof_merged["pred_xgb"] == pred_val), ID_COL])
        union = s_rf | s_xgb
        inter = s_rf & s_xgb
        overlap_summary[err_type] = {
            "rf_count": len(s_rf),
            "xgb_count": len(s_xgb),
            "shared_count": len(inter),
            "jaccard_overlap": round(len(inter) / max(len(union), 1), 4),
        }

    return {"mann_whitney": mw_results, "error_overlap_oof": overlap_summary, "inspect_cols": inspect_cols}


def compute_subgroup_metrics(
    df_sub: pd.DataFrame,
    df_rest: pd.DataFrame,
    overall_fpr: float,
    model: str,
    threshold: float,
    n_boot: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    y = df_sub[TARGET].astype(int).to_numpy()
    p = df_sub[f"proba_{model}"].to_numpy()
    pred = (p >= threshold).astype(int)

    n = int(len(y))
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    unreliable = (n < 30) or (n_pos < 10) or (n_neg < 10)

    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())

    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    prec = float(precision_score(y, pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y, pred, pos_label=1, zero_division=0))
    f1_val = float(f1_score(y, pred, pos_label=1, zero_division=0))

    b_fpr, b_fnr, b_f1 = [], [], []
    if n >= 5:
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            yb, pb = y[idx], pred[idx]
            fpb = int(((yb == 0) & (pb == 1)).sum())
            tnb = int(((yb == 0) & (pb == 0)).sum())
            fnb = int(((yb == 1) & (pb == 0)).sum())
            tpb = int(((yb == 1) & (pb == 1)).sum())
            if (fpb + tnb) > 0:
                b_fpr.append(fpb / (fpb + tnb))
            if (fnb + tpb) > 0:
                b_fnr.append(fnb / (fnb + tpb))
            b_f1.append(float(f1_score(yb, pb, pos_label=1, zero_division=0)))

    def _ci(arr: List[float], fallback: float) -> Tuple[float, float]:
        if len(arr) < 10:
            return (fallback, fallback)
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    fpr_ci = _ci(b_fpr, fpr)
    fnr_ci = _ci(b_fnr, fnr)
    f1_ci = _ci(b_f1, f1_val)

    y_r = df_rest[TARGET].astype(int).to_numpy()
    p_r = (df_rest[f"proba_{model}"].to_numpy() >= threshold).astype(int)
    fp_r = int(((y_r == 0) & (p_r == 1)).sum())
    tn_r = int(((y_r == 0) & (p_r == 0)).sum())
    _, fisher_p = fisher_exact([[fp, tn], [fp_r, tn_r]], alternative="two-sided")

    return {
        "n": n,
        "fraud_share": round(n_pos / max(n, 1), 4),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "unreliable": unreliable,
        "fpr": round(fpr, 4),
        "fpr_ci_low": round(fpr_ci[0], 4),
        "fpr_ci_high": round(fpr_ci[1], 4),
        "fnr": round(fnr, 4),
        "fnr_ci_low": round(fnr_ci[0], 4),
        "fnr_ci_high": round(fnr_ci[1], 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1_val, 4),
        "f1_ci_low": round(f1_ci[0], 4),
        "f1_ci_high": round(f1_ci[1], 4),
        "fpr_ratio_vs_overall": round(fpr / max(overall_fpr, 1e-6), 3),
        "fisher_p_raw": float(fisher_p),
    }


def build_subgroup_definitions(df: pd.DataFrame) -> List[Tuple[str, pd.Series]]:
    subgroups: List[Tuple[str, pd.Series]] = []

    if {"G_temp_sent_tx_count", "G_temp_recv_tx_count"}.issubset(df.columns):
        tot_tx = df["G_temp_sent_tx_count"] + df["G_temp_recv_tx_count"]
        subgroups.extend([
            ("Tx Count: <5 (Sparse)", tot_tx < 5),
            ("Tx Count: 5-19", (tot_tx >= 5) & (tot_tx <= 19)),
            ("Tx Count: 20-99", (tot_tx >= 20) & (tot_tx <= 99)),
            ("Tx Count: >=100", tot_tx >= 100),
        ])
    else:
        logger.warning("Skipping tx count tiers: missing G_temp_sent/recv_tx_count.")

    if "G_temp_burstiness_max_1h" in df.columns:
        p90 = float(df["G_temp_burstiness_max_1h"].quantile(0.90))
        subgroups.extend([
            ("Burstiness: Top Decile", df["G_temp_burstiness_max_1h"] >= p90),
            ("Burstiness: Bottom 90%", df["G_temp_burstiness_max_1h"] < p90),
        ])
    else:
        logger.warning("Skipping burstiness subgroup: missing G_temp_burstiness_max_1h.")

    if {"G_val_total_eth_sent", "G_val_total_eth_recv"}.issubset(df.columns):
        tot_val = df["G_val_total_eth_sent"] + df["G_val_total_eth_recv"]
        q33, q67 = float(tot_val.quantile(1 / 3)), float(tot_val.quantile(2 / 3))
        subgroups.extend([
            ("Value Tercile: Low", tot_val <= q33),
            ("Value Tercile: Mid", (tot_val > q33) & (tot_val <= q67)),
            ("Value Tercile: High", tot_val > q67),
        ])
    else:
        logger.warning("Skipping value terciles: missing G_val_total_eth_sent/recv.")

    if "G_temp_lifespan_days" in df.columns:
        ls = df["G_temp_lifespan_days"]
        subgroups.extend([
            ("Lifespan: <7 days", ls < 7.0),
            ("Lifespan: 7-90 days", (ls >= 7.0) & (ls <= 90.0)),
            ("Lifespan: >90 days", ls > 90.0),
        ])
    else:
        logger.warning("Skipping lifespan tiers: missing G_temp_lifespan_days.")

    if "is_contract" in df.columns:
        subgroups.extend([
            ("Entity: EOA (0)", df["is_contract"] == 0),
            ("Entity: Contract (1)", df["is_contract"] == 1),
        ])
    else:
        logger.warning("Skipping is_contract subgroup: column not found in sampled_wallets.csv.")

    return subgroups


def plot_bias_and_error_figures(
    bias_df: pd.DataFrame,
    mw_results: Dict[str, Any],
    inspect_cols: List[str],
    fig_dir: Path,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    oof_bias = bias_df[bias_df["eval_set"] == "oof"].copy()

    sub_names = list(dict.fromkeys(oof_bias["subgroup"].tolist()))
    y_pos = np.arange(len(sub_names))
    height = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(10.5, max(4.5, len(sub_names) * 0.42)), sharey=True, gridspec_kw={"wspace": 0.14})
    for ax, metric, title in [
        (axes[0], "fpr", "(a) False Positive Rate (FPR) by Subgroup"),
        (axes[1], "fnr", "(b) False Negative Rate (FNR) by Subgroup"),
    ]:
        for offset, model, color, label in [
            (-height / 2, "rf", COLOR_RF, "Random Forest"),
            (+height / 2, "xgb", COLOR_XGB, "XGBoost"),
        ]:
            sub_m = oof_bias[oof_bias["model"] == model].set_index("subgroup").reindex(sub_names)
            vals = sub_m[metric].to_numpy()
            err_lo = np.maximum(0.0, vals - sub_m[f"{metric}_ci_low"].to_numpy())
            err_hi = np.maximum(0.0, sub_m[f"{metric}_ci_high"].to_numpy() - vals)
            ax.barh(y_pos + offset, vals, height=height, xerr=[err_lo, err_hi], color=color, alpha=0.9, label=label, capsize=2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub_names, fontsize=8.0)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Rate (with 95% Bootstrap CI)")
        ax.set_xlim(0.0, 1.0)
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[1].legend(loc="lower right", frameon=True)
    fig.savefig(fig_dir / "bias_fpr_fnr_by_subgroup.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "bias_fpr_fnr_by_subgroup.png", bbox_inches="tight")
    plt.close(fig)

    xgb_mw = mw_results.get("xgb", {})
    cols_plot = [c for c in inspect_cols if c in xgb_mw.get("fn_vs_tp", {})]
    if cols_plot:
        y_c = np.arange(len(cols_plot))
        r_fn = [xgb_mw["fn_vs_tp"][c]["rank_biserial_r"] for c in cols_plot]
        r_fp = [xgb_mw["fp_vs_tn"][c]["rank_biserial_r"] for c in cols_plot]

        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        ax.barh(y_c - 0.18, r_fn, height=0.35, color=COLOR_XGB, label="Missed Fraud (FN vs. TP)")
        ax.barh(y_c + 0.18, r_fp, height=0.35, color=COLOR_RF, label="False Flagged (FP vs. TN)")
        ax.axvline(0.0, color="black", lw=0.9)
        ax.set_yticks(y_c)
        ax.set_yticklabels(cols_plot, fontsize=8.0)
        ax.set_xlabel("Rank-Biserial Effect Size $r$ (Mann-Whitney U)")
        ax.set_title("XGBoost Error Profile: Key Feature Shifts (OOF)", fontweight="bold")
        ax.set_xlim(-1.05, 1.05)
        ax.legend(loc="lower right", frameon=True)
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.savefig(fig_dir / "errors_feature_comparison.pdf", bbox_inches="tight")
        fig.savefig(fig_dir / "errors_feature_comparison.png", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    test_preds_path = EVAL_DIR / "test_predictions.csv"
    oof_preds_path = EVAL_DIR / "oof_predictions.csv"
    unscaled_path = FEATURES_DIR / "features_unscaled.parquet"
    feat_report_path = FEATURES_DIR / "feature_report.json"

    for p in (test_preds_path, oof_preds_path, unscaled_path, feat_report_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    test_df = pd.read_csv(test_preds_path)
    oof_df = pd.read_csv(oof_preds_path)
    feat_unscaled = pd.read_parquet(unscaled_path)
    with open(feat_report_path, "r", encoding="utf-8") as f:
        feat_report = json.load(f)

    log1p_cols = feat_report.get("log1p_transformed_features", [])
    logger.info(
        "Loaded features_unscaled.parquet (%d rows; %d columns were log1p-transformed downstream in scaled splits).",
        len(feat_unscaled), len(log1p_cols),
    )

    if SAMPLED_WALLETS_PATH.exists():
        sw_df = pd.read_csv(SAMPLED_WALLETS_PATH)
        addr_col = "address" if "address" in sw_df.columns else ID_COL
        if addr_col in sw_df.columns and "is_contract" in sw_df.columns:
            sw_map = sw_df[[addr_col, "is_contract"]].drop_duplicates().copy()
            sw_map[ID_COL] = sw_map[addr_col].astype(str).str.strip().str.lower()
            feat_unscaled = feat_unscaled.merge(sw_map[[ID_COL, "is_contract"]], on=ID_COL, how="left")

    err_summary = analyze_errors_and_overlap(oof_df, test_df, feat_unscaled, threshold=args.threshold)
    with open(EVAL_DIR / "error_analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(err_summary, f, indent=2)

    rng = np.random.default_rng(args.seed)
    oof_full = oof_df.merge(feat_unscaled, on=[ID_COL, TARGET], how="inner")
    test_full = test_df.rename(columns={"y_true": TARGET}).merge(feat_unscaled, on=[ID_COL, TARGET], how="inner")

    bias_rows: List[Dict[str, Any]] = []
    for eval_name, df_eval in [("oof", oof_full), ("test", test_full)]:
        subgroups = build_subgroup_definitions(df_eval)
        for model in ["rf", "xgb"]:
            y_all = df_eval[TARGET].astype(int).to_numpy()
            p_all = (df_eval[f"proba_{model}"].to_numpy() >= args.threshold).astype(int)
            overall_fpr = float(((y_all == 0) & (p_all == 1)).sum()) / max(int((y_all == 0).sum()), 1)

            raw_pvals = []
            temp_rows = []
            for sg_name, mask in subgroups:
                sub_df = df_eval.loc[mask]
                if eval_name == "test" and len(sub_df) < 30:
                    continue
                rest_df = df_eval.loc[~mask]
                res = compute_subgroup_metrics(
                    sub_df, rest_df, overall_fpr=overall_fpr,
                    model=model, threshold=args.threshold, n_boot=args.n_boot, rng=rng,
                )
                res["eval_set"] = eval_name
                res["model"] = model
                res["subgroup"] = sg_name
                raw_pvals.append(res["fisher_p_raw"])
                temp_rows.append(res)

            holm_pvals = holm_bonferroni(raw_pvals)
            for r, hp in zip(temp_rows, holm_pvals):
                r["fisher_p_holm"] = round(hp, 5)
                r["fpr_significantly_elevated"] = bool((hp < 0.05) and (r["fpr_ratio_vs_overall"] > 1.0))
                bias_rows.append(r)

    bias_df = pd.DataFrame(bias_rows)
    bias_df.to_csv(EVAL_DIR / "bias_table.csv", index=False)

    oof_tex = bias_df[bias_df["eval_set"] == "oof"].copy()
    lines = [
        r"\begin{table*}[!htbp]",
        r"\centering",
        r"\caption{Subgroup Fairness and Error Disparity Audit on Out-of-Fold Predictions ($N=2{,}522$)}",
        r"\label{tab:bias_subgroups}",
        r"\footnotesize",
        r"\begin{tabularx}{\textwidth}{l l r c c c c c l}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Subgroup} & \textbf{$n$} & \textbf{Fraud \%} & \textbf{FPR} & \textbf{FNR} & \textbf{$F_1$} & \textbf{FPR Ratio} & \textbf{Flag} \\",
        r"\midrule",
    ]
    for _, r in oof_tex.iterrows():
        flags = []
        if r["unreliable"]:
            flags.append("Unreliable ($n<30$)")
        if r["fpr_significantly_elevated"]:
            flags.append("High FPR*")
        flag_str = ", ".join(flags) if flags else "OK"
        sg_esc = str(r["subgroup"]).replace("_", r"\_").replace("%", r"\%").replace("<", "$<$").replace(">", "$>$")
        lines.append(
            f"{r['model'].upper()} & {sg_esc} & {int(r['n'])} & {100*r['fraud_share']:.1f}\\% & "
            f"{r['fpr']:.3f} & {r['fnr']:.3f} & {r['f1']:.3f} & {r['fpr_ratio_vs_overall']:.2f} & {flag_str} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table*}"])
    (EVAL_DIR / "bias_table.tex").write_text("\n".join(lines), encoding="utf-8")

    plot_bias_and_error_figures(bias_df, err_summary["mann_whitney"], err_summary["inspect_cols"], FIG_DIR)

    print("\n=== Plain-Language Error & Bias Summary ===")
    ov_fp = err_summary["error_overlap_oof"]["fp"]
    ov_fn = err_summary["error_overlap_oof"]["fn"]
    print(f"- Error overlap (OOF): RF & XGB share {ov_fp['shared_count']} FPs (Jaccard={ov_fp['jaccard_overlap']}) and {ov_fn['shared_count']} FNs (Jaccard={ov_fn['jaccard_overlap']}).")
    elevated = oof_tex[oof_tex["fpr_significantly_elevated"]]
    if elevated.empty:
        print("- Subgroup FPR disparity: No subgroup showed statistically elevated FPR after Holm-Bonferroni correction (p < 0.05).")
    else:
        for _, er in elevated.iterrows():
            print(f"- Elevated FPR ({er['model'].upper()}): {er['subgroup']} (n={int(er['n'])}, FPR={er['fpr']:.3f}, ratio={er['fpr_ratio_vs_overall']:.2f}x, Holm p={er['fisher_p_holm']:.4f}).")
    unrel = oof_tex[oof_tex["unreliable"]]
    if not unrel.empty:
        print(f"- Sample size caution: {unrel['subgroup'].nunique()} subgroup(s) had n<30 or <10 class examples and are marked unreliable.")


if __name__ == "__main__":
    main()