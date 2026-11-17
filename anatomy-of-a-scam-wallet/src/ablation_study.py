#!/usr/bin/env python3
"""
Phase 6 (Step 6A) — Grouped Behavioral Ablation Study (LOGO, ONLY, & Random Controls)
Project: Anatomy of a Scam Wallet (Ethereum Fraud Detection with Random Forest & XGBoost)

Design:
  - Controlled experiment: tuned 1-SE hyperparameters (`outputs/tuning/best_params.json`),
    seed=42, preprocessing, and splits are held constant; only the feature subset varies.
  - Primary protocol: 5x5 repeated stratified CV (`seed = 42 + repeat`) over all 2,522 wallets.
  - Secondary protocol: single fit on `train_scaled`, single evaluation on `test_scaled`.
  - Size-matched random controls (`k in {4, 6, 12}`) separate behavioral signal from group size.
  - Paired bootstrap (1,000 resamples) + Holm-Bonferroni correction test H1 and H3.
  - Resumable per-configuration caching in `outputs/ablation/_cache/`.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

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
    get_features,
)
from src.models import make_pipeline

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

ABLATION_DIR = ROOT_DIR / "outputs" / "ablation"
CACHE_DIR = ABLATION_DIR / "_cache"
GROUP_ORDER = ["G_temp", "G_val", "G_net", "G_tok"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ablation_study")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 6: Grouped Behavioral Ablation Study")
    parser.add_argument("--repeats", type=int, default=5, help="CV repeats (default: 5)")
    parser.add_argument("--folds", type=int, default=5, help="Stratified CV folds (default: 5)")
    parser.add_argument("--n-random", type=int, default=10, help="Random control subsets per k (default: 10)")
    parser.add_argument("--n-boot", type=int, default=1000, help="Bootstrap resamples (default: 1000)")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Estimator parallel jobs (default: -1)")
    parser.add_argument("--seed", type=int, default=SEED, help="Base random seed (default: config SEED)")
    parser.add_argument("--skip-shap", action="store_true", help="Skip TreeExplainer SHAP computation")
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


def build_configurations(n_random: int, seed: int) -> List[Dict[str, Any]]:
    """Build FULL, LOGO, ONLY, EXTRA (no-lifespan), and size-matched random control configs."""
    configs: List[Dict[str, Any]] = [
        {"name": "FULL", "family": "FULL", "group": "ALL", "k": len(ALL_FEATURES), "features": list(ALL_FEATURES)}
    ]
    for grp in GROUP_ORDER:
        feats_logo = get_features(exclude_groups=[grp])
        configs.append({
            "name": f"LOGO_minus_{grp}",
            "family": "LOGO",
            "group": grp,
            "k": len(feats_logo),
            "features": feats_logo,
        })
    for grp in GROUP_ORDER:
        feats_only = get_features(only_groups=[grp])
        configs.append({
            "name": f"ONLY_{grp}",
            "family": "ONLY",
            "group": grp,
            "k": len(feats_only),
            "features": feats_only,
        })

    no_life_feats = [f for f in ALL_FEATURES if f != "G_temp_lifespan_days"]
    configs.append({
        "name": "NO_LIFESPAN",
        "family": "EXTRA",
        "group": "G_temp_lifespan_days",
        "k": len(no_life_feats),
        "features": no_life_feats,
    })

    for k in [4, 6, 12]:
        rng = np.random.default_rng(seed + k)
        for r_idx in range(n_random):
            chosen = sorted(rng.choice(ALL_FEATURES, size=k, replace=False).tolist())
            leave_out = [f for f in ALL_FEATURES if f not in set(chosen)]
            configs.append({
                "name": f"RAND_ONLY_k{k}_r{r_idx}",
                "family": f"RAND_ONLY_k{k}",
                "group": f"rand_k{k}",
                "k": k,
                "features": chosen,
            })
            configs.append({
                "name": f"RAND_LEAVE_k{k}_r{r_idx}",
                "family": f"RAND_LEAVE_k{k}",
                "group": f"rand_k{k}",
                "k": len(leave_out),
                "features": leave_out,
            })

    return configs


def evaluate_config_with_cache(
    model_name: str,
    params: Dict[str, Any],
    cfg: Dict[str, Any],
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    repeats: int,
    folds: int,
    n_jobs: int,
    seed: int,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Run 5x5 CV (primary) and train->test (secondary) for one config, caching to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_json = CACHE_DIR / f"{model_name}__{cfg['name']}.json"
    cache_npz = CACHE_DIR / f"{model_name}__{cfg['name']}.npz"

    if cache_json.exists() and cache_npz.exists():
        with open(cache_json, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("repeats") == repeats and cached.get("folds") == folds:
            oof_arr = np.load(cache_npz)["oof_probs"]
            return cached, oof_arr

    feats = cfg["features"]
    est_params = dict(params)
    est_params["n_jobs"] = n_jobs

    X_all = full_df[ALL_FEATURES]
    y_all = full_df[TARGET].astype(int).to_numpy()
    n_samples = len(full_df)

    oof_matrix = np.zeros((repeats, n_samples), dtype=np.float32)
    rep_f1, rep_prec, rep_rec, rep_auc = [], [], [], []

    for r in range(repeats):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + r)
        oof_r = np.zeros(n_samples, dtype=np.float32)
        for tr_idx, va_idx in skf.split(X_all, y_all):
            pipe = make_pipeline(model_name, params=est_params, features=feats)
            pipe.fit(X_all.iloc[tr_idx], y_all[tr_idx])
            oof_r[va_idx] = pipe.predict_proba(X_all.iloc[va_idx])[:, 1].astype(np.float32)

        oof_matrix[r, :] = oof_r
        pred_r = (oof_r >= 0.5).astype(int)
        rep_f1.append(float(f1_score(y_all, pred_r, pos_label=1, zero_division=0)))
        rep_prec.append(float(precision_score(y_all, pred_r, pos_label=1, zero_division=0)))
        rep_rec.append(float(recall_score(y_all, pred_r, pos_label=1, zero_division=0)))
        rep_auc.append(float(roc_auc_score(y_all, oof_r)))

    # Secondary: train_scaled -> test_scaled once
    pipe_sec = make_pipeline(model_name, params=est_params, features=feats)
    pipe_sec.fit(train_df[ALL_FEATURES], train_df[TARGET].astype(int).to_numpy())
    y_test = test_df[TARGET].astype(int).to_numpy()
    p_test = pipe_sec.predict_proba(test_df[ALL_FEATURES])[:, 1]
    pred_test = (p_test >= 0.5).astype(int)

    record = {
        "model": model_name,
        "config": cfg["name"],
        "family": cfg["family"],
        "group": cfg["group"],
        "k": cfg["k"],
        "repeats": repeats,
        "folds": folds,
        "cv_f1_mean": float(np.mean(rep_f1)),
        "cv_f1_std": float(np.std(rep_f1)),
        "cv_precision_mean": float(np.mean(rep_prec)),
        "cv_precision_std": float(np.std(rep_prec)),
        "cv_recall_mean": float(np.mean(rep_rec)),
        "cv_recall_std": float(np.std(rep_rec)),
        "cv_auc_mean": float(np.mean(rep_auc)),
        "cv_auc_std": float(np.std(rep_auc)),
        "test_f1": float(f1_score(y_test, pred_test, pos_label=1, zero_division=0)),
        "test_precision": float(precision_score(y_test, pred_test, pos_label=1, zero_division=0)),
        "test_recall": float(recall_score(y_test, pred_test, pos_label=1, zero_division=0)),
        "test_auc": float(roc_auc_score(y_test, p_test)),
    }

    np.savez_compressed(cache_npz, oof_probs=oof_matrix)
    with open(cache_json, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return record, oof_matrix


def paired_bootstrap_delta(
    y_true: np.ndarray,
    oof_full: np.ndarray,
    oof_sub: np.ndarray,
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    """Paired bootstrap over wallets using mean OOF probabilities across repeats."""
    p_full_mean = oof_full.mean(axis=0)
    p_sub_mean = oof_sub.mean(axis=0)
    pred_full = (p_full_mean >= 0.5).astype(int)
    pred_sub = (p_sub_mean >= 0.5).astype(int)

    f1_full_pt = float(f1_score(y_true, pred_full, pos_label=1, zero_division=0))
    f1_sub_pt = float(f1_score(y_true, pred_sub, pos_label=1, zero_division=0))

    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_sub, boot_delta = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        if len(np.unique(yb)) < 2:
            continue
        f1_f = float(f1_score(yb, pred_full[idx], pos_label=1, zero_division=0))
        f1_s = float(f1_score(yb, pred_sub[idx], pos_label=1, zero_division=0))
        boot_sub.append(f1_s)
        boot_delta.append(f1_s - f1_f)

    delta_arr = np.asarray(boot_delta, dtype=float)
    sub_arr = np.asarray(boot_sub, dtype=float)
    p_val = min(float(2.0 * min(np.mean(delta_arr <= 0.0), np.mean(delta_arr >= 0.0))), 1.0)
    return {
        "oof_f1_pooled": f1_sub_pt,
        "oof_f1_ci95_low": float(np.percentile(sub_arr, 2.5)),
        "oof_f1_ci95_high": float(np.percentile(sub_arr, 97.5)),
        "delta_f1_vs_full": f1_sub_pt - f1_full_pt,
        "delta_ci95_low": float(np.percentile(delta_arr, 2.5)),
        "delta_ci95_high": float(np.percentile(delta_arr, 97.5)),
        "p_raw": p_val,
    }


def test_h1_and_h3(
    y_true: np.ndarray,
    oof_store: Dict[Tuple[str, str], np.ndarray],
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    """Run paired bootstrap tests for H1 and H3 per model."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    verdicts: Dict[str, Any] = {}

    for model in ["rf", "xgb"]:
        p_full = (oof_store[(model, "FULL")].mean(axis=0) >= 0.5).astype(int)
        p_temp = (oof_store[(model, "LOGO_minus_G_temp")].mean(axis=0) >= 0.5).astype(int)
        p_net = (oof_store[(model, "LOGO_minus_G_net")].mean(axis=0) >= 0.5).astype(int)
        p_val = (oof_store[(model, "LOGO_minus_G_val")].mean(axis=0) >= 0.5).astype(int)

        only_preds = {
            grp: (oof_store[(model, f"ONLY_{grp}")].mean(axis=0) >= 0.5).astype(int)
            for grp in GROUP_ORDER
        }
        only_f1s = {
            grp: float(f1_score(y_true, pred, pos_label=1, zero_division=0))
            for grp, pred in only_preds.items()
        }
        best_only_grp = max(only_f1s, key=only_f1s.get)

        h1_temp_diffs, h1_net_diffs, h3_gaps = [], [], []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            yb = y_true[idx]
            if len(np.unique(yb)) < 2:
                continue
            f_full = f1_score(yb, p_full[idx], pos_label=1, zero_division=0)
            drop_temp = f_full - f1_score(yb, p_temp[idx], pos_label=1, zero_division=0)
            drop_net = f_full - f1_score(yb, p_net[idx], pos_label=1, zero_division=0)
            drop_val = f_full - f1_score(yb, p_val[idx], pos_label=1, zero_division=0)
            h1_temp_diffs.append(drop_temp - drop_val)
            h1_net_diffs.append(drop_net - drop_val)

            best_only_b = max(
                f1_score(yb, only_preds[g][idx], pos_label=1, zero_division=0) for g in GROUP_ORDER
            )
            h3_gaps.append(f_full - best_only_b)

        t_arr = np.asarray(h1_temp_diffs)
        n_arr = np.asarray(h1_net_diffs)
        g_arr = np.asarray(h3_gaps)

        t_ci = (float(np.percentile(t_arr, 2.5)), float(np.percentile(t_arr, 97.5)))
        n_ci = (float(np.percentile(n_arr, 2.5)), float(np.percentile(n_arr, 97.5)))
        h3_ci = (float(np.percentile(g_arr, 2.5)), float(np.percentile(g_arr, 97.5)))

        h1_temp_ok = t_ci[0] >= 0.03
        h1_net_ok = n_ci[0] >= 0.03
        h1_point_ok = (float(np.mean(t_arr)) >= 0.03) or (float(np.mean(n_arr)) >= 0.03)
        h3_point = float(f1_score(y_true, p_full, pos_label=1) - only_f1s[best_only_grp])

        verdicts[model] = {
            "H1_temp_minus_val": {
                "mean": round(float(np.mean(t_arr)), 4),
                "ci95": [round(t_ci[0], 4), round(t_ci[1], 4)],
                "ci_excludes_below_0.03": bool(h1_temp_ok),
            },
            "H1_net_minus_val": {
                "mean": round(float(np.mean(n_arr)), 4),
                "ci95": [round(n_ci[0], 4), round(n_ci[1], 4)],
                "ci_excludes_below_0.03": bool(h1_net_ok),
            },
            "H1_verdict": "SUPPORTED" if (h1_temp_ok or h1_net_ok) else ("POINT_ONLY_NOT_CI" if h1_point_ok else "NOT SUPPORTED"),
            "H3_best_single_group": best_only_grp,
            "H3_best_single_f1": round(only_f1s[best_only_grp], 4),
            "H3_full_minus_best_single": {
                "point": round(h3_point, 4),
                "ci95": [round(h3_ci[0], 4), round(h3_ci[1], 4)],
                "ci_excludes_below_0.05": bool(h3_ci[0] >= 0.05),
            },
            "H3_verdict": "SUPPORTED" if (h3_ci[0] >= 0.05) else ("POINT_ONLY_NOT_CI" if h3_point >= 0.05 else "NOT SUPPORTED"),
        }
    return verdicts


def compute_grouped_perm_and_shap(
    rf_params: Dict[str, Any],
    xgb_params: Dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    skip_shap: bool,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute grouped permutation importance (30 repeats) and TreeExplainer/pred_contribs SHAP on test."""
    X_tr, y_tr = train_df[ALL_FEATURES], train_df[TARGET].astype(int).to_numpy()
    X_te, y_te = test_df[ALL_FEATURES], test_df[TARGET].astype(int).to_numpy()

    rng = np.random.default_rng(seed)
    grp_rows: List[Dict[str, Any]] = []
    feat_rows: List[Dict[str, Any]] = []

    for model_name, params in [("rf", rf_params), ("xgb", xgb_params)]:
        pipe = make_pipeline(model_name, params=params, features=ALL_FEATURES)
        pipe.fit(X_tr, y_tr)
        base_f1 = float(f1_score(y_te, (pipe.predict_proba(X_te)[:, 1] >= 0.5).astype(int), pos_label=1))

        # Grouped permutation importance (30 repeats)
        perm_means: Dict[str, float] = {}
        perm_stds: Dict[str, float] = {}
        for grp in GROUP_ORDER:
            cols = FEATURE_GROUPS[grp]
            drops = []
            for _ in range(30):
                X_perm = X_te.copy()
                perm_idx = rng.permutation(len(X_perm))
                X_perm[cols] = X_perm[cols].iloc[perm_idx].to_numpy()
                f1_p = float(f1_score(y_te, (pipe.predict_proba(X_perm)[:, 1] >= 0.5).astype(int), pos_label=1))
                drops.append(base_f1 - f1_p)
            perm_means[grp] = float(np.mean(drops))
            perm_stds[grp] = float(np.std(drops))

        # SHAP / pred_contribs
        shap_abs_mean = np.zeros(len(ALL_FEATURES), dtype=float)
        shap_used = False
        est = pipe.named_steps["clf"]
        if not skip_shap:
            if model_name == "xgb":
                booster = est.get_booster()
                dmat = xgb.DMatrix(X_te[ALL_FEATURES].to_numpy(), feature_names=ALL_FEATURES)
                contribs = booster.predict(dmat, pred_contribs=True)
                shap_abs_mean = np.mean(np.abs(contribs[:, :-1]), axis=0)
                shap_used = True
            elif model_name == "rf" and SHAP_AVAILABLE:
                explainer = shap.TreeExplainer(est)
                sv = explainer.shap_values(X_te[ALL_FEATURES])
                if isinstance(sv, list):
                    sv_arr = np.asarray(sv[1])
                elif sv.ndim == 3:
                    sv_arr = sv[:, :, 1]
                else:
                    sv_arr = sv
                shap_abs_mean = np.mean(np.abs(sv_arr), axis=0)
                shap_used = True
            else:
                logger.warning("`shap` not installed; skipping RF TreeExplainer SHAP (using native Gini importance).")

        gini_imp = est.feature_importances_
        feat_to_grp = {c: g for g in GROUP_ORDER for c in FEATURE_GROUPS[g]}
        for idx_f, fname in enumerate(ALL_FEATURES):
            feat_rows.append({
                "model": model_name,
                "feature": fname,
                "group": feat_to_grp[fname],
                "native_importance": float(gini_imp[idx_f]),
                "mean_abs_shap": float(shap_abs_mean[idx_f]) if shap_used else np.nan,
            })

        for grp in GROUP_ORDER:
            grp_mask = [i for i, f in enumerate(ALL_FEATURES) if feat_to_grp[f] == grp]
            grp_shap = float(np.sum(shap_abs_mean[grp_mask])) if shap_used else np.nan
            grp_rows.append({
                "model": model_name,
                "group": grp,
                "k": len(grp_mask),
                "perm_f1_drop_mean": round(perm_means[grp], 4),
                "perm_f1_drop_std": round(perm_stds[grp], 4),
                "sum_mean_abs_shap": round(grp_shap, 4) if shap_used else np.nan,
            })

    return pd.DataFrame(grp_rows), pd.DataFrame(feat_rows)


def write_ablation_latex_tables(
    res_df: pd.DataFrame,
    ctrl_df: pd.DataFrame,
    grp_imp_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Write ablation_table.tex, controls_table.tex, and importance_perm_shap.tex."""
    main_configs = ["FULL"] + [f"LOGO_minus_{g}" for g in GROUP_ORDER] + [f"ONLY_{g}" for g in GROUP_ORDER] + ["NO_LIFESPAN"]

    lines = [
        r"\begin{table*}[!htbp]",
        r"\centering",
        r"\caption{Grouped Behavioral Ablation Results on 5$\times$5 Repeated Stratified CV ($N=2{,}522$)}",
        r"\label{tab:ablation}",
        r"\footnotesize",
        r"\begin{tabularx}{\textwidth}{l r c c c c}",
        r"\toprule",
        r"\textbf{Configuration} & \textbf{$k$} & \textbf{RF $F_1$ [95\% CI]} & \textbf{RF $\Delta F_1$} & \textbf{XGB $F_1$ [95\% CI]} & \textbf{XGB $\Delta F_1$} \\",
        r"\midrule",
    ]
    for cname in main_configs:
        r_rf = res_df[(res_df["model"] == "rf") & (res_df["config"] == cname)].iloc[0]
        r_xgb = res_df[(res_df["model"] == "xgb") & (res_df["config"] == cname)].iloc[0]
        c_esc = cname.replace("_", r"\_")
        rf_f1 = f"{r_rf['cv_f1_mean']:.4f} [{r_rf['oof_f1_ci95_low']:.3f}, {r_rf['oof_f1_ci95_high']:.3f}]"
        xgb_f1 = f"{r_xgb['cv_f1_mean']:.4f} [{r_xgb['oof_f1_ci95_low']:.3f}, {r_xgb['oof_f1_ci95_high']:.3f}]"
        lines.append(
            f"\\texttt{{{c_esc}}} & {int(r_rf['k'])} & {rf_f1} & {r_rf['delta_f1_vs_full']:+.4f} & {xgb_f1} & {r_xgb['delta_f1_vs_full']:+.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table*}"])
    (out_dir / "ablation_table.tex").write_text("\n".join(lines), encoding="utf-8")

    # Controls table
    c_lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Size-Matched Random Feature Controls vs. Real Behavioral Groups}",
        r"\label{tab:controls}",
        r"\footnotesize",
        r"\begin{tabularx}{\linewidth}{l l r c c}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Control Family} & \textbf{$k$} & \textbf{Mean $F_1$} & \textbf{2.5--97.5\% Band} \\",
        r"\midrule",
    ]
    for _, cr in ctrl_df.iterrows():
        fam_esc = str(cr["family"]).replace("_", r"\_")
        c_lines.append(
            f"{cr['model'].upper()} & \\texttt{{{fam_esc}}} & {int(cr['k'])} & {cr['rand_f1_mean']:.4f} & [{cr['rand_f1_p025']:.4f}, {cr['rand_f1_p975']:.4f}] \\\\"
        )
    c_lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    (out_dir / "controls_table.tex").write_text("\n".join(c_lines), encoding="utf-8")

    # Importance table
    i_lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Grouped Permutation Importance ($\Delta F_1$, 30 Repeats) and Summed $|\text{SHAP}|$ on Test}",
        r"\label{tab:group_imp}",
        r"\footnotesize",
        r"\begin{tabularx}{\linewidth}{l c c c c}",
        r"\toprule",
        r"\textbf{Group} & \textbf{RF Perm $\Delta F_1$} & \textbf{RF $|\text{SHAP}|$} & \textbf{XGB Perm $\Delta F_1$} & \textbf{XGB $|\text{SHAP}|$} \\",
        r"\midrule",
    ]
    for grp in GROUP_ORDER:
        gr_rf = grp_imp_df[(grp_imp_df["model"] == "rf") & (grp_imp_df["group"] == grp)].iloc[0]
        gr_xgb = grp_imp_df[(grp_imp_df["model"] == "xgb") & (grp_imp_df["group"] == grp)].iloc[0]
        rf_s = f"{gr_rf['sum_mean_abs_shap']:.4f}" if pd.notna(gr_rf["sum_mean_abs_shap"]) else "N/A"
        xgb_s = f"{gr_xgb['sum_mean_abs_shap']:.4f}" if pd.notna(gr_xgb["sum_mean_abs_shap"]) else "N/A"
        g_sym = f"$G_{{\\text{{{grp.split('_')[1]}}}}}$"
        i_lines.append(
            f"{g_sym} & {gr_rf['perm_f1_drop_mean']:.4f} & {rf_s} & {gr_xgb['perm_f1_drop_mean']:.4f} & {xgb_s} \\\\"
        )
    i_lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    (out_dir / "importance_perm_shap.tex").write_text("\n".join(i_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    best_params_path = TUNING_OUTPUT_DIR / "best_params.json"
    for p in (TRAIN_SCALED_PATH, TEST_SCALED_PATH, best_params_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_parquet(TRAIN_SCALED_PATH)
    test_df = pd.read_parquet(TEST_SCALED_PATH)
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    y_all = full_df[TARGET].astype(int).to_numpy()

    with open(best_params_path, "r", encoding="utf-8") as f:
        best_params = json.load(f)

    configs = build_configurations(n_random=args.n_random, seed=args.seed)
    est_min = round((len(configs) * 2 * args.repeats * args.folds) * 0.025 / 60.0, 1)
    logger.info(
        "Running %d configs x 2 models x %dx%d CV (est. ~%.1f-%.1f min on CPU, cached per config)...",
        len(configs), args.repeats, args.folds, max(est_min, 0.5), max(est_min * 2.5, 1.5),
    )

    records: List[Dict[str, Any]] = []
    oof_store: Dict[Tuple[str, str], np.ndarray] = {}

    for model_name in ["rf", "xgb"]:
        params = best_params[model_name]["chosen_params"]
        for cfg in configs:
            rec, oof_arr = evaluate_config_with_cache(
                model_name=model_name, params=params, cfg=cfg,
                full_df=full_df, train_df=train_df, test_df=test_df,
                repeats=args.repeats, folds=args.folds, n_jobs=args.n_jobs, seed=args.seed,
            )
            records.append(rec)
            oof_store[(model_name, cfg["name"])] = oof_arr

    # Paired bootstrap vs FULL & Holm correction within LOGO and ONLY families
    for rec in records:
        m = rec["model"]
        c = rec["config"]
        boot = paired_bootstrap_delta(
            y_true=y_all, oof_full=oof_store[(m, "FULL")], oof_sub=oof_store[(m, c)],
            n_boot=args.n_boot, seed=args.seed,
        )
        rec.update(boot)

    res_df = pd.DataFrame(records)
    res_df["p_holm"] = np.nan
    for m in ["rf", "xgb"]:
        for fam in ["LOGO", "ONLY"]:
            mask = (res_df["model"] == m) & (res_df["family"] == fam)
            res_df.loc[mask, "p_holm"] = holm_bonferroni(res_df.loc[mask, "p_raw"].tolist())

    res_df.to_csv(ABLATION_DIR / "ablation_results.csv", index=False)

    # Save compact OOF npz for main configs
    npz_payload = {
        f"{m}__{c}": oof_store[(m, c)]
        for (m, c) in oof_store
        if not c.startswith("RAND_")
    }
    np.savez_compressed(ABLATION_DIR / "oof_probs.npz", **npz_payload)

    # Random controls summary
    ctrl_rows = []
    for m in ["rf", "xgb"]:
        for k in [4, 6, 12]:
            for fam in [f"RAND_ONLY_k{k}", f"RAND_LEAVE_k{k}"]:
                sub = res_df[(res_df["model"] == m) & (res_df["family"] == fam)]
                ctrl_rows.append({
                    "model": m,
                    "family": fam,
                    "k": k,
                    "rand_f1_mean": round(float(sub["cv_f1_mean"].mean()), 4),
                    "rand_f1_p025": round(float(np.percentile(sub["cv_f1_mean"], 2.5)), 4),
                    "rand_f1_p975": round(float(np.percentile(sub["cv_f1_mean"], 97.5)), 4),
                })
    ctrl_df = pd.DataFrame(ctrl_rows)
    ctrl_df.to_csv(ABLATION_DIR / "controls_table.csv", index=False)

    # H1 & H3 Hypothesis tests
    hypotheses = test_h1_and_h3(y_all, oof_store, n_boot=args.n_boot, seed=args.seed)

    # Cross-model rank correlation
    logo_rf = res_df[(res_df["model"] == "rf") & (res_df["family"] == "LOGO")].set_index("group").reindex(GROUP_ORDER)
    logo_xgb = res_df[(res_df["model"] == "xgb") & (res_df["family"] == "LOGO")].set_index("group").reindex(GROUP_ORDER)
    only_rf = res_df[(res_df["model"] == "rf") & (res_df["family"] == "ONLY")].set_index("group").reindex(GROUP_ORDER)
    only_xgb = res_df[(res_df["model"] == "xgb") & (res_df["family"] == "ONLY")].set_index("group").reindex(GROUP_ORDER)

    rho_logo, p_logo = spearmanr(-logo_rf["delta_f1_vs_full"], -logo_xgb["delta_f1_vs_full"])
    rho_only, p_only = spearmanr(only_rf["cv_f1_mean"], only_xgb["cv_f1_mean"])
    hypotheses["cross_model_spearman"] = {
        "logo_drop_rho": round(float(rho_logo), 4),
        "logo_drop_p": round(float(p_logo), 4),
        "only_f1_rho": round(float(rho_only), 4),
        "only_f1_p": round(float(p_only), 4),
    }
    with open(ABLATION_DIR / "hypotheses.json", "w", encoding="utf-8") as f:
        json.dump(hypotheses, f, indent=2)

    cross_rows = []
    for g in GROUP_ORDER:
        r_f, r_x = logo_rf.loc[g], logo_xgb.loc[g]
        ci_overlap = not (r_f["delta_ci95_high"] < r_x["delta_ci95_low"] or r_x["delta_ci95_high"] < r_f["delta_ci95_low"])
        cross_rows.append({
            "group": g,
            "rf_logo_delta": round(float(r_f["delta_f1_vs_full"]), 4),
            "xgb_logo_delta": round(float(r_x["delta_f1_vs_full"]), 4),
            "delta_ci_overlap": bool(ci_overlap),
            "rf_only_f1": round(float(only_rf.loc[g, "cv_f1_mean"]), 4),
            "xgb_only_f1": round(float(only_xgb.loc[g, "cv_f1_mean"]), 4),
        })
    pd.DataFrame(cross_rows).to_csv(ABLATION_DIR / "cross_model_comparison.csv", index=False)

    # Grouped permutation & SHAP
    grp_imp_df, feat_imp_df = compute_grouped_perm_and_shap(
        rf_params=best_params["rf"]["chosen_params"],
        xgb_params=best_params["xgb"]["chosen_params"],
        train_df=train_df, test_df=test_df, skip_shap=args.skip_shap, seed=args.seed,
    )
    grp_imp_df.to_csv(ABLATION_DIR / "group_importance.csv", index=False)
    feat_imp_df.to_csv(ABLATION_DIR / "feature_importance.csv", index=False)

    write_ablation_latex_tables(res_df, ctrl_df, grp_imp_df, ABLATION_DIR)

    # 10-line plain-language summary
    print("\n=== Ablation Study Summary ===")
    for m in ["rf", "xgb"]:
        full_f1 = res_df[(res_df["model"] == m) & (res_df["config"] == "FULL")].iloc[0]["cv_f1_mean"]
        worst_logo = res_df[(res_df["model"] == m) & (res_df["family"] == "LOGO")].sort_values("delta_f1_vs_full").iloc[0]
        best_only = res_df[(res_df["model"] == m) & (res_df["family"] == "ONLY")].sort_values("cv_f1_mean", ascending=False).iloc[0]
        h = hypotheses[m]
        print(f"- {m.upper()} FULL 5x5 CV F1 = {full_f1:.4f} | Biggest LOGO drop: {worst_logo['group']} ({worst_logo['delta_f1_vs_full']:+.4f}) | Best ONLY: {best_only['group']} ({best_only['cv_f1_mean']:.4f}).")
        print(f"- {m.upper()} Verdicts: H1={h['H1_verdict']} (temp-val={h['H1_temp_minus_val']['mean']:+.4f}, net-val={h['H1_net_minus_val']['mean']:+.4f}); H3={h['H3_verdict']} (gap={h['H3_full_minus_best_single']['point']:+.4f}).")
    print(f"- Cross-model Spearman rho: LOGO drops={rho_logo:.3f} (p={p_logo:.3f}), ONLY F1={rho_only:.3f} (p={p_only:.3f}).")
    print(f"- All ablation tables, OOF arrays, and importance CSVs saved to {ABLATION_DIR}.")


if __name__ == "__main__":
    main()