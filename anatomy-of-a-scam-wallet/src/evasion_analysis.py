#!/usr/bin/env python3
"""
Phase 6 (Step 6C) — Deep-Dive Evasion Analysis & Time/Cap Confound Audit
Project: Anatomy of a Scam Wallet (Ethereum Fraud Detection with Random Forest & XGBoost)

Outputs (in outputs/ablation/):
  - evasion_summary.json
  - evasion_table.tex
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from src.config import FEATURES_DIR, ID_COL, ROOT_DIR, TARGET

ABLATION_DIR = ROOT_DIR / "outputs" / "ablation"
EVAL_DIR = ROOT_DIR / "outputs" / "eval"
SAMPLED_WALLETS_PATH = ROOT_DIR / "data" / "raw" / "sampled_wallets.csv"

KEY_PROFILE_COLS = [
    "G_temp_lifespan_days",
    "G_temp_sent_tx_count",
    "G_temp_recv_tx_count",
    "G_temp_burstiness_max_1h",
    "G_val_total_eth_sent",
    "G_val_total_eth_recv",
    "G_net_unique_sent_to",
    "G_net_unique_recv_from",
    "G_tok_mean_tx_fee_eth",
    "G_tok_contract_creation_count",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evasion_analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 6: Evasion & Observation-Window Confound Analysis")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    oof_path = EVAL_DIR / "oof_predictions.csv"
    unscaled_path = FEATURES_DIR / "features_unscaled.parquet"

    for p in (oof_path, unscaled_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    oof_df = pd.read_csv(oof_path)
    feat_df = pd.read_parquet(unscaled_path)
    merged = oof_df.merge(feat_df, on=[ID_COL, TARGET], how="inner")

    pred_rf = (merged["proba_rf"] >= args.threshold).astype(int)
    pred_xgb = (merged["proba_xgb"] >= args.threshold).astype(int)

    fraud_df = merged[merged[TARGET] == 1].copy()
    fn_both = fraud_df[(pred_rf[fraud_df.index] == 0) & (pred_xgb[fraud_df.index] == 0)].copy()
    tp_both = fraud_df[(pred_rf[fraud_df.index] == 1) & (pred_xgb[fraud_df.index] == 1)].copy()

    # Quantify evasion archetypes among fraud wallets missed by BOTH models vs caught by BOTH
    def _archetypes(sub: pd.DataFrame) -> Dict[str, Any]:
        n = max(len(sub), 1)
        tot_tx = sub["G_temp_sent_tx_count"] + sub["G_temp_recv_tx_count"]
        return {
            "count": int(len(sub)),
            "pct_tx_lt_5": round(100.0 * float((tot_tx < 5).sum()) / n, 2),
            "pct_lifespan_lt_1d": round(100.0 * float((sub["G_temp_lifespan_days"] < 1.0).sum()) / n, 2),
            "pct_zero_outgoing": round(100.0 * float((sub["G_temp_sent_tx_count"] == 0).sum()) / n, 2),
            "pct_only_contract_creation": round(
                100.0 * float(((tot_tx > 0) & (sub["G_tok_contract_creation_count"] == tot_tx)).sum()) / n, 2
            ),
        }

    archetypes_summary = {
        "missed_by_both_fn": _archetypes(fn_both),
        "caught_by_both_tp": _archetypes(tp_both),
    }

    # Mann-Whitney U comparison between missed_by_both and caught_by_both
    mw_comp: Dict[str, Dict[str, Any]] = {}
    for col in [c for c in KEY_PROFILE_COLS if c in merged.columns]:
        a = fn_both[col].to_numpy()
        b = tp_both[col].to_numpy()
        if len(a) >= 2 and len(b) >= 2:
            u_stat, p_val = mannwhitneyu(a, b, alternative="two-sided")
            r_eff = 1.0 - (2.0 * float(u_stat)) / float(len(a) * len(b))
            mw_comp[col] = {
                "missed_median": round(float(np.median(a)), 4),
                "caught_median": round(float(np.median(b)), 4),
                "rank_biserial_r": round(r_eff, 4),
                "p_value": float(p_val),
            }

    # Time & Cap Confound Check (Fraud vs Normal across all 2,522 wallets)
    fraud_all = merged[merged[TARGET] == 1]
    norm_all = merged[merged[TARGET] == 0]
    tot_tx_all = merged["G_temp_sent_tx_count"] + merged["G_temp_recv_tx_count"]

    confound_check: Dict[str, Any] = {
        "lifespan_days_median_fraud": round(float(fraud_all["G_temp_lifespan_days"].median()), 3),
        "lifespan_days_median_normal": round(float(norm_all["G_temp_lifespan_days"].median()), 3),
        "mean_fee_eth_median_fraud": round(float(fraud_all["G_tok_mean_tx_fee_eth"].median()), 6),
        "mean_fee_eth_median_normal": round(float(norm_all["G_tok_mean_tx_fee_eth"].median()), 6),
        "pct_at_1000_tx_cap_fraud": round(100.0 * float((tot_tx_all.loc[fraud_all.index] >= 1000).mean()), 2),
        "pct_at_1000_tx_cap_normal": round(100.0 * float((tot_tx_all.loc[norm_all.index] >= 1000).mean()), 2),
    }

    if SAMPLED_WALLETS_PATH.exists():
        sw = pd.read_csv(SAMPLED_WALLETS_PATH)
        if {"window_first_ts", "window_last_ts", "window_sent_cnt", "window_recv_cnt"}.issubset(sw.columns):
            sw_tot = sw["window_sent_cnt"] + sw["window_recv_cnt"]
            sw_f = sw[sw["is_scam"] == 1]
            sw_n = sw[sw["is_scam"] == 0]
            confound_check["exact_window_first_ts_median_fraud"] = int(sw_f["window_first_ts"].median())
            confound_check["exact_window_first_ts_median_normal"] = int(sw_n["window_first_ts"].median())
            confound_check["exact_window_last_ts_median_fraud"] = int(sw_f["window_last_ts"].median())
            confound_check["exact_window_last_ts_median_normal"] = int(sw_n["window_last_ts"].median())
            confound_check["exact_pct_ge_1000_tx_fraud"] = round(100.0 * float((sw_tot[sw["is_scam"] == 1] >= 1000).mean()), 2)
            confound_check["exact_pct_ge_1000_tx_normal"] = round(100.0 * float((sw_tot[sw["is_scam"] == 0] >= 1000).mean()), 2)

    _, ls_p = mannwhitneyu(fraud_all["G_temp_lifespan_days"], norm_all["G_temp_lifespan_days"], alternative="two-sided")
    confound_check["lifespan_mannwhitney_p"] = float(ls_p)

    summary = {
        "archetypes": archetypes_summary,
        "mann_whitney_missed_vs_caught": mw_comp,
        "time_and_cap_confound_check": confound_check,
    }
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    with open(ABLATION_DIR / "evasion_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Behavioral Profile of Fraud Wallets Missed vs. Caught by Both Models (OOF)}",
        r"\label{tab:evasion}",
        r"\footnotesize",
        r"\begin{tabularx}{\linewidth}{X r r r}",
        r"\toprule",
        r"\textbf{Metric / Archetype} & \textbf{Missed ($n=" + str(archetypes_summary["missed_by_both_fn"]["count"]) + r"$)} & \textbf{Caught ($n=" + str(archetypes_summary["caught_by_both_tp"]["count"]) + r"$)} & \textbf{Effect $r$} \\",
        r"\midrule",
        f"Burner ($<5$ stored txs) & {archetypes_summary['missed_by_both_fn']['pct_tx_lt_5']}\\% & {archetypes_summary['caught_by_both_tp']['pct_tx_lt_5']}\\% & --- \\\\",
        f"Lifespan $<1$ day & {archetypes_summary['missed_by_both_fn']['pct_lifespan_lt_1d']}\\% & {archetypes_summary['caught_by_both_tp']['pct_lifespan_lt_1d']}\\% & --- \\\\",
        f"Zero outgoing txs & {archetypes_summary['missed_by_both_fn']['pct_zero_outgoing']}\\% & {archetypes_summary['caught_by_both_tp']['pct_zero_outgoing']}\\% & --- \\\\",
        r"\midrule",
    ]
    for col, info in list(mw_comp.items())[:6]:
        c_esc = col.replace("_", r"\_")
        lines.append(
            f"\\texttt{{{c_esc}}} (med.) & {info['missed_median']:.3f} & {info['caught_median']:.3f} & {info['rank_biserial_r']:+.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    (ABLATION_DIR / "evasion_table.tex").write_text("\n".join(lines), encoding="utf-8")

    print("\n=== Evasion & Observation-Window Confound Summary ===")
    print(f"- Missed by both models (OOF FN): {archetypes_summary['missed_by_both_fn']}")
    print(f"- Caught by both models (OOF TP): {archetypes_summary['caught_by_both_tp']}")
    print(f"- Time/cap confound check: {json.dumps(confound_check)}")
    if ls_p < 0.05:
        print("- Conservative conclusion: Fraud and normal wallets differ significantly in observed lifespan (p < 0.05); part of G_temp's signal may reflect campaign duration within the 19-month sampling window.")


if __name__ == "__main__":
    main()