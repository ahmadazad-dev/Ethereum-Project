# Viva Preparation Guide: Anatomy of a Scam Wallet
**Author:** Ahmad Azad (BS Data Science, FAST NUCES Lahore)  
**Target Length:** <= 6 pages printed | **Project Seed:** `42` | **Dataset:** `fesevu/ethereum_fraud_dataset_by_activity` (DOI `10.57967/hf/6373`, CC-BY-4.0)

---

## 1. Talk Tracks

### 2-Minute Elevator Pitch
> "Ethereum phishing and scam wallets steal billions, yet most detectors rely on opaque graph embeddings or raw transaction dumps. In *Anatomy of a Scam Wallet*, I ask which human-interpretable on-chain behavioral groups actually drive detection. From the 96 GB Hugging Face Ethereum fraud dataset, I built a low-RAM DuckDB pipeline to extract a contract-matched sample of **2,522 wallets** (1,261 fraud, 1,261 normal; 1,053 EOAs and 208 contracts per class) and engineered **28 features** across four behavioral groups: Temporal ($G_{\text{temp}}$, 6), Value ($G_{\text{val}}$, 12), Network ($G_{\text{net}}$, 4), and Fee/Contract Interaction ($G_{\text{tok}}$, 6).
>
> Comparing tuned Random Forest and XGBoost on a strict 80/20 split (2,017 train / 505 test) with train-only `RobustScaler`, both models achieve virtually identical held-out Fraud-Class F1 (**0.8172** RF vs. **0.8175** XGB, $p = 0.956$), refuting the hypothesis that gradient boosting outperforms bagging here. Grouped ablation and SHAP show that **Temporal cadence** (`G_temp_lifespan_days`, importance 0.149) and **Value asymmetry** carry the signal, while **newly created normal wallets** (< 7 days) suffer a **0.70–0.85 false-positive rate**, highlighting a critical deployment risk."

### 5-Minute Structured Presentation
1. **Problem & Motivation (0:45):** Why behavioral interpretability matters for compliance/triage, and why naive class-imbalanced or contract-confounded datasets mislead models.
2. **Low-RAM Data & Matched Design (1:00):** Streaming a 96 GB dataset on < 4 GB RAM via DuckDB predicate pushdown; matching `is_contract` (1,053 EOA / 208 contract per class) and computing uncapped per-wallet stats before shard deletion.
3. **28-Feature Behavioral Taxonomy (1:00):** $G_{\text{temp}}$ (6), $G_{\text{val}}$ (12), $G_{\text{net}}$ (4), $G_{\text{tok}}$ (6; explicitly noting absence of `gas_price`, `is_error`, and ERC-20 logs). Leakage prevention (`is_contract` dropped; `log1p` + `RobustScaler` fit on train only).
4. **Model Parity & Ablation Results (1:15):** 5x5 CV (RF `0.8445`, XGB `0.8554`), Test F1 (RF `0.8172`, XGB `0.8175`, $H_2$ rejected $p = 0.956$), LOGO vs. single-group-only vs. random controls with Holm-Bonferroni correction.
5. **Bias Audit, Evasion & Deployment Reality (1:00):** Short-lifespan FPR (`0.70` RF, `0.85` XGB), analytical base-rate precision drop (`prevalence_table`), and human-in-the-loop recommendation.

---

## 2. 10-Slide Outline & Figure Mapping

| Slide | Title | Key Content | Report Figure / Table |
| :---: | :--- | :--- | :--- |
| 1 | Title & Research Question | Which behavioral feature groups drive Ethereum scam detection? | — |
| 2 | Dataset & Low-RAM Pipeline | 96 GB HF dataset -> DuckDB shard streaming -> < 4 GB RAM | `report/figures/class_balance.pdf` |
| 3 | Matched Design & Cleaning | 2,522 wallets (1,261/1,261; 1,053 EOA, 208 contract); p99.9 winsorization | `report/figures/value_distribution.pdf` |
| 4 | 4 Behavioral Feature Groups | $G_{\text{temp}}$ (6), $G_{\text{val}}$ (12), $G_{\text{net}}$ (4), $G_{\text{tok}}$ (6) = 28 features | `report/figures/eda_top10_features.pdf` |
| 5 | Leakage-Free Preprocessing | 80/20 split (2,017/505), `log1p` (`|skew|>2`), train-only `RobustScaler` | `report/figures/eda_corr_heatmap.pdf` |
| 6 | Tuning & Model Parity ($H_2$) | 1-SE CV selection; Test F1: RF `0.8172` vs. XGB `0.8175` ($p=0.956$) | `report/figures/eval_roc_pr.pdf` |
| 7 | Grouped Ablation (LOGO & ONLY) | Leave-one-group-out & single-group-only vs. random controls | `report/figures/ablation_logo_only.pdf` |
| 8 | Feature & Group Importance | Permutation importance + SHAP; `G_temp_lifespan_days` (`0.149`) | `report/figures/ablation_importance.pdf` |
| 9 | Subgroup Bias & Evasion | Short-lifespan (<7d) FPR `0.70`/`0.85`; FN evasion archetypes | `report/figures/bias_subgroup_fpr.pdf` |
| 10 | Prevalence, Limits & Next Steps | Real-world base-rate precision drop; missing ERC-20/gas_price | `outputs/eval/prevalence_table.csv` |

---

## 3. Comprehensive 40-Question Q&A

### A. Data Collection & Schema (Q1–Q7)
1. **Why not download the full 96 GB dataset?** It has ~1.62B rows and exceeds 8–16 GB laptop RAM. DuckDB predicate pushdown + `--max-gb 4.0` streams monthly shards sequentially under 4 GB RAM.
2. **Why match fraud and normal wallets on `is_contract`?** In the raw labels, 73% of scam wallets are contracts vs. 50% of normal wallets. Matching (1,053 EOA + 208 contract per class) prevents models from taking a trivial contract-vs-EOA shortcut.
3. **Why did you compute uncapped stats before deleting shards?** Each wallet is capped at 1,000 rows in `raw_transactions.parquet`; computing `sent_cnt`, `recv_cnt`, `first_ts`, `last_ts`, and ETH sums on the full shard preserves true lifespan and counts.
4. **Why avoid `OR`-joins in DuckDB when aggregating stats?** `src_id = wallet OR dst_id = wallet` prevents hash-join optimization and spikes memory; aggregating `src` and `dst` separately and unioning keeps RAM low.
5. **What happens if a wallet has 0 sent or 0 received transactions?** Inter-tx times default cleanly (0/median-safe) and ratios use epsilon/zero-safe denominators so `nan_audit` is `0` across all splits.
6. **Why drop `is_contract` from the 28 model features?** Because both classes are identically matched on `is_contract`, keeping it adds zero legitimate signal and risks spurious splits.
7. **What fields are missing from the Hugging Face dataset?** `gas_price`, `is_error` (failed tx flag), and ERC-20 token transfer logs; `G_tok` therefore captures `tx_fee_eth`, contract-creation, and zero-value call proxies.

### B. Feature Engineering & Leakage Prevention (Q8–Q14)
8. **Why 28 features across 4 groups?** They map raw logs to interpretable behavioral pillars: Temporal (6), Value (12), Network (4), Fee/Contract (6).
9. **Why `log1p` on 23 of 28 features?** Raw blockchain counts and ETH values are heavily right-skewed (skewness up to 44.9 for contract creation); `log1p` on train features with `|skew| > 2.0` stabilizes distributions.
10. **Why `RobustScaler` instead of `StandardScaler`?** Even after `log1p`, heavy-tailed whale outliers distort mean/std; `RobustScaler` uses median and IQR, which are outlier-resistant.
11. **How do you guarantee zero data leakage?** The 80/20 stratified split (2,017/505) happens *before* `log1p` threshold checks, `RobustScaler.fit()`, hyperparameter tuning, and CV.
12. **Why winsorize at p99.9 in cleaning?** Extreme Wei/ETH outliers (e.g., exchange sweeps) can dominate tree splits or plots; winsorizing caps only the top 0.1%.
13. **What is `G_temp_burstiness_max_1h`?** Maximum number of transactions a wallet executes within any 1-hour window — capturing automated drainer/phishing bursts.
14. **Why is `G_net` only 4 features?** The dataset provides sender/recipient IDs per tx without full multi-hop graph neighborhoods, yielding `unique_sent_to`, `unique_recv_from`, `unique_cp_per_tx`, and `fan_in_out_ratio`.

### C. Modeling & Hyperparameter Tuning (Q15–Q21)
15. **Why compare Random Forest and XGBoost?** RF represents parallel variance-reducing bagging; XGBoost represents sequential bias-reducing gradient boosting. Both handle non-linear tabular interactions well.
16. **What is the 1-SE rule in `tune_hyperparameters.py`?** Instead of picking the rank-1 CV score (which may overfit), we pick the simplest model within 1 standard error of the best mean CV Fraud-F1.
17. **Why optimize Fraud-Class F1 instead of Accuracy?** Even in a balanced eval, false negatives (missed scams) and false positives (blocked users) both carry high cost; Fraud-F1 balances precision and recall on class 1.
18. **How does `make_pipeline` ensure fair ablation?** `ColumnTransformer` projects only the selected feature subset inside the scikit-learn `Pipeline`, keeping estimator hyperparameters and CV folds identical.
19. **Why use `scale_pos_weight` in XGBoost tuning?** Even on a 50/50 train split (1,008/1,009), testing `[0.5, 1.0, 2.0]` tunes the precision-recall trade-off for Fraud-F1.
20. **Why fix `SEED = 42` everywhere?** Ensures deterministic sampling, splits, CV folds, RF/XGB initialization, and bootstrap resampling.
21. **How is leakage sanity-checked during tuning?** `run_leakage_sanity_check` verifies no ID/target columns enter `X` and that train/test wallet sets are disjoint.

### D. Evaluation, Ablation & Statistics (Q22–Q30)
22. **Why did $H_2$ (XGB > RF) fail?** Held-out test Fraud-F1 is `0.8172` (RF) vs. `0.8175` (XGB), $\Delta = +0.0002$, paired bootstrap $p = 0.956$. With 2,017 well-engineered tabular rows, bagging and boosting extract nearly identical decision boundaries.
23. **Why is 5x5 CV F1 (~0.845–0.855) higher than Test F1 (~0.817)?** Small-sample variance on $n=505$ test wallets plus slight optimism in CV; both models show consistent ~3% generalization gap.
24. **What is Leave-One-Group-Out (LOGO) ablation?** Removing one group at a time (e.g., 28 -> 22 features) to measure unique, non-redundant predictive contribution ($\Delta F_1$).
25. **What is Single-Group-Only ablation?** Training with only one group (e.g., 6 features) to measure standalone sufficiency.
26. **Why include random feature-count controls?** A group with 12 features ($G_{\text{val}}$) has more capacity than one with 4 ($G_{\text{net}}$); comparing against 10 random subsets of size $k$ separates *group semantics* from *feature count*.
27. **Why apply Holm-Bonferroni correction?** Testing multiple ablation hypotheses ($H_1, H_3$) inflates family-wise Type I error; Holm-Bonferroni controls it without being as overly conservative as raw Bonferroni.
28. **What is the single most important feature?** `G_temp_lifespan_days` (RF importance `0.149`), because scam wallets typically operate in short, intense campaigns before abandonment.
29. **Why use both Permutation Importance and SHAP?** Permutation importance measures global loss in F1 when shuffling a feature group; TreeExplainer SHAP provides local additive attributions per wallet.
30. **Why use paired bootstrap for $\Delta F_1$?** Both models are evaluated on the exact same 505 test wallets; paired resampling accounts for sample correlation and yields exact 95% CIs and $p$-values.

### E. Bias, Evasion & Deployment (Q31–Q40)
31. **What is the biggest fairness/bias failure found?** Normal wallets with `lifespan < 7 days` have a **0.70 (RF)** and **0.85 (XGB)** false-positive rate because short lifespan mimics hit-and-run scam behavior.
32. **How does a 50/50 matched design affect real-world deployment?** At a 0.1%–1% real-world scam base rate, a 0.70 FPR on new wallets would flood compliance queues with false alarms (`prevalence_table.csv`).
33. **How should this model be deployed safely?** As a **risk-triage score** combined with wallet age gating, allowlists, and human review — never autonomous account freezing.
34. **What evasion archetypes appear in False Negatives (`evasion_analysis.py`)?** Low-frequency "sleeper" scams that mimic normal lifespan/cadence and low-value dust/split transfers that stay below typical value thresholds.
35. **Can a scammer evade `G_temp_lifespan_days`?** Yes, by aging wallets with benign transactions before attacking ("aged-wallet evasion"), which is why multi-group ensembles ($G_{\text{val}} + G_{\text{net}}$) remain necessary.
36. **What if we had ERC-20 token transfer data?** Approval-phishing (`approve`/`transferFrom`) and rug-pull liquidity drains would make $G_{\text{tok}}$ much stronger.
37. **Why not use a Graph Neural Network (GNN)?** GNNs require full multi-hop adjacency matrices (huge RAM/compute) and lack direct feature-group interpretability for compliance teams.
38. **How fast is inference?** Once features are computed, RF/XGB inference on 505 wallets takes milliseconds on CPU.
39. **How do we know the numbers in the paper match the code?** `tools/check_numbers.py` cross-checks `report/main.tex` against `feature_report.json` and evaluation outputs.
40. **What is the #1 future improvement?** Incorporating ERC-20 `Transfer`/`Approval` logs and temporal out-of-time validation splits across newer Ethereum blocks.

---

## 4. Eight Hard / Adversarial Questions
1. **"Your test F1 is 0.817 on a 50/50 dataset, which means terrible precision in the real world. Isn't this unusable?"**  
   *Answer:* Yes if deployed as an automated blocker — `prevalence_table` explicitly proves this. It is designed as a **triage ranker** and an **ablation study** of behavioral signals, not an uncalibrated production classifier.
2. **"Why did you drop `is_contract` if 73% of scams are contracts?"**  
   *Answer:* We matched classes on `is_contract` *specifically* to remove that confound so the model learns *behavioral* differences rather than simply flagging smart contracts.
3. **"Your `G_tok` group doesn't actually have ERC-20 tokens or gas price. Isn't the name misleading?"**  
   *Answer:* Good catch — we renamed it to **Fee & Smart Contract Interaction** in the final paper and explicitly document the 3 missing fields in `feature_report.json` and Section IV.
4. **"Why didn't XGBoost beat Random Forest ($p = 0.956$)?"**  
   *Answer:* With 2,017 training rows and 28 clean, log-transformed tabular features, both tree ensembles saturate the same axis-aligned signal; boosting's residual fitting adds no significant edge over bagging.
5. **"A 0.85 FPR on <7-day normal wallets means you flag almost every new Ethereum user!"**  
   *Answer:* Correct — `G_temp_lifespan_days` (`0.149` importance) is a double-edged sword. Production systems must exempt or down-weight newly funded wallets or require corroborating $G_{\text{val}}/G_{\text{net}}$ signals.
6. **"Isn't 2,522 wallets tiny compared to 114,739 labeled nodes?"**  
   *Answer:* It is a deliberate trade-off to keep shard streaming under 4 GB RAM and 4 GB download budget on consumer hardware while preserving exact uncapped per-wallet stats.
7. **"How do you know your 1,000-tx cap didn't bias `G_val` or `G_net`?"**  
   *Answer:* Counts and ETH sums use the **uncapped** shard aggregates (`*_uncapped`), while distribution shapes (min/max/mean/std) for high-volume wallets (>1,000 tx/month) are bounded by the 1,000-tx sample.
8. **"Why random-split 80/20 instead of a strict temporal train/test split?"**  
   *Answer:* Monthly shards are sampled across time for candidate coverage; a temporal out-of-time split is a key next step to measure concept drift against newer phishing kits.

---

## 5. Key Terms Glossary (25 Terms)
1. **Ablation Study:** Systematically removing or isolating feature groups to measure their marginal contribution.
2. **Bagging (Bootstrap Aggregating):** Training parallel trees on bootstrap samples to reduce variance (Random Forest).
3. **Burstiness:** Maximum transaction count in a short window (1 hour), indicating automated activity.
4. **Class Imbalance:** Disproportionate ratio of normal to scam wallets in real-world Ethereum traffic.
5. **ColumnTransformer:** Scikit-learn utility projecting exact feature subsets inside a `Pipeline`.
6. **Contract Matching:** Sampling equal EOA/contract proportions in both classes to eliminate entity-type confounding.
7. **Data Leakage:** Using test-set statistics (e.g., scaling/imputation) during training, inflating apparent performance.
8. **DuckDB Predicate Pushdown:** Filtering parquet row groups on disk before loading into memory.
9. **EOA (Externally Owned Account):** User-controlled Ethereum wallet (private key) vs. smart contract.
10. **Fan-in / Fan-out Ratio:** Ratio of unique incoming senders to unique outgoing recipients.
11. **Fraud-Class F1:** Harmonic mean of precision and recall specifically for the positive (`is_scam=1`) class.
12. **Gradient Boosting:** Sequential additive tree fitting on pseudo-residuals (XGBoost).
13. **Holm-Bonferroni Correction:** Step-down multiple-testing adjustment controlling family-wise error rate.
14. **LOGO (Leave-One-Group-Out):** Ablation removing one behavioral group at a time from the full 28-feature set.
15. **log1p Transform:** $\ln(1 + x)$ applied to right-skewed (`|skew| > 2`) non-negative features.
16. **Matched Design:** Constructing a 50/50 class-balanced sample paired on confounding covariates (`is_contract`).
17. **1-SE Rule:** Selecting the simplest hyperparameter config within one standard error of the best CV score.
18. **Paired Bootstrap:** Resampling test predictions with replacement to compute CIs and $p$-values for $\Delta F_1$.
19. **Permutation Importance:** Drop in F1 when a feature (or group) is randomly shuffled on held-out data.
20. **Prevalence Projection:** Analytical recalculation of precision under realistic base rates (0.1%–1%) via Bayes' rule.
21. **RobustScaler:** Scaling by subtracting median and dividing by IQR (resistant to heavy-tailed outliers).
22. **SHAP (SHapley Additive exPlanations):** Game-theoretic local feature attribution for tree ensembles.
23. **Uncapped Stats:** Exact per-wallet counts/sums computed across full monthly shards prior to 1,000-row capping.
24. **Wei / Gwei / ETH:** Ethereum base units ($1\text{ ETH} = 10^{18}\text{ Wei} = 10^9\text{ Gwei}$).
25. **Winsorization (p99.9):** Capping extreme values at the 99.9th percentile to limit outlier distortion.
