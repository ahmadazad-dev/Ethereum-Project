# Threats to Validity

Running inventory of construct, internal, external, and conclusion validity threats (`AGENTS.md` §6, §11, §16). Every item is tracked with concrete empirical checks and mitigations.

## 1. Internal Validity & Data Leakage Hazards

- **Identifier and index leakage:** Raw datasets frequently include row indices (`Unnamed: 0`, `Index`), address hex strings, or label-ordered rows. *Mitigation:* Drop all identifier/address columns before feature matrix construction and assert exclusion in code and tests.
- **Duplicate / near-duplicate wallets across folds:** Exact or near-duplicate feature/address records can straddle train/test or CV folds. *Mitigation:* Detect exact and near duplicates prior to splitting and group them in stratified splits (`StratifiedGroupKFold` / grouped holdout), logging exact counts.
- **Duplicate and constant columns:** Public Ethereum tabular benchmarks are suspected to contain duplicate `.1`-suffixed or zero-variance columns (`AGENTS.md` §6, §16). *Mitigation:* Programmatically detect and log dropped constant (`std == 0`) and identical columns during Phase 2 cleaning.
- **Post-flagging snapshot leakage:** Features such as current ETH balance (`Total Ether Balance`) may reflect post-scam / post-flagging wallet depletion rather than active scam behavior. *Mitigation:* Audit candidate snapshot features and run paired leakage-sensitivity ablations with vs. without them (RQ5).
- **Preprocessing leakage:** Fitting imputers, scalers, resamplers, or threshold tuning outside training folds inflates CV estimates. *Mitigation:* Encapsulate all preprocessing inside `sklearn.pipeline.Pipeline` (or `imblearn.pipeline.Pipeline`) and verify with an explicit leakage unit test.
- **Hyperparameter freezing bias:** Tuning once on `{G1..G5}` and freezing across ablations avoids confounding tuning variance with group removal, but may advantage the full set. *Mitigation:* Run a budget-limited re-tuning sensitivity check on key ablated subsets (`AGENTS.md` §5).

## 2. Construct Validity

- **Positive-unlabeled (PU) / incomplete label noise:** Negatives represent "accounts not known to be flagged" (e.g., unflagged by Etherscan community tags), not verified benign users. False positives may include genuine unflagged scammers, while label noise bounds achievable precision. *Mitigation:* Frame conclusions explicitly around the label source and consider label-noise sensitivity checks.
- **Structural vs. missing-at-random NaNs:** Missing ERC-20 or internal-trace values often denote zero activity rather than missing observations. *Mitigation:* Include explicit activity indicator flags and compare zero-fill vs. median imputation (`AGENTS.md` §6).
- **Non-orthogonality of behavior groups:** Groups `G1`–`G5` are conceptually separated but empirically correlated (e.g., ETH sent to contracts spans value, contract activity, and topology). *Mitigation:* Compute within- and cross-group Spearman correlation and VIF diagnostics; compare Group-Only (sufficiency) vs. Drop-One-Group (necessity) side by side (`AGENTS.md` §5, §16).

## 3. External & Statistical Conclusion Validity

- **Class prevalence mismatch:** Benchmark fraud prevalence (often ~20%+) dramatically exceeds real-world Ethereum prevalence, inflating observed precision relative to live deployment. *Mitigation:* Report prevalence-reweighted precision alongside benchmark precision (`AGENTS.md` §13).
- **Era / dataset shift:** Scam mechanics evolve (e.g., classic ETH Ponzis vs. modern ERC-20 approval drainers and DEX rug pulls). If benchmark lacks timestamps, temporal generalization cannot be claimed (`AGENTS.md` §6).
- **Non-independence of repeated CV folds:** 5×5 CV produces 25 overlapping train/test evaluations; naive paired t-tests or Wilcoxon tests are anti-conservative. *Mitigation:* Use corrected resampled tests (Nadeau & Bengio) or report effect sizes with confidence intervals and Holm correction (`AGENTS.md` §5).
- **Non-causality:** Group ablations quantify predictive dependence given the feature engineering, estimator family, and label source—not causal mechanisms of fraud (`AGENTS.md` §3).

