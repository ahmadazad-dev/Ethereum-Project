# Phase 4 Learning Notes: Decision Trees, Bagging, Boosting, & Bias-Variance Tradeoff

## 1. How Decision Trees Split
A classification tree recursively partitions the feature space $\mathcal{X} \subset \mathbb{R}^{28}$ into axis-aligned regions using greedy binary splits $(j, t)$ that maximize impurity reduction:
- **Gini Impurity (Random Forest):** For node $m$ with fraud proportion $p_m$, $G(m) = 2 p_m (1 - p_m)$. A split maximizes $\Delta G = G(m) - \frac{N_L}{N_m}G(m_L) - \frac{N_R}{N_m}G(m_R)$.
- **Second-Order Gradient Gain (XGBoost):** Instead of Gini, XGBoost fits regression trees to the first-order gradient $g_i$ and second-order Hessian $h_i$ of binary log-loss, scoring splits via $\text{Gain} = \frac{1}{2}\left[\frac{(\sum_{i\in L} g_i)^2}{\sum_{i\in L} h_i + \lambda} + \frac{(\sum_{i\in R} g_i)^2}{\sum_{i\in R} h_i + \lambda} - \frac{(\sum_{i\in m} g_i)^2}{\sum_{i\in m} h_i + \lambda}\right] - \gamma$.

## 2. Bagging (Random Forest) vs. Gradient Boosting (XGBoost)
- **Bagging (Variance Reduction):** Random Forest trains $B$ deep, low-bias trees in parallel on bootstrap resamples, sampling only `max_features` columns per split to decorrelate trees. Averaging their probabilities cancels out individual tree variance.
- **Gradient Boosting (Bias Reduction):** XGBoost trains $T$ shallow, high-bias trees *sequentially*: each new tree $f_t(x)$ corrects the residual log-loss errors of the ensemble $\hat{y}_i^{(t-1)}$, shrinking updates by `learning_rate` ($\eta$) and penalizing leaf weights via $L_2$ regularization (`reg_lambda`).

## 3. Hyperparameter Effects on the Bias-Variance Tradeoff

| Model | Hyperparameter | Effect When Increased | Effect When Decreased |
| :--- | :--- | :--- | :--- |
| **RF** | `n_estimators` | Lowers variance (smoother ensemble average) without overfitting | Higher variance; faster training |
| **RF** | `max_depth` | Lower bias, higher variance (deeper, memorizing splits) | Higher bias, lower variance (simpler trees) |
| **RF** | `min_samples_split` | Higher bias, lower variance (prevents small-node splits) | Lower bias, higher variance |
| **RF** | `min_samples_leaf` | Higher bias, lower variance (enforces larger leaf support) | Lower bias, higher variance |
| **RF** | `max_features` | Lower bias, more correlated trees (higher variance) | Stronger tree decorrelation, lower variance |
| **RF** | `class_weight` | `"balanced"` upweights minority errors; `None` treats 50/50 equally | — |
| **XGB** | `n_estimators` | Lower bias, higher variance if `learning_rate` is large | Higher bias (underfitting) |
| **XGB** | `learning_rate` | Faster fit, higher variance (risk of overshooting) | Lower variance, requires more `n_estimators` |
| **XGB** | `max_depth` | Lower bias, higher variance (captures deeper interactions) | Higher bias, lower variance |
| **XGB** | `subsample` | `1.0` uses all rows (higher variance); `<1.0` adds stochastic bagging | Lower variance, slight bias increase |
| **XGB** | `colsample_bytree` | Uses more features/tree (lower bias, higher tree correlation) | Stronger feature decorrelation, lower variance |
| **XGB** | `min_child_weight` | Higher bias, lower variance (requires larger Hessian sum per leaf) | Lower bias, higher variance |
| **XGB** | `reg_lambda` | Higher bias, lower variance (shrinks leaf weights toward zero) | Lower bias, higher variance |
| **XGB** | `scale_pos_weight` | `>1` boosts Fraud recall (risks false positives); `<1` boosts precision | `1.0` = neutral weighting |
