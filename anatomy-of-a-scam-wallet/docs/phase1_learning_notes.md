# Phase 1 Research Learning Notes: *Anatomy of a Scam Wallet*

*Mentor's Guide for Undergraduate/Graduate Blockchain Security Research (Track B)*

## 1. How to Use Google Scholar Effectively for Security Research

Conducting a rigorous literature review requires moving beyond keyword guessing to structured bibliographic discovery:

1. **Boolean and Exact-Phrase Search Operators:** Use quotation marks for exact technical phrases, `OR` for synonyms, and `intitle:` to restrict scope. For example:
   `("Ethereum" AND ("phishing" OR "illicit accounts")) ("XGBoost" OR "Random Forest" OR "ablation")`.
   Use the year filter (e.g., *Since 2022*) to capture modern threat vectors like EvolveGCN (`zhang2024`) and XAI ensemble stacking (`chithanuru2025`).

2. **Forward and Backward Snowballing ("Cited by" & "Related articles"):** Once you find a foundational anchor paper—such as [Farrugia et al. (2020)](https://www.um.edu.mt/library/oar/handle/123456789/136426) or [Wu et al. (2020)](https://arxiv.org/abs/1911.09259)—click **"Cited by"** on Google Scholar and check *"Search within citing articles"* to see how newer papers (2022–2025) critique its dataset or features. Use **"Related articles"** to uncover parallel communities (e.g., ACM WWW vs. IEEE TSMC).

3. **The 3-Pass Reading Protocol (Abstract** $\rightarrow$ **Methods** $\rightarrow$ **Limitations):** Never cite a paper from its title alone. Read the **Abstract** first to identify the task and headline metric; jump directly to **Methodology / Experimental Setup** to inspect the exact dataset size, feature list, and evaluation metric (e.g., whether they report minority Fraud-Class $F_1$ or just overall accuracy); finally, read the **Discussion / Limitations** to find what the authors left unsolved.

## 2. Why Static Heuristics Fail vs. Multi-Dimensional Statistical Learning

Rule-based systems rely on fixed, single-variable thresholds (e.g., `IF tx_value > 10 ETH AND time_since_received < 10 min THEN flag_fraud`). Because blockchain code and compliance thresholds are predictable, adaptive adversaries defeat static rules using **transaction structuring (smurfing)** and rapid address rotation.

### Concrete Worked Example: Transaction Splitting (Structuring)

Suppose an exchange compliance rule flags any wallet that forwards more than $\tau_{\text{val}} = 5.0\text{ ETH}$ in a single outgoing transaction within $\tau_{\text{time}} = 30\text{ minutes}$ of receiving funds.

* **Adversary Evasion:** A phishing drainer steals $12.0\text{ ETH}$ from a victim. Instead of sending one $12.0\text{ ETH}$ transfer, the drainer's automated script splits the haul into **30 outgoing transfers of** $0.40\text{ ETH}$ **each**, spaced $65\text{ seconds}$ apart, routed to 30 freshly generated intermediary wallets.

* **Why the Heuristic Fails:** Every individual transfer ($0.40\text{ ETH} < 5.0\text{ ETH}$) passes the threshold rule silently.

* **Why Multi-Dimensional ML Succeeds:** Random Forest and XGBoost evaluate the **joint distribution** across multiple engineered behavior groups simultaneously:

  * *Group 1 (Timing):* Variance of inter-transaction arrival times is near zero ($\sigma_{\Delta t} \approx 0\text{ s}$), exposing automated scripting.

  * *Group 2 (Value Flow):* Ratio of total outgoing to incoming ETH equals $12.0 / 12.0 = 1.00$ with near-zero retained balance.

  * *Group 3 (Counterparty Network):* Out-degree to in-degree ratio spikes ($30 : 1$), forming a classic star-dispersion motif.
    Even though each single feature looks benign in isolation, their non-linear combination inside a decision tree partition (`out_in_degree_ratio > 15 AND balance_retention < 0.01 AND inter_tx_std < 5s`) catches the scam wallet immediately.

## 3. Step-by-Step Identification of the Research Gap & Hypothesis Derivation

1. **Step 1 — Audit Prior Art:** Reviewing 11 key studies revealed two dominant camps: GNN models ([Chen et al., 2020](https://www.semanticscholar.org/paper/Phishing-Scams-Detection-in-Ethereum-Transaction-Chen-Peng/ed9b9a3001c801579d8b9be1ef1a9bd309ae1e4d); [Li et al., 2022](https://arxiv.org/abs/2204.13442); [Zhang et al., 2024](https://www.computer.org/csdl/journal/sc/2024/06/10552120/1XAp46W7rig)) that learn opaque node embeddings, and tabular ML papers ([Farrugia et al., 2020](https://www.um.edu.mt/library/oar/handle/123456789/136426); [Chithanuru & Ramaiah, 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11935758/); [Gu & Dib, 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11888915/)) that dump 17–42 engineered features into XGBoost/RF to maximize accuracy.

2. **Step 2 — Spot the Methodological Flaw:** When tabular papers inspect feature importance, they rank *individual* variables. However, on-chain summary statistics within the same behavioral domain (e.g., `mean_val_sent`, `max_val_sent`, `total_val_sent`) are heavily correlated. Decision trees split importance across correlated features, masking the true impact of the underlying behavioral category.

3. **Step 3 — Formulate the Gap & Testable Hypotheses:** Nobody has systematically grouped features into behavioral domains (Timing, Value, Counterparty Network, Contract/Token) and performed a **Leave-One-Group-Out (LOGO) ablation study** measured by minority **Fraud-Class** $F_1$. From this gap, **RQ1–RQ3** and falsifiable hypotheses $H_1$**–**$H_3$ follow directly: each hypothesis specifies an exact direction and measurable $\Delta F_1$ threshold (e.g., $\Delta F_1 \ge 0.03$) that Phase 2 experiments can either confirm or refute.

## 4. Student Self-Quiz (5 Questions & Answers)

1. **Q1:** *Why is overall classification accuracy a misleading metric for Ethereum phishing detection, and why do we use Fraud-Class* $F_1$ *instead?*

   * **Answer:** Phishing accounts make up a tiny fraction ($<1\%$) of Ethereum addresses. A dummy classifier that predicts "Benign" for every wallet achieves $>99\%$ accuracy while catching zero scammers ($0\%$ recall). Fraud-Class $F_1$ computes the harmonic mean of Precision and Recall specifically on the positive (fraud) class, penalizing both false alarms and missed scammers.

2. **Q2:** *Why can't Random Forest or XGBoost ingest raw Ethereum transaction logs directly?*

   * **Answer:** Raw blockchain logs are variable-length chronological sequences of edges $(u, v, t, \text{value}, \text{gas})$ where one wallet may have 2 transactions and another may have 50,000. Tree ensembles require a fixed-width feature vector $\mathbf{x}_i \in \mathbb{R}^d$ (one row per wallet), requiring summary aggregation (means, standard deviations, ratios, degree counts).

3. **Q3:** *Why is single-feature importance (e.g., standard Gini importance or top-3 gain) insufficient to determine which on-chain behavior matters most?*

   * **Answer:** Because engineered features within a behavioral group are collinear (e.g., `total_eth_received` and `avg_eth_received`), trees split their importance weights across those correlated variables, artificially deflating each individual feature's rank. Grouped ablation removes the entire behavioral category at once, revealing its true unique contribution.

4. **Q4:** *What specific algorithmic mechanism allows XGBoost to handle severe class imbalance differently than standard Random Forest?*

   * **Answer:** XGBoost uses sequential second-order gradient boosting and supports cost-sensitive instance weighting via `scale_pos_weight`, which scales the gradient and hessian of minority fraud examples so the loss function heavily penalizes misclassifying scam wallets.

5. **Q5:** *Why are smart-contract Ponzi detectors like SADPonzi ([Chen et al., 2021](https://research.polyu.edu.hk/en/publications/sadponzi-detecting-and-characterizing-ponzi-schemes-in-ethereum-s-2/)) unable to catch standard phishing or wallet-drainer EOAs?*

   * **Answer:** SADPonzi relies on symbolic execution of deployed EVM bytecode. Standard phishing recipient wallets and EOA drainers are Externally Owned Accounts controlled by private keys; they store zero contract bytecode on-chain and can only be detected via their transactional and token-interaction behavior.