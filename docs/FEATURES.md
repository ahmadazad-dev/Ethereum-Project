# Feature Definitions and Behavioral Group Partition

Single source of truth for group membership in code: `FEATURE_GROUPS` in `src/features.py`.
Every engineered/retained model feature must belong to **exactly one** group (`G1`–`G5`). Changing group membership invalidates downstream ablation results and requires an ADR in `docs/DECISIONS.md` plus a full rerun (`AGENTS.md` §1).

## Behavioral Groups (`G1`–`G5`)

| Group ID | Name | Core Behavioral Hypothesis |
|---|---|---|
| `G1` | Temporal & Cadence | Captures transaction timing, lifespan, and inter-arrival burstiness distinguishing automated/hit-and-run scam campaigns from organic user activity. |
| `G2` | Native ETH Value Flow | Captures incoming/outgoing native ETH magnitude, dispersion, and flow asymmetry (e.g., many small victim inflows consolidated into large cash-out sweeps). |
| `G3` | Contract & Internal-Trace Activity | Captures smart-contract deployment/interaction, internal message calls, and gas/execution complexity characteristic of Ponzis, honeypots, and drainers. |
| `G4` | Counterparty & Fan-In/Fan-Out Topology | Captures local ego-graph structure (unique senders/recipients, fan-in/fan-out asymmetry, counterparty concentration). |
| `G5` | ERC-20 Token Behavior | Captures ERC-20 token transfer counts, token contract diversity, token counterparty topology, and token value statistics. |

## Feature Ledger

*(To be populated once the dataset schema is registered in `data/MANIFEST.md` and `FEATURE_GROUPS` is finalized in `src/features.py` via ADR.)*

| Feature Name | Group | Formula / Definition | On-Chain Mechanism & Scam Type | Failure Modes / Legitimate Lookalikes | Leakage Risk |
|---|---|---|---|---|---|
| *(pending dataset registration)* | — | — | — | — | — |

