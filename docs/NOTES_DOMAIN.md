# Ethereum Domain Notes

Verified first-principles notes on Ethereum execution semantics relevant to feature engineering and error analysis (`AGENTS.md` §8). Unverified assumptions must be checked against official Ethereum documentation or observed raw transaction data before relying on them.

## Checklist of Core Concepts to Verify & Document

- [ ] **EOA vs. Smart Contract accounts:** Key-controlled nonces vs. deployed bytecode; contracts cannot initiate top-level transactions directly (`tx.from` is always an EOA).
- [ ] **Normal transactions vs. Internal traces vs. ERC-20 `Transfer` logs:** Why contract-initiated ETH payouts appear only in internal traces (`CALL`, `DELEGATECALL`, `CREATE`), and why ERC-20 transfers appear as normal `value = 0` calls to the token contract with `Transfer(...)` event logs.
- [ ] **Units & Gas mechanics:** Wei vs. Gwei vs. ETH (`1 ETH = 10^18 Wei`); `gasUsed` vs. `gasLimit`; EIP-1559 `baseFeePerGas` + `maxPriorityFeePerGas` vs. legacy `gasPrice`.
- [ ] **Legitimate lookalikes (expected false-positive archetypes):** Exchanges/hot wallets, MEV bots, high-frequency arbitrageurs, payment processors, airdrop distributors, privacy mixer users, and newly created benign accounts with 1–2 transactions.

