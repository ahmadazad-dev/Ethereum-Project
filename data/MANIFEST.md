# Data Provenance Manifest

All raw datasets are immutable. Never commit raw or derived data files (or files >100 MB) to Git; store them under `data/raw/` and `data/derived/` (both gitignored) and record every source, access date, query/parameters, row/column counts, and SHA-256 hash here.

## Registered Datasets

| Dataset ID | Source / Provenance | Access Date (UTC) | Parameters / Version | Local Path (gitignored) | Rows | Columns | Class Counts (Fraud / Non-Fraud) | SHA-256 | Notes |
|---|---|---|---|---|---|---|---|---|---|
| *(none registered yet)* | — | — | — | — | — | — | — | — | Pending dataset source selection per `AGENTS.md` §7 and §15 |

## Cleaning & Transformation Logs

Machine-written before/after cleaning summaries (rows, columns, nulls, duplicates, constants, outliers) will be linked or appended below when `src/data_scripts/` executes on a registered raw dataset.

