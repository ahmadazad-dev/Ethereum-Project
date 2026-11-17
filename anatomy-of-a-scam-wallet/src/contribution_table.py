#!/usr/bin/env python3
"""
Phase 5 (Step 5C) — Git-Derived Author Contribution Table
Project: Anatomy of a Scam Wallet (Ethereum Fraud Detection with Random Forest & XGBoost)
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import pandas as pd

from src.config import ROOT_DIR

MODULES = ["data_scripts/", "src/", "report/", "docs/"]
OUT_DIR = ROOT_DIR / "outputs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("contribution_table")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate git-derived author contribution table")
    parser.add_argument("--alias", action="append", default=[], help="Optional 'Old Name=Canonical Name' rule")
    return parser.parse_args()


def parse_alias_map(alias_list: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in alias_list:
        if "=" not in item:
            raise ValueError(f"Invalid --alias `{item}`. Expected format: 'Old Name=Canonical Name'.")
        old, new = item.split("=", 1)
        mapping[old.strip()] = new.strip()
    return mapping


def map_module(filepath: str) -> str:
    norm = filepath.replace("\\", "/")
    for mod in MODULES:
        if norm.startswith(mod):
            return mod
    return "other"


def main() -> None:
    args = parse_args()
    alias_map = parse_alias_map(args.alias)

    cmd = ["git", "log", "--use-mailmap", "--numstat", "--format=COMMIT:%H|%aN"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"git log failed (exit {proc.returncode}): {proc.stderr.strip()}. "
            "Ensure this script is run inside an initialized git repository with at least one commit."
        )

    commits_seen: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    added_lines: Dict[Tuple[str, str], int] = defaultdict(int)
    deleted_lines: Dict[Tuple[str, str], int] = defaultdict(int)

    current_hash = ""
    current_author = ""

    for line in proc.stdout.splitlines():
        if line.startswith("COMMIT:"):
            parts = line[len("COMMIT:"):].split("|", 1)
            current_hash = parts[0].strip()
            raw_author = parts[1].strip() if len(parts) > 1 else "Unknown"
            current_author = alias_map.get(raw_author, raw_author)
        elif "\t" in line and current_author:
            cols = line.split("\t")
            if len(cols) == 3:
                add_s, del_s, fpath = cols
                mod = map_module(fpath.strip())
                if mod == "other":
                    continue
                commits_seen[(current_author, mod)].add(current_hash)
                if add_s.isdigit():
                    added_lines[(current_author, mod)] += int(add_s)
                if del_s.isdigit():
                    deleted_lines[(current_author, mod)] += int(del_s)

    rows = [
        {
            "author": author,
            "module": mod,
            "commits": len(c_set),
            "lines_added": added_lines[(author, mod)],
            "lines_deleted": deleted_lines[(author, mod)],
        }
        for (author, mod), c_set in sorted(commits_seen.items())
    ]

    if not rows:
        raise RuntimeError("No git commits touching data_scripts/, src/, report/, or docs/ were found.")

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "contribution_table.csv", index=False)

    if df["author"].nunique() == 1:
        print(f"Note: Single contributor detected in git history ({df['author'].iloc[0]}).")

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Git-Derived Author Contributions by Project Module}",
        r"\label{tab:contributions}",
        r"\footnotesize",
        r"\begin{tabularx}{\linewidth}{X l r r}",
        r"\toprule",
        r"\textbf{Author} & \textbf{Module} & \textbf{Commits} & \textbf{Lines ($+/-$)} \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        auth_esc = str(r["author"]).replace("_", r"\_")
        mod_esc = str(r["module"]).replace("_", r"\_")
        lines.append(
            f"{auth_esc} & \\texttt{{{mod_esc}}} & {int(r['commits'])} & $+{int(r['lines_added'])} / -{int(r['lines_deleted'])}$ \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    (OUT_DIR / "contribution_table.tex").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved contribution_table.csv and contribution_table.tex (%d rows).", len(df))


if __name__ == "__main__":
    main()