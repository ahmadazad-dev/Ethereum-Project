"""Phases 3–5 (Deliverables 2 & 3): Central configuration, paths, and random seeds."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "report"
FIGURES_DIR = REPORT_DIR / "figures"

RANDOM_SEED = 42
