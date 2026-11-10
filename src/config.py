"""Configuration loading, hashing, and runtime provenance utilities."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and require a top-level mapping.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ImportError: If ``PyYAML`` is not installed.
        ValueError: If the parsed YAML root is not a dictionary.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load YAML configs. Install via `pip install -r requirements.txt`."
        ) from exc

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at root of {config_path}, got {type(data).__name__}")
    return data


def load_seeds_config(path: str | Path = "configs/seeds.yaml") -> dict[str, Any]:
    """Load and validate the fixed split and CV seed configuration."""
    cfg = load_yaml_config(path)
    required_top = ("holdout", "cv", "tuning", "model")
    for key in required_top:
        if key not in cfg:
            raise ValueError(f"Missing required key '{key}' in {path}")

    cv_cfg = cfg["cv"]
    if not isinstance(cv_cfg, dict) or "seeds" not in cv_cfg or "n_splits" not in cv_cfg:
        raise ValueError(f"Invalid 'cv' section in {path}; expected 'n_splits' and 'seeds'")
    if not isinstance(cv_cfg["seeds"], list) or len(cv_cfg["seeds"]) == 0:
        raise ValueError(f"'cv.seeds' must be a non-empty list in {path}")

    holdout_cfg = cfg["holdout"]
    if "test_size" not in holdout_cfg or "random_state" not in holdout_cfg:
        raise ValueError(f"Invalid 'holdout' section in {path}")

    return cfg


def compute_file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute hex SHA-256 checksum of a file without loading it fully into memory."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Cannot hash missing file: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def compute_dict_sha256(payload: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash of a JSON-serializable dictionary."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def get_git_commit(repo_root: str | Path = ".") -> str:
    """Return current HEAD commit SHA (or 'UNCOMMITTED' / 'NO_GIT' if unavailable)."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "NO_COMMIT"


def get_library_versions(
    packages: tuple[str, ...] = ("numpy", "pandas", "scikit-learn", "xgboost", "scipy", "shap"),
) -> str:
    """Return semicolon-delimited Python and package versions for registry logging."""
    parts = [f"python={platform.python_version()}"]
    for pkg in packages:
        try:
            ver = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            ver = "not-installed"
        parts.append(f"{pkg}={ver}")
    return ";".join(parts)

