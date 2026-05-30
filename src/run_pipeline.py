#!/usr/bin/env python3
"""
src/run_pipeline.py
--------------------
Production-grade pipeline orchestrator for trace-based discipline evaluation.

Implements the six-stage pipeline from the Remyx architecture:
    ingestion → parsing → feature extraction → discipline computation →
    aggregation → validation.

Usage:
    python src/run_pipeline.py --config /path/to/config.yaml

The config file (YAML) specifies all input/output paths and stage parameters.
"""

import argparse
import json
import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_ROLL_WINDOW: int = 5
DEFAULT_REWARD_THRESHOLD: float = 0.0
DEFAULT_DISCIPLINE_SIGMA: float = 0.5
LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_ROTATION: str = "midnight"
LOG_BACKUP_COUNT: int = 7
DEFAULT_LOG_DIR: str = "logs"
MAX_RECORDS_WARN: int = 10_000_000  # warn if dataset too large for memory

# ---------------------------------------------------------------------------
# Logging setup (singleton per module, thread‑safe via double‑check)
# ---------------------------------------------------------------------------
_logger_initialized = False
_logger_lock = __import__("threading").Lock()


def _init_logger() -> logging.Logger:
    """Initialize the pipeline logger with a rotating file handler and console handler.

    Returns:
        The configured logger instance.
    """
    global _logger_initialized
    logger = logging.getLogger("discipline_pipeline")
    if _logger_initialized:
        return logger

    with _logger_lock:
        if _logger_initialized:
            return logger
        logger.setLevel(logging.DEBUG)

        # Console handler (INFO+)
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(console)

        # Rotating file handler (DEBUG+) – log to ./logs/pipeline.log
        log_dir = Path(DEFAULT_LOG_DIR)
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_dir / "pipeline.log",
            when=LOG_ROTATION,
            backupCount=LOG_BACKUP_COUNT,
            utc=True,
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(file_handler)

        _logger_initialized = True
    return logger


logger = _init_logger()


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------
class PipelineError(Exception):
    """Base exception for pipeline‑specific errors."""
    pass


class ConfigurationError(PipelineError):
    """Raised when configuration is invalid."""
    pass


class DataValidationError(PipelineError):
    """Raised when input data fails validation checks."""
    pass


class StageExecutionError(PipelineError):
    """Raised when a pipeline stage fails."""
    pass


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _validate_input_path(path: Path, description: str = "Input") -> None:
    """Ensure the given path exists and is a file.

    Args:
        path: Path to validate.
        description: Human‑readable label for error messages.

    Raises:
        FileNotFoundError: If the path does not exist.
        IsADirectoryError: If the path is a directory instead of a file.
        PermissionError: If the file is not readable.
    """
    if not path.exists():
        raise FileNotFoundError(f"{description} file does not exist: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"{description} path is a directory, expected a file: {path}")
    if not os.access(str(path), os.R_OK):
        raise PermissionError(f"{description} file is not readable: {path}")


def _validate_output_path(path: Path) -> None:
    """Ensure the parent directory of the output path exists and is writable.

    Args:
        path: Output file path; parent directory is created if missing.

    Raises:
        PermissionError: If parent directory cannot be created or is not writable.
    """
    parent = path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        # Test write access
        test_file = parent / ".write_test"
        test_file.touch()
        test_file.unlink()
    except (OSError, PermissionError) as e:
        raise PermissionError(f"Cannot write to output directory {parent}: {e}")


def _load_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate a YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary containing validated configuration.

    Raises:
        ConfigurationError: If the config file is missing, malformed, or missing required keys.
    """
    _validate_input_path(config_path, "Configuration")
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"YAML parse error in {config_path}: {e}")

    if not isinstance(config, dict):
        raise ConfigurationError("Configuration must be a YAML mapping (dictionary).")

    required_keys = {"stages", "paths"}
    missing = required_keys - set(config.keys())
    if missing:
        raise ConfigurationError(f"Configuration missing required top‑level keys: {missing}")

    # Validate that each stage referenced in 'stages' has a corresponding entry in 'paths'
    stages = config.get("stages", {})
    paths = config.get("paths", {})
    for stage_name in stages:
        if stage_name not in paths:
            raise ConfigurationError(f"Stage '{stage_name}' defined but no path entry found in 'paths'")
        path_entry = paths[stage_name]
        if not isinstance(path_entry, dict):
            raise ConfigurationError(f"Path entry for stage '{stage_name}' must be a dict with 'input' and/or 'output'")
        # input may be None for first stage, output must exist
        if "output" not in path_entry:
            raise ConfigurationError(f"Stage '{stage_name}' path entry missing 'output' key")

    logger.debug("Configuration loaded successfully from %s", config_path)
    return config


def _detect_and_load(
    path: Path,
    required_columns: Optional[List[str]] = None,
    dtype_map: Optional[Dict[str, type]] = None,
) -> pd.DataFrame:
    """Load a DataFrame from CSV, Parquet, or JSON, with optional validation.

    Performs format detection via file extension.  If JSON, expects a list of records.

    Args:
        path: Input file path.
        required_columns: If provided, checks that all these columns exist in the DataFrame.
        dtype_map: Optional dictionary mapping column names to numpy/pandas dtypes for efficient loading.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the format is unsupported or required columns are missing.
        pd.errors.EmptyDataError: If the file is empty.
        RuntimeError: For unexpected errors during loading.
    """
    _validate_input_path(path)
    suffix = path.suffix.lower()

    # Warn if file is very large
    try:
        file_size_bytes = os.path.getsize(str(path))
        if file_size_bytes > 1_000_000_000:  # >1GB
            logger.warning("Loading large file (%s GB), ensure sufficient memory.",
                           f"{file_size_bytes / 1e9:.2f}")
    except OSError:
        pass  # Best effort warning

    read_kwargs: Dict[str, Any] = {}
    if dtype_map:
        read_kwargs["dtype"] = dtype_map

    try:
        if suffix == ".csv":
            df = pd.read_csv(path, **read_kwargs)
        elif suffix == ".parquet":
            df = pd.read_parquet(path)
        elif suffix == ".json":
            with open(path, "r") as f:
                raw = json.load(f)
            if not isinstance(raw, list):
                raise ValueError("JSON input must be a list of records")
            df = pd.DataFrame(raw)
        else:
            raise ValueError(f"Unsupported input format: {suffix}. Use .csv, .parquet, or .json")

        logger.debug("Loaded %d rows from %s", len(df), path)
    except FileNotFoundError:
        raise
    except pd.errors.EmptyDataError as e:
        logger.error("File %s is empty: %s", path, e)
        raise
    except ValueError as e:
        logger.error("Data format error in %s: %s", path, e)
        raise
    except Exception as e:
        logger.exception("Unexpected error while loading %s", path)
        raise RuntimeError(f"Data loading failed: {e}") from e

    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise DataValidationError(f"Missing required columns: {missing} in file {path}")

    return df


def _save_dataframe(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Save a DataFrame to a file, auto‑detecting format from extension.

    Performs basic sanity checks (non‑empty, no excessively large columns).

    Args:
        df: DataFrame to save.
        path: Output file path (suffix must be .csv or .parquet).
        index: Whether to include the index.

    Raises:
        ValueError: If the output format is unsupported.
        RuntimeError: If saving fails unexpectedly.
    """
    _validate_output_path(path)
    suffix = path.suffix.lower()

    if suffix not in (".csv", ".parquet"):
        raise ValueError(f"Unsupported output format: {suffix}. Use .csv or .parquet")

    # Warn if saving a very large DataFrame
    if len(df) > MAX_RECORDS_WARN:
        logger.warning("Saving %d rows – may be large.", len(df))

    try:
        if suffix == ".csv":
            df.to_csv(path, index=index)
        elif suffix == ".parquet":
            df.to_parquet(path, index=index)
        logger.debug("Saved %d rows to %s", len(df), path)
    except ValueError as e:
        logger.error("Output format error: %s", e)
        raise
    except Exception as e:
        logger.exception("Failed to save data to %s", path)
        raise RuntimeError(f"Data saving failed: {e}") from e


def _validate_episode_continuity(df: pd.DataFrame, episode_col: str = "episode_id") -> None:
    """Check that episodes have monotonic timestamps and no gaps.

    Args:
        df: DataFrame with at least 'episode_id' and 'timestamp' columns.
        episode_col: Name of the episode identifier column.

    Raises:
        DataValidationError: If continuity violations are found.
    """
    if "timestamp" not in df.columns:
        return  # skip if no timestamp

    # For each episode, check that timestamps are strictly increasing
    for ep_id, group in df.groupby(episode_col):
        ts = group["timestamp"].values
        if len(ts) > 1:
            diffs = np.diff(ts)
            if np.any(diffs <= 0):
                raise DataValidationError(
                    f"Non‑monotonic timestamps in episode {ep_id}"
                )


# ---------------------------------------------------------------------------
# Stage 1: Ingestion
# ---------------------------------------------------------------------------

def stage_ingestion(
    input_path: Path,
    output_path: Path,
    **kwargs: Any
) -> None:
    """Load raw agent traces and produce a canonical CSV.

    Expects a list of records (each record = one step episode) with keys:
        - timestamp (ISO or numeric)
        - action (str or dict)
        - reward (float)
        - competitor_context (bool or int)
        - state_hidden (optional dict)

    Produces a flat CSV with required fields and explicit types.

    Args:
        input_path: Input file path (JSON, CSV, or Parquet).
        output_path: Output CSV path.
        **kwargs: Additional stage parameters (ignored, but reserved for future).

    Raises:
        FileNotFoundError: If input_path does not exist.
        DataValidationError: If required columns are missing or data is malformed.
        StageExecutionError: For unexpected failures during ingestion.
    """
    logger.info("Stage 1: Ingestion — reading from %s", input_path)

    required_columns = ["timestamp", "action", "reward", "competitor_context"]
    optional_columns = ["state_hidden"]

    try:
        df = _detect_and_load(input_path, required_columns=required_columns)

        # Ensure optional columns with defaults
        for col in optional_columns:
            if col not in df.columns:
                df[col] = None
                logger.debug("Added missing optional column '%s' with default None", col)

        # Type enforcement
        df["reward"] = pd.to_numeric(df["reward"], errors="coerce")
        if df["reward"].isna().any():
            logger.warning("Some 'reward' values could not be parsed; those rows will be dropped.")
            df = df.dropna(subset=["reward"])

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if df["timestamp"].isna().any():
            logger.warning("Some 'timestamp' values could not be parsed; those rows will be dropped.")
            df = df.dropna(subset=["timestamp"])

        # Validate at least one row remains
        if df.empty:
            raise DataValidationError("After cleaning, no valid records remain.")

        # Save canonical output
        _save_dataframe(df, output_path)
        logger.info("Ingestion complete: %d records saved to %s", len(df), output_path)

    except (FileNotFoundError, DataValidationError):
        raise
    except Exception as e:
        logger.exception("Stage 1 (ingestion) failed unexpectedly")
        raise StageExecutionError(f"Ingestion stage failed: {e}") from e


# ---------------------------------------------------------------------------
# Stage 2: Parsing
# ---------------------------------------------------------------------------

def stage_parsing(
    input_path: Path,
    output_path: Path,
    **kwargs: Any
) -> None:
    """Parse action strings/dicts into structured numeric components.

    Expected input: canonical CSV from ingestion stage.
    Produces a DataFrame with parsed action embeddings (mean, std, etc.).

    Args:
        input_path: Input CSV (or Parquet) from ingestion.
        output_path: Output CSV/Parquet with parsed fields.
        **kwargs: Additional parameters (e.g., parsing strategy).

    Raises:
        FileNotFoundError: If input_path does not exist.
        DataValidationError: If action column is missing or unparseable.
        StageExecutionError: For unexpected failures.
    """
    logger.info("Stage 2: Parsing — reading from %s", input_path)

    try:
        df = _detect_and_load(input_path, required_columns=["action"])

        # Example parsing logic: extract numeric components from action
        # This is a placeholder; real implementation depends on action format.
        def _parse_action(action_val: Any) -> Dict[str, float]:
            """Parse a single action into numeric components."""
            if isinstance(action_val, dict):
                # Expect dict like {'mean': 0.5, 'std': 0.1}
                return {
                    "action_mean": action_val.get("mean", 0.0),
                    "action_std": action_val.get("std", 1.0),
                    "action_logprob": action_val.get("logprob", 0.0),
                }
            elif isinstance(action_val, (int, float)):
                return {
                    "action_mean": float(action_val),
                    "action_std": 0.0,
                    "action_logprob": 0.0,
                }
            elif isinstance(action_val, str):
                # Attempt to parse as JSON
                try:
                    parsed = json.loads(action_val)
                    return _parse_action(parsed)
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Cannot parse action as JSON: %s", action_val)
                    return {"action_mean": 0.0, "action_std": 1.0, "action_logprob": 0.0}
            else:
                logger.warning("Unknown action type: %s", type(action_val))
                return {"action_mean": 0.0, "action_std": 1.0, "action_logprob": 0.0}

        # Vectorized parsing using .apply
        parsed = df["action"].apply(_parse_action)
        parsed_df = pd.json_normalize(parsed)
        df = pd.concat([df, parsed_df], axis=1).drop(columns=["action"])

        # Add derived columns if needed
        df["action_entropy"] = 0.5 * np.log(2 * np.pi * np.e * (df["action_std"] ** 2 + 1e-8))

        # Check for required numeric columns
        required_numeric = ["action_mean", "action_std", "action_logprob"]
        missing_numeric = [col for col in required_numeric if col not in df.columns]
        if missing_numeric:
            raise DataValidationError(f"Parsing did not produce required numeric columns: {missing_numeric}")

        _save_dataframe(df, output_path)
        logger.info("Parsing complete: %d records with %d parsed features",
                    len(df), len(required_numeric) + 1)  # +1 for entropy

    except (FileNotFoundError, DataValidationError):
        raise
    except Exception as e:
        logger.exception("Stage 2 (parsing) failed unexpectedly")
        raise StageExecutionError(f"Parsing stage failed: {e}") from e


# ---------------------------------------------------------------------------
# Stage 3: Feature Extraction
# ---------------------------------------------------------------------------

def stage_feature_extraction(
    input_path: Path,
    output_path: Path,
    roll_window: int = DEFAULT_ROLL_WINDOW,
    **kwargs: Any
) -> None:
    """Compute rolling statistics and derived features for discipline evaluation.

    Adds rolling means and standard deviations for `reward` and `action_*` columns
    over the specified window. Also adds a binary 'discipline_violation' flag.

    Args:
        input_path: Input CSV/Parquet from parsing stage.
        output_path: Output CSV/Parquet with extracted features.
        roll_window: Rolling window size (number of steps).
        **kwargs: Additional parameters (ignored).

    Raises:
        FileNotFoundError: If input_path does not exist.
        DataValidationError: If required numeric columns are missing.
        StageExecutionError: For unexpected failures.
    """
    logger.info("Stage 3: Feature extraction — roll_window=%d", roll_window)

    requires_columns = ["reward", "action_mean", "action_std"]
    try:
        df = _detect_and_load(input_path, required_columns=requires_columns)
    except Exception as e:
        raise DataValidationError(f"Missing required columns for feature extraction: {e}") from e

    try:
        # Rolling statistics
        df["reward_roll_mean"] = df["reward"].rolling(window=roll_window, min_periods=1).mean()
        df["reward_roll_std"] = df["reward"].rolling(window=roll_window, min_periods=1).std().fillna(0.0)
        df["action_mean_roll"] = df["action_mean"].rolling(window=roll_window, min_periods=1).mean()
        df["action_std_roll"] = df["action_std"].rolling(window=roll_window, min_periods=1).std().fillna(0.0)

        # Discipline violation flag: action deviates more than threshold
        # Example: violation if |action_mean - roll_mean| > 2 * roll_std + epsilon
        threshold = 2.0 * df["action_std_roll"] + 1e-6
        df["discipline_violation"] = (np.abs(df["action_mean"] - df["action_mean_roll"]) > threshold).astype(int)

        # Rolling violation rate
        df["violation_rate"] = df["discipline_violation"].rolling(window=roll_window, min_periods=1).mean()

        # Normalized reward: (reward - rolling mean) / (rolling std + epsilon)
        epsilon = 1e-8
        df["reward_norm"] = (df["reward"] - df["reward_roll_mean"]) / (df["reward_roll_std"] + epsilon)

        # Competitor context flag if available
        if "competitor_context" in df.columns:
            df["competitor_active"] = df["competitor_context"].astype(int)

        _save_dataframe(df, output_path)
        logger.info("Feature extraction complete: %d records, %d features",
                    len(df), len(df.columns))

    except Exception as e:
        logger.exception("Stage 3 (feature extraction) failed")
        raise StageExecutionError(f"Feature extraction failed: {e}") from e


# ---------------------------------------------------------------------------
# Stage 4: Discipline Computation
# ---------------------------------------------------------------------------

def stage_discipline_computation(
    input_path: Path,
    output_path: Path,
    sigma: float = DEFAULT_DISCIPLINE_SIGMA,
    **kwargs: Any
) -> None:
    """Compute the 'discipline stability' metric per time step.

    Discipline is defined as: 1 - tanh( |deviation| / sigma ), where
    deviation = action_mean - action_mean_roll.
    Also computes economic outcome (cumulative reward) for comparison.

    Args:
        input_path: Input CSV/Parquet from feature extraction.
        output_path: Output CSV/Parquet with discipline scores.
        sigma: Scale parameter for tanh normalization.
        **kwargs: Additional parameters (ignored).

    Raises:
        FileNotFoundError: If input_path does not exist.
        DataValidationError: If required columns missing.
        StageExecutionError: For unexpected failures.
    """
    logger.info("Stage 4: Discipline computation — sigma=%.3f", sigma)

    required_cols = ["action_mean", "action_mean_roll", "reward", "timestamp"]
    try:
        df = _detect_and_load(input_path, required_columns=required_cols)
    except Exception as e:
        raise DataValidationError(f"Cannot load data for discipline computation: {e}") from e

    try:
        # Deviation and discipline
        deviation = df["action_mean"].values - df["action_mean_roll"].values
        df["deviation"] = deviation
        df["discipline"] = 1.0 - np.tanh(np.abs(deviation) / max(sigma, 1e-8))

        # Economic outcome: cumulative sum of reward (global and per episode if episode_id exists)
        if "episode_id" in df.columns:
            df["cumulative_reward"] = df.groupby("episode_id")["reward"].cumsum()
            # Normalize episode progress
            df["episode_progress"] = df.groupby("episode_id").cumcount() + 1
        else:
            df["cumulative_reward"] = df["reward"].cumsum()
            df["episode_progress"] = np.arange(1, len(df) + 1)

        # Rolling discipline average
        df["discipline_roll"] = df["discipline"].rolling(window=DEFAULT_ROLL_WINDOW, min_periods=1).mean()

        _save_dataframe(df, output_path)
        logger.info("Discipline computation complete: discipline range [%.4f, %.4f]",
                    df["discipline"].min(), df["discipline"].max())

    except Exception as e:
        logger.exception("Stage 4 (discipline computation) failed")
        raise StageExecutionError(f"Discipline computation failed: {e}") from e


# ---------------------------------------------------------------------------
# Stage 5: Aggregation
# ---------------------------------------------------------------------------

def stage_aggregation(
    input_path: Path,
    output_path: Path,
    **kwargs: Any
) -> None:
    """Aggregate per‑step discipline and reward statistics across episodes or time.

    Produces a summary DataFrame with episode‑level or global statistics:
        - mean/median discipline
        - mean/std cumulative reward
        - violation count/rate
        - discipline stability (std of discipline within episode)

    Args:
        input_path: Input CSV/Parquet from discipline computation.
        output_path: Output CSV/Parquet with aggregated results.
        **kwargs: Additional parameters (ignored).

    Raises:
        FileNotFoundError: If input_path does not exist.
        DataValidationError: If required columns missing.
        StageExecutionError: For unexpected failures.
    """
    logger.info("Stage 5: Aggregation")

    required_cols = ["discipline", "cumulative_reward"]
    try:
        df = _detect_and_load(input_path, required_columns=required_cols)
    except Exception as e:
        raise DataValidationError(f"Cannot load data for aggregation: {e}") from e

    try:
        if "episode_id" not in df.columns:
            # No episode grouping – treat entire dataset as single episode
            df["episode_id"] = 0
            logger.warning("No 'episode_id' column found; treating all rows as one episode.")

        # Episode‑level aggregation
        agg_funcs = {
            "discipline": ["mean", "std", "min", "max"],
            "cumulative_reward": ["max", "std"],
            "reward": ["mean", "std", "sum"],
        }
        # Only use columns that exist
        existing = {col: funcs for col, funcs in agg_funcs.items() if col in df.columns}
        if not existing:
            raise DataValidationError("No aggregation‑relevant columns found.")

        episode_stats = df.groupby("episode_id").agg(existing)
        episode_stats.columns = ["_".join(col).strip() for col in episode_stats.columns]
        episode_stats = episode_stats.reset_index()

        # Add discipline stability (std of discipline) – already in the aggregation above

        # Global aggregation (over all episodes)
        global_stats = {
            "global_mean_discipline": float(episode_stats["discipline_mean"].mean()),
            "global_mean_cumulative_reward": float(episode_stats["cumulative_reward_max"].mean()),
            "num_episodes": len(episode_stats),
            "mean_episode_length": float(len(df) / len(episode_stats)) if len(episode_stats) > 0 else 0.0,
        }

        # Combine into a single DataFrame:
        # Episode stats with global stats appended as attributes
        result_df = episode_stats
        for key, value in global_stats.items():
            result_df.attrs[key] = value

        # Add a row with global statistics? Better to save separate file.
        # For simplicity, we save episode stats and also write global stats as a JSON sidecar.
        base = output_path.with_suffix("")
        global_json_path = base.with_name(base.stem + "_global.json")
        _save_dataframe(result_df, output_path)
        with open(global_json_path, "w") as f:
            json.dump(global_stats, f, indent=2)

        logger.info("Aggregation complete: %d episodes, global mean discipline=%.4f",
                    len(episode_stats), global_stats["global_mean_discipline"])

    except Exception as e:
        logger.exception("Stage 5 (aggregation) failed")
        raise StageExecutionError(f"Aggregation failed: {e}") from e


# ---------------------------------------------------------------------------
# Stage 6: Validation
# ---------------------------------------------------------------------------

def stage_validation(
    input_path: Path,
    output_path: Path,
    reward_threshold: float = DEFAULT_REWARD_THRESHOLD,
    **kwargs: Any
) -> None:
    """Validate the aggregated results for economic safety and discipline adherence.

    Checks:
    - Global mean discipline >= threshold (default 0.5)
    - Variance of discipline low (discipline_stability <= sigma)
    - Correlation between discipline and reward is positive
    - No critical violations (violation rate > 0.2)

    Outputs a validation report (JSON) and a summary plot (optional).

    Args:
        input_path: Input CSV/Parquet from aggregation (episode stats) or the per‑step data.
        output_path: Output JSON report path.
        reward_threshold: Minimum acceptable mean reward for "economically safe" classification.
        **kwargs: Additional parameters (e.g., discipline_threshold).

    Raises:
        FileNotFoundError: If input_path does not exist.
        DataValidationError: If required columns missing.
        StageExecutionError: For unexpected failures.
    """
    logger.info("Stage 6: Validation — reward_threshold=%.3f", reward_threshold)

    try:
        df = _detect_and_load(input_path, required_columns=["discipline_mean", "cumulative_reward_max"])
    except Exception as e:
        raise DataValidationError(f"Cannot load aggregation results: {e}") from e

    try:
        discipline_threshold = kwargs.get("discipline_threshold", 0.5)
        discipline_stability_threshold = kwargs.get("discipline_stability_threshold", 0.3)

        report: Dict[str, Any] = {
            "status": "PASS",
            "checks": {},
            "summary": {},
        }

        # Check 1: mean discipline
        mean_disc = df["discipline_mean"].mean()
        disc_check = mean_disc >= discipline_threshold
        report["checks"]["mean_discipline"] = {
            "value": float(mean_disc),
            "threshold": discipline_threshold,
            "pass": disc_check,
        }
        if not disc_check:
            report["status"] = "FAIL"
            logger.warning("Validation FAILED: mean discipline %.4f < %.2f", mean_disc, discipline_threshold)

        # Check 2: discipline stability (std of discipline across episodes)
        disc_std = df["discipline_std"].mean() if "discipline_std" in df.columns else 0.0
        stab_check = disc_std <= discipline_stability_threshold
        report["checks"]["discipline_stability"] = {
            "value": float(disc_std),
            "threshold": discipline_stability_threshold,
            "pass": stab_check,
        }
        if not stab_check:
            report["status"] = "FAIL"
            logger.warning("Validation FAILED: discipline stability %.4f > %.2f", disc_std, discipline_stability_threshold)

        # Check 3: correlation between discipline and cumulative reward
        if "discipline_mean" in df.columns and "cumulative_reward_max" in df.columns:
            corr = df["discipline_mean"].corr(df["cumulative_reward_max"])
            corr_check = not np.isnan(corr) and corr > 0.0
            report["checks"]["discipline_reward_correlation"] = {
                "value": float(corr) if not np.isnan(corr) else None,
                "pass": corr_check,
            }
            if not corr_check:
                if np.isnan(corr):
                    logger.warning("Correlation could not be computed (few episodes).")
                else:
                    report["status"] = "FAIL"
                    logger.warning("Validation FAILED: discipline‑reward correlation %.4f is not positive", corr)

        # Check 4: violation rate (if available)
        if "discipline_violation" in df.columns:
            viol_rate = df["discipline_violation"].mean()
            viol_check = viol_rate <= 0.2
            report["checks"]["violation_rate"] = {
                "value": float(viol_rate),
                "threshold": 0.2,
                "pass": viol_check,
            }
            if not viol_check:
                report["status"] = "FAIL"
                logger.warning("Validation FAILED: violation rate %.4f > 0.2", viol_rate)

        # Economically safe flag
        mean_reward = df["cumulative_reward_max"].mean() if "cumulative_reward_max" in df.columns else 0.0
        economic_safe = mean_reward >= reward_threshold
        report["summary"]["economically_safe"] = economic_safe
        report["summary"]["mean_cumulative_reward"] = float(mean_reward)
        report["summary"]["threshold"] = reward_threshold
        if not economic_safe:
            report["status"] = "FAIL"
            logger.warning("Validation FAILED: mean cumulative reward %.4f < %.2f", mean_reward, reward_threshold)

        # Final verdict
        report["overall_pass"] = (report["status"] == "PASS")

        _validate_output_path(output_path)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info("Validation complete: overall status = %s", report["status"])
        # If validation failed, raise an exception (optional – configurable)
        if kwargs.get("fail_on_validation_error", False) and not report["overall_pass"]:
            raise DataValidationError("Validation checks failed. See report for details.")

    except (FileNotFoundError, DataValidationError):
        raise
    except Exception as e:
        logger.exception("Stage 6 (validation) failed")
        raise StageExecutionError(f"Validation stage failed: {e}") from e


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

# Stage registry: maps stage name to (function, description)
STAGE_REGISTRY: Dict[str, Tuple[Any, str]] = {
    "ingestion": (stage_ingestion, "Load and canonicalize raw traces"),
    "parsing": (stage_parsing, "Parse actions into numeric components"),
    "feature_extraction": (stage_feature_extraction, "Compute rolling features and violation flags"),
    "discipline_computation": (stage_discipline_computation, "Compute discipline stability metric"),
    "aggregation": (stage_aggregation, "Aggregate per episode and globally"),
    "validation": (stage_validation, "Validate economic safety and discipline adherence"),
}

# Default order of execution (can be overridden in config)
DEFAULT_STAGE_ORDER = [
    "ingestion",
    "parsing",
    "feature_extraction",
    "discipline_computation",
    "aggregation",
    "validation",
]


def run_pipeline(config: Dict[str, Any]) -> None:
    """Execute the pipeline stages in order as specified in the configuration.

    The configuration dictionary must have:
        - 'stages': dict with stage names as keys, each containing optional parameters (e.g., roll_window)
        - 'paths': dict mapping stage names to dicts with 'input' and 'output' paths.
                  The input for the first stage may be omitted if None; it is expected to be provided
                  as a command line argument or default.

    Args:
        config: A validated configuration dictionary.

    Raises:
        ConfigurationError: If the pipeline order is invalid or a stage function is missing.
        StageExecutionError: If any stage fails.
    """
    stages_config = config.get("stages", {})
    paths_config = config.get("paths", {})

    # Determine execution order
    stage_order = config.get("pipeline_order", DEFAULT_STAGE_ORDER)

    # Validate stage order
    for stage in stage_order:
        if stage not in STAGE_REGISTRY:
            raise ConfigurationError(f"Unknown stage '{stage}' in pipeline_order. Valid stages: {list(STAGE_REGISTRY.keys())}")

    logger.info("Pipeline order: %s", ", ".join(stage_order))

    # Execute each stage sequentially
    previous_output = None
    for stage_name in stage_order:
        func, desc = STAGE_REGISTRY[stage_name]
        stage_params = stages_config.get(stage_name, {})
        stage_paths = paths_config.get(stage_name, {})

        input_path = None
        output_path = None

        # Input path: either from config or previous stage's output
        if "input" in stage_paths and stage_paths["input"] is not None:
            input_path = Path(stage_paths["input"])
        elif previous_output is not None:
            input_path = previous_output
        elif stage_name == stage_order[0]:
            # First stage without input – assume it's from command line argument
            raise ConfigurationError(f"First stage '{stage_name}' requires an input path in config or as argument.")

        # Output path: must exist
        if "output" in stage_paths:
            output_path = Path(stage_paths["output"])
            _validate_output_path(output_path)
        else:
            raise ConfigurationError(f"Stage '{stage_name}' missing 'output' path in configuration.")

        logger.info("----- Stage: %s (%s) -----", stage_name, desc)
        logger.debug("Input: %s", input_path)
        logger.debug("Output: %s", output_path)
        logger.debug("Parameters: %s", stage_params)

        try:
            func(
                input_path=input_path,
                output_path=output_path,
                **stage_params
            )
            previous_output = output_path
        except Exception as e:
            raise StageExecutionError(f"Pipeline failed at stage '{stage_name}': {e}") from e

    logger.info("Pipeline execution completed successfully.")


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: List of arguments (default: sys.argv[1:]).

    Returns:
        Parsed arguments Namespace.
    """
    parser = argparse.ArgumentParser(
        description="Discipline Evaluation Pipeline for Trace‑Based Agent Analysis."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Override input path for the first stage (ingestion)."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)."
    )
    parser.add_argument(
        "--fail-on-violation",
        action="store_true",
        help="If set, the pipeline will exit with code 1 if validation fails."
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the pipeline.

    Args:
        argv: Command line arguments (default: sys.argv[1:]).

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    args = parse_arguments(argv)

    # Set root logger level
    logging.getLogger("discipline_pipeline").setLevel(getattr(logging, args.log_level.upper()))

    try:
        config = _load_config(args.config)

        # Override input for first stage if provided
        if args.input is not None:
            stages_config = config.get("stages", {})
            first_stage = config.get("pipeline_order", DEFAULT_STAGE_ORDER)[0]
            config.setdefault("paths", {})
            config["paths"].setdefault(first_stage, {})
            config["paths"][first_stage]["input"] = str(args.input)

        # Optionally pass fail_on_validation_error
        if args.fail_on_violation:
            # Inject into validation stage params
            config.setdefault("stages", {})
            config["stages"].setdefault("validation", {})
            config["stages"]["validation"]["fail_on_validation_error"] = True

        run_pipeline(config)
        return 0

    except (ConfigurationError, DataValidationError, StageExecutionError, FileNotFoundError) as e:
        logger.critical("Pipeline execution failed: %s", e)
        return 1
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        return 130
    except Exception as e:
        logger.exception("Unexpected fatal error")
        return 1


if __name__ == "__main__":
    sys.exit(main())