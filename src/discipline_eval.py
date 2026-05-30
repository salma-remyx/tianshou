# src/discipline_eval.py
"""
Compute trace-based discipline stability metrics for agent evaluation.

Inspired by "When Outcome Looks Right But Discipline Fails: Trace-Based
Evaluation Under Hidden Competitor State" (arXiv:2605.18580v1).

This module provides functions to load per-step feature vectors, compute
per-trace discipline metrics (discipline_score, economic_alignment,
hidden_state_influence), and aggregate results across traces.

Typical usage:
    python -m src.discipline_eval --input processed/features_vectors.csv
    python -m src.discipline_eval --input processed/features_vectors.parquet \\
                                  --output processed/discipline_metrics.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants and type aliases
# ---------------------------------------------------------------------------

EXPECTED_FEATURE_COLUMNS: Dict[str, type] = {
    "trace_id": str,
    "step": int,
    "action_consistency": float,
    "competitor_present": bool,
    "rule_adherence": bool,
    "deviation_magnitude": float,
    "reward": float,
}

REQUIRED_COLUMNS: set = {"trace_id", "step"}

DEFAULT_COLUMNS: Dict[str, Any] = {
    "action_consistency": 1.0,
    "competitor_present": False,
    "rule_adherence": True,
    "deviation_magnitude": 0.0,
    "reward": 0.0,
}

SUPPORTED_EXTENSIONS: set = {".csv", ".parquet"}

# Type alias for a dictionary of per-trace metrics
TraceMetrics = Dict[str, Any]

# Public API
__all__ = [
    "load_features",
    "compute_discipline_metrics",
    "DEFAULT_CONFIG",
]


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class DisciplineMetricsConfig:
    """Configuration parameters for discipline stability computation.

    Attributes
    ----------
    max_deviation_clamp : float
        Maximum value for deviation magnitude before clamping (default 1.0).
    hidden_state_influence_clamp : Tuple[float, float]
        Min and max clamp values for hidden state influence (default (0.0, 5.0)).
    correlation_min_variance : float
        Minimum variance required to compute Pearson correlation (default 1e-6).
    """

    max_deviation_clamp: float = 1.0
    hidden_state_influence_clamp: Tuple[float, float] = (0.0, 5.0)
    correlation_min_variance: float = 1e-6


# Default configuration
DEFAULT_CONFIG = DisciplineMetricsConfig()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_features(path: Union[str, Path]) -> pd.DataFrame:
    """Load per‑step feature vectors from CSV or Parquet file.

    Parameters
    ----------
    path : Union[str, Path]
        Path to features file (``.csv`` or ``.parquet``). Must exist, be a
        regular file, and have a supported extension.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns expected by ``compute_discipline_metrics``.
        Missing optional columns are filled with sensible defaults.

    Raises
    ------
    FileNotFoundError
        If the specified path does not exist.
    PermissionError
        If the file exists but the current process lacks read permissions.
    ValueError
        If the file format is unsupported, required columns are missing, or
        type coercion fails.

    Notes
    -----
    The returned DataFrame is sorted by ``trace_id`` and ``step``.
    """
    # --- Path validation ---
    try:
        file_path = Path(path).resolve(strict=False)
    except Exception as exc:
        raise ValueError(f"Invalid path '{path}': {exc}") from exc

    if not file_path.exists():
        raise FileNotFoundError(f"Feature file not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a regular file: {file_path}")

    # --- Extension check ---
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    # --- Read file ---
    try:
        if suffix == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)
    except Exception as exc:
        raise ValueError(f"Failed to read file '{file_path}': {exc}") from exc

    logger.info("Loaded features from %s: shape %s", file_path, df.shape)

    # --- Required columns ---
    actual_columns = set(df.columns)
    missing_required = REQUIRED_COLUMNS - actual_columns
    if missing_required:
        raise ValueError(
            f"Required columns missing in '{file_path}': {missing_required}. "
            f"Available: {sorted(actual_columns)}"
        )

    # --- Fill missing optional columns ---
    for col, default in DEFAULT_COLUMNS.items():
        if col not in df.columns:
            df[col] = default
            logger.debug("Filled missing column '%s' with default=%s", col, default)

    # --- Type coercion with per‑column errors ---
    for col, dtype in EXPECTED_FEATURE_COLUMNS.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Column '{col}' in '{file_path}' could not be cast "
                    f"to {dtype}: {e}"
                ) from e

    # --- Sort for reproducibility ---
    if {"trace_id", "step"}.issubset(df.columns):
        df = df.sort_values(["trace_id", "step"]).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Core metrics computation (per trace)
# ---------------------------------------------------------------------------


def _compute_trace_metrics(
    trace_group: pd.DataFrame,
    config: DisciplineMetricsConfig = DEFAULT_CONFIG,
) -> TraceMetrics:
    """Compute discipline stability metrics for a single trace.

    Parameters
    ----------
    trace_group : pd.DataFrame
        All steps belonging to one trace. Must contain at least the columns
        ``trace_id``, ``step``, ``action_consistency``, ``rule_adherence``,
        ``deviation_magnitude``, ``competitor_present``, ``reward``.
    config : DisciplineMetricsConfig
        Configuration for clamping and variance thresholds.

    Returns
    -------
    TraceMetrics
        Dictionary with keys:
        - ``trace_id`` : str
        - ``num_steps`` : int
        - ``discipline_score`` : float
        - ``economic_alignment`` : float
        - ``hidden_state_influence`` : float

    Notes
    -----
    - ``discipline_score`` is the product of mean action consistency, mean
      rule adherence, and (1 - clamped mean deviation magnitude), clamped to [0,1].
    - ``economic_alignment`` is the Pearson correlation between rule adherence
      and reward when both have sufficient variance; otherwise it falls back
      to a normalised mean reward difference.
    - ``hidden_state_influence`` is the ratio of mean deviation magnitude
      when a competitor is present versus absent, clamped according to config.
      Returns 0.0 when no comparison is possible.
    """
    n = len(trace_group)
    if n == 0:
        logger.warning("Empty trace group encountered, returning zero metrics.")
        return {
            "trace_id": "unknown",
            "num_steps": 0,
            "discipline_score": 0.0,
            "economic_alignment": 0.0,
            "hidden_state_influence": 0.0,
        }

    trace_id = str(trace_group["trace_id"].iloc[0])

    # --- Clamp deviation magnitude ---
    deviation = np.clip(
        trace_group["deviation_magnitude"].values,
        0.0,
        config.max_deviation_clamp,
    )

    # --- Component means ---
    mean_consistency = float(np.mean(trace_group["action_consistency"].values))
    mean_adherence = float(np.mean(trace_group["rule_adherence"].values.astype(float)))
    mean_deviation = float(np.mean(deviation))

    # discipline_score = consistency * adherence * (1 - mean_deviation)
    discipline_score = mean_consistency * mean_adherence * (1.0 - mean_deviation)
    discipline_score = max(0.0, min(1.0, discipline_score))  # clamp to [0,1]

    # --- economic_alignment ---
    adherence_arr = trace_group["rule_adherence"].values.astype(float)
    reward_arr = trace_group["reward"].values

    var_adherence = np.var(adherence_arr)
    var_reward = np.var(reward_arr)

    if var_adherence > config.correlation_min_variance and var_reward > config.correlation_min_variance:
        corr_matrix = np.corrcoef(adherence_arr, reward_arr)
        economic_alignment = float(corr_matrix[0, 1])
        # handle NaN from constant arrays (should not happen due to variance check, but guard)
        if np.isnan(economic_alignment):
            economic_alignment = 0.0
    else:
        # fallback: normalised mean reward difference when adherence is low vs high
        median_adherence = float(np.median(adherence_arr))
        high_adherence_mask = adherence_arr >= median_adherence
        low_adherence_mask = ~high_adherence_mask

        if high_adherence_mask.sum() > 0 and low_adherence_mask.sum() > 0:
            high_mean_reward = float(np.mean(reward_arr[high_adherence_mask]))
            low_mean_reward = float(np.mean(reward_arr[low_adherence_mask]))
            diff = high_mean_reward - low_mean_reward
            # normalise by max absolute reward range
            reward_range = float(np.max(reward_arr) - np.min(reward_arr))
            if reward_range > 0.0:
                economic_alignment = diff / reward_range
            else:
                economic_alignment = 0.0
        else:
            economic_alignment = 0.0

    # --- hidden_state_influence ---
    with_comp_mask = trace_group["competitor_present"].values.astype(bool)
    without_comp_mask = ~with_comp_mask

    if with_comp_mask.sum() > 0 and without_comp_mask.sum() > 0:
        mean_dev_with = float(np.mean(trace_group["deviation_magnitude"].values[with_comp_mask]))
        mean_dev_without = float(np.mean(trace_group["deviation_magnitude"].values[without_comp_mask]))
        if mean_dev_without > 0.0:
            influence = mean_dev_with / mean_dev_without
        else:
            influence = config.hidden_state_influence_clamp[1]  # max clamp
        influence = float(np.clip(
            influence,
            config.hidden_state_influence_clamp[0],
            config.hidden_state_influence_clamp[1],
        ))
    else:
        influence = 0.0

    return {
        "trace_id": trace_id,
        "num_steps": n,
        "discipline_score": round(discipline_score, 6),
        "economic_alignment": round(economic_alignment, 6),
        "hidden_state_influence": round(influence, 6),
    }


# ---------------------------------------------------------------------------
# Aggregation over all traces
# ---------------------------------------------------------------------------


def compute_discipline_metrics(
    features: pd.DataFrame,
    config: Optional[DisciplineMetricsConfig] = None,
) -> pd.DataFrame:
    """Compute discipline stability metrics across all traces.

    Parameters
    ----------
    features : pd.DataFrame
        Feature vectors DataFrame as returned by ``load_features``.
    config : Optional[DisciplineMetricsConfig]
        Configuration parameters. If ``None``, uses ``DEFAULT_CONFIG``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``trace_id``, ``num_steps``,
        ``discipline_score``, ``economic_alignment``,
        ``hidden_state_influence``. Sorted by ``trace_id``.

    Raises
    ------
    ValueError
        If ``features`` is empty or missing required columns after loading
        (defensive check, though ``load_features`` should have handled this).
    """
    if config is None:
        config = DEFAULT_CONFIG

    if features.empty:
        logger.error("Empty features DataFrame provided.")
        raise ValueError("Input features DataFrame is empty.")

    # Defensive: ensure required columns exist
    missing = REQUIRED_COLUMNS - set(features.columns)
    if missing:
        raise ValueError(f"Input features missing required columns: {missing}")

    # Group by trace_id and compute metrics
    trace_metrics_list: List[TraceMetrics] = []
    for trace_id, group in features.groupby("trace_id", sort=False):
        metrics = _compute_trace_metrics(group, config)
        trace_metrics_list.append(metrics)

    if not trace_metrics_list:
        logger.warning("No traces found in features; returning empty DataFrame.")
        return pd.DataFrame(columns=[
            "trace_id", "num_steps", "discipline_score",
            "economic_alignment", "hidden_state_influence",
        ])

    result_df = pd.DataFrame(trace_metrics_list)
    result_df = result_df.sort_values("trace_id").reset_index(drop=True)
    return result_df


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def _configure_logging(verbose: bool = False) -> None:
    """Set up module logging with a simple console handler."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))

    root_logger = logging.getLogger(__name__)
    root_logger.setLevel(level)
    root_logger.addHandler(handler)
    # Prevent propagation to root logger to avoid duplicate messages
    root_logger.propagate = False


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : Optional[Sequence[str]]
        Argument list (default: None, uses sys.argv[1:]).

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compute trace-based discipline stability metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python -m src.discipline_eval --input features.csv --output metrics.csv --verbose"
        ),
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input features file (.csv or .parquet).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to output metrics file (.csv or .parquet). If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )
    parser.add_argument(
        "--max-deviation-clamp",
        type=float,
        default=DEFAULT_CONFIG.max_deviation_clamp,
        help="Maximum deviation magnitude clamping value (default: %(default)s).",
    )
    parser.add_argument(
        "--hidden-influence-min",
        type=float,
        default=DEFAULT_CONFIG.hidden_state_influence_clamp[0],
        help="Minimum hidden state influence clamp (default: %(default)s).",
    )
    parser.add_argument(
        "--hidden-influence-max",
        type=float,
        default=DEFAULT_CONFIG.hidden_state_influence_clamp[1],
        help="Maximum hidden state influence clamp (default: %(default)s).",
    )
    parser.add_argument(
        "--correlation-min-variance",
        type=float,
        default=DEFAULT_CONFIG.correlation_min_variance,
        help="Minimum variance to compute Pearson correlation (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main entry point for CLI.

    Parameters
    ----------
    argv : Optional[Sequence[str]]
        Argument list.

    Returns
    -------
    int
        Exit code (0 on success, 1 on error).
    """
    args = _parse_args(argv)
    _configure_logging(args.verbose)

    try:
        # Load features
        features_df = load_features(args.input)

        # Build configuration with overrides
        config = DisciplineMetricsConfig(
            max_deviation_clamp=args.max_deviation_clamp,
            hidden_state_influence_clamp=(args.hidden_influence_min, args.hidden_influence_max),
            correlation_min_variance=args.correlation_min_variance,
        )

        # Compute discipline metrics
        metrics_df = compute_discipline_metrics(features_df, config)

        # Output
        if args.output:
            output_path = Path(args.output)
            suffix = output_path.suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                logger.warning(
                    "Output extension '%s' not in supported formats (%s). "
                    "Falling back to CSV.",
                    suffix,
                    sorted(SUPPORTED_EXTENSIONS),
                )
                output_path = output_path.with_suffix(".csv")

            try:
                if output_path.suffix.lower() == ".parquet":
                    metrics_df.to_parquet(output_path, index=False)
                else:
                    metrics_df.to_csv(output_path, index=False)
                logger.info("Metrics saved to %s", output_path.resolve())
            except Exception as exc:
                logger.error("Failed to write output file: %s", exc)
                return 1
        else:
            # Print to stdout as CSV without index
            sys.stdout.write(metrics_df.to_csv(index=False))

        return 0

    except (FileNotFoundError, PermissionError, ValueError) as exc:
        logger.error("Error: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())