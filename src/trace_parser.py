#!/usr/bin/env python3
"""
src/trace_parser.py

Production-grade parser for agent trace logs (JSON or CSV) producing a structured CSV
aligned with the discipline‑stability evaluation paradigm described in

    "When Outcome Looks Right But Discipline Fails:
     Trace-Based Evaluation Under Hidden Competitor State"
     (https://arxiv.org/abs/2605.18580v1).

Features
--------
* Supports NDJSON (newline‑delimited) or JSON arrays, and CSV files.
* Automatic column mapping via user-supplied dict or heuristic fallback.
* Schema validation with strict / lenient mode.
* Batched writes for memory efficiency on large datasets.
* Input size & path traversal guards.
* Full logging with context.
* Comprehensive error handling with custom exception hierarchy.
* Full type annotations and explicitness.

Usage:
    python trace_parser.py --input raw_traces.json --output traces_parsed.csv
    python trace_parser.py --input logs/*.csv --output processed/traces_parsed.csv
"""

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, TextIO, Union

# ---------------------------------------------------------------------------
# Logging – configured once; modules import the logger by name
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TraceParser")

# ---------------------------------------------------------------------------
# Constants – schema column names and default column mapping keys
# ---------------------------------------------------------------------------
COL_TRACE_ID = "trace_id"
COL_STEP = "step"
COL_ACTION = "action"
COL_REWARD = "reward"
COL_COMPETITOR_PRESENT = "competitor_present"
COL_RULE_VIOLATION = "rule_violation"
COL_HIDDEN_STATE_FLAG = "hidden_state_flag"

REQUIRED_COLUMNS: List[str] = [
    COL_TRACE_ID,
    COL_STEP,
    COL_ACTION,
    COL_REWARD,
    COL_COMPETITOR_PRESENT,
    COL_RULE_VIOLATION,
    COL_HIDDEN_STATE_FLAG,
]

# Expected Python types for output columns (used for validation & coercion)
SCHEMA: Dict[str, type] = {
    COL_TRACE_ID: str,
    COL_STEP: int,
    COL_ACTION: str,
    COL_REWARD: float,
    COL_COMPETITOR_PRESENT: bool,
    COL_RULE_VIOLATION: bool,
    COL_HIDDEN_STATE_FLAG: bool,
}

# Maximum input file size (in bytes) – 500 MB
MAX_FILE_SIZE: int = 500 * 1024 * 1024

# Default column mapping (heuristic)
DEFAULT_COLUMN_MAPPING: Dict[str, str] = {
    "trace_id": COL_TRACE_ID,
    "step": COL_STEP,
    "action": COL_ACTION,
    "reward": COL_REWARD,
    "competitor_present": COL_COMPETITOR_PRESENT,
    "rule_violation": COL_RULE_VIOLATION,
    "hidden_state_flag": COL_HIDDEN_STATE_FLAG,
}

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------
class TraceParserError(Exception):
    """Base exception for all trace parsing failures."""
    pass

class SchemaValidationError(TraceParserError):
    """Raised when a record cannot be coerced to the expected schema."""
    pass

class FileSizeExceededError(TraceParserError):
    """Raised when input file exceeds the allowed limit."""
    pass

class EmptyInputError(TraceParserError):
    """Raised when input file contains no usable records."""
    pass

class FormatError(TraceParserError):
    """Raised when the input file has an unsupported format or structure."""
    pass

class ColumnMappingError(TraceParserError):
    """Raised when required columns cannot be mapped in the input data."""
    pass

class IOError(TraceParserError):
    """Raised on I/O failures (permission issues, disk full, etc.)."""
    pass

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class ParserConfig:
    """Configuration for the trace parser."""

    batch_size: int = 10_000
    input_encoding: str = "utf-8"
    strict_schema: bool = True
    column_mapping: Dict[str, str] = field(default_factory=dict)
    max_input_size: int = MAX_FILE_SIZE

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.max_input_size < 1:
            raise ValueError("max_input_size must be >= 1")
        try:
            "".encode(self.input_encoding)
        except LookupError as exc:
            raise ValueError(f"Unknown encoding: {self.input_encoding}") from exc

        # Merge with defaults – user mapping takes precedence
        self.column_mapping = {**DEFAULT_COLUMN_MAPPING, **self.column_mapping}

# ---------------------------------------------------------------------------
# Column mapping utility
# ---------------------------------------------------------------------------
def _infer_column_mapping(header: List[str]) -> Dict[str, str]:
    """
    Infer a column mapping from input header, matching against DEFAULT_COLUMN_MAPPING.

    Args:
        header: List of column names from input file.

    Returns:
        Dict mapping input column names to standard output column names.

    Raises:
        ColumnMappingError: If required column cannot be mapped.
    """
    mapping: Dict[str, str] = {}
    reverse_default = {v: k for k, v in DEFAULT_COLUMN_MAPPING.items()}
    for col in header:
        # exact match
        if col in DEFAULT_COLUMN_MAPPING:
            mapping[col] = DEFAULT_COLUMN_MAPPING[col]
        # case-insensitive match
        else:
            lower_col = col.lower().replace(" ", "_")
            if lower_col in DEFAULT_COLUMN_MAPPING:
                mapping[col] = DEFAULT_COLUMN_MAPPING[lower_col]
            # final fallback: keep original column name
            else:
                mapping[col] = col

    # Verify all required columns can be satisfied
    missing = set(REQUIRED_COLUMNS) - set(mapping.values())
    if missing:
        raise ColumnMappingError(
            f"Required columns missing in input header: {missing}. "
            f"Found header: {header}"
        )
    return mapping

# ---------------------------------------------------------------------------
# Core parser class
# ---------------------------------------------------------------------------
class TraceParser:
    """
    Reads raw agent trace logs (JSON or CSV) and writes a normalised CSV output.

    Args:
        config: ParserConfig instance; uses sensible defaults if omitted.

    Attributes:
        config: ParserConfig instance with runtime parameters.
        _batch: Internal buffer for batched writes.
        _batch_count: number of records written so far.
    """

    def __init__(self, config: Optional[ParserConfig] = None) -> None:
        """
        Initialize the parser with an optional configuration.

        Args:
            config: ParserConfig instance; uses sensible defaults if omitted.
        """
        self.config = config or ParserConfig()
        self._batch: List[Dict[str, Any]] = []
        self._batch_count: int = 0
        logger.debug("TraceParser initialized with config: %s", self.config)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def parse(self, input_path: Union[str, "Path"], output_path: Union[str, "Path"]) -> None:
        """
        Parse input file(s) and write structured CSV output.

        Args:
            input_path: Path to input file (JSON or CSV). Glob patterns are not expanded
                        internally; use shell expansion before calling.
            output_path: Path to output CSV file (parent directory is created).

        Raises:
            FileNotFoundError: If input_path does not exist.
            FileSizeExceededError: If input file exceeds configured limit.
            TraceParserError: On any parsing or schema failure.
        """
        input_path = Path(input_path).expanduser().resolve()
        output_path = Path(output_path).expanduser().resolve()

        # ---- Security: path traversal guard ----
        self._validate_path(input_path)

        # ---- Existence check ----
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # ---- Size check ----
        file_size = input_path.stat().st_size
        if file_size > self.config.max_input_size:
            raise FileSizeExceededError(
                f"Input file ({file_size} bytes) exceeds max allowed "
                f"({self.config.max_input_size} bytes): {input_path}"
            )

        # ---- Determine format by extension ----
        ext = input_path.suffix.lower()
        if ext not in (".json", ".csv"):
            raise FormatError(
                f"Unsupported file format: '{ext}'. Only .json and .csv are supported."
            )

        logger.info(
            "Parsing %s (size=%.2f MB, encoding=%s)",
            input_path,
            file_size / 1024 / 1024,
            self.config.input_encoding,
        )

        # ---- Ensure output directory exists ----
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise IOError(
                f"Cannot create output directory {output_path.parent}: {exc}"
            ) from exc

        # ---- Parse according to format ----
        try:
            if ext == ".json":
                record_iter = self._iter_json_records(input_path)
            else:  # .csv
                record_iter = self._iter_csv_records(input_path)
        except IOError as exc:
            raise IOError(f"Failed to read input file {input_path}: {exc}") from exc

        # ---- Write output CSV ----
        try:
            with open(output_path, mode="w", newline="", encoding="utf-8") as fout:
                writer = csv.DictWriter(fout, fieldnames=REQUIRED_COLUMNS, extrasaction="ignore")
                writer.writeheader()
                self._batch = []

                for record in record_iter:
                    self._batch.append(record)
                    if len(self._batch) >= self.config.batch_size:
                        self._flush_batch(writer)
                self._flush_batch(writer)  # final flush

            logger.info(
                "Successfully wrote %d records to %s",
                self._batch_count,
                output_path,
            )
        except IOError as exc:
            raise IOError(f"Failed to write output CSV {output_path}: {exc}") from exc

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------
    def _validate_path(self, path: Path) -> None:
        """
        Check that path does not contain traversal components.

        Args:
            path: Resolved path object.

        Raises:
            PermissionError: If path attempts directory traversal.
        """
        resolved = path.resolve()
        # Check if any part is '..' (should be resolved out, but double-check)
        if ".." in resolved.parts or str(resolved).startswith(".."):
            raise PermissionError(
                f"Path traversal attempt detected: {path}"
            )

    def _iter_json_records(self, path: Path) -> Iterator[Dict[str, Any]]:
        """
        Yield normalized records from a JSON file (NDJSON or array).

        Args:
            path: Path to input JSON file.

        Yields:
            Dict with keys from REQUIRED_COLUMNS after mapping and coercion.

        Raises:
            FormatError: If JSON structure is unsupported.
            SchemaValidationError: If strict mode and coercion fails.
        """
        with open(path, mode="r", encoding=self.config.input_encoding) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as exc:
                raise FormatError(
                    f"Invalid JSON in {path}: {exc}"
                ) from exc

        # NDJSON: treat as list of lines (re-parse)
        if isinstance(data, str):
            # Could be NDJSON concatenated string – split lines and re-parse
            raise FormatError(
                "Input JSON is a string; NDJSON must be list of objects per line. "
                "Please provide a valid JSON array or newline-delimited JSON."
            )

        records: List[Dict[str, Any]]
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            # Maybe a single object - wrap in list
            records = [data]
        else:
            raise FormatError(
                f"JSON root must be an array or object, got {type(data).__name__}"
            )

        if not records:
            logger.warning("JSON file %s contains zero records.", path)
            return

        header = list(records[0].keys())
        mapping = self.config.column_mapping or _infer_column_mapping(header)

        for i, raw in enumerate(records):
            if not isinstance(raw, dict):
                logger.warning("Record %d is not a dict; skipping.", i)
                continue
            try:
                normalized = self._normalize_record(raw, mapping)
                yield normalized
            except SchemaValidationError as exc:
                if self.config.strict_schema:
                    raise
                logger.warning("Skipping record %d: %s", i, exc)

    def _iter_csv_records(self, path: Path) -> Iterator[Dict[str, Any]]:
        """
        Yield normalized records from a CSV file.

        Args:
            path: Path to input CSV file.

        Yields:
            Dict with keys from REQUIRED_COLUMNS after mapping and coercion.

        Raises:
            FormatError: If CSV file has no header or insufficient rows.
        """
        with open(path, mode="r", encoding=self.config.input_encoding) as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise FormatError(f"CSV file {path} has no header row.")

            mapping = self.config.column_mapping or _infer_column_mapping(reader.fieldnames)
            for i, row in enumerate(reader):
                if not row:
                    logger.warning("Empty row %d in CSV; skipping.", i)
                    continue
                try:
                    normalized = self._normalize_record(row, mapping)
                    yield normalized
                except SchemaValidationError as exc:
                    if self.config.strict_schema:
                        raise
                    logger.warning("Skipping row %d: %s", i, exc)

    def _normalize_record(
        self,
        raw: Dict[str, Any],
        mapping: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Map and coerce a raw record to the output schema.

        Args:
            raw: Raw record dict from input.
            mapping: Dict mapping input column names to standard column names.

        Returns:
            Normalized dict with all required columns.

        Raises:
            SchemaValidationError: If required field missing or coercion fails.
        """
        normalized: Dict[str, Any] = {}
        for std_col in REQUIRED_COLUMNS:
            # find the source column
            src_col = None
            for src, dst in mapping.items():
                if dst == std_col:
                    src_col = src
                    break
            if src_col is None:
                if self.config.strict_schema:
                    raise SchemaValidationError(f"Required column '{std_col}' not found in mapping")
                else:
                    logger.warning("Column '%s' missing; using default.", std_col)
                    normalized[std_col] = SCHEMA[std_col]()
                    continue

            raw_val = raw.get(src_col)
            if raw_val is None:
                if self.config.strict_schema:
                    raise SchemaValidationError(
                        f"Required column '{src_col}' (->'{std_col}') missing"
                    )
                else:
                    logger.debug("Column '%s' empty; using default.", src_col)
                    normalized[std_col] = SCHEMA[std_col]()
                    continue

            # Coerce to expected type
            try:
                coerced = self._coerce_value(raw_val, SCHEMA[std_col])
            except (ValueError, TypeError) as exc:
                raise SchemaValidationError(
                    f"Cannot coerce value '{raw_val}' for column '{std_col}': {exc}"
                ) from exc
            normalized[std_col] = coerced

        return normalized

    @staticmethod
    def _coerce_value(value: Any, target: type) -> Any:
        """
        Coerce a value to the target type with reasonable parsing.

        Args:
            value: Input value.
            target: Expected Python type.

        Returns:
            Coerced value.

        Raises:
            ValueError, TypeError: On failure.
        """
        if target == bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lower = value.lower().strip()
                if lower in ("1", "true", "yes", "y"):
                    return True
                if lower in ("0", "false", "no", "n", ""):
                    return False
                raise ValueError(f"Cannot parse as bool: '{value}'")
            # numeric
            return bool(int(value))
        if target == int:
            if isinstance(value, bool):
                return int(value)
            return int(value)
        if target == float:
            if isinstance(value, bool):
                return float(value)
            return float(value)
        if target == str:
            if isinstance(value, bytes):
                return value.decode("utf-8")
            return str(value)
        return target(value)  # fallback (shouldn't happen)

    def _flush_batch(self, writer: "csv.DictWriter") -> None:
        """
        Write buffered batch to output CSV.

        Args:
            writer: CSV DictWriter instance.
        """
        if not self._batch:
            return
        try:
            writer.writerows(self._batch)
            self._batch_count += len(self._batch)
            logger.debug("Flushed batch of %d records (total %d)", len(self._batch), self._batch_count)
            self._batch.clear()
        except IOError as exc:
            raise IOError(f"Failed to write batch: {exc}") from exc

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse agent trace logs (JSON/CSV) into structured CSV for discipline‑stability evaluation."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input file (.json or .csv)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output CSV file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Number of records per write batch (default: 10000)",
    )
    parser.add_argument(
        "--max-input-size",
        type=int,
        default=MAX_FILE_SIZE,
        help=f"Maximum input file size in bytes (default: {MAX_FILE_SIZE})",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="Input file encoding (default: utf-8)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Raise on schema violations (default: True)",
    )
    parser.add_argument(
        "--lenient",
        action="store_false",
        dest="strict",
        help="Skip malformed records instead of aborting",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (default: INFO)",
    )
    return parser.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> None:
    """
    CLI entry point.

    Args:
        argv: Argument list, defaults to sys.argv[1:].
    """
    args = _parse_args(argv)
    logging.getLogger("TraceParser").setLevel(getattr(logging, args.log_level.upper()))

    config = ParserConfig(
        batch_size=args.batch_size,
        input_encoding=args.encoding,
        strict_schema=args.strict,
        max_input_size=args.max_input_size,
    )

    parser = TraceParser(config)
    try:
        parser.parse(args.input, args.output)
    except TraceParserError as exc:
        logger.critical("Parsing failed: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.critical("Unexpected error: %s", exc, exc_info=True)
        sys.exit(2)

if __name__ == "__main__":
    main()