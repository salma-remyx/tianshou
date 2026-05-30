"""
src/csv_writer.py

Production-grade CSV writer with schema enforcement, type coercion, and auxiliary metadata.
Designed for the Trace-Based Discipline Evaluation Pipeline.

Usage:
    writer = CsvWriter("output/metrics.csv", schema={"reward": float, "step": int})
    writer.write([{"step": 1, "reward": 0.5}, {"step": 2, "reward": 0.8}])
    writer.write_metadata({"pipeline": "discipline", "run_id": "run_001"})
"""

import csv
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Type, Union, get_origin, get_args

logger = logging.getLogger(__name__)

# Mapping from Python types to CSV-compatible names for metadata
TYPE_MAP: Dict[Type, str] = {
    int: "integer",
    float: "float",
    str: "string",
    bool: "boolean",
    datetime: "datetime",
    type(None): "null",
}


def _resolve_type(t: Type) -> str:
    """Return a human-readable type name for metadata.

    Args:
        t: A Python type.

    Returns:
        A string describing the type.
    """
    if t in TYPE_MAP:
        return TYPE_MAP[t]
    # Handle Optional[X], Union[X, None] etc.
    origin = get_origin(t)
    if origin is Union:
        args = get_args(t)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _resolve_type(non_none[0])
    return str(t)


def _coerce_value(value: Any, target_type: Type) -> Any:
    """Coerce a single value to the target type, with sane defaults for None.

    Args:
        value: The raw value to coerce.
        target_type: The desired type.

    Returns:
        Coerced value.

    Raises:
        ValueError: If coercion is not possible or None is not allowed.
        TypeError: If the target type is unsupported.
    """
    if value is None:
        # Check if target_type is Optional
        origin = get_origin(target_type)
        if origin is Union and type(None) in get_args(target_type):
            return None
        # For plain types or non-optional unions, None is invalid
        raise ValueError(f"Cannot coerce None to non-optional type {target_type!r}")
    try:
        if target_type is int:
            return int(value)
        elif target_type is float:
            return float(value)
        elif target_type is str:
            return str(value)
        elif target_type is bool:
            if isinstance(value, str):
                if value.lower() in ("true", "1", "yes"):
                    return True
                elif value.lower() in ("false", "0", "no"):
                    return False
                else:
                    raise ValueError(f"Cannot convert {value!r} to bool")
            return bool(value)
        elif target_type is datetime:
            if isinstance(value, datetime):
                return value
            if isinstance(value, str):
                # Try common ISO formats
                return datetime.fromisoformat(value)
            raise ValueError(f"Cannot convert {type(value)} to datetime")
        else:
            # For custom types, attempt direct construction
            return target_type(value)
    except (ValueError, TypeError, OverflowError) as e:
        raise ValueError(f"Value {value!r} cannot be coerced to {target_type!r}: {e}") from e


class CsvWriter:
    """Write metrics data to a CSV file with proper formatting, column headers,
    and data type enforcement. Optionally writes an auxiliary metadata file
    (JSON) describing the schema and any custom metadata.

    Attributes:
        filepath: Path to the output CSV file.
        schema: Dict mapping column name to expected Python type.
        metadata: Dict of auxiliary metadata.
        append: Whether to append to an existing file.
        _metadata_filepath: Path to the auxiliary metadata file (derived from CSV path).
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        schema: Optional[Dict[str, Type]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        append: bool = False,
    ) -> None:
        """Initialize the CsvWriter.

        Args:
            filepath: Path to the output CSV file.
            schema: Optional dict mapping column name to expected Python type.
                If given, values are coerced to that type on write.
            metadata: Optional dict of auxiliary metadata to embed in the metadata file.
            append: If True, data is appended to existing file (no header rewrite).

        Raises:
            OSError: If the parent directory cannot be created.
            TypeError: If any schema key is not a string.
        """
        self.filepath = Path(filepath)
        self.schema = schema if schema is not None else {}
        self._metadata: Dict[str, Any] = metadata if metadata is not None else {}
        self.append = append

        # Validate schema keys are strings
        if not all(isinstance(k, str) for k in self.schema.keys()):
            raise TypeError("All schema keys must be strings.")

        # Ensure the parent directory exists
        try:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(f"Cannot create directory for {self.filepath}: {e}") from e

        # Derive metadata filepath (same stem, .json extension)
        self._metadata_filepath = self.filepath.with_suffix(".json")

        logger.debug("CsvWriter initialized: path=%s, schema_keys=%s, append=%s",
                      self.filepath, list(self.schema.keys()), self.append)

    def write(
        self,
        data: Sequence[Dict[str, Any]],
        mode: Optional[str] = None
    ) -> int:
        """Write a sequence of row dicts to the CSV file.

        If no mode is given, defaults to 'a' if ``self.append`` is True,
        otherwise 'w'.  If the file already exists in append mode and has
        content, the header is not rewritten.

        Args:
            data: List of dictionaries, each representing a row.
            mode: File open mode ('w' for overwrite, 'a' for append).
                Overrides the instance's ``append`` setting for this call.

        Returns:
            Number of rows written.

        Raises:
            ValueError: If schema columns are missing in data or type coercion fails,
                or if an invalid mode is provided.
            OSError: If file cannot be opened or written.
        """
        if mode is None:
            mode = "a" if self.append else "w"

        if mode not in ("w", "a"):
            raise ValueError(f"Invalid mode {mode!r}; must be 'w' or 'a'")

        if not data:
            logger.warning("No data provided to write. Skipping.")
            return 0

        # Determine fieldnames: use schema keys if present, otherwise infer from first row
        fieldnames: List[str]
        if self.schema:
            fieldnames = list(self.schema.keys())
        else:
            # Infer from first row, ensure all rows have consistent keys
            first_row_keys = set(data[0].keys())
            for idx, row in enumerate(data[1:], start=1):
                if set(row.keys()) != first_row_keys:
                    raise ValueError(
                        f"Row {idx} has keys {set(row.keys())} but expected {first_row_keys}. "
                        "All rows must have the same keys when schema is not provided."
                    )
            fieldnames = list(data[0].keys())

        # Normalize rows: ensure all keys exist, coerce types according to schema
        normalized: List[Dict[str, Any]] = []
        for row_idx, row in enumerate(data):
            cleaned: Dict[str, Any] = {}
            for col in fieldnames:
                val = row.get(col)  # missing key -> None
                if col in self.schema:
                    try:
                        cleaned[col] = _coerce_value(val, self.schema[col])
                    except (ValueError, TypeError) as e:
                        raise ValueError(
                            f"Row {row_idx}, column {col!r}: {e}"
                        ) from e
                else:
                    # No schema enforcement, pass as-is (but ensure at least None-safe)
                    cleaned[col] = val
            normalized.append(cleaned)

        # Prepare dialect with Unix line endings and proper quoting
        dialect = csv.excel
        dialect.lineterminator = '\n'

        try:
            file_exists = self.filepath.exists() and self.filepath.stat().st_size > 0
            with open(self.filepath, mode, newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, dialect=dialect)

                # Write header only if file is empty or we are overwriting
                if mode == 'w' or (mode == 'a' and not file_exists):
                    writer.writeheader()

                for row in normalized:
                    writer.writerow(row)
        except OSError as e:
            raise OSError(f"Failed to write CSV file {self.filepath}: {e}") from e

        logger.info("Wrote %d rows to %s (mode=%s)", len(normalized), self.filepath, mode)
        return len(normalized)

    def write_metadata(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Write (or update) the auxiliary metadata JSON file.

        Merges the given metadata dict with any previously stored metadata.
        The final dict always includes ``_written_at`` (ISO timestamp),
        ``_source_csv`` (relative path), and ``_schema`` (human-readable type map).

        Args:
            metadata: Optional additional metadata to merge.

        Raises:
            OSError: If the metadata file cannot be written.
            TypeError: If metadata content is not JSON-serializable.
        """
        if metadata:
            self._metadata.update(metadata)

        # Build the complete metadata payload
        payload: Dict[str, Any] = {
            "_written_at": datetime.now(timezone.utc).isoformat(),
            "_source_csv": str(self.filepath.resolve()),
            "_schema": {
                col: _resolve_type(t) for col, t in self.schema.items()
            },
        }
        payload.update(self._metadata)

        try:
            with open(self._metadata_filepath, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, default=str)
        except (OSError, TypeError) as e:
            raise type(e)(f"Failed to write metadata file {self._metadata_filepath}: {e}") from e

        logger.debug("Metadata written to %s", self._metadata_filepath)

    def close(self) -> None:
        """Finalize writer: ensure metadata file is written.

        This method should be called when the writer is no longer needed.
        It writes the current metadata to disk.
        """
        self.write_metadata()
        logger.info("CsvWriter closed for %s", self.filepath)

    def __enter__(self) -> 'CsvWriter':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Optional[type],
                 exc_val: Optional[BaseException],
                 exc_tb: Optional[object]) -> None:
        """Context manager exit: flush metadata and close."""
        if exc_type is None:
            self.close()
        else:
            # If an exception occurred, still try to write metadata but log warning
            logger.warning("CsvWriter context exiting with exception, writing metadata anyway")
            try:
                self.close()
            except Exception as e:
                logger.error("Failed to write metadata during exception handling: %s", e)