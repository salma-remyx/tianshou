"""src/data_validation.py - CSV schema validator with JSON report generation.

Production-grade implementation with comprehensive error handling, logging,
type annotations, input validation, and performance optimizations.
"""

from __future__ import annotations

import csv
import json
import logging
import typing as t
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum

__all__ = [
    "ColumnType",
    "ValidationStatus",
    "SchemaRule",
    "ValidationReport",
    "validate_csv",
    "validate_csv_to_json",
    "validate_csv_to_file",
]

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------

class ColumnType(str, Enum):
    """Supported column types for schema validation."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    STRING = "string"
    ANY = "any"


class ValidationStatus(str, Enum):
    """Overall validation status of a CSV file."""
    PASSED = "passed"
    FAILED = "failed"


@dataclass(frozen=True)
class SchemaRule:
    """Schema definition for a single column.

    Attributes:
        type: Expected column type.
        nullable: Whether empty values are allowed.
        min_value: Minimum numeric value (only for numeric type).
        max_value: Maximum numeric value (only for numeric type).
        allowed_values: Allowed categorical values (only for categorical type).
    """
    type: ColumnType = ColumnType.ANY
    nullable: bool = False
    min_value: t.Optional[float] = None
    max_value: t.Optional[float] = None
    allowed_values: t.Optional[t.List[t.Any]] = None

    def __post_init__(self) -> None:
        """Validate constraints after initialization."""
        if self.min_value is not None and self.max_value is not None:
            if self.min_value > self.max_value:
                raise ValueError(
                    f"min_value ({self.min_value}) must be <= max_value ({self.max_value})"
                )
        if self.allowed_values is not None and self.type != ColumnType.CATEGORICAL:
            logger.warning(
                "allowed_values provided for non-categorical column '%s'; ignoring.",
                self.type.value
            )


# Schema is a dictionary mapping column names to SchemaRule objects
Schema = t.Dict[str, SchemaRule]


@dataclass
class ValidationReport:
    """Complete validation result for a CSV file.

    Attributes:
        file_path: Absolute path of the validated file.
        valid: True if all rows passed validation.
        row_count: Number of data rows parsed (excluding header).
        errors: Mapping of column name to list of error messages.
        columns: Per-column summary statistics (e.g., non-null count).
        status: Overall validation status string ("passed" or "failed").
    """
    file_path: str
    valid: bool
    row_count: int
    errors: t.Dict[str, t.List[str]] = field(default_factory=dict)
    columns: t.Dict[str, t.Dict[str, t.Any]] = field(default_factory=dict)
    status: str = "passed"

    def __post_init__(self) -> None:
        """Set status based on validity."""
        if self.valid:
            self.status = ValidationStatus.PASSED.value
        else:
            self.status = ValidationStatus.FAILED.value

    def to_dict(self) -> t.Dict[str, t.Any]:
        """Convert report to a dictionary (JSON-serializable)."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Return JSON string representation of the report."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def write_json(self, output_path: t.Union[str, Path]) -> None:
        """Write the report as JSON to a file.

        Args:
            output_path: Destination file path.

        Raises:
            OSError: If the file cannot be written.
        """
        path = Path(output_path)
        path.write_text(self.to_json(), encoding="utf-8")
        logger.info("Validation report written to %s", path.resolve())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_value(
    value: str,
    column: str,
    row_number: int,
    rule: SchemaRule,
    column_stats: t.Dict[str, t.Any],
) -> t.Optional[str]:
    """Check a single value against its schema rule.

    Args:
        value: Raw string from the CSV.
        column: Column name.
        row_number: 1-indexed row number (for error messages).
        rule: Schema rule for the column.
        column_stats: Mutable dictionary tracking per-column statistics.

    Returns:
        Error message string if validation fails, else None.
    """
    # --- Empty / whitespace check ---
    if not value or value.strip() == "":
        if rule.nullable:
            column_stats["missing"] = column_stats.get("missing", 0) + 1
            return None
        column_stats["missing"] = column_stats.get("missing", 0) + 1
        return (
            f"Row {row_number}: column '{column}' is empty but not nullable."
        )

    stripped = value.strip()
    column_stats["non_null"] = column_stats.get("non_null", 0) + 1

    # --- Numeric validation ---
    if rule.type == ColumnType.NUMERIC:
        try:
            num = float(stripped)
        except ValueError:
            return (
                f"Row {row_number}: column '{column}' has non-numeric value "
                f"'{stripped}'."
            )
        if rule.min_value is not None and num < rule.min_value:
            return (
                f"Row {row_number}: column '{column}' value {num} is less "
                f"than minimum {rule.min_value}."
            )
        if rule.max_value is not None and num > rule.max_value:
            return (
                f"Row {row_number}: column '{column}' value {num} is greater "
                f"than maximum {rule.max_value}."
            )
        return None

    # --- Categorical validation ---
    if rule.type == ColumnType.CATEGORICAL:
        if rule.allowed_values is not None and stripped not in rule.allowed_values:
            return (
                f"Row {row_number}: column '{column}' value '{stripped}' "
                f"not in allowed values {rule.allowed_values}."
            )
        return None

    # --- String / any types: pass through (no additional checks) ---
    return None


def _check_columns(
    schema_cols: t.Set[str],
    file_cols: t.List[str],
) -> t.Optional[str]:
    """Verify file columns exactly match schema columns.

    Args:
        schema_cols: Set of column names defined in the schema.
        file_cols: List of column names from the CSV header.

    Returns:
        Error message if mismatch, otherwise None.
    """
    file_cols_set = set(file_cols)
    missing = schema_cols - file_cols_set
    if missing:
        return f"Missing column(s): {sorted(missing)}"
    extra = file_cols_set - schema_cols
    if extra:
        return f"Extra column(s) not in schema: {sorted(extra)}"
    return None


# ---------------------------------------------------------------------------
# Public validation functions
# ---------------------------------------------------------------------------

def validate_csv(
    file_path: t.Union[str, Path],
    schema: Schema,
    *,
    encoding: str = "utf-8-sig",
    max_rows: t.Optional[int] = None,
) -> ValidationReport:
    """Validate a CSV file against a schema.

    Args:
        file_path: Path to the CSV file (must exist and be non-empty).
        schema: Dictionary mapping column names to SchemaRule.
        encoding: File encoding (default: ``utf-8-sig`` to handle BOM).
        max_rows: Optional maximum number of rows to process. If ``None``,
            all rows are processed.

    Returns:
        ValidationReport with detailed results.

    Raises:
        TypeError: If ``schema`` is not a dict or contains non-SchemaRule values.
        ValueError: If schema contains non-string column names.
        OSError: If the file cannot be opened or read.
    """
    # --- Input validation ---
    if not isinstance(schema, dict):
        raise TypeError(
            "schema must be a dictionary mapping column names to SchemaRule, "
            f"got {type(schema)}"
        )
    for col, rule in schema.items():
        if not isinstance(col, str) or not col:
            raise ValueError(f"Column keys must be non-empty strings, got {col!r}")
        if not isinstance(rule, SchemaRule):
            raise TypeError(
                f"Schema value for column '{col}' must be a SchemaRule, "
                f"got {type(rule)}"
            )

    path = Path(file_path)
    logger.debug("Validating CSV file: %s (encoding=%s)", path, encoding)

    # --- File existence and emptiness checks ---
    if not path.exists():
        logger.error("File does not exist: %s", path)
        return ValidationReport(
            file_path=str(path.resolve()),
            valid=False,
            row_count=0,
            errors={"__file__": ["File does not exist."]},
        )

    try:
        file_size = path.stat().st_size
    except OSError as e:
        logger.error("Cannot stat file %s: %s", path, e)
        return ValidationReport(
            file_path=str(path.resolve()),
            valid=False,
            row_count=0,
            errors={"__file__": [f"Cannot access file: {e}"]},
        )

    if file_size == 0:
        logger.error("File is empty: %s", path)
        return ValidationReport(
            file_path=str(path.resolve()),
            valid=False,
            row_count=0,
            errors={"__file__": ["File is empty."]},
        )

    # --- Initialize report accumulators ---
    errors: t.Dict[str, t.List[str]] = {}
    column_stats: t.Dict[str, t.Dict[str, t.Any]] = {
        col: {"non_null": 0, "missing": 0} for col in schema
    }
    row_count = 0
    valid = True

    # --- Open and parse CSV -------------------------------------------------
    try:
        with path.open("r", encoding=encoding, newline="") as fh:
            reader = csv.reader(fh)

            # Read header
            try:
                header = next(reader)
            except StopIteration:
                # File has no header
                logger.error("CSV file has no header row: %s", path)
                return ValidationReport(
                    file_path=str(path.resolve()),
                    valid=False,
                    row_count=0,
                    errors={"__file__": ["CSV file has no header row."]},
                )

            header = [col.strip() for col in header]  # strip whitespace

            # Validate header matches schema
            schema_cols = set(schema.keys())
            col_error = _check_columns(schema_cols, header)
            if col_error:
                logger.error("Column mismatch in %s: %s", path, col_error)
                return ValidationReport(
                    file_path=str(path.resolve()),
                    valid=False,
                    row_count=0,
                    errors={"__header__": [col_error]},
                )

            # Build column index mapping
            col_index = {col: idx for idx, col in enumerate(header)}

            # Process rows
            for row_num, row in enumerate(reader, start=1):
                if max_rows is not None and row_num > max_rows:
                    logger.debug("Reached max_rows=%d, stopping", max_rows)
                    break

                # Skip completely empty rows (but not partial)
                if not any(cell.strip() for cell in row):
                    continue

                row_count += 1

                # Validate each column
                for col_name in schema:
                    idx = col_index[col_name]
                    raw_value = row[idx] if idx < len(row) else ""
                    err_msg = _validate_value(
                        raw_value, col_name, row_num, schema[col_name],
                        column_stats[col_name],
                    )
                    if err_msg is not None:
                        valid = False
                        errors.setdefault(col_name, []).append(err_msg)

            # Record final per-column statistics
            for col in schema:
                column_stats[col] = {
                    "non_null": column_stats[col]["non_null"],
                    "missing": column_stats[col]["missing"],
                    "total_expected": row_count,
                }

    except UnicodeDecodeError as e:
        logger.error("Encoding error reading %s: %s", path, e)
        return ValidationReport(
            file_path=str(path.resolve()),
            valid=False,
            row_count=row_count,
            errors={"__file__": [f"Encoding error: {e}"]},
        )
    except csv.Error as e:
        logger.error("CSV parsing error in %s: %s", path, e)
        return ValidationReport(
            file_path=str(path.resolve()),
            valid=False,
            row_count=row_count,
            errors={"__file__": [f"CSV parsing error: {e}"]},
        )
    except OSError as e:
        logger.error("I/O error reading %s: %s", path, e)
        return ValidationReport(
            file_path=str(path.resolve()),
            valid=False,
            row_count=row_count,
            errors={"__file__": [f"I/O error: {e}"]},
        )

    report = ValidationReport(
        file_path=str(path.resolve()),
        valid=valid,
        row_count=row_count,
        errors=errors,
        columns=column_stats,
    )
    logger.info(
        "Validation finished: valid=%s, rows=%d, errors=%d",
        valid, row_count, sum(len(v) for v in errors.values())
    )
    return report


def validate_csv_to_json(
    file_path: t.Union[str, Path],
    schema: Schema,
    *,
    encoding: str = "utf-8-sig",
    max_rows: t.Optional[int] = None,
    indent: int = 2,
) -> str:
    """Validate a CSV file and return the report as a JSON string.

    This is a convenience wrapper around :func:`validate_csv`.

    Args:
        file_path: Path to the CSV file.
        schema: Schema dictionary.
        encoding: File encoding (default: ``utf-8-sig``).
        max_rows: Optional maximum number of rows to process.
        indent: JSON indentation level (default: 2).

    Returns:
        JSON string of the validation report.
    """
    report = validate_csv(
        file_path, schema, encoding=encoding, max_rows=max_rows
    )
    return report.to_json(indent=indent)


def validate_csv_to_file(
    file_path: t.Union[str, Path],
    schema: Schema,
    output_path: t.Union[str, Path],
    *,
    encoding: str = "utf-8-sig",
    max_rows: t.Optional[int] = None,
    indent: int = 2,
) -> None:
    """Validate a CSV file and write the JSON report to a file.

    Args:
        file_path: Path to the CSV file.
        schema: Schema dictionary.
        output_path: Destination file for JSON report.
        encoding: File encoding for CSV (default: ``utf-8-sig``).
        max_rows: Optional maximum number of rows to process.
        indent: JSON indentation level (default: 2).

    Raises:
        OSError: If the output file cannot be written.
    """
    report = validate_csv(
        file_path, schema, encoding=encoding, max_rows=max_rows
    )
    report.write_json(output_path)