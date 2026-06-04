# tests/test_validation.py
"""Unit tests for data_validation.py with malformed CSV examples."""

import csv
import io
import json
import os
import tempfile
from pathlib import Path

import pytest

# The module under test; adjust import path according to project structure.
# Assuming src/data_validation.py exists.
from src.data_validation import (
    validate_trace_csv,
    validate_discipline_summary,
    validate_schema,
    ValidationError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_csv_string(rows: list[list[str]], header: list[str]) -> str:
    """Produce a CSV string from a header and list of row lists."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(header)
    writer.writerows(rows)
    return output.getvalue()


def _write_csv(tmpdir: str, filename: str, csv_string: str) -> str:
    """Write a CSV string to a temporary file and return its full path."""
    filepath = os.path.join(tmpdir, filename)
    with open(filepath, "w", newline="") as f:
        f.write(csv_string)
    return filepath


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_csv_dir():
    """Provide a temporary directory for test CSV files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ---------------------------------------------------------------------------
# Tests – validate_trace_csv
# ---------------------------------------------------------------------------

class TestValidateTraceCSV:
    """Tests for the `validate_trace_csv` function."""

    # Expected schema for trace records (as used in the discipline pipeline).
    TRACE_SCHEMA = {
        "timestamp": float,
        "action_type": str,
        "competitor_present": bool,
        "outcome": str,
    }

    def test_valid_traces(self, tmp_csv_dir):
        """A complete, well-formed trace file should pass validation."""
        header = list(self.TRACE_SCHEMA.keys())
        rows = [
            ["0.0", "move", "true", "success"],
            ["0.5", "attack", "false", "failure"],
        ]
        csv_string = _make_csv_string(rows, header)
        path = _write_csv(tmp_csv_dir, "valid_traces.csv", csv_string)

        # Should not raise
        result = validate_trace_csv(path, self.TRACE_SCHEMA)
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_missing_required_column(self, tmp_csv_dir):
        """Missing a required column should fail validation."""
        header = ["timestamp", "competitor_present", "outcome"]  # missing action_type
        rows = [["0.0", "true", "success"]]
        csv_string = _make_csv_string(rows, header)
        path = _write_csv(tmp_csv_dir, "missing_col.csv", csv_string)

        with pytest.raises(ValidationError, match="Missing required columns"):
            validate_trace_csv(path, self.TRACE_SCHEMA)

    def test_extra_column_ignored(self, tmp_csv_dir):
        """Extra columns beyond the schema may be allowed (or warn)."""
        header = list(self.TRACE_SCHEMA.keys()) + ["extra_col"]
        rows = [["0.0", "move", "true", "success", "ignore_me"]]
        csv_string = _make_csv_string(rows, header)
        path = _write_csv(tmp_csv_dir, "extra_col.csv", csv_string)

        result = validate_trace_csv(path, self.TRACE_SCHEMA, strict_columns=False)
        # Should still pass, with warnings
        assert result["valid"] is True
        assert "extra_col" in result["warnings"][0]

    def test_malformed_types(self, tmp_csv_dir):
        """A field that cannot be cast to the required type should be flagged."""
        header = list(self.TRACE_SCHEMA.keys())
        # timestamp is "abc", not a float
        rows = [["abc", "move", "true", "success"]]
        csv_string = _make_csv_string(rows, header)
        path = _write_csv(tmp_csv_dir, "bad_type.csv", csv_string)

        with pytest.raises(ValidationError, match="Row 1: Field 'timestamp'"):
            validate_trace_csv(path, self.TRACE_SCHEMA)

    def test_empty_file(self, tmp_csv_dir):
        """An empty file (only header) should be valid (no data rows)."""
        header = list(self.TRACE_SCHEMA.keys())
        csv_string = _make_csv_string([], header)
        path = _write_csv(tmp_csv_dir, "empty.csv", csv_string)

        result = validate_trace_csv(path, self.TRACE_SCHEMA)
        assert result["valid"] is True
        assert result["row_count"] == 0

    def test_completely_empty_file(self, tmp_csv_dir):
        """A file with no header at all should fail."""
        path = _write_csv(tmp_csv_dir, "empty_no_header.csv", "")
        with pytest.raises(ValidationError, match="Empty or missing header"):
            validate_trace_csv(path, self.TRACE_SCHEMA)

    def test_missing_value_in_row(self, tmp_csv_dir):
        """A row with fewer fields than header should raise an error."""
        header = list(self.TRACE_SCHEMA.keys())
        rows = [["0.0", "move", "true"]]  # missing outcome
        csv_string = _make_csv_string(rows, header)
        path = _write_csv(tmp_csv_dir, "missing_field.csv", csv_string)

        with pytest.raises(ValidationError, match="Row 1: expected 4 fields, got 3"):
            validate_trace_csv(path, self.TRACE_SCHEMA)

    def test_bool_field_invalid_string(self, tmp_csv_dir):
        """Competitor_present must be one of 'true'/'false' (case-insensitive)."""
        header = list(self.TRACE_SCHEMA.keys())
        rows = [["0.0", "move", "yes", "success"]]  # invalid boolean
        csv_string = _make_csv_string(rows, header)
        path = _write_csv(tmp_csv_dir, "bad_bool.csv", csv_string)

        with pytest.raises(ValidationError, match="Row 1: Field 'competitor_present'"):
            validate_trace_csv(path, self.TRACE_SCHEMA)

    def test_future_timestamps_rejected(self, tmp_csv_dir):
        """Timestamps beyond a reasonable bound should be rejected."""
        header = list(self.TRACE_SCHEMA.keys())
        rows = [["999999.0", "move", "true", "success"]]
        csv_string = _make_csv_string(rows, header)
        path = _write_csv(tmp_csv_dir, "future_ts.csv", csv_string)

        with pytest.raises(ValidationError, match="Out-of-range timestamp"):
            validate_trace_csv(path, self.TRACE_SCHEMA, max_timestamp=1000.0)

    def test_negative_timestamp(self, tmp_csv_dir):
        """Negative timestamps are invalid."""
        header = list(self.TRACE_SCHEMA.keys())
        rows = [["-1.0", "move", "true", "success"]]
        csv_string = _make_csv_string(rows, header)
        path = _write_csv(tmp_csv_dir, "neg_ts.csv", csv_string)

        with pytest.raises(ValidationError, match="Negative timestamp"):
            validate_trace_csv(path, self.TRACE_SCHEMA)

    def test_outcome_not_in_allowed_set(self, tmp_csv_dir):
        """Outcome must be from an allowed set (e.g., 'success', 'failure')."""
        header = list(self.TRACE_SCHEMA.keys())
        rows = [["0.0", "move", "true", "unknown_outcome"]]
        csv_string = _make_csv_string(rows, header)
        path = _write_csv(tmp_csv_dir, "bad_outcome.csv", csv_string)

        allowed_outcomes = {"success", "failure", "partial"}
        with pytest.raises(ValidationError, match="Row 1: Field 'outcome'"):
            validate_trace_csv(
                path, self.TRACE_SCHEMA, allowed_outcomes=allowed_outcomes
            )

    def test_multiple_errors_reported(self, tmp_csv_dir):
        """All row-level errors should be collected and reported together."""
        header = list(self.TRACE_SCHEMA.keys())
        rows = [
            ["0.0", "move", "true", "success"],     # good
            ["abc", "attack", "yes", "invalid_out"], # three issues
        ]
        csv_string = _make_csv_string(rows, header)
        path = _write_csv(tmp_csv_dir, "multi_errors.csv", csv_string)

        with pytest.raises(ValidationError) as excinfo:
            validate_trace_csv(
                path,
                self.TRACE_SCHEMA,
                allowed_outcomes={"success", "failure"},
            )
        # Expect three distinct error entries in the exception message
        error_msg = str(excinfo.value)
        assert error_msg.count("Row 2") >= 3


# ---------------------------------------------------------------------------
# Tests – validate_discipline_summary
# ---------------------------------------------------------------------------

class TestValidateDisciplineSummary:
    """Tests for the aggregated metrics summary CSV."""

    SUMMARY_SCHEMA = {
        "agent_id": (str, True),
        "run_id": (str, True),
        "economic_score": (float, (0.0, 1.0)),
        "behavioral_compliance": (float, (0.0, 1.0)),
        "discipline_stability": (float, (-1.0, 1.0)),
        "num_steps": (int, (1, 100000)),
    }

    def test_valid_summary(self, tmp_csv_dir):
        header = list(self.SUMMARY_SCHEMA.keys())
        rows = [
            ["agent_1", "run_001", "0.85", "0.92", "0.73", "1200"],
        ]
        csv_string = _make_csv_string(rows, header)
        path = _write_csv(tmp_csv_dir, "valid_summary.csv", csv_string)
        result = validate_discipline_summary(path, self.SUMMARY_SCHEMA)
        assert result["valid"] is True

    def test_economic_score_out_of_range(self, tmp_csv_dir):
        header = list(self.SUMMARY_SCHEMA.keys())
        rows = [
            ["agent_1", "run_001", "1.5", "0.5", "0.0", "100"],
        ]
        csv_string = _make_csv_string(rows, header)
        path = _write_csv(tmp_csv_dir, "bad_score.csv", csv_string)
        with pytest.raises(ValidationError, match="out of range"):
            validate_discipline_summary(path, self.SUMMARY_SCHEMA)

    def test_missing_agent_id(self, tmp_csv_dir):
        header = list(self.SUMMARY_SCHEMA.keys())
        rows = [
            ["", "run_001", "0.5", "0.5", "0.0", "100"],
        ]
        csv_string = _make_csv_string(rows, header)
        path = _write_csv(tmp_csv_dir, "missing_id.csv", csv_string)
        with pytest.raises(ValidationError, match="empty"):
            validate_discipline_summary(path, self.SUMMARY_SCHEMA)

    def test_non_numeric_steps(self, tmp_csv_dir):
        """num_steps must be castable to int."""
        header = list(self.SUMMARY_SCHEMA.keys())
        rows = [
            ["agent_1", "run_001", "0.5", "0.5", "0.0", "many"],
        ]
        csv_string = _make_csv_string(rows, header)
        path = _write_csv(tmp_csv_dir, "bad_steps.csv", csv_string)
        with pytest.raises(ValidationError, match="cannot be interpreted"):
            validate_discipline_summary(path, self.SUMMARY_SCHEMA)


# ---------------------------------------------------------------------------
# Tests – validate_schema (lower-level)
# ---------------------------------------------------------------------------

class TestValidateSchema:
    """Tests for generic schema validation (used internally)."""

    def test_non_existent_file(self):
        with pytest.raises(FileNotFoundError):
            validate_schema("/nonexistent/file.csv")

    def test_unicode_encoding(self, tmp_csv_dir):
        """Should handle BOM or UTF-8 BOM."""
        header = "timestamp,action_type\n"
        data = "0.0,move\n"
        path = _write_csv(tmp_csv_dir, "utf8bom.csv", "\ufeff" + header + data)
        schema = {"timestamp": float, "action_type": str}
        result = validate_schema(path, schema)
        assert result["valid"] is True

    def test_delimiter_override(self, tmp_csv_dir):
        """Semicolon-delimited files should be handled with a delimiter argument."""
        rows = [["0.0;move;true;success"]]
        header = "timestamp;action_type;competitor_present;outcome"
        csv_string = header + "\n" + rows[0]
        path = _write_csv(tmp_csv_dir, "semicol.csv", csv_string)
        schema = {
            "timestamp": float,
            "action_type": str,
            "competitor_present": bool,
            "outcome": str,
        }
        result = validate_schema(path, schema, delimiter=";")
        assert result["valid"] is True

    def test_blank_lines_ignored(self, tmp_csv_dir):
        """Blank lines in the middle of a file should be skipped."""
        content = "a,b\n1,2\n\n3,4\n\n"
        path = _write_csv(tmp_csv_dir, "blank_lines.csv", content)
        result = validate_schema(path, {"a": int, "b": int})
        assert result["valid"] is True
        assert result["row_count"] == 2


# ---------------------------------------------------------------------------
# Integration test with actual discipline pipeline context
# ---------------------------------------------------------------------------

class TestDisciplinePipelineValidation:
    """
    End-to-end validation inspired by the trace-based discipline evaluation paper.
    Simulates the entire pipeline from raw traces to summary validation.
    """

    PIPELINE_ORDER = [
        "ingestion", "parsing", "feature_extraction",
        "discipline_computation", "aggregation", "validation",
    ]

    @pytest.mark.parametrize("stage", PIPELINE_ORDER)
    def test_each_stage_csv_is_validatable(self, stage, tmp_csv_dir):
        """
        For each stage we create a minimal but correct CSV file (or a malformed one)
        and verify that the validator either passes or raises appropriately.
        Here we test with a valid minimal file to confirm the validator can accept it.
        """
        # Build a common schema for all stages (simplified)
        if stage in ("ingestion", "parsing"):
            schema = {
                "trace_id": str,
                "timestamp": float,
                "action": str,
                "competitor_flag": bool,
                "reward": float,
            }
            header = list(schema.keys())
            rows = [["trace_1", "0.0", "move", "false", "1.0"]]
        elif stage == "feature_extraction":
            schema = {
                "trace_id": str,
                "action_consistency": float,
                "competitor_presence": bool,
                "rule_adherence": bool,
                "deviation": float,
            }
            header = list(schema.keys())
            rows = [["trace_1", "0.95", "true", "true", "0.02"]]
        elif stage in ("discipline_computation", "aggregation"):
            schema = {
                "agent_id": str,
                "behavioral_consistency": float,
                "hidden_state_misalignment": float,
                "economic_compliance": float,
                "behavioral_compliance": float,
            }
            header = list(schema.keys())
            rows = [["agent_1", "0.8", "0.1", "0.9", "0.85"]]
        else:  # validation stage
            schema = {
                "summary_id": str,
                "score": float,
                "status": str,
            }
            header = list(schema.keys())
            rows = [["summary_1", "0.75", "passed"]]

        csv_string = _make_csv_string(rows, header)
        path = _write_csv(tmp_csv_dir, f"{stage}.csv", csv_string)
        result = validate_schema(path, schema)
        assert result["valid"] is True

    def test_malformed_intermediate_file_in_pipeline(self, tmp_csv_dir):
        """
        A deliberately malformed feature CSV with integer expected but string given
        should be caught.
        """
        schema = {
            "trace_id": str,
            "action_consistency": float,  # expecting float
        }
        header = list(schema.keys())
        rows = [["trace_1", "not_a_number"]]  # malformed
        csv_string = _make_csv_string(rows, header)
        path = _write_csv(tmp_csv_dir, "malformed_feature.csv", csv_string)

        with pytest.raises(ValidationError, match="cannot be interpreted"):
            validate_schema(path, schema)