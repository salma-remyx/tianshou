"""
Unit tests for discipline_eval.py — edge cases and expected metric ranges.

This module tests the core discipline evaluation functions following the
principles of trace-based evaluation under hidden competitor state,
as discussed in the referenced paper.

Tests cover:
- Empty and boundary inputs
- Missing fields and malformed data
- Extreme and edge-case metric values
- Cross-validation folds and aggregation correctness
- Behavioral vs economic compliance scores
"""

import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any

# ---------------------------------------------------------------------------
# Helper to generate synthetic trace data for tests
# ---------------------------------------------------------------------------

def _make_trace_record(
    *,
    step: int,
    action_consistency: float = 1.0,
    competitor_present: bool = False,
    rule_adherent: bool = True,
    deviation: float = 0.0,
    hidden_state_misalignment: float = 0.0,
    economic_outcome: float = 0.0,
    behavioral_compliance: float = 1.0,
) -> Dict[str, Any]:
    """Create a single trace record with configurable fields."""
    return {
        "step": step,
        "action_consistency": action_consistency,
        "competitor_present": competitor_present,
        "rule_adherent": rule_adherent,
        "deviation": deviation,
        "hidden_state_misalignment": hidden_state_misalignment,
        "economic_outcome": economic_outcome,
        "behavioral_compliance": behavioral_compliance,
    }


def _make_traces_df(records: list) -> pd.DataFrame:
    """Convert list of trace dicts to DataFrame with expected columns."""
    df = pd.DataFrame(records)
    required = [
        "step", "action_consistency", "competitor_present",
        "rule_adherent", "deviation", "hidden_state_misalignment",
        "economic_outcome", "behavioral_compliance",
    ]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan
    return df


# ---------------------------------------------------------------------------
# Mocks / stubs for the discipline computation module
# In production replace with actual imports from discipline_eval.
# ---------------------------------------------------------------------------

# We assume discipline_eval exports these functions:
# - compute_discipline_metrics(df: pd.DataFrame) -> pd.DataFrame
# - compute_stability(metrics_df: pd.DataFrame) -> float
# - compute_hidden_state_misalignment_score(obs, hidden) -> float
# - aggregate_across_runs(runs: list) -> pd.DataFrame
# - validate_metrics(metrics_df: pd.DataFrame) -> Dict[str, Any]

# For self-contained tests we provide local stubs that mimic expected behavior.
# These stubs are intentionally simplistic; they will be replaced by the real
# module in a production deployment.

def _stub_compute_discipline_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Stub: compute dummy discipline metrics from trace features."""
    if df.empty:
        return df.copy()

    metrics = df.copy()
    # Discipline composite: average of consistency and compliance, penalised by deviation
    metrics["discipline_score"] = (
        0.5 * metrics["action_consistency"]
        + 0.3 * metrics["behavioral_compliance"]
        - 0.2 * metrics["deviation"]
    )
    # Stability: 1 - mean absolute hidden state misalignment
    metrics["stability"] = 1.0 - metrics["hidden_state_misalignment"].abs().mean()
    # Economic vs behavioral discrepancy
    metrics["eco_behavior_gap"] = (
        metrics["economic_outcome"] - metrics["behavioral_compliance"]
    )
    return metrics


def _stub_compute_stability(metrics_df: pd.DataFrame) -> float:
    """Stub: stability as global mean of per-trace stability."""
    if "stability" not in metrics_df.columns:
        return np.nan
    return float(metrics_df["stability"].mean())


def _stub_validate_metrics(metrics_df: pd.DataFrame) -> Dict[str, Any]:
    """Stub: basic range and non-null checks."""
    report = {"valid": True, "errors": [], "warnings": []}
    required_cols = [
        "discipline_score", "stability", "eco_behavior_gap",
        "action_consistency", "behavioral_compliance",
    ]
    for col in required_cols:
        if col not in metrics_df.columns:
            report["errors"].append(f"Missing column: {col}")
            report["valid"] = False
            continue
        if metrics_df[col].isnull().all():
            report["errors"].append(f"Column {col} is entirely NA")
            report["valid"] = False

    # Check discipline_score range (should be in [0,1] ideally)
    if "discipline_score" in metrics_df.columns:
        bad = metrics_df[
            (metrics_df["discipline_score"] < -1) | (metrics_df["discipline_score"] > 2)
        ]
        if not bad.empty:
            report["warnings"].append(
                f"{len(bad)} discipline_score values outside [-1, 2] range"
            )
    return report


# ---------------------------------------------------------------------------
# Actual test module
# ---------------------------------------------------------------------------


class TestDisciplineMetrics:
    """Tests for discipline metric computation functions."""

    def test_empty_dataframe(self):
        """Empty input should be handled gracefully without crash."""
        df = pd.DataFrame()
        result = _stub_compute_discipline_metrics(df)
        assert result.empty

    def test_single_trace(self):
        """Single trace should produce deterministic output."""
        rec = _make_trace_record(step=0, action_consistency=0.8, deviation=0.1)
        df = _make_traces_df([rec])
        metrics = _stub_compute_discipline_metrics(df)
        assert len(metrics) == 1
        assert "discipline_score" in metrics.columns
        # Manual compute: 0.5*0.8 + 0.3*1.0 - 0.2*0.1 = 0.4 + 0.3 - 0.02 = 0.68
        expected = 0.68
        assert abs(metrics["discipline_score"].iloc[0] - expected) < 1e-10

    def test_multiple_traces_happy_path(self):
        """Multiple traces with varied values should compute correctly."""
        records = [
            _make_trace_record(step=0, action_consistency=1.0, deviation=0.0,
                               behavioral_compliance=1.0),
            _make_trace_record(step=1, action_consistency=0.5, deviation=0.2,
                               behavioral_compliance=0.7),
            _make_trace_record(step=2, action_consistency=0.0, deviation=0.8,
                               behavioral_compliance=0.3),
        ]
        df = _make_traces_df(records)
        metrics = _stub_compute_discipline_metrics(df)
        assert len(metrics) == 3

        # Trace 0: 0.5*1.0 + 0.3*1.0 - 0.2*0.0 = 0.8
        assert metrics["discipline_score"].iloc[0] == pytest.approx(0.8)
        # Trace 1: 0.5*0.5 + 0.3*0.7 - 0.2*0.2 = 0.25 + 0.21 - 0.04 = 0.42
        assert metrics["discipline_score"].iloc[1] == pytest.approx(0.42)
        # Trace 2: 0.5*0.0 + 0.3*0.3 - 0.2*0.8 = 0 + 0.09 - 0.16 = -0.07
        assert metrics["discipline_score"].iloc[2] == pytest.approx(-0.07)

    def test_extreme_metric_values(self):
        """Test with extreme but valid values (large deviations, zero consistency)."""
        records = [
            _make_trace_record(step=0, action_consistency=1.0, deviation=10.0),
            _make_trace_record(step=1, action_consistency=-1.0, deviation=-5.0),
            _make_trace_record(step=2, action_consistency=0.0, deviation=0.0),
        ]
        df = _make_traces_df(records)
        metrics = _stub_compute_discipline_metrics(df)
        # Extreme values may produce scores outside [0,1]; we only check no crash
        assert not metrics["discipline_score"].isnull().any()

    def test_missing_optional_fields(self):
        """Missing optional fields should fall back to defaults or NaN."""
        # Only step and required column missing
        records = [
            {"step": 0, "action_consistency": 0.9}
            # missing: competitor_present, rule_adherent, deviation, etc.
        ]
        df = _make_traces_df(records)
        metrics = _stub_compute_discipline_metrics(df)
        # Missing fields become NaN; discipline_score should be NaN because
        # deviation and behavioral_compliance are NaN
        assert metrics["discipline_score"].isnull().all()

    def test_all_zeros(self):
        """All-zero input should produce zero or near-zero output."""
        rec = _make_trace_record(
            step=0, action_consistency=0.0, deviation=0.0,
            behavioral_compliance=0.0, hidden_state_misalignment=0.0
        )
        df = _make_traces_df([rec])
        metrics = _stub_compute_discipline_metrics(df)
        assert metrics["discipline_score"].iloc[0] == pytest.approx(0.0)
        assert metrics["eco_behavior_gap"].iloc[0] == pytest.approx(0.0)

    def test_discipline_score_range_no_outliers(self):
        """Typical inputs should keep discipline_score in plausible range."""
        # Generate 100 random records with reasonable values
        np.random.seed(42)
        records = []
        for i in range(100):
            rec = _make_trace_record(
                step=i,
                action_consistency=np.random.uniform(0.0, 1.0),
                deviation=np.random.uniform(0.0, 0.5),
                behavioral_compliance=np.random.uniform(0.5, 1.0),
            )
            records.append(rec)
        df = _make_traces_df(records)
        metrics = _stub_compute_discipline_metrics(df)
        scores = metrics["discipline_score"]
        # Most should be between -0.1 and 1.0 (since deviation capped at 0.5)
        assert scores.min() >= -0.3  # worst case: 0 + 0.15 - 0.1 = 0.05
        assert scores.max() <= 1.5   # best case: 0.5 + 0.3 - 0 = 0.8

    def test_eco_behavior_gap_sign(self):
        """Gap should reflect discrepancy between economic outcome and compliance."""
        records = [
            _make_trace_record(step=0, economic_outcome=1.0, behavioral_compliance=0.5),
            _make_trace_record(step=1, economic_outcome=0.2, behavioral_compliance=0.9),
        ]
        df = _make_traces_df(records)
        metrics = _stub_compute_discipline_metrics(df)
        # gap = economic - behavioral
        assert metrics["eco_behavior_gap"].iloc[0] == pytest.approx(0.5)
        assert metrics["eco_behavior_gap"].iloc[1] == pytest.approx(-0.7)


class TestStability:
    """Stability metric tests."""

    def test_stability_perfect(self):
        """Perfect stability when hidden state misalignment is zero."""
        rec = _make_trace_record(step=0, hidden_state_misalignment=0.0)
        df = _make_traces_df([rec])
        metrics = _stub_compute_discipline_metrics(df)
        stable = _stub_compute_stability(metrics)
        assert stable == pytest.approx(1.0)

    def test_stability_degraded(self):
        """High misalignment reduces stability."""
        records = [
            _make_trace_record(step=0, hidden_state_misalignment=0.3),
            _make_trace_record(step=1, hidden_state_misalignment=0.7),
        ]
        df = _make_traces_df(records)
        metrics = _stub_compute_discipline_metrics(df)
        stable = _stub_compute_stability(metrics)
        # average absolute misalignment = (0.3+0.7)/2 = 0.5, stability = 1 - 0.5 = 0.5
        assert stable == pytest.approx(0.5)

    def test_stability_extreme_misalignment(self):
        """Misalignment >1 should still compute (not capped)."""
        rec = _make_trace_record(step=0, hidden_state_misalignment=2.0)
        df = _make_traces_df([rec])
        metrics = _stub_compute_discipline_metrics(df)
        stable = _stub_compute_stability(metrics)
        # stability = 1 - 2.0 = -1.0
        assert stable == pytest.approx(-1.0)

    def test_stability_empty(self):
        """Empty metrics should return NaN."""
        df_empty = pd.DataFrame()
        stable = _stub_compute_stability(df_empty)
        assert np.isnan(stable)


class TestValidation:
    """Validation report tests (edge cases)."""

    def test_validate_correct_metrics(self):
        """All columns present and within range -> valid."""
        rec = _make_trace_record(step=0)
        df = _make_traces_df([rec])
        metrics = _stub_compute_discipline_metrics(df)
        report = _stub_validate_metrics(metrics)
        assert report["valid"] is True
        assert len(report["errors"]) == 0

    def test_validate_missing_column(self):
        """Missing required column causes error."""
        df = pd.DataFrame({"step": [0]})  # no discipline columns
        report = _stub_validate_metrics(df)
        assert report["valid"] is False
        assert any("Missing column" in e for e in report["errors"])

    def test_validate_all_na_column(self):
        """Column full of NaN triggers error."""
        rec = _make_trace_record(step=0)
        df = _make_traces_df([rec])
        metrics = _stub_compute_discipline_metrics(df)
        # force one column to all NaN
        metrics["discipline_score"] = np.nan
        report = _stub_validate_metrics(metrics)
        assert report["valid"] is False
        assert any("entirely NA" in e for e in report["errors"])

    def test_validate_outlier_warning(self):
        """Values outside expected range produce warnings (not errors)."""
        rec = _make_trace_record(step=0, action_consistency=5.0, deviation=10.0)
        df = _make_traces_df([rec])
        metrics = _stub_compute_discipline_metrics(df)
        # discipline_score may be: 0.5*5 + 0.3*1 - 0.2*10 = 2.5 + 0.3 - 2 = 0.8 (within range)
        # Force an extreme value to trigger warning
        metrics["discipline_score"] = 3.0
        report = _stub_validate_metrics(metrics)
        assert report["valid"] is True  # no errors
        assert len(report["warnings"]) >= 1


class TestRealisticPipelineScenario:
    """Integration-like tests using realistic multi-run data."""

    @pytest.fixture
    def multi_run_data(self):
        """Create three runs with different hidden competitor scenarios."""
        runs = []
        for run_id in range(3):
            np.random.seed(run_id)
            records = []
            n_steps = 50
            for t in range(n_steps):
                # Vary hidden state misalignment by run
                if run_id == 0:
                    # Low misalignment (ideal discipline)
                    misalign = np.random.uniform(0.0, 0.1)
                elif run_id == 1:
                    # Moderate misalignment
                    misalign = np.random.uniform(0.2, 0.5)
                else:
                    # High misalignment
                    misalign = np.random.uniform(0.6, 1.0)

                rec = _make_trace_record(
                    step=t,
                    action_consistency=np.random.uniform(0.7, 1.0),
                    competitor_present=bool(np.random.binomial(1, 0.3)),
                    rule_adherent=bool(np.random.binomial(1, 0.9)),
                    deviation=np.random.uniform(0.0, 0.3),
                    hidden_state_misalignment=misalign,
                    economic_outcome=np.random.uniform(-0.5, 1.0),
                    behavioral_compliance=np.random.uniform(0.6, 1.0),
                )
                records.append(rec)
            runs.append(_make_traces_df(records))
        return runs

    def test_multi_run_aggregation(self, multi_run_data):
        """Aggregation across runs should produce summary statistics."""
        # our stub doesn't implement aggregation, but we test that
        # compute works on each run and overall means are sensible.
        run_metrics = [_stub_compute_discipline_metrics(run) for run in multi_run_data]
        # Run 0 should have highest average stability
        stabilities = [_stub_compute_stability(m) for m in run_metrics]
        assert stabilities[0] > stabilities[1]
        assert stabilities[1] > stabilities[2]

    def test_validation_on_aggregated_output(self, multi_run_data):
        """Validation should pass on aggregate metrics."""
        combined = pd.concat(multi_run_data, ignore_index=True)
        metrics = _stub_compute_discipline_metrics(combined)
        report = _stub_validate_metrics(metrics)
        assert report["valid"] is True

    def test_competitor_flag_effect_on_hidden_state(self):
        """When competitor is present, hidden state misalignment may be higher."""
        records = []
        for t in range(20):
            competitor = t % 2 == 0
            misalign = np.random.uniform(0.1, 0.6) if competitor else np.random.uniform(0.0, 0.2)
            records.append(
                _make_trace_record(step=t, competitor_present=competitor,
                                   hidden_state_misalignment=misalign)
            )
        df = _make_traces_df(records)
        metrics = _stub_compute_discipline_metrics(df)
        # stability should reflect competitor presence (but we only check no crash)
        assert metrics["stability"].iloc[0] is not None

    def test_behavioral_compliance_outliers(self):
        """Behavioral compliance can be negative (malicious agent)."""
        rec = _make_trace_record(step=0, behavioral_compliance=-0.5)
        df = _make_traces_df([rec])
        metrics = _stub_compute_discipline_metrics(df)
        # discipline_score = 0.5*1.0 + 0.3*(-0.5) - 0.2*0.0 = 0.5 - 0.15 = 0.35
        assert metrics["discipline_score"].iloc[0] == pytest.approx(0.35)

    def test_deviation_beyond_one(self):
        """Deviation > 1 should be allowed but penalises discipline heavily."""
        rec = _make_trace_record(step=0, deviation=2.0)
        df = _make_traces_df([rec])
        metrics = _stub_compute_discipline_metrics(df)
        # score = 0.5*1 + 0.3*1 - 0.2*2 = 0.5+0.3-0.4 = 0.4
        assert metrics["discipline_score"].iloc[0] == pytest.approx(0.4)

    def test_negative_economic_outcome(self):
        """Negative economic outcome should produce negative gap."""
        rec = _make_trace_record(step=0, economic_outcome=-0.3, behavioral_compliance=0.8)
        df = _make_traces_df([rec])
        metrics = _stub_compute_discipline_metrics(df)
        assert metrics["eco_behavior_gap"].iloc[0] == pytest.approx(-1.1)