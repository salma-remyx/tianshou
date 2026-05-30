# tests/test_parser.py
"""
Unit tests for trace_parser module using synthetic small traces.

These tests verify the correctness of trace parsing, feature extraction,
and discipline stability computations as described in the paper
"When Outcome Looks Right But Discipline Fails: Trace-Based Evaluation
Under Hidden Competitor State" (arXiv:2605.18580v1).

The tests use fully synthetic data and do not depend on any pre-existing
Tianshou modules.  A minimal trace_parser module is mocked/inlined here for
self-containment; in production the real module would be imported.
"""

import csv
import io
import json
from datetime import datetime, timezone
from typing import Any, Dict, List

import pytest

# ---------------------------------------------------------------------------
# Minimal inline implementation of trace_parser (production version would be
# a separate module, but we keep everything in one file to satisfy the
# "no pre-existing module" constraint of this test pack).
# ---------------------------------------------------------------------------

# ---------- Constants ----------
TIMESTAMP_KEY = "timestamp"
ACTION_KEY = "action"
COMPETITOR_KEY = "competitor_present"
OUTCOME_KEY = "outcome"
RULE_ADHERENCE_KEY = "rule_adherence"
DEVIATION_KEY = "deviation"

# ---------- Valid values ----------
VALID_ACTIONS = {"up", "down", "left", "right", "stay"}
VALID_RULE_ADHERENCE = {"compliant", "non_compliant", "uncertain"}
VALID_OUTCOMES = {"win", "lose", "draw"}


def _validate_raw_trace(trace: Dict[str, Any]) -> None:
    """Raise ValueError if a raw trace record has invalid structure."""
    for key in (TIMESTAMP_KEY, ACTION_KEY, COMPETITOR_KEY, OUTCOME_KEY):
        if key not in trace:
            raise ValueError(f"Missing required key: {key}")
    if trace[ACTION_KEY] not in VALID_ACTIONS:
        raise ValueError(f"Invalid action: {trace[ACTION_KEY]}")
    if trace[OUTCOME_KEY] not in VALID_OUTCOMES:
        raise ValueError(f"Invalid outcome: {trace[OUTCOME_KEY]}")
    try:
        datetime.fromisoformat(trace[TIMESTAMP_KEY])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid timestamp: {trace[TIMESTAMP_KEY]}") from exc


def parse_raw_traces(
    raw_data: str,
    input_format: str = "json",
) -> List[Dict[str, Any]]:
    """Parse raw traces (JSON or CSV string) into structured records.

    Parameters
    ----------
    raw_data : str
        Raw trace data as a JSON array of objects (input_format='json')
        or a CSV string (input_format='csv').
    input_format : str, optional
        'json' or 'csv' (default 'json').

    Returns
    -------
    List[Dict[str, Any]]
        List of validated parsed records.

    Raises
    ------
    ValueError
        If input_format is unsupported or individual traces are invalid.
    """
    if input_format == "json":
        traces = json.loads(raw_data)
    elif input_format == "csv":
        reader = csv.DictReader(io.StringIO(raw_data))
        traces = list(reader)
    else:
        raise ValueError(f"Unsupported input format: {input_format}")

    parsed = []
    for trace in traces:
        _validate_raw_trace(trace)
        # Ensure boolean conversion for competitor field
        trace[COMPETITOR_KEY] = (
            trace[COMPETITOR_KEY]
            if isinstance(trace[COMPETITOR_KEY], bool)
            else trace[COMPETITOR_KEY].strip().lower() == "true"
        )
        parsed.append(trace)
    return parsed


def extract_trace_features(
    parsed_traces: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Extract per-step features from parsed traces.

    Features computed:
      - action_consistency: 1 if action equals previous step's action else 0
      - competitor_effect: 1 if competitor present and outcome is 'lose' else 0
      - rule_adherence: taken from raw data (default 'uncertain')
      - deviation: absolute numeric deviation from a reference (here: dummy 0)

    Parameters
    ----------
    parsed_traces : List[Dict[str, Any]]
        Parsed trace records from `parse_raw_traces`.

    Returns
    -------
    List[Dict[str, Any]]
        Feature vectors with at least the keys above.
    """
    features = []
    prev_action = None
    for trace in parsed_traces:
        action = trace[ACTION_KEY]
        consistency = 1 if (prev_action is not None and action == prev_action) else 0
        prev_action = action

        competitor = trace[COMPETITOR_KEY]
        outcome = trace[OUTCOME_KEY]
        competitor_effect = 1 if (competitor and outcome == "lose") else 0

        adherence = trace.get(RULE_ADHERENCE_KEY, "uncertain")
        if adherence not in VALID_RULE_ADHERENCE:
            adherence = "uncertain"

        # Dummy deviation (in production this would be computed from a reference)
        deviation = 0.0

        features.append(
            {
                TIMESTAMP_KEY: trace[TIMESTAMP_KEY],
                ACTION_KEY: action,
                COMPETITOR_KEY: competitor,
                OUTCOME_KEY: outcome,
                "action_consistency": consistency,
                "competitor_effect": competitor_effect,
                RULE_ADHERENCE_KEY: adherence,
                DEVIATION_KEY: deviation,
            }
        )
    return features


def compute_discipline_metrics(
    features: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute paper‑inspired discipline stability metrics.

    Returns a dict with keys:
      - behavioral_consistency: fraction of steps with action consistency = 1
      - hidden_state_misalignment: fraction of steps where competitor_effect = 1
      - compliance_score: fraction of steps where rule_adherence = 'compliant'
      - economic_vs_behavioral_gap: abs(win_rate - compliance_score)

    Parameters
    ----------
    features : List[Dict[str, Any]]
        Feature vectors from `extract_trace_features`.

    Returns
    -------
    Dict[str, float]
        Computed discipline metrics.
    """
    n = len(features)
    if n == 0:
        return {
            "behavioral_consistency": 0.0,
            "hidden_state_misalignment": 0.0,
            "compliance_score": 0.0,
            "economic_vs_behavioral_gap": 0.0,
        }

    consistency_count = sum(f["action_consistency"] for f in features)
    misalignment_count = sum(f["competitor_effect"] for f in features)
    compliant_count = sum(1 for f in features if f[RULE_ADHERENCE_KEY] == "compliant")
    win_count = sum(1 for f in features if f[OUTCOME_KEY] == "win")

    behavioral_consistency = consistency_count / n
    hidden_state_misalignment = misalignment_count / n
    compliance_score = compliant_count / n
    win_rate = win_count / n
    gap = abs(win_rate - compliance_score)

    return {
        "behavioral_consistency": round(behavioral_consistency, 4),
        "hidden_state_misalignment": round(hidden_state_misalignment, 4),
        "compliance_score": round(compliance_score, 4),
        "economic_vs_behavioral_gap": round(gap, 4),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_json_traces() -> str:
    """Synthetic trace data in JSON format (3 steps)."""
    return json.dumps(
        [
            {
                "timestamp": "2026-05-30T10:00:00+00:00",
                "action": "up",
                "competitor_present": False,
                "outcome": "win",
                "rule_adherence": "compliant",
            },
            {
                "timestamp": "2026-05-30T10:00:01+00:00",
                "action": "left",
                "competitor_present": True,
                "outcome": "lose",
                "rule_adherence": "non_compliant",
            },
            {
                "timestamp": "2026-05-30T10:00:02+00:00",
                "action": "left",
                "competitor_present": False,
                "outcome": "draw",
                "rule_adherence": "uncertain",
            },
        ]
    )


@pytest.fixture
def sample_csv_traces() -> str:
    """Synthetic trace data in CSV format (3 steps)."""
    return (
        "timestamp,action,competitor_present,outcome,rule_adherence\n"
        "2026-05-30T10:00:00+00:00,up,false,win,compliant\n"
        "2026-05-30T10:00:01+00:00,left,true,lose,non_compliant\n"
        "2026-05-30T10:00:02+00:00,left,false,draw,uncertain\n"
    )


class TestParseRawTraces:
    """Tests for parse_raw_traces."""

    def test_json_input(self, sample_json_traces: str) -> None:
        """Parse valid JSON returns list of dicts with correct keys."""
        parsed = parse_raw_traces(sample_json_traces, input_format="json")
        assert len(parsed) == 3
        for record in parsed:
            assert TIMESTAMP_KEY in record
            assert ACTION_KEY in record
            assert COMPETITOR_KEY in record
            assert OUTCOME_KEY in record

    def test_csv_input(self, sample_csv_traces: str) -> None:
        """Parse valid CSV returns list of dicts with converted fields."""
        parsed = parse_raw_traces(sample_csv_traces, input_format="csv")
        assert len(parsed) == 3
        # CSV reader returns strings; parse_raw_traces will convert competitor to bool
        assert isinstance(parsed[0][COMPETITOR_KEY], bool)
        assert parsed[0][COMPETITOR_KEY] is False
        assert parsed[1][COMPETITOR_KEY] is True

    def test_invalid_format(self) -> None:
        """Unsupported format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported input format"):
            parse_raw_traces("[]", input_format="xml")

    def test_missing_key_raises(self) -> None:
        """Trace missing required key raises ValueError."""
        bad_json = json.dumps([{"timestamp": "now", "action": "up"}])
        with pytest.raises(ValueError, match="Missing required key"):
            parse_raw_traces(bad_json)

    def test_invalid_action_raises(self) -> None:
        """Invalid action raises ValueError."""
        bad_json = json.dumps(
            [
                {
                    "timestamp": "2026-05-30T10:00:00+00:00",
                    "action": "teleport",
                    "competitor_present": False,
                    "outcome": "win",
                }
            ]
        )
        with pytest.raises(ValueError, match="Invalid action"):
            parse_raw_traces(bad_json)

    def test_empty_input(self) -> None:
        """Empty array returns empty list."""
        parsed = parse_raw_traces("[]")
        assert parsed == []


class TestExtractTraceFeatures:
    """Tests for extract_trace_features."""

    def test_basic_features(self, sample_json_traces: str) -> None:
        """Check action_consistency and competitor_effect values."""
        parsed = parse_raw_traces(sample_json_traces)
        features = extract_trace_features(parsed)
        assert len(features) == 3

        # first step: no previous action → consistency = 0
        assert features[0]["action_consistency"] == 0
        # no competitor and win → competitor_effect = 0
        assert features[0]["competitor_effect"] == 0

        # second step: action "left" vs previous "up" → 0
        assert features[1]["action_consistency"] == 0
        # competitor + lose → 1
        assert features[1]["competitor_effect"] == 1

        # third step: action "left" vs previous "left" → 1
        assert features[2]["action_consistency"] == 1
        # no competitor → 0
        assert features[2]["competitor_effect"] == 0

    def test_missing_rule_adherence_defaults_uncertain(self) -> None:
        """If rule_adherence is missing, it defaults to 'uncertain'."""
        trace = json.dumps(
            [
                {
                    "timestamp": "2026-05-30T10:00:00+00:00",
                    "action": "right",
                    "competitor_present": False,
                    "outcome": "win",
                }
            ]
        )
        parsed = parse_raw_traces(trace)
        features = extract_trace_features(parsed)
        assert features[0][RULE_ADHERENCE_KEY] == "uncertain"

    def test_all_features_present(self) -> None:
        """Output dict contains all expected keys."""
        trace = json.dumps(
            [
                {
                    "timestamp": "2026-05-30T10:00:00+00:00",
                    "action": "up",
                    "competitor_present": True,
                    "outcome": "lose",
                    "rule_adherence": "compliant",
                }
            ]
        )
        parsed = parse_raw_traces(trace)
        features = extract_trace_features(parsed)
        expected_keys = {
            TIMESTAMP_KEY,
            ACTION_KEY,
            COMPETITOR_KEY,
            OUTCOME_KEY,
            "action_consistency",
            "competitor_effect",
            RULE_ADHERENCE_KEY,
            DEVIATION_KEY,
        }
        assert set(features[0].keys()) == expected_keys


class TestComputeDisciplineMetrics:
    """Tests for compute_discipline_metrics."""

    def test_known_values(self, sample_json_traces: str) -> None:
        """Validate metrics from the sample data (3 steps)."""
        parsed = parse_raw_traces(sample_json_traces)
        features = extract_trace_features(parsed)
        metrics = compute_discipline_metrics(features)

        # consistency: only step 3 is consistent → 1/3 ≈ 0.3333
        assert metrics["behavioral_consistency"] == pytest.approx(1 / 3, abs=1e-4)
        # misalignment: only step 2 has competitor+lose → 1/3
        assert metrics["hidden_state_misalignment"] == pytest.approx(1 / 3, abs=1e-4)
        # compliance: only step 1 is compliant → 1/3
        assert metrics["compliance_score"] == pytest.approx(1 / 3, abs=1e-4)
        # win rate: only step 1 is win → 1/3
        # gap = |win_rate - compliance| = |0.3333 - 0.3333| = 0.0
        assert metrics["economic_vs_behavioral_gap"] == pytest.approx(0.0, abs=1e-4)

    def test_empty_features(self) -> None:
        """Empty input returns all zeros."""
        metrics = compute_discipline_metrics([])
        assert all(v == 0.0 for v in metrics.values())

    def test_all_compliant_and_win(self) -> None:
        """When all steps are compliant and win, gap is 0."""
        traces = [
            {
                TIMESTAMP_KEY: "2026-05-30T10:00:00+00:00",
                ACTION_KEY: "up",
                COMPETITOR_KEY: False,
                OUTCOME_KEY: "win",
                RULE_ADHERENCE_KEY: "compliant",
            }
            for _ in range(5)
        ]
        features = extract_trace_features(traces)
        metrics = compute_discipline_metrics(features)
        assert metrics["compliance_score"] == 1.0
        assert metrics["economic_vs_behavioral_gap"] == 0.0

    def test_all_non_compliant_win_still_gap(self) -> None:
        """When wins happen despite non-compliance, gap > 0."""
        traces = [
            {
                TIMESTAMP_KEY: f"2026-05-30T10:00:0{i}+00:00",
                ACTION_KEY: "up",
                COMPETITOR_KEY: False,
                OUTCOME_KEY: "win",
                RULE_ADHERENCE_KEY: "non_compliant",
            }
            for i in range(3)
        ]
        features = extract_trace_features(traces)
        metrics = compute_discipline_metrics(features)
        assert metrics["compliance_score"] == 0.0
        assert metrics["economic_vs_behavioral_gap"] == 1.0


class TestIntegration:
    """Integration tests: raw → parse → features → metrics."""

    def test_full_pipeline_json(self, sample_json_traces: str) -> None:
        """End-to-end trace parsing and discipline computation."""
        parsed = parse_raw_traces(sample_json_traces)
        features = extract_trace_features(parsed)
        metrics = compute_discipline_metrics(features)
        # basic sanity: metrics should be floats between 0 and 1
        for key, value in metrics.items():
            assert isinstance(value, float), f"{key} is not float"
            assert 0.0 <= value <= 1.0, f"{key} out of bounds: {value}"

    def test_full_pipeline_csv(self, sample_csv_traces: str) -> None:
        """Same as above with CSV input."""
        parsed = parse_raw_traces(sample_csv_traces, input_format="csv")
        features = extract_trace_features(parsed)
        metrics = compute_discipline_metrics(features)
        for value in metrics.values():
            assert isinstance(value, float)
            assert 0.0 <= value <= 1.0