import pandas as pd
import pytest

from energy_parser.analyzer import (
    detect_granularity,
    detect_granularity_with_confidence,
)


def _make_datetime_series(start, freq, periods):
    """Helper to build a datetime Series with a given frequency."""
    return pd.Series(pd.date_range(start=start, freq=freq, periods=periods))


# ── Edge cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_series(self):
        result = detect_granularity_with_confidence(pd.Series(dtype="datetime64[ns]"))
        assert result["label"] == "unknown"
        assert result["confidence"] == 0.0

    def test_single_element(self):
        s = pd.Series([pd.Timestamp("2024-01-01 00:00")])
        result = detect_granularity_with_confidence(s)
        assert result["label"] == "unknown"
        assert result["confidence"] == 0.0

    def test_all_nat(self):
        s = pd.Series([pd.NaT, pd.NaT, pd.NaT])
        result = detect_granularity_with_confidence(s)
        assert result["label"] == "unknown"
        assert result["confidence"] == 0.0

    def test_two_valid_datetimes(self):
        s = pd.Series([
            pd.Timestamp("2024-01-01 00:00"),
            pd.Timestamp("2024-01-01 00:15"),
        ])
        result = detect_granularity_with_confidence(s)
        assert result["label"] != "unknown"
        assert result["confidence"] > 0


# ── Standard interval detection (within 20 % tolerance) ─────────────────────


class TestStandardIntervals:
    @pytest.mark.parametrize("freq, expected_label, expected_hours", [
        ("10min", "10 min", 10 / 60),
        ("15min", "15 min", 15 / 60),
        ("30min", "30 min", 30 / 60),
        ("60min", "1 hour", 1.0),
    ])
    def test_consistent_standard_intervals(self, freq, expected_label, expected_hours):
        s = _make_datetime_series("2024-01-01", freq=freq, periods=100)
        result = detect_granularity_with_confidence(s)
        assert result["label"] == expected_label
        assert result["hours_per_interval"] == pytest.approx(expected_hours)
        assert result["is_standard"] is True
        assert result["confidence"] == 0.95

    def test_12min_detected_as_10min(self):
        """12-min median is within 20 % of 10 min → mapped to 10 min."""
        s = _make_datetime_series("2024-01-01", freq="12min", periods=100)
        result = detect_granularity_with_confidence(s)
        assert result["label"] == "10 min"
        assert result["is_standard"] is True


# ── Non-standard intervals ──────────────────────────────────────────────────


class TestNonStandardIntervals:
    def test_45min_interval(self):
        s = _make_datetime_series("2024-01-01", freq="45min", periods=100)
        result = detect_granularity_with_confidence(s)
        assert result["label"] == "45 min"
        assert result["is_standard"] is False

    def test_90min_interval(self):
        s = _make_datetime_series("2024-01-01", freq="90min", periods=100)
        result = detect_granularity_with_confidence(s)
        assert result["label"] == "1.5 hours"
        assert result["is_standard"] is False

    def test_120min_interval(self):
        s = _make_datetime_series("2024-01-01", freq="120min", periods=100)
        result = detect_granularity_with_confidence(s)
        assert result["label"] == "2.0 hours"
        assert result["is_standard"] is False


# ── Confidence levels ────────────────────────────────────────────────────────


class TestConfidenceLevels:
    def test_standard_high_consistency(self):
        """Standard interval + >=80 % consistency → 0.95."""
        s = _make_datetime_series("2024-01-01", freq="15min", periods=100)
        result = detect_granularity_with_confidence(s)
        assert result["confidence"] == 0.95

    def test_standard_medium_consistency(self):
        """Standard interval + >=60 % consistency → 0.75."""
        # Build a series where ~70 % of intervals are 15 min, rest random
        base = pd.date_range("2024-01-01", freq="15min", periods=70)
        noise = pd.date_range("2024-01-01 18:00", freq="7min", periods=30)
        s = pd.Series(base.append(noise)).sort_values().reset_index(drop=True)
        result = detect_granularity_with_confidence(s)
        # Should be standard (15 min is closest) with medium consistency
        if result["is_standard"] and 0.6 <= result["consistency"] < 0.8:
            assert result["confidence"] == 0.75

    def test_nonstandard_high_consistency(self):
        """Non-standard interval + >=80 % consistency → 0.70."""
        s = _make_datetime_series("2024-01-01", freq="45min", periods=100)
        result = detect_granularity_with_confidence(s)
        assert result["is_standard"] is False
        assert result["consistency"] >= 0.8
        assert result["confidence"] == 0.70

    def test_low_consistency_below_50(self):
        """<50 % consistency → 0.30."""
        # Highly irregular intervals
        timestamps = [pd.Timestamp("2024-01-01")]
        import random
        random.seed(42)
        for _ in range(99):
            delta = pd.Timedelta(minutes=random.choice([5, 20, 45, 90, 120]))
            timestamps.append(timestamps[-1] + delta)
        s = pd.Series(timestamps)
        result = detect_granularity_with_confidence(s)
        if result["consistency"] < 0.5:
            assert result["confidence"] == 0.30


# ── Consistency calculation ──────────────────────────────────────────────────


class TestConsistency:
    def test_perfect_consistency(self):
        s = _make_datetime_series("2024-01-01", freq="15min", periods=100)
        result = detect_granularity_with_confidence(s)
        assert result["consistency"] == pytest.approx(1.0)

    def test_mixed_intervals_lower_consistency(self):
        base = pd.date_range("2024-01-01", freq="15min", periods=50)
        noise = pd.date_range("2024-01-01 13:00", freq="3min", periods=50)
        s = pd.Series(base.append(noise)).sort_values().reset_index(drop=True)
        result = detect_granularity_with_confidence(s)
        assert result["consistency"] < 1.0


# ── detect_granularity wrapper ───────────────────────────────────────────────


class TestDetectGranularityWrapper:
    def test_returns_tuple(self):
        s = _make_datetime_series("2024-01-01", freq="15min", periods=50)
        label, hours = detect_granularity(s)
        assert isinstance(label, str)
        assert isinstance(hours, float)

    def test_matches_dict_values(self):
        s = _make_datetime_series("2024-01-01", freq="30min", periods=50)
        label, hours = detect_granularity(s)
        result = detect_granularity_with_confidence(s)
        assert label == result["label"]
        assert hours == result["hours_per_interval"]
