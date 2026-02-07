from unittest.mock import patch

import pytest

from energy_parser.analyzer import STANDARD_GRANULARITIES
from energy_parser.cli import prompt_granularity_selection


# ── STANDARD_GRANULARITIES constant ─────────────────────────────────────────


class TestStandardGranularitiesConstant:
    def test_has_four_entries(self):
        assert len(STANDARD_GRANULARITIES) == 4

    @pytest.mark.parametrize("idx, expected_label, expected_minutes, expected_hours", [
        (0, "10 minutes", 10, 10 / 60),
        (1, "15 minutes", 15, 15 / 60),
        (2, "30 minutes", 30, 30 / 60),
        (3, "60 minutes (1 hour)", 60, 1.0),
    ])
    def test_entry_values(self, idx, expected_label, expected_minutes, expected_hours):
        label, minutes, hours = STANDARD_GRANULARITIES[idx]
        assert label == expected_label
        assert minutes == expected_minutes
        assert hours == pytest.approx(expected_hours)


# ── prompt_granularity_selection ─────────────────────────────────────────────


class TestPromptGranularitySelection:
    @patch("energy_parser.cli.Prompt.ask", return_value="1")
    def test_choice_1_returns_10min(self, mock_ask):
        label, hours = prompt_granularity_selection()
        assert label == "10 minutes"
        assert hours == pytest.approx(10 / 60)

    @patch("energy_parser.cli.Prompt.ask", return_value="2")
    def test_choice_2_returns_15min(self, mock_ask):
        label, hours = prompt_granularity_selection()
        assert label == "15 minutes"
        assert hours == pytest.approx(15 / 60)

    @patch("energy_parser.cli.Prompt.ask", return_value="3")
    def test_choice_3_returns_30min(self, mock_ask):
        label, hours = prompt_granularity_selection()
        assert label == "30 minutes"
        assert hours == pytest.approx(30 / 60)

    @patch("energy_parser.cli.Prompt.ask", return_value="4")
    def test_choice_4_returns_60min(self, mock_ask):
        label, hours = prompt_granularity_selection()
        assert label == "60 minutes (1 hour)"
        assert hours == pytest.approx(1.0)

    @patch("energy_parser.cli.Prompt.ask", return_value="2")
    def test_with_detected_info(self, mock_ask):
        """Passing detected_info should not change return value."""
        detected = {
            "label": "unknown",
            "confidence": 0.3,
            "consistency": 0.2,
        }
        label, hours = prompt_granularity_selection(detected_info=detected)
        assert label == "15 minutes"
        assert hours == pytest.approx(15 / 60)
