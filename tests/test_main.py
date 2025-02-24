"""This file runs the test for the main application."""

import pytest
from src.main import MainApp


class TestMainApp:
    """Test class for testing the current application for any
    errors or bugs."""

    """Value Section."""

    @pytest.mark.parametrize(
        """ModelName, Creativity, Prompt, SummaryPrompt,
        expected_Main""",
        [
            (
                "llama3.1:8b",
                1,
                "This is a test prompt.",
                "This is a summary test prompt.",
                True,
            ),
        ],
    )
    def test_MainApp(self, ModelName, Creativity, Prompt, SummaryPrompt, expected_Main):
        """Test the main application."""
        self.MainApp = MainApp(ModelName, Creativity, Prompt, SummaryPrompt)
        assert (self.MainApp is not None) == expected_Main
