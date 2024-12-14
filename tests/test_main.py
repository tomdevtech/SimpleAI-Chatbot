"""This file runs the test for the main application."""

import pytest
from src.main import MainApp


class TestMainApp:
    """Test class for testing the current application for any
    errors or bugs."""

    """Value Section."""

    @pytest.mark.parametrize(
        "expected_Main",
        [
            (True),
        ],
    )
    def test_MainApp(self, expected_Main):
        """Test the main application."""
        self.MainApp = MainApp()
        assert (self.MainApp is not None) == expected_Main
