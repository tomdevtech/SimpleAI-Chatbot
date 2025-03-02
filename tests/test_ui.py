"""This file runs the tests for the UI."""

import pytest
from src.ai import AIAssistant
from src.ui import StreamlitUI


class TestUI:
    """Test class for testing the current UI for any errors or bugs."""

    """Value Section."""

    @pytest.mark.parametrize(
        """ expected_UI""",
        [
            (
                True,
            ),
        ],
    )
    def test_UI(
        self,
        expected_UI,
    ):
        """Initialize the UI."""
        self.UI = StreamlitUI()
        assert (self.UI is not None) == expected_UI
