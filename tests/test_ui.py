"""This file runs the tests for the UI."""

import pytest
from src.ai import AIAssistant
from src.ui import StreamlitUI


class TestUI:
    """Test class for testing the current UI for any errors or bugs."""

    """Value Section."""

    @pytest.mark.parametrize(
        "ModelName, Context, Creativity, expected_AIAssistant, expected_UI",
        [
            (
                "llama3.2",
                """You are a smart assistant called Bob and you know everything
            about computer science.""",
                1,
                True,
                True,
            ),
        ],
    )
    def test_UI(
        self, ModelName, Context, Creativity, expected_AIAssistant, expected_UI
    ):
        """Initialize the UI."""
        self.AIAssistant = AIAssistant(ModelName, Context, Creativity)
        self.UI = StreamlitUI(self.AIAssistant)
        assert (self.AIAssistant is not None) == expected_AIAssistant
        assert (self.UI is not None) == expected_UI
