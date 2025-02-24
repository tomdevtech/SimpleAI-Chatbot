"""This file runs the tests for the UI."""

import pytest
from src.ai import AIAssistant
from src.ui import StreamlitUI


class TestUI:
    """Test class for testing the current UI for any errors or bugs."""

    """Value Section."""

    @pytest.mark.parametrize(
        """ModelName, Creativity, Prompt, SummaryPrompt, FileTypes,
        expected_AIAssistant, expected_UI""",
        [
            (
                "llama3.1:8b",
                1,
                "This is a test prompt.",
                "This is a summary test prompt.",
                [".py", ".js", ".java", ".md", ".txt"],
                True,
                True,
            ),
        ],
    )
    def test_UI(
        self,
        ModelName,
        Creativity,
        Prompt,
        SummaryPrompt,
        FileTypes,
        expected_AIAssistant,
        expected_UI,
    ):
        """Initialize the UI."""
        self.AIAssistant = AIAssistant(
            ModelName, Creativity, Prompt, SummaryPrompt, FileTypes
        )
        self.UI = StreamlitUI(self.AIAssistant)
        assert (self.AIAssistant is not None) == expected_AIAssistant
        assert (self.UI is not None) == expected_UI
