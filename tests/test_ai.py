"""This file runs the test for the AI model."""

import pytest
from src.ai import AIAssistant


class TestAI:
    """Test class for testing the current AI for any errors or bugs."""

    """Value Section."""

    @pytest.mark.parametrize(
        "ModelName, Creativity, Prompt, SummaryPrompt, FileTypes, expected_AIAssistant",
        [
            (
                "llama3.1:8b",
                1,
                "This is a test prompt.",
                "This is a summary test prompt.",
                [".py", ".js", ".java", ".md", ".txt"],
            ),
        ],
    )
    def test_AI(
        self,
        ModelName,
        Creativity,
        Prompt,
        SummaryPrompt,
        FileTypes,
        expected_AIAssistant,
    ):
        """Initialize the AI."""
        self.AIAssistant = AIAssistant(
            ModelName, Creativity, Prompt, SummaryPrompt, FileTypes
        )
        assert (self.AIAssistant is not None) == expected_AIAssistant
