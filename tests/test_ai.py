"""This file runs the test for the AI model."""

import pytest
from src.ai import AIAssistant


class TestAI:
    """Test class for testing the current AI for any errors or bugs."""

    """Value Section."""

    @pytest.mark.parametrize(
        "ModelName, Context, Creativity, expected_AIAssistant",
        [
            (
                "llama3.2",
                """You are a smart assistant called Bob and you know everything
            about computer science.""",
                1,
                True,
            ),
        ],
    )
    def test_AI(self, ModelName, Context, Creativity, expected_AIAssistant):
        """Initialize the AI."""
        self.AIAssistant = AIAssistant(ModelName, Context, Creativity)
        assert (self.AIAssistant is not None) == expected_AIAssistant
