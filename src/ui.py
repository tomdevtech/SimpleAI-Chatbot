"""This file provides the UI for the AI model."""

from ai import AIAssistant
import gradio as gr


class GradioUI:
    """Class for managing Gradio-based user interface."""

    def __init__(self, AIAssistant: AIAssistant):
        """Initialize Gradio UI with an AI Assistant instance."""
        self.AIAssistant = AIAssistant

    def Chat(self, UserInput):
        """
        Handle user input and return AI response.

        Args:
            UserInput (str): User's input question.

        Returns:
            str: AI's response.
        """
        return self.AIAssistant.GetResponse(UserInput)

    def BuildInterface(self):
        """
        Build the Gradio interface.

        Returns:
            gr.Interface: Configured Gradio interface.
        """
        return gr.Interface(
            fn=self.Chat,
            inputs="text",
            outputs="text",
            title="AI Assistant",
            description="Ask Bob everything!",
        )

    def Launch(self):
        """Launch the Gradio app."""
        self.BuildInterface().launch()
