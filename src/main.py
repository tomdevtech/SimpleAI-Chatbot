"""This file runs the ai model and lets you interact with it."""

from ai import AIAssistant
from ui import StreamlitUI


class MainApp:
    """Main Application class to combine AI Assistant and UI."""

    def __init__(self, ModelName, Creativity, PromptTemplate, 
        SummaryPromptTemplate):
        """Initialize the application."""
        self.AIAssistant = AIAssistant(
            ModelName, Creativity, PromptTemplate, SummaryPromptTemplate
        )
        self.UI = StreamlitUI(self.AIAssistant)

    def Run(self):
        """Run the application."""
        self.UI.Run()


if __name__ == "__main__":
    App = MainApp("llama3.1:8b", 1.0, "", "")
    App.Run()
