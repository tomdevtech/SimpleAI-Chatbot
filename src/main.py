"""This file runs the ai model and lets you interact with it."""

from ai import AIAssistant
from ui import StreamlitUI


class MainApp:
    """Main Application class to combine AI Assistant and UI."""

    def __init__(self):
        """Initialize the application."""
        self.AIAssistant = AIAssistant(
            "llama3.2",
            """You are a smart assistant called Bob and you know everything
            about computer science.""",
            1.0
        )
        self.UI = StreamlitUI(self.AIAssistant)

    def Run(self):
        """Run the application."""
        self.UI.Run()


if __name__ == "__main__":
    App = MainApp()
    App.Run()
