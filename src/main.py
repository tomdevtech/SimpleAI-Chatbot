"""This file is bringing all components together 
and is running the actual application."""

from ai import AIAssistant
from ui import GradioUI


class MainApp:
    """Main Application class to combine AI Assistant and Gradio UI."""

    def __init__(self):
        """Initialize the application."""
        self.AIAssistant = AIAssistant(
            "llama3.2",
            "You are a smart assistant called Bob and you know everything about computer science.",
            1.0
        )
        self.UI = GradioUI(self.AIAssistant)

    def Run(self):
        """Run the application."""
        self.UI.Launch()


if __name__ == "__main__":
    App = MainApp()
    App.Run()
