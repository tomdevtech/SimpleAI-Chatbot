"""This file runs the AI model and lets you interact with it."""

from ai import AIAssistant
from ui import StreamlitUI


class MainApp:
    """Main Application class to combine AI Assistant and UI."""

    def __init__(
        self,
        model_name,
        creativity,
        prompt_template,
        summary_prompt_template
    ):
        """Initialize the application."""
        self.ai_assistant = AIAssistant(
            model_name,
            creativity,
            prompt_template,
            summary_prompt_template
        )
        self.ui = StreamlitUI(self.ai_assistant)

    def run(self):
        """Run the application."""
        self.ui.Run()


if __name__ == "__main__":
    app = MainApp("llama3.1:8b", 1.0, "", "")
    app.run()
