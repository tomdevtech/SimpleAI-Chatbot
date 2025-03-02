"""This file runs the AI model and lets you interact with it."""

import streamlit as st
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
        if "AI_Assistant" not in st.session_state:
            st.session_state.AI_Assistant = AIAssistant(
                model_name,
                creativity,
                prompt_template,
                summary_prompt_template
            )
            st.session_state.AI_Assistant.ManageOllama()
            print("Starting AI...")

        self.ai_assistant = st.session_state.AI_Assistant
        self.ui = StreamlitUI()

    def Run(self):
        """Run the application."""
        self.ui.Run()
        print("Starting UI...")


if __name__ == "__main__":
    if "MainApp" not in st.session_state:
        st.session_state.MainApp = MainApp("llama3.1:8b", 1.0, "", "")

    st.session_state.MainApp.Run()
