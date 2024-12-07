"""This file provides the UI for the AI model."""

from ai import AIAssistant
import streamlit as st


class StreamlitUI:
    """Class for managing the Streamlit user interface."""

    def __init__(self, AIAssistant: AIAssistant):
        """Initialization."""
        self.AIAssistant = AIAssistant
        self.ChatHistory = ""

    def Run(self):
        """Run the Streamlit UI."""
        st.title("AI Assistant")
        st.write("Ask Bob everything!")

        # User Input
        UserInput = st.text_input("Your question:")
        if st.button("Send Question"):
            if UserInput:
                Response = self.AIAssistant.GetResponse(UserInput)
                self.ChatHistory += f"User: {UserInput}\nAI: {Response}\n\n"
                st.write(Response)

        # Show Conversation History
        st.text_area("Chat History", self.ChatHistory, height=300)
