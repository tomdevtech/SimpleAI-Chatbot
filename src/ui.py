"""This file provides the UI for the AI model."""

from ai import AIAssistant
import streamlit as st


class StreamlitUI:
    """Class for managing the Streamlit user interface."""

    def __init__(self, AIAssistant: AIAssistant):
        """Initialize Process."""
        self.AIAssistant = AIAssistant
        self.ChatHistory = ""
        self.RepoPath = ""

    def Run(self):
        """Run the Streamlit UI."""
        st.title("AI Assistant")
        st.write(
            """Welcome to the AI Repo Summarizer!\n
            Please enter the repository path first!"""
        )

        # Path Input
        self.RepoPath = st.text_input("Set Repository Path:", self.RepoPath)
        if st.button("Set Path"):
            if self.RepoPath:
                st.write(f"Repository path set to: {self.RepoPath}")
                self.AIAssistant.set_repo_path(self.ChatHistory)
                st.write("Analyzing repository...")
                result = self.AIAssistant.analyze_repository()
                st.write(result)

        # User Input
        UserInput = st.text_input("Your question:")
        if st.button("Send Question"):
            if UserInput:
                Response = self.AIAssistant.ask_question(UserInput)
                self.ChatHistory += f"User: {UserInput}\nAI: {Response}\n\n"
                st.write(Response)

        # Show Conversation History
        st.text_area("Chat History", self.ChatHistory, height=300)
