"""This file provides the UI for the AI model."""

import streamlit as st


class StreamlitUI:
    """Class for managing the Streamlit user interface."""

    def __init__(self):
        """Initialize Process."""
        if "ChatHistory" not in st.session_state:
            st.session_state.ChatHistory = ""
        if "RepoPath" not in st.session_state:
            st.session_state.RepoPath = ""

    def Run(self):
        """Run the Streamlit UI."""
        st.set_page_config("AI Assistant")
        st.title("AI Assistant")
        st.write(
            """Welcome to the AI Repo Summarizer!\n
            Please enter the repository path first!"""
        )

        # Path Input
        st.session_state.RepoPath = st.text_input("Set Repository Path:",
                                                  st.session_state.RepoPath)
        if st.button("Set Path"):
            if st.session_state.RepoPath:
                st.write(f"""Repository path set to: 
                         {st.session_state.RepoPath}""")
                st.session_state.AI_Assistant.SetRepoPath(
                    st.session_state.RepoPath)
                st.write("Analyzing repository...")
                result = st.session_state.AI_Assistant.AnalyzeRepository()
                st.write(result)

        # User Input
        UserInput = st.text_input("Your question:")
        if st.button("Send Question"):
            if UserInput:
                Response = st.session_state.AI_Assistant.AskQuestion(UserInput)
                st.session_state.ChatHistory += f"User: {UserInput}\nAI: {Response}\n\n"

        # Show Conversation History
        st.text_area("Chat History", st.session_state.ChatHistory, height=300)
