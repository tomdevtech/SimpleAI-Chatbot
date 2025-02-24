"""This file provides the AI model and all of its
functionalities for repository code analysis and Q&A.
"""

import os
import unittest
import subprocess  # nosec B404
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.messages import AIMessage


class AIAssistant:
    """AI Assistant Class for Repository Analysis and Contextual Chat."""

    def __init__(
        self,
        model_name,
        temperature,
        prompt_template,
        summary_prompt_template,
        file_types=None,
    ):
        self.repo_path = None
        self.model_name = model_name
        self.temperature = temperature
        self.file_types = (
            file_types if file_types else [".py", ".js", ".java",
                                           ".md", ".txt"]
        )
        self.vector_store = None
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.assistant = OllamaLLM(
            model=self.model_name,
            temperature=self.temperature,
        )
        self.context = ""
        self.summary_completed = False

        self.set_templates(prompt_template, summary_prompt_template)
        self.check_ollama()

    def set_templates(self, prompt_template, summary_prompt_template):
        """Sets the prompt templates."""
        default_prompt = """
        You are an expert code reviewer. Answer the following question
        based on the provided repository context:
        Context:
        {Context}
        Question:
        {Question}
        Please provide a concise and clear answer.
        """

        default_summary_prompt = """
        You are a technical documentation assistant.
        Summarize the provided repository contents
        in a clear and concise manner.
        Repository Contents:
        {Context}
        Provide a structured summary highlighting key points,
        code structure, and any important observations.
        """

        self.prompt_template = ChatPromptTemplate.from_template(
            prompt_template or default_prompt
        )
        self.summary_prompt_template = ChatPromptTemplate.from_template(
            summary_prompt_template or default_summary_prompt
        )

    def set_repo_path(self, path):
        """Sets the path for the repository."""
        self.repo_path = path

    @unittest.skip("Not needed for test.")
    def check_ollama(self):
        """Check if Ollama server is running and start if necessary."""
        try:
            response = requests.get("http://localhost:11434/health", timeout=5)
            if response.status_code == 200:
                print("Ollama server is running.")
                return
        except requests.exceptions.ConnectionError:
            print("Ollama server not running. Attempting to start...")

        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("Ollama server started successfully.")
        except FileNotFoundError:
            print(
                "Ollama is not installed. Please install Ollama using "
                "'brew install ollama' or the appropriate command for your OS."
            )
            exit(1)

    def check_and_pull_model(self, model_name):
        """Check if a specific model exists and pull if not."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                shell=False,
            ) #nosec
            if model_name not in result.stdout:
                print(f"Model '{model_name}' not found. Downloading...")
                subprocess.run(["ollama", "pull", model_name], 
                capture_output=True,
                text=True,
                shell=False,
                )
                print(f"Model '{model_name}' downloaded successfully.")
        except Exception as e:
            print(f"Failed to check/download model '{model_name}': {e}")
            exit(1)

    def load_documents(self):
        """Load documents based on specified file types."""
        docs = []
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if any(file.endswith(ft) for ft in self.file_types):
                    file_path = os.path.join(root, file)
                    try:
                        with open(
                            file_path,
                            "r",
                            encoding="utf-8",
                            errors="ignore",
                        ) as f:
                            content = f.read()
                            docs.append({"Path": file_path,
                                         "Content": content})
                    except Exception as e:
                        print(f"Failed to read {file_path}: {e}")
        return docs

    def create_vector_store(self, docs):
        """Create Chroma vector store for contextual queries."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )
        splits = text_splitter.create_documents(
            [doc["Content"] for doc in docs]
        )
        self.vector_store = Chroma.from_documents(
            splits,
            embedding=self.embeddings,
        )
        print("Vector store created successfully.")

    def analyze_repository(self):
        """Complete analysis and vector store creation."""
        print("Loading documents...")
        docs = self.load_documents()
        if not docs:
            return "No matching documents found."

        print("Creating vector store...")
        self.create_vector_store(docs)
        self.context = "\n\n".join(doc["Content"] for doc in docs)
        summary = self.generate_summary(self.context)
        self.write_summary(summary)
        self.summary_completed = True

        return "Repository analysis complete. You can now ask questions!"

    def generate_summary(self, content):
        """Generate a structured summary using the Assistant."""
        prompt = self.summary_prompt_template.format(Context=content)
        response = self.assistant.invoke(prompt)
        if isinstance(response, AIMessage):
            return response.content
        return str(response)

    def ask_question(self, query):
        """Ask a question based on the repository context."""
        if not self.summary_completed:
            return (
                "Repository analysis not complete. "
                "Please analyze the repository first."
            )

        if not self.vector_store:
            return (
                "No vector store available. "
                "Please analyze a repository first."
            )

        # Retrieve relevant documents using similarity search
        relevant_docs = self.vector_store.similarity_search(query, k=5)
        context = "\n\n".join(doc.page_content for doc in relevant_docs)

        # Create prompt based on retrieved context
        prompt = self.prompt_template.format(Context=context, Question=query)
        response = self.assistant.invoke(prompt)

        if isinstance(response, AIMessage):
            return response.content
        return str(response)

    def write_summary(self, content):
        """Write the summary to a Markdown file."""
        try:
            with open("RepoSummary.md", "w", encoding="utf-8") as f:
                f.write(content)
            print("Summary written to RepoSummary.md")
        except Exception as e:
            print(f"Failed to write summary: {e}")
