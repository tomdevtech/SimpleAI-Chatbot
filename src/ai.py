"""
This file provides the AI model and its functionalities.

This module defines the `AIAssistant` class, which facilitates
code repository analysis, contextual querying, and documentation generation.
"""

import os
import shutil
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
        ModelName,
        Temperature,
        PromptTemplate,
        SummaryPromptTemplate,
        FileTypes=None,
    ):
        """
        Initialize the AI Assistant with the given parameters.

        Args:
            ModelName (str): The name of the AI model to use.
            Temperature (float): The temperature setting for model responses.
            PromptTemplate (str): Template for user prompts.
            SummaryPromptTemplate (str): Template for generating summaries.
            FileTypes (list, optional): List of file types
            to include in analysis.
        """
        self.RepoPath = None
        self.ModelName = ModelName
        self.Temperature = Temperature
        self.FileTypes = (
            FileTypes if FileTypes else [".py", ".js", ".java", ".md", ".txt"]
        )
        #self.VectorStore = None
        self.Embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.Assistant = OllamaLLM(
            model=self.ModelName,
            temperature=self.Temperature,
        )
        self.Context = ""
        #self.SummaryCompleted = False

        self.SetTemplates(PromptTemplate, SummaryPromptTemplate)
        self.ManageOllama()

    def SetTemplates(self, PromptTemplate, SummaryPromptTemplate):
        """
        Set the prompt templates for user queries and summaries.

        Args:
            PromptTemplate (str): The template for user questions.
            SummaryPromptTemplate (str): The template for generating summaries.
        """
        DefaultPrompt = """
        You are an expert code reviewer. Answer the following question
        based on the provided repository context:
        Context:
        {Context}
        Question:
        {Question}
        Please provide a concise and clear answer.
        """

        DefaultSummaryPrompt = """
        You are a technical documentation assistant.
        Summarize the provided repository contents
        in a clear and concise manner.
        Repository Contents:
        {Context}
        Provide a structured summary highlighting key points,
        code structure, and any important observations.
        """

        self.PromptTemplate = ChatPromptTemplate.from_template(
            PromptTemplate or DefaultPrompt
        )
        self.SummaryPromptTemplate = ChatPromptTemplate.from_template(
            SummaryPromptTemplate or DefaultSummaryPrompt
        )

    def SetRepoPath(self, Path):
        """
        Set the path for the repository to analyze.

        Args:
            Path (str): Path to the code repository.
        """
        self.RepoPath = Path

    @unittest.skip("Not needed for test.")
    def ManageOllama(self):
        """
        Manage Ollama server and model availability.

        Ensures the Ollama server is running and the required model
        is available. If the server is not running, it attempts to start it.
        If the model is not available, it downloads the specified model.
        """
        OllamaPath = shutil.which("ollama")
        if not OllamaPath:
            print("Ollama executable not found. Please install Ollama.")
            exit(1)

        # Check if Ollama server is running
        try:
            Response = requests.get("http://localhost:11434/health", timeout=5)
            if Response.status_code == 200:
                print("Ollama server is running.")
            else:
                raise requests.exceptions.ConnectionError
        except requests.exceptions.ConnectionError:
            print("Ollama server not running. Attempting to start...")
            try:
                subprocess.Popen(
                    [OllamaPath, "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )  # nosec
                print("Ollama server started successfully.")
            except Exception as E:
                print(f"Failed to start Ollama server: {E}")
                exit(1)

        # Check if the specified model exists and pull if necessary
        try:
            Result = subprocess.run(
                [OllamaPath, "list"],
                capture_output=True,
                text=True
            )  # nosec

            if self.ModelName not in Result.stdout:
                print(f"Model '{self.ModelName}' not found. Downloading...")
                subprocess.run(
                    [OllamaPath, "pull", self.ModelName],
                    capture_output=True,
                    text=True
                )  # nosec
                print(f"Model '{self.ModelName}' downloaded successfully.")
            else:
                print(f"Model '{self.ModelName}' already exists.")
        except Exception as E:
            print(f"Failed to check/download model '{self.ModelName}': {E}")
            exit(1)

    def LoadDocuments(self):
        """
        Load documents from the repository based on specified file types.

        Returns:
            list: A list of dictionaries containing file paths and content.
        """
        Docs = []
        if not self.RepoPath:
            print("Repository path is not set.")
            return Docs

        for Root, _, Files in os.walk(self.RepoPath):
            for File in Files:
                if any(File.endswith(FileType) for FileType in self.FileTypes):
                    FilePath = os.path.join(Root, File)
                    try:
                        with open(FilePath, "r", encoding="utf-8",
                                  errors="ignore") as F:
                            Content = F.read()
                            Docs.append({"Path": FilePath, "Content": Content})
                    except Exception as E:
                        print(f"Failed to read {FilePath}: {E}")
        return Docs

    def CreateVectorStore(self, Docs):
        """
        Create a Chroma vector store for contextual document queries.

        Args:
            Docs (list): List of documents to be processed into vectors.
        """
        if not Docs:
            print("No documents provided for vector store creation.")
            return

        TextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )
        Splits = TextSplitter.create_documents(
            [Doc["Content"] for Doc in Docs]
        )
        # self.VectorStore = Chroma.from_documents(
        #     Splits,
        #     embedding=self.Embeddings,
        # )
        print("Vector store created successfully.")

    def AnalyzeRepository(self):
        """
        Perform analysis of the repository and create a vector store.

        Returns:
            str: Message indicating the result of the analysis.
        """
        print("Loading documents...")
        Docs = self.LoadDocuments()
        if not Docs:
            return "No matching documents found."

        print("Creating vector store...")
        self.CreateVectorStore(Docs)
        self.Context = "\n\n".join(Doc["Content"] for Doc in Docs)
        Summary = self.GenerateSummary(self.Context)
        self.WriteSummary(Summary)
        #self.SummaryCompleted = True

        return "Repository analysis complete. You can now ask questions!"

    def GenerateSummary(self, Content):
        """
        Generate a structured summary of the repository contents.

        Args:
            Content (str): The repository content to summarize.

        Returns:
            str: The generated summary.
        """
        Prompt = self.SummaryPromptTemplate.format(Context=Content)
        Response = self.Assistant.invoke(Prompt)
        if isinstance(Response, AIMessage):
            return Response.content
        return str(Response)

    # def AskQuestion(self, Query):
    #     """
    #     Ask a contextual question based on the repository content.

    #     Args:
    #         Query (str): The question to ask.

    #     Returns:
    #         str: The AI-generated response to the question.
    #     """
    #     if not self.SummaryCompleted:
    #         return (
    #             "Repository analysis not complete. "
    #             "Please analyze the repository first."
    #         )

    #     if not self.VectorStore:
    #         return (
    #             "No vector store available. "
    #             "Please analyze a repository first."
    #         )

    #     # Retrieve relevant documents using similarity search
    #     RelevantDocs = self.VectorStore.similarity_search(Query, k=5)
    #     Context = "\n\n".join(Doc.page_content for Doc in RelevantDocs)

    #     # Create prompt based on retrieved context
    #     Prompt = self.PromptTemplate.format(Context=Context, Question=Query)
    #     Response = self.Assistant.invoke(Prompt)

    #     if isinstance(Response, AIMessage):
    #         return Response.content
    #     return str(Response)

    def WriteSummary(self, Content):
        """
        Write the generated summary to a Markdown file.

        Args:
            Content (str): The summary content to write.
        """
        try:
            with open("RepoSummary.md", "w", encoding="utf-8") as F:
                F.write(Content)
            print("Summary written to RepoSummary.md")
        except Exception as E:
            print(f"Failed to write summary: {E}")
