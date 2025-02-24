"""This file provides the AI model and all of its functionalities for repository code ananalysis and Q&A."""

import os
import subprocess
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.messages import AIMessage


class AIAssistant:
    """AI Assistent Class for Repository Analysis and Contextual Chat."""

    def __init__(
        self,
        ModelName,
        Temperature,
        PromptTemplate,
        SummaryPromptTemplate,
        FileTypes=None,
    ):
        self.RepoPath = None
        self.ModelName = ModelName
        self.Temperature = Temperature
        self.FileTypes = (
            FileTypes if FileTypes else [".py", ".js", ".java", ".md", ".txt"]
        )
        self.VectorStore = None
        self.Embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.Assistant = OllamaLLM(model=self.ModelName, temperature=self.Temperature)
        self.Context = ""
        self.SummaryCompleted = False

        self.SetTemplates(PromptTemplate, SummaryPromptTemplate)
        self.CheckOllama()

    def SetTemplates(self, PromptTemplate, SummaryPromptTemplate):
        """Sets the prompt for the templates."""
        if not PromptTemplate:
            self.PromptTemplate = ChatPromptTemplate.from_template(
                """
            You are an expert code reviewer. Answer the following question 
            based on the provided repository context:
            
            Context:
            {Context}
            
            Question:
            {Question}
            
            Please provide a concise and clear answer.
            """
            )
        else:
            self.PromptTemplate = ChatPromptTemplate.from_template(PromptTemplate)

        if not SummaryPromptTemplate:
            self.SummaryPromptTemplate = ChatPromptTemplate.from_template(
                """
            You are a technical documentation assistant. 
            Summarize the provided repository contents in a clear and concise manner.
            
            Repository Contents:
            {Context}
            
            Provide a structured summary highlighting key points, 
            code structure, and any important observations.
            """
            )
        else:
            self.SummaryPromptTemplate = ChatPromptTemplate.from_template(
                SummaryPromptTemplate
            )

    def SetRepoPath(self, Path):
        """Sets the path for the repository."""
        self.RepoPath = Path

    def CheckOllama(self):
        """Check if Ollama server is running and start if necessary."""
        try:
            response = requests.get("http://localhost:11434/health")
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
                """Ollama is not installed. Please install 
                Ollama using 'brew install ollama' or appropriate for your OS."""
            )
            exit(1)

    def CheckAndPullModel(self, ModelName):
        """Check if a specific model exists and pull if not."""
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if ModelName not in result.stdout:
                print(f"Model '{ModelName}' not found. Downloading...")
                subprocess.run(["ollama", "pull", ModelName])
                print(f"Model '{ModelName}' downloaded successfully.")
        except Exception as e:
            print(f"Failed to check/download model '{ModelName}': {e}")
            exit(1)

    def LoadDocuments(self):
        """Load documents based on specified file types."""
        docs = []
        for root, _, files in os.walk(self.RepoPath):
            for file in files:
                if any(file.endswith(ft) for ft in self.FileTypes):
                    file_path = os.path.join(root, file)
                    try:
                        with open(
                            file_path, "r", encoding="utf-8", errors="ignore"
                        ) as f:
                            content = f.read()
                            docs.append({"Path": file_path, "Content": content})
                    except Exception as e:
                        print(f"Failed to read {file_path}: {e}")
        return docs

    def CreateVectorStore(self, docs):
        """Create Chroma vector store for contextual queries."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        splits = text_splitter.create_documents([doc["Content"] for doc in docs])
        self.VectorStore = Chroma.from_documents(splits, embedding=self.Embeddings)
        print("Vector store created successfully.")

    def AnalyzeRepository(self):
        """Complete analysis and vector store creation."""
        print("Loading documents...")
        docs = self.LoadDocuments()
        if not docs:
            return "No matching documents found."
        print("Creating vector store...")
        self.CreateVectorStore(docs)
        self.Context = "\n\n".join(doc["Content"] for doc in docs)
        summary = self.GenerateSummary(self.Context)
        self.WriteSummary(summary)
        self.SummaryCompleted = True
        return "Repository analysis complete. You can now ask questions!"

    def GenerateSummary(self, content):
        """Generate a structured summary using the Assistant with a specific prompt."""
        prompt = self.SummaryPromptTemplate.format(Context=content)
        response = self.Assistant.invoke(prompt)
        return response.content if isinstance(response, AIMessage) else str(response)

    def AskQuestion(self, query):
        """Ask a question based on the repository context using the vector store."""
        if not self.SummaryCompleted:
            return (
                "Repository analysis not complete. Please analyze the repository first."
            )

        if not self.VectorStore:
            return "No vector store available. Please analyze a repository first."

        # Retrieve relevant documents using similarity search
        relevant_docs = self.VectorStore.similarity_search(query, k=5)
        context = "\n\n".join(doc.page_content for doc in relevant_docs)

        # Create prompt based on retrieved context
        prompt = self.PromptTemplate.format(Context=context, Question=query)
        response = self.Assistant.invoke(prompt)

        return response.content if isinstance(response, AIMessage) else str(response)

    def WriteSummary(self, content):
        """Write the summary to a Markdown file."""
        try:
            with open("RepoSummary.md", "w", encoding="utf-8") as f:
                f.write(content)
            print("Summary written to RepoSummary.md")
        except Exception as e:
            print(f"Failed to write summary: {e}")
