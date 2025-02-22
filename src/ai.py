import os
import subprocess
import requests
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama, OllamaLLM
from langchain_core.messages import AIMessage

class AIAssistent:
    """Unified AI Class for Repository Analysis and Contextual Chat."""

    def __init__(self, RepoPath, ModelName="llama3.1:8b", Temperature=0.7, FileTypes=None):
        self.RepoPath = RepoPath
        self.ModelName = ModelName
        self.Temperature = Temperature
        self.FileTypes = FileTypes if FileTypes else [".py", ".js", ".java", ".md", ".txt"]
        self.VectorStore = None
        self.Embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.Assistant = OllamaLLM(model=self.ModelName, temperature=self.Temperature)
        self.Context = ""
        self.PromptTemplate = ChatPromptTemplate.from_template(
            """
            You are an expert code reviewer. Summarize the main contents of the repository by extracting key elements
            such as classes, functions, and important comments from the following documents:

            Format the output in Markdown, providing clear section headers and concise explanations.
            
            Context: {context}
            Question: {question}
            """
        )
        self.CheckOllama()

    def CheckOllama(self):
        """Check if Ollama server is running and start if necessary."""
        try:
            Response = requests.get("http://localhost:XXX/XXX")
            if Response.status_code == 200:
                print("Ollama server is running.")
                return
        except requests.exceptions.ConnectionError:
            print("Ollama server not running. Attempting to start...")

        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("Ollama server started successfully.")
        except FileNotFoundError:
            print("Ollama is not installed. Please install Ollama using 'brew install ollama'.")
            exit(1)

        self.CheckAndPullModel("llama3.1:8b")
        self.CheckAndPullModel("nomic-embed-text")

    def CheckAndPullModel(self, ModelName):
        """Check if a specific model exists and pull if not."""
        try:
            Result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if ModelName not in Result.stdout:
                print(f"Model '{ModelName}' not found. Downloading...")
                subprocess.run(["ollama", "pull", ModelName])
                print(f"Model '{ModelName}' downloaded successfully.")
        except Exception as e:
            print(f"Failed to check/download model '{ModelName}': {e}")
            exit(1)

    def LoadDocuments(self):
        """Load documents based on specified file types."""
        Docs = []
        for Root, _, Files in os.walk(self.RepoPath):
            for File in Files:
                if any(File.endswith(FT) for FT in self.FileTypes):
                    FilePath = os.path.join(Root, File)
                    try:
                        with open(FilePath, "r", encoding="utf-8", errors="ignore") as F:
                            Content = F.read()
                            Docs.append({"Path": FilePath, "Content": Content})
                    except Exception as e:
                        print(f"Failed to read {FilePath}: {e}")
        return Docs

    def CreateVectorStore(self, Docs):
        """Create Chroma vector store for contextual queries."""
        TextSplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        Splits = TextSplitter.create_documents([Doc['Content'] for Doc in Docs])
        self.VectorStore = Chroma.from_documents(Splits, embedding=self.Embeddings)

    def AnalyzeRepository(self):
        """Complete analysis and vector store creation."""
        print("Loading documents...")
        Docs = self.LoadDocuments()
        if not Docs:
            return "No matching documents found."
        print("Creating vector store...")
        self.CreateVectorStore(Docs)
        self.Context = "\n\n".join(Doc['Content'] for Doc in Docs)
        self.WriteSummary(self.Context)
        return "Repository analysis complete. You can now ask questions!"

    def AskQuestion(self, Query):
        """Ask a question based on the repository context."""
        if not self.Context:
            return "No context available. Please analyze a repository first."
        Prompt = self.PromptTemplate.format(context=self.Context, question=Query)
        Response = self.Assistant.invoke(Prompt)
        return Response.content if isinstance(Response, AIMessage) else Response

    def ResetContext(self):
        """Reset the current context."""
        self.Context = ""

    def WriteSummary(self, Content):
        """Write the summary to a Markdown file."""
        try:
            with open("RepoSummary.md", "w", encoding="utf-8") as F:
                F.write(Content)
            print("Summary written to RepoSummary.md")
        except Exception as e:
            print(f"Failed to write summary: {e}")