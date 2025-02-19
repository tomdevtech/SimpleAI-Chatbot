import os
import subprocess
import requests
import langchain as lc
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage

class AutoRepoAnalyzer:
    """Class for Automatically Analyzing Repositories and Creating a Markdown Summary."""

    def __init__(self, repo_path, filetypes=None):
        self.repo_path = repo_path
        self.filetypes = filetypes if filetypes else [".py", ".js", ".java", ".md", ".txt"]
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are an expert code reviewer. Summarize the main contents of the repository by extracting key elements
            such as classes, functions, and important comments from the following documents:
            {docs}

            Format the output in Markdown, providing clear section headers and concise explanations.
            """
        )
        self.check_ollama()

    def check_ollama(self):
        """Check if Ollama server is running and start it if necessary."""
        try:
            response = requests.get("http://localhost:XXX/")
            if response.status_code == 200:
                print("Ollama server is running.")
                return
        except requests.exceptions.ConnectionError:
            print("Ollama server not running. Attempting to start...")

        # Try to start the Ollama server
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("Ollama server started successfully.")
        except FileNotFoundError:
            print("Ollama is not installed. Please install Ollama using 'brew install ollama' or the official installer.")
            exit(1)

        # Check if the models exist
        self.check_and_pull_model("llama3.1:8b")
        self.check_and_pull_model("nomic-embed-text")

    def check_and_pull_model(self, model_name):
        """Check if a specific model exists and pull it if not."""
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if model_name not in result.stdout:
                print(f"Model '{model_name}' not found. Downloading...")
                subprocess.run(["ollama", "pull", model_name])
                print(f"Model '{model_name}' downloaded successfully.")
            else:
                print(f"Model '{model_name}' is already available.")
        except Exception as e:
            print(f"Failed to check or download model '{model_name}': {e}")
            exit(1)

    def load_documents(self):
        """Load documents based on specified file types from the given directory."""
        docs = []
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if any(file.endswith(ft) for ft in self.filetypes):
                    filepath = os.path.join(root, file)
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        docs.append({"path": filepath, "content": content})
        return docs

    def format_docs(self, docs):
        """Format documents for further processing."""
        return "\n\n".join(f"### {doc['path']}\n{doc['content']}" for doc in docs)

    def create_vectorstore(self, docs):
        """Create a Chroma vectorstore from the loaded documents."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.create_documents([doc['content'] for doc in docs])
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = Chroma.from_documents(splits, embedding=embeddings)
        return vectorstore

    def generate_summary(self, formatted_docs):
        """Generate a summary using the language model."""
        model = ChatOllama(model="llama3.1:8b")
        chain = RunnablePassthrough() | self.prompt | model
        return chain.invoke({"docs": formatted_docs})

    def write_summary(self, content):
        """Write the summary to a Markdown file."""
        if isinstance(content, AIMessage):
            content = content.content
        with open("RepoSummary.md", "w", encoding="utf-8") as f:
            f.write(content)

    def run(self):
        """Run the complete analysis process."""
        print("Loading documents...")
        docs = self.load_documents()

        if not docs:
            print("No matching documents found.")
            return

        print("Creating vectorstore...")
        self.create_vectorstore(docs)

        print("Generating summary...")
        formatted_docs = self.format_docs(docs)
        summary = self.generate_summary(formatted_docs)

        print("Writing summary to RepoSummary.md...")
        self.write_summary(summary)
        print("Analysis complete!")

if __name__ == "__main__":
    repo_path = ""
    analyzer = AutoRepoAnalyzer(repo_path)
    analyzer.run()