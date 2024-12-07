"""This file provides the AI model and its functionality."""

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


class AIAssistant:
    """AI Assistent Class."""

    def __init__(self, ModelName, ModelFile, Temperature):
        """Initialize Process."""
        
        self.ModelName = ModelName
        self.ModelFile = ModelFile
        self.Temperature = Temperature
        self.Prompt = any
        self.Chain = any
        self.Context = ""
        self.Template = (
            f"Context: {self.ModelFile}"
            + """
        That`s the current conversation history: {context}
        Answer to this question: {question}"""
        )
        self.Result = any
        self.CreateModel()

    def CreateModel(self):
        """Creation of the AI model with given specifications."""
        self.Model = OllamaLLM(model=self.ModelName,
                               temperature=self.Temperature)
        self.Prompt = ChatPromptTemplate.from_template(self.Template)
        self.Chain = self.Prompt | self.Model

    def GetResponse(self, UserInput):
        """Get AI response for user input."""
        self.Result = self.Chain.invoke(
            {"context": self.Context, "question": UserInput}
        )
        self.Context += f"\nUser: {UserInput}\nAI Assistent: {self.Result}"
        return self.Result
