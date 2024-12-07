"""This file runs the ai model and lets you interact with it."""

import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


class AIAssistent:
    """AI Assistent Class."""

    def __init__(self, ModelName, Modelfile, Temperature):
        """Initilization Process."""
        self.ModelName = ModelName
        self.ModelFile = Modelfile
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

    def CreateModel(self):
        """Creation of the AI model with given specifications."""
        self.Model = OllamaLLM(model=self.ModelName, temperature=self.Temperature)
        self.Prompt = self.Prompt = ChatPromptTemplate.from_template(self.Template)
        self.Chain = self.Prompt | self.Model

    def RunConversation(self):
        """Run the AI model."""
        print("Welcome to the AI Assistent! Type 'exit' to quit the program.")
        while True:
            UserInput = input("You: ")
            if UserInput.lower() == "exit":
                break
            Result = self.Chain.invoke({"context": self.Context, "question": UserInput})
            print("AI Assistent: ", Result)
            self.Context += f"\nUser: {UserInput}\nAI Assistent: {Result}"

    def Main(self):
        """Execute all methods."""
        self.CreateModel()
        self.RunConversation()


if __name__ == "__main__":
    load_dotenv("config/.env")
    AI = AIAssistent(
        "llama3.2",
        """You are a smart assistent called Bob and you know 
        everything about computer science and want help to solve 
        every problem about that.""",
        1.0,
    )
    AI.Main()
