from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


class AIAssistent():

    def __init__(self, ModelName):
        self.ModelName = ModelName
        self.Prompt = any
        self.Chain = any
        self.Context = ""
        self.Template = """
    You are a smart programming and planning assistent and
    help to build projects. Answer the question below as
    accurate as possible.
    That`s the current conversation history: {context}
    Answer this question: {question}"""

    def CreateModel(self):
        """Creation of the AI model with given specifications."""
        self.Model = OllamaLLM(model=self.ModelName)
        self.Prompt = ChatPromptTemplate.from_template(self.Template)
        self.Chain = self.Prompt | self.Model

    def RunConversation(self):
        """Method for running the AI."""
        print("Welcome to the AI Assistent! Type 'exit' to quit the program.")
        while (True):
            UserInput = input("You: ")
            if UserInput.lower() == "exit":
                break
            Result = self.Chain.invoke({"context": self.Context,
                                        "question": UserInput})
            print("AI Assistent: ", Result)
            self.Context += f"\nUser: {UserInput}\nAI Assistent: {Result}"

    def Main(self):
        """Method for calling all other methods."""
        self.CreateModel()
        self.RunConversation()


if __name__ == "__main__":
    load_dotenv(".env")
    AI = AIAssistent("llama3.2")
    AI.Main()
