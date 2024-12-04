from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Template Definition for the AI Model
Template = """
    You are a smart programming and planning assistent and 
    help to build projects. Answer the question below as 
    accurate as possible.

    That`s the current conversation history: {context}
    Answer this question: {question}
    """


# Variable Declaration of the AI Model & Chaining
model = OllamaLLM(model="llama3.2")
prompt = ChatPromptTemplate.from_template(Template)
chain = prompt | model


def RunConversation():
    """Method for running the AI."""
    context = ""
    print("Welcome to the AI Assistent! Type 'exit' to quit the program.")
    while (True):
        userInput = input("You: ")
        if userInput.lower() == "exit":
            break
        result = chain.invoke({"context": context, "question": userInput})
        print("AI Assistent: ", result)
        context += f"\nUser: {userInput}\nAI Assistent: {result}"


def Main():
    """Method for calling all other methods."""
    RunConversation()


if __name__ == "__main__":
    load_dotenv(".env")
    Main()
