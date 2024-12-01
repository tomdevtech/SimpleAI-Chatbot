import os
from dotenv import load_dotenv
from ollama import chat
from ollama import ChatResponse

def CreateResponse(Input: str):
    response: ChatResponse = chat(model='llama3.2', messages=[
    {
        'role': 'user',
        'content': Input,
    },])
    print(response.message.content)

def CallAPI():
    CreateResponse("Tell me that you are an assistent and want to help me!")
    Input = input()
    CreateResponse(Input)
    CallAPI()

if __name__ == "__main__":
    load_dotenv(".env")
    CallAPI()