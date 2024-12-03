import os
from dotenv import load_dotenv
import ollama


def CreateModel(ModelName: str,Modelfile: str):
    ollama.create(model=ModelName, modelfile=Modelfile)


def CreateResponse(Prompt: str, ModelName: str, Modelfile: str):
    CreateModel(Modelfile)
    response = ollama.generate(model=ModelName, prompt=Prompt)
    print(response.get("response"))

def CallAPI():
    CreateResponse("Tell me that you are an assistent and want to help me!")
    Input = input()
    CreateResponse(Input)
    CallAPI()

if __name__ == "__main__":
    load_dotenv(".env")
    CallAPI()