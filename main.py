import os
from openai import OpenAI
from dotenv import load_dotenv

def CreateResponse(Input: str):
    client = OpenAI(
    api_key=os.getenv("API_KEY"))

    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": Input,
        }
    ])
    print(completion.choices[0].message.content)
    CallAPI()

def CallAPI():
    print("How can I help you?")
    Input = input()
    CreateResponse(Input)

if __name__ == "__main__":
    load_dotenv(".env")
    CallAPI()