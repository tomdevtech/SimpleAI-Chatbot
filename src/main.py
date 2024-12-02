from dotenv import load_dotenv
import os
import json
import httpx


def CreateResponse(Input: str, URL: str):
    data = {
        "model": "llama2",
        "prompt": Input,
    }
    response = httpx.post(url=URL, data=json.dumps(data), headers={
                          "Content-Type": "application/json"})

    # Split the response by newlines and filter out empty lines
    response_lines = [line for line in response.text.strip().split('\n') if line]

    # Parse each line as JSON
    response_dicts = [json.loads(line) for line in response_lines]

    print(''.join(response_dict.get('response', '') for response_dict in response_dicts))


def CallAPI():
    print(os.getenv("URL"))
    CreateResponse(
        "Tell me that you are an assistent and want to help me!", os.getenv("URL"))
    Input = input()
    CreateResponse(Input, os.getenv("URL"))
    CallAPI()


if __name__ == "__main__":
    load_dotenv("config/.env")
    CallAPI()
