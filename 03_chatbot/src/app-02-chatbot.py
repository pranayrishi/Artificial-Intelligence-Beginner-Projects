import chainlit as cl
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv, find_dotenv
import os
import asyncio

load_dotenv(find_dotenv())
MODEL_NAME = "gpt-3.5-turbo" # in ths program we are using a v1/chat/completion endpoint however text davinci 
#is a regular completion model not able to support chat conversations
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

system_prompt = """
You are a helpful assistant that truthfully respond to a user's query or question.

If you don't know the answer, simply answer: I don't know. 
Most importantly, do not respond with false information.
"""
# multi-line string that serves as the system prompt; 
# it defines the introduction message given to the assistant before each user's query
#you can customize the behaviour of the langauge model based on the use case
@cl.on_message
def main(query: str):
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': query}
    ]
    response_text = ""
    try:
        chat = ChatOpenAI(temperature=0, model=MODEL_NAME)
        response = chat.predict_messages([
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ])
        response_text = response.content
    except Exception as e:
        response_text = f"no response: {e}"

    asyncio.run(
        cl.Message(content=response_text).send()
    )

@cl.on_chat_start
def start():
    asyncio.run(
        cl.Message(content="Hello there!").send()
    )
