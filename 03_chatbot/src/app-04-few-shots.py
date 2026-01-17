import asyncio
import os
import chainlit as cl
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.llms import OpenAI
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv

#asyncio is used for asnchornous programming
#os is for interacting with the operating system
#lanchain helps with language processing tasks
#dotenv helps load the environment using our api keys

load_dotenv(find_dotenv(".env"))
MODEL_NAME = "text-davinci-003"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

examples = [
    {"word": "happy",     "antonym": "sad"},
    {"word": "content",   "antonym": "dissatisfied"},
    {"word": "peaceful",  "antonym": "belligerent"},
    {"word": "tall",      "antonym": "short"},
    {"word": "high",      "antonym": "low"},
    {"word": "energetic", "antonym": "lethargic"},
    {"word": "fast",      "antonym": "slow"},
    {"word": "sunny",     "antonym": "gloomy"},
    {"word": "clear",     "antonym": "cloudy"},
    {"word": "windy",     "antonym": "calm"},
]

#we fare done defining example data

example_formatter_template = "Word: {word}\nAntonym: {antonym}"

example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template=example_formatter_template,
)

#defining the example prompt templates; specifies how the input variables will the formatted in the prompt

@cl.on_message
def main(input_word: str):
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        OpenAIEmbeddings(),
        Chroma,
        k=2
    )
#So, overall, the example_selector is created to find the most similar examples to the user's input word based on semantic 
# similarity using word embeddings and the Chroma vector store. These similar examples will be used to create the few-shot 
# prompt for the chatbot's response.

    fewshot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix="Give the antonym of every input word",
        suffix="Word: {word}\nAntonym:",
        input_variables=["word"],
    )

#creating the example selector and few-shots prompt teplate
# Few-Shot Learning (FSL) is a Machine Learning framework that enables a pre-trained model to generalize over new categories 
# of data (that the pre-trained model has not seen during training) using only a few labeled samples per class.

    llm = OpenAI(openai_api_key=OPENAI_API_KEY, model=MODEL_NAME)
    response = llm(fewshot_prompt.format(word=input_word))
    response += "\n\n=> Enter a word:"
#initiating the OpenAI language model and generating a response

    asyncio.run(cl.Message(content=response).send())
#sending the response to the user

@cl.on_chat_start
def start():
    output = "\n".join(f"word: {e['word']} <=> antonym: {e['antonym']}" for e in examples)
    output += "\n\n=> Enter a word:"
    asyncio.run(cl.Message(content=output).send())

# to start the conversation