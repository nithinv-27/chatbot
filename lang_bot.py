# LangChain supports many other chat models. Here, we're using Ollama
import os
import signal
import langchain
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from app import text_prompt
output_parser = StrOutputParser()

def get_out_for_text_inp(text_input):
    llm = Ollama(model="llama2")
    langchain.verbose = False

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Be precise."),
        ("user", "{input}")
    ])

    chain = prompt | llm | output_parser

    input = text_input
    # input=text_prompt
    return chain.invoke({"input": input})

def stop_responding():
    os.kill(os.getpid(), signal.SIGTERM)
