# LangChain supports many other chat models. Here, we're using Ollama
import os
import signal
import langchain
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores import FAISS


langchain.verbose = False

output_parser = StrOutputParser()

llm = Ollama(model="llama2")

def without_retrieval_chain(text_input):

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Be precise."),
        ("user", "{input}")
    ])

    chain = prompt | llm | output_parser

    # input=text_prompt
    return chain.invoke({"input": text_input})

#Retrieval chain
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")

docs = loader.load()

##Embeddings
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings()

from langchain_text_splitters import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
db = FAISS.from_documents(documents, embeddings)


#save the embeddings into FAISS vector store
db.save_local("./dbs/documentation/faiss_index")

#load the faiss vector store we saved into memory
vectorStore = FAISS.load_local("./dbs/documentation/faiss_index", embeddings)

#use the faiss vector store we saved to search the local document
retriever = vectorStore.as_retriever(search_type="similarity", search_kwargs={"k":2})


from langchain.prompts import PromptTemplate

QUESTION_PROMPT = PromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:""")

from langchain.chains import ConversationalRetrievalChain


qa = ConversationalRetrievalChain.from_llm(llm=llm,
                                            retriever=retriever,
                                            condense_question_prompt=QUESTION_PROMPT,
                                            return_source_documents=True,
                                            verbose=False)

def ask_question_with_context(qa, question, chat_history):
    result = qa({"question": question, "chat_history": chat_history})
    chat_history = [(question, result["answer"])]
    return result["answer"]


chat_history = []

def chat_with_bot(query, chat_history = chat_history):
    result = ask_question_with_context(qa, query, chat_history)
    return result

