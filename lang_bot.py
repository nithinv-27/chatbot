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
# from langchain_community.document_loaders import WebBaseLoader
# loader = WebBaseLoader("    ")
##Embeddings
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings()

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("input_pdfs/Profile.pdf")
pages = loader.load_and_split()

 #Use Langchain to create the embeddings using text-embedding-ada-002
db = FAISS.from_documents(documents=pages, embedding=embeddings)
# docs = loader.load()

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
    query = "what is the full name of Nithin?"
    result = qa({"question": question, "chat_history": chat_history})
    print("answer:", result["answer"])
    chat_history = [(query, result["answer"])]
    return chat_history

chat_history = []
while True:
    query = input('you: ')
    if query == 'q':
        break
    chat_history = ask_question_with_context(qa, query, chat_history)


# from langchain_community.vectorstores import FAISS
# from langchain_text_splitters import RecursiveCharacterTextSplitter


# text_splitter = RecursiveCharacterTextSplitter()
# documents = text_splitter.split_documents(docs)
# vector = FAISS.from_documents(documents, embeddings)

# from langchain.chains.combine_documents import create_stuff_documents_chain

# prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

# <context>
# {context}
# </context>

# Question: {input}""")

# document_chain = create_stuff_documents_chain(llm, prompt)

# from langchain.chains import create_retrieval_chain

# def with_retrieval_chain(text_input):

#     retriever = vector.as_retriever()
#     retrieval_chain = create_retrieval_chain(retriever, document_chain)

#     response = retrieval_chain.invoke({"input": text_input})
#     return response["answer"]

# LangSmith offers several features that can help with testing:...