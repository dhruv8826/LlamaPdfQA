# Using Chroma vectorstore
from langchain.vectorstores import Chroma
# embeddings are numerical representations of the question and answer text
from langchain.embeddings import HuggingFaceEmbeddings
# use a common text splitter to split text into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

from generate_log import log_update


def split_doc(documents):
    # split the documents into chunks, this helps in creating embeddings and storing on the vector db
    log_update("creating chunks", 2)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    all_splits = text_splitter.split_documents(documents)
    log_update("all splits/chunks created successfully", 2)
    return all_splits


def create_embedding():
    # creating embeddings for all the chunks
    log_update("Creating embeddings", 2)
    return HuggingFaceEmbeddings()


def store_on_vectordb(all_splits, embeddings):
    # storing data on Chroma vectordb
    log_update("Storing documents on vectorstore", 2)
    log_update("vector store - Chroma", 2)
    log_update("embedding - HuggingFaceEmbedding", 2)
    log_update("Starting to store", 1)
    vectordb = Chroma.from_documents(
        documents=all_splits,
        embedding=embeddings,
    )
    log_update("Vector store has been successfully set", 1)
    return vectordb


def save_embeddings_on_vector_store(documents):
    log_update("Creating chunks of the doc to create embeddings and loading on a vector store.", 2)
    # Split the documents into smaller chunks
    all_splits = split_doc(documents)
    # Create Embedding to load chunks into vector store
    embeddings = create_embedding()
    # Store the data on vector store
    return store_on_vectordb(all_splits, embeddings)


