import streamlit as st
from langchain_community.llms import Ollama
from langchain.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --- Initialize heavy components once ---
# This remains global as it's constant and heavy to load.
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# --- Cached function for expensive processing ---
@st.cache_resource
def create_vector_store(_file_paths):
    """
    Loads one or more documents, creates vector embeddings, and returns a FAISS vector store.
    This is the most computationally expensive part and is cached.
    The cache is invalidated if the list of file paths changes.
    """
    all_documents = []
    # Loop through each file path and load its documents
    for file_path in _file_paths:
        loader = UnstructuredFileLoader(file_path)
        all_documents.extend(loader.load())

    text_splitter = CharacterTextSplitter(separator="\n",
                                          chunk_size=1000,
                                          chunk_overlap=200)
    text_chunks = text_splitter.split_documents(all_documents)

    vector_store = FAISS.from_documents(text_chunks, embeddings)
    return vector_store


def create_conversational_chain(vector_store, model_name="gemma:2b"):
    """
    Creates a conversational retrieval chain with dynamic model selection.
    """
    # Initialize the LLM dynamically based on user selection
    llm = Ollama(
        model=model_name,
        temperature=0
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Create the conversational chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        return_source_documents=True
    )

    return conversation_chain

