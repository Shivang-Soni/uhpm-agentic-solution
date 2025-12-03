import chromadb
from langchain_community.vectorstores import Chroma 
from langchain_community.embeddings import SentenceTransformerEmbeddings
from core.config import settings


# Load embeddings model
embedding_function = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
    )


def get_vectorstore():
    """
    Returns a Chroma vector store instance.
    """
    return Chroma(
        embedding_function=embedding_function,
        persist_directory=settings.PERSIST_DIRECTORY
    )


def add_document(text: str, metadata: dict = None):
    """
    Adds a document to the vector store.
    """
    vector_store = get_vectorstore()
    vector_store.add_texts(
        [text],
        metadatas=[metadata or {}]
    )
    vector_store.persist()


def search(query: str, k: int = 3):
    """
    Searches the vector store using semantic search.
    """
    vector_store = get_vectorstore()
    return vector_store.similarity_search(query, k=k)
