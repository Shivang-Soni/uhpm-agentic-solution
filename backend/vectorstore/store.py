# backend/vectorstore/store.py
import uuid
from typing import Optional, Dict

import chromadb
from sentence_transformers import SentenceTransformer
from core.config import settings

# Lade Embedding-Modell
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialisiere persistenten Chroma-Client
chroma_client = chromadb.PersistentClient(path=settings.PERSIST_DIRECTORY)

# Lade oder erstelle Collection
collection = chroma_client.get_or_create_collection(
    name="uhpm_collection",
    metadata={"hnsw:space": "cosine"}  # cosine similarity
)


def add_document(text: str, metadata: Optional[Dict] = None) -> Dict:
    """
    Fügt ein Dokument in den Vector Store ein.
    """
    doc_id = str(uuid.uuid4())
    embedding = embedding_model.encode(text).tolist()

    collection.add(
        ids=[doc_id],
        documents=[text],
        embeddings=[embedding],
        metadatas=[metadata or {"source": "manual"}],
    )

    return {"id": doc_id, "text": text}


def search(query: str, k: int = 3) -> Dict:
    """
    Sucht die k ähnlichsten Dokumente zum Query.
    """
    query_embedding = embedding_model.encode(query).tolist()

    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    return {
        "ids": raw.get("ids", [[]])[0],
        "documents": raw.get("documents", [[]])[0],
        "metadatas": raw.get("metadatas", [[]])[0],
        "distances": raw.get("distances", [[]])[0],
    }
