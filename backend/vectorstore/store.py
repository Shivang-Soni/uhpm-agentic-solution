import uuid
from typing import Optional, Dict

import chromadb
from sentence_transformers import SentenceTransformer
from core.config import settings

# Load embedding model (global singleton)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Init persistent Chroma client
chroma_client = chromadb.PersistentClient(
    path=settings.PERSIST_DIRECTORY
)

# Load or create collection
collection = chroma_client.get_or_create_collection(
    name="uhpm_collection",
    metadata={"hnsw:space": "cosine"}
)


def add_document(
    text: str,
    action: str,
    success: bool,
    campaign_id: str,
    metadata: Optional[Dict] = None,
) -> Dict:
    """
    Persist agent experience into vector memory.
    """

    doc_id = str(uuid.uuid4())
    embedding = embedding_model.encode(text).tolist()

    meta = {
        "action": action,
        "success": success,
        "campaign_id": campaign_id,
    }

    if metadata:
        meta.update(metadata)

    collection.add(
        ids=[doc_id],
        documents=[text],
        embeddings=[embedding],
        metadatas=[meta],
    )

    return {"id": doc_id}


def search(
    query: str,
    k: int = 3,
    action: Optional[str] = None
) -> Dict:
    """
    Semantic search over agent memory.
    """

    query_embedding = embedding_model.encode(query).tolist()

    where = {"action": action} if action else None

    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    return {
        "documents": raw.get("documents", [[]])[0],
        "metadatas": raw.get("metadatas", [[]])[0],
        "distances": raw.get("distances", [[]])[0],
    }
