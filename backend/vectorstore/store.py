import uuid

import chromadb
from sentence_transformers import SentenceTransformer
from core.config import settings

# load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# init persistent client
chroma_client = chromadb.PersistentClient(path=settings.PERSIST_DIRECTORY)

# load/create collection
collection = chroma_client.get_or_create_collection(
    name="uhpm_collection",
    metadata={"hnsw:space": "cosine"}
)


def add_document(text: str, metadata: dict | None = None):
    doc_id = str(uuid.uuid4())
    embedding = embedding_model.encode(text).tolist()

    collection.add(
        ids=[doc_id],
        documents=[text],
        embeddings=[embedding],
        metadatas=[metadata or {"source": "manual"}],
    )

    return {"id": doc_id, "text": text}


def search(query: str, k: int = 3):
    query_embedding = embedding_model.encode(query).tolist()

    # MUST SPECIFY include param, or Chroma returns a non-serializable object
    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    # ALWAYS convert to primitive dict
    return {
        "ids": raw.get("ids", [[]])[0],
        "documents": raw.get("documents", [[]])[0],
        "metadatas": raw.get("metadatas", [[]])[0],
        "distances": raw.get("distances", [[]])[0],
    }
