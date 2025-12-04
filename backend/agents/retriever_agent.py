import logging

from vectorstore.store import search

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrieverAgent:
    def __init__(self):
        pass

    def search_docs(self, query: str, top_k: int = 3):
        """
        Searches the vector store for relevant documents based on the query.
        """
        logger.info(f"Searching memory for query: {query}")

        raw_data = search(query, k=top_k)

        formatted_results = []

        documents = raw_data.get("documents", [])
        metadatas = raw_data.get("metadatas", [])
        distances = raw_data.get("distances", [])

        for doc, meta, dist in zip(documents, metadatas, distances):
            formatted_results.append(
                {
                    "text": doc,
                    "metadata": meta,
                    "distance": dist
                }
            )
        
        return formatted_results
