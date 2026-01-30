from typing import List, Dict
from vectorstore.store import search


def extract_reflection(state: Dict, action: str) -> List[Dict]:
    """
    Retrieve relevant past agent experiences from vector memory
    for the current action.
    """

    # Build semantic query from current context
    objective = state.get("objective", "")
    query = f"{action} {objective}"

    results = search(
        query=query,
        k=3,
        action=action,
    )

    reflections = []

    for doc, meta, dist in zip(
        results["documents"],
        results["metadatas"],
        results["distances"],
    ):
        reflections.append(
            {
                "text": doc,
                "distance": dist,
                "success": meta.get("success"),
                "campaign_id": meta.get("campaign_id"),
            }
        )

    return reflections
