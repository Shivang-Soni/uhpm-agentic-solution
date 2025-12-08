import pytest
from unittest.mock import patch
from backend.agents.retriever_agent import RetrieverAgent


@pytest.fixture
def agent():
    return RetrieverAgent()


def test_search_docs_returns_formatted_results(agent):
    # Mocks return value of vectorstore.search
    mock_search_return = {
        "documents": ["doc1", "doc2"],
        "metadatas": [{"type": "research"}, {"type": "persona"}],
        "distances": [0.1, 0.2]
    }

    with patch("backend.agents.retriever_agent.search", return_value=mock_search_return) \
            as mock_search:

        results = agent.search_docs("some query", top_k=2)

        mock_search.assert_called_once_with("some query", k=2)

        # Check result format
        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0]["text"] == "doc1"
        assert results[0]["metadata"] == {"type": "research"}
        assert results[0]["distance"] == 0.1
        assert results[1]["text"] == "doc2"
        assert results[1]["metadata"] == {"type": "persona"}
        assert results[1]["distance"] == 0.2


def test_search_docs_empty_results(agent):
    # Mock empty search result
    mock_search_return = {"documents": [], "metadatas": [], "distances": []}

    with patch("backend.agents.retriever_agent.search", return_value=mock_search_return):
        results = agent.search_docs("empty query")

        assert results == []
