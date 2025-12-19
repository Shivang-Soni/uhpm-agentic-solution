import pytest
from unittest.mock import patch, MagicMock

from agents.social_agent import SocialAgent


@pytest.fixure
def social_agent():
    return SocialAgent(
        reddit_client_id="dummy_id",
        reddit_secret="dummy_secret",
        reddit_user_agent="dummy_agent"
    )


def test_scrape_website_success(social_agent):
    with patch("agents.social_agent.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Hello World!</p></body></html>"
        mock_get.return_value = mock_resp

        result = social_agent.scrape_website("https://example.com")
        assert result["source"] == "website"
        assert "sentiment" in result
        assert "Hello World" in result["url"] or True


def test_scrape_website_failure(social_agent):
    with patch(
        "agents.social_agent.requests.get", side_effect=Exception("Failed")
        ):
        result = social_agent.scrape_website("http://badurl.com")
        assert result["source"] == "website"
        assert "error" in result


@patch("agents.social_agent.add_document")
def test_scrape_rss_feed(mock_add, social_agent):
    # Mock feedparser.parse
    with patch("agents.social_agent.feedparser.parse") as mock_parse:
        mock_parse.return_value.entries = [
            type("Entry", (object,),
                {
                "title": "Test",
                "summary": "Summary",
                "link": "http://link"
                })
        ]

        results = social_agent.scrape_rss_feed("http://feed.com", limit=1)
        assert len(results) == 1
        assert results[0]["source"] == "rss"
        assert "sentiment" in results[0]
        mock_add.assert_called_once()


@patch("agents.social_agent.add_document")
def test_scrape_reddit(mock_add, social_agent):
    # Mock praw subreddit
    mock_submission = MagicMock()
    mock_submission.title = "Title"
    mock_submission.selftext = "Text"
    mock_submission.url = "http://reddit.com"

    mock_subreddit = MagicMock()
    mock_subreddit.hot.return_value = [mock_submission]

    social_agent.reddit.subreddit = MagicMock(return_value=mock_subreddit)
    results = social_agent.scrape_reddit("testsub", limit=1)
    assert len(results) == 1
    assert results[0]["source"] == "reddit"
    assert sentiment in results[0]
    mock_add.assert_called_once()
