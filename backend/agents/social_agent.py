from typing import List
import requests
from bs4 import BeautifulSoup
import feedparser
import praw
from textblob import TextBlob  # For sentiment analysis

from vectorstore.store import add_document


class SocialAgent:
    """
    Scrape forums, Reddit, blogs, RSS feeds and perform sentiment analysis
    to generate marketing insights and store them in the vector store.
    """

    def __init__(
        self,
        reddit_client_id: str,
        reddit_secret: str,
        reddit_user_agent: str
    ):
        self.reddit = praw.Reddit(
            client_id=reddit_client_id,
            client_secret=reddit_secret,
            user_agent=reddit_user_agent
        )

    def scrape_reddit(self, subreddit_name: str, limit: int = 10) -> List[dict]:
        results = []
        subreddit = self.reddit.subreddit(subreddit_name)
        for submission in subreddit.hot(limit=limit):
            text = submission.title + "\n" + (submission.selftext or "")
            sentiment = TextBlob(text).sentiment.polarity
            metadata = {
                "source": "reddit",
                "subreddit": subreddit_name,
                "url": submission.url,
                "sentiment": sentiment
            }
            add_document(text, metadata)
            results.append(metadata)
        return results

    def scrape_rss_feed(self, feed_url: str, limit: int = 10) -> List[dict]:
        feed = feedparser.parse(feed_url)
        results = []
        for entry in feed.entries[:limit]:
            text = entry.title + "\n" + getattr(entry, "summary", "")
            sentiment = TextBlob(text).sentiment.polarity
            metadata = {
                "source": "rss",
                "feed_url": feed_url,
                "entry_link": entry.link,
                "sentiment": sentiment
            }
            add_document(text, metadata)
            results.append(metadata)
        return results

    def scrape_website(self, url: str) -> dict:
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            paragraphs = soup.find_all("p")
            text = "\n".join(p.get_text() for p in paragraphs)
            sentiment = TextBlob(text).sentiment.polarity
            metadata = {
                "source": "website",
                "url": url,
                "sentiment": sentiment
            }
            add_document(text, metadata)
            return metadata
        except Exception as e:
            return {"source": "website", "url": url, "error": str(e)}

    def gather_insights(
        self,
        reddit_subs: List[str],
        rss_feeds: List[str],
        websites: List[str]
    ) -> List[dict]:
        insights = []
        for sub in reddit_subs:
            insights.extend(self.scrape_reddit(sub))
        for feed in rss_feeds:
            insights.extend(self.scrape_rss_feed(feed))
        for url in websites:
            insights.append(self.scrape_website(url))
        return insights
