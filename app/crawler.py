"""Web crawler module for RSS and manual crawling."""
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional

import feedparser
import requests
from bs4 import BeautifulSoup

from app.config import (
    ARXIV_RSS_URLS,
    HF_DAILY_PAPERS_RSS,
    AI_LAB_RSS_URLS,
    CN_AI_RSS_URLS,
    AI_NEWS_RSS_URLS,
    REDDIT_RSS_URLS,
    AI_TREND_APIS,
    MAX_ARTICLES_PER_RUN,
)

logger = logging.getLogger(__name__)


class ArticleData:
    """Data transfer object for crawled articles."""

    def __init__(self, title: str, summary: str, url: str, source: str,
                 authors: Optional[str] = None, published_at: Optional[datetime] = None):
        self.title = title
        self.summary = summary
        self.url = url
        self.source = source
        self.authors = authors
        self.published_at = published_at
        self._hash = None

    @property
    def content_hash(self) -> str:
        """Generate content hash for deduplication."""
        if self._hash is None:
            content = f"{self.title}:{self.url}"
            self._hash = hashlib.md5(content.encode()).hexdigest()
        return self._hash


class GenericRssCrawler:
    """Generic crawler for any RSS/Atom feed."""

    def __init__(self, source_name: str, timeout: int = 15):
        self.source_name = source_name
        self.timeout = timeout

    def crawl(self, url: str, limit: int = 10) -> List[ArticleData]:
        """Crawl a single RSS feed."""
        articles = []
        try:
            logger.info(f"[RSS] Crawling {self.source_name}: {url}")
            feed = feedparser.parse(url, request_headers={
                "User-Agent": "AI-Self-Evolution Bot/1.0 (Research)"
            })
            for entry in feed.entries[:limit]:
                article = self._parse_entry(entry)
                if article:
                    articles.append(article)
        except Exception as e:
            logger.error(f"[RSS] Error crawling {url}: {e}")
        return articles

    def _parse_entry(self, entry) -> Optional[ArticleData]:
        """Parse a single RSS entry."""
        try:
            title = entry.get("title", "").strip()
            summary = entry.get("summary", "") or entry.get("description", "")
            summary = summary.strip()
            url = entry.get("link", "").strip()

            authors = ""
            if entry.get("authors"):
                authors = ", ".join(a.get("name", "") for a in entry["authors"])
            elif entry.get("author"):
                authors = entry["author"]

            published = entry.get("published_parsed") or entry.get("updated_parsed")
            published_at = datetime(*published[:6]) if published else None

            if title and url:
                return ArticleData(
                    title=title,
                    summary=summary[:2000],
                    url=url,
                    source=self.source_name,
                    authors=authors[:500] if authors else None,
                    published_at=published_at,
                )
        except Exception as e:
            logger.error(f"[RSS] Error parsing entry: {e}")
        return None


class ArxivCrawler(GenericRssCrawler):
    """Crawler for arXiv RSS feeds."""

    def __init__(self):
        super().__init__("arxiv")
        self.urls = ARXIV_RSS_URLS

    def crawl(self, limit: int = MAX_ARTICLES_PER_RUN) -> List[ArticleData]:
        """Crawl all arXiv RSS feeds."""
        articles = []
        per_feed = max(3, limit // max(len(self.urls), 1))
        for url in self.urls:
            feed_articles = super().crawl(url, limit=per_feed)
            articles.extend(feed_articles)
        return articles[:limit]


class WebCrawler:
    """Generic web crawler for manual URL submission."""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.headers = {
            "User-Agent": "AI-Self-Evolution Bot/1.0 (Research Purpose)"
        }

    def crawl_url(self, url: str, source: str = "manual") -> Optional[ArticleData]:
        """Crawl a single URL."""
        try:
            logger.info(f"[Web] Crawling URL: {url}")
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")
            title = self._extract_title(soup)
            summary = self._extract_summary(soup)

            if title and url:
                return ArticleData(
                    title=title,
                    summary=summary[:2000] if summary else "",
                    url=url,
                    source=source,
                )
        except Exception as e:
            logger.error(f"[Web] Error crawling {url}: {e}")
        return None

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract title from HTML."""
        for meta in soup.find_all("meta", property=["og:title", "twitter:title"]):
            if meta.get("content"):
                return meta["content"].strip()
        if soup.title:
            return soup.title.string.strip() if soup.title.string else ""
        h1 = soup.find("h1")
        if h1:
            return h1.get_text().strip()
        return ""

    def _extract_summary(self, soup: BeautifulSoup) -> str:
        """Extract summary from HTML."""
        for meta in soup.find_all("meta", attrs={"name": "description"}):
            if meta.get("content"):
                return meta["content"].strip()
        for meta in soup.find_all("meta", property=["og:description", "twitter:description"]):
            if meta.get("content"):
                return meta["content"].strip()
        p = soup.find("p")
        if p:
            return p.get_text().strip()
        return ""


class TrendApiCrawler:
    """Crawler for AI trend/leaderboard APIs."""

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.headers = {"User-Agent": "AI-Self-Evolution Bot/1.0"}

    def crawl_trends(self) -> List[ArticleData]:
        """Crawl all trend APIs."""
        articles = []
        for name, config in AI_TREND_APIS.items():
            try:
                logger.info(f"[Trend] Fetching {name}: {config['url']}")
                response = requests.get(
                    config["url"],
                    headers=self.headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()

                parsed = self._parse_response(name, config, data)
                articles.extend(parsed)
            except Exception as e:
                logger.error(f"[Trend] Error fetching {name}: {e}")
        return articles

    def _parse_response(self, name: str, config: dict, data) -> List[ArticleData]:
        """Parse API response into ArticleData."""
        articles = []

        if name == "huggingface_trending" and isinstance(data, list):
            for model in data[:10]:
                model_id = model.get("id", model.get("modelId", ""))
                if model_id:
                    articles.append(ArticleData(
                        title=f"Trending Model: {model_id}",
                        summary=f"Downloads: {model.get('downloads', 0)} | Likes: {model.get('likes', 0)}",
                        url=f"https://huggingface.co/{model_id}",
                        source="huggingface_trending",
                    ))

        elif name == "hf_daily_papers" and isinstance(data, list):
            for paper in data[:10]:
                title = paper.get("title", "")
                paper_id = paper.get("paperId", "")
                if title:
                    articles.append(ArticleData(
                        title=title,
                        summary=paper.get("summary", paper.get("abstract", ""))[:500],
                        url=f"https://huggingface.co/papers/{paper_id}",
                        source="hf_daily_papers",
                    ))

        elif name == "semantic_scholar_trending" and isinstance(data, dict):
            for paper in data.get("data", [])[:10]:
                title = paper.get("title", "")
                paper_url = paper.get("url", "")
                if title and paper_url:
                    articles.append(ArticleData(
                        title=title,
                        summary=paper.get("abstract", "")[:500],
                        url=paper_url,
                        source="semantic_scholar",
                    ))

        elif name == "github_trending_ai" and isinstance(data, dict):
            for repo in data.get("items", [])[:10]:
                articles.append(ArticleData(
                    title=f"GitHub Trending: {repo.get('full_name', '')}",
                    summary=f"Stars: {repo.get('stargazers_count', 0)} | {repo.get('description', '')}",
                    url=repo.get("html_url", ""),
                    source="github_trending",
                ))

        return articles


class CrawlerManager:
    """Manager for all crawlers."""

    def __init__(self):
        self.arxiv = ArxivCrawler()
        self.web = WebCrawler()
        self.trend = TrendApiCrawler()

    def crawl_all(self) -> List[ArticleData]:
        """Run all crawlers."""
        articles = []

        # Tier 1: arXiv (core)
        articles.extend(self.arxiv.crawl())

        # Tier 1: HuggingFace Daily Papers
        hf_crawler = GenericRssCrawler("huggingface_papers")
        articles.extend(hf_crawler.crawl(HF_DAILY_PAPERS_RSS, limit=10))

        # Tier 1: AI Lab Blogs
        for lab_name, rss_url in AI_LAB_RSS_URLS.items():
            crawler = GenericRssCrawler(lab_name)
            articles.extend(crawler.crawl(rss_url, limit=5))

        # Tier 1: Chinese AI Media
        for name, rss_url in CN_AI_RSS_URLS.items():
            crawler = GenericRssCrawler(name)
            articles.extend(crawler.crawl(rss_url, limit=5))

        # Tier 1: AI News
        for name, rss_url in AI_NEWS_RSS_URLS.items():
            crawler = GenericRssCrawler(name)
            articles.extend(crawler.crawl(rss_url, limit=5))

        # Tier 1: Reddit
        for name, rss_url in REDDIT_RSS_URLS.items():
            crawler = GenericRssCrawler(name)
            articles.extend(crawler.crawl(rss_url, limit=5))

        # Tier 2: Trend APIs
        articles.extend(self.trend.crawl_trends())

        logger.info(f"Total articles crawled: {len(articles)}")
        return articles

    def crawl_manual(self, url: str, source: str = "manual") -> Optional[ArticleData]:
        """Crawl a manually submitted URL."""
        return self.web.crawl_url(url, source)
