"""Content processing: cleaning, deduplication, and leveling."""
import hashlib
import logging
import re
from typing import List, Optional, Tuple

from app.models import Article
from app.crawler import ArticleData

logger = logging.getLogger(__name__)


class ContentCleaner:
    """Clean raw HTML/text content."""

    @staticmethod
    def clean(text: str) -> str:
        """Clean text content."""
        if not text:
            return ""

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,;:!?-]', '', text)

        return text.strip()

    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
        """Extract simple keywords from text."""
        # Simple keyword extraction based on frequency
        words = re.findall(r'\b[A-Za-z]{4,}\b', text.lower())

        # Common stop words
        stop_words = {
            'this', 'that', 'with', 'from', 'they', 'have', 'will',
            'been', 'their', 'were', 'said', 'each', 'which',
            'paper', 'study', 'research', 'using', 'based', 'model'
        }

        # Count word frequency
        word_counts = {}
        for word in words:
            if word not in stop_words:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Return top keywords
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:max_keywords]]


class Deduplicator:
    """Deduplicate articles."""

    @staticmethod
    def compute_simhash(text: str) -> str:
        """Compute simple simhash for deduplication."""
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return ""

        # Simple hash based on word frequencies
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        # Create hash from top words
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        hash_input = " ".join(f"{w}:{c}" for w, c in top_words)
        return hashlib.md5(hash_input.encode()).hexdigest()

    @staticmethod
    def is_duplicate(article: ArticleData, existing_articles: List[Article]) -> bool:
        """Check if article is a duplicate."""
        # Check URL match
        for existing in existing_articles:
            if article.url == existing.url:
                return True

        # Check title similarity
        for existing in existing_articles:
            if Deduplicator._title_similarity(article.title, existing.title) > 0.85:
                return True

        return False

    @staticmethod
    def _title_similarity(title1: str, title2: str) -> float:
        """Calculate title similarity using Jaccard index."""
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)


class ContentLeveler:
    """Determine content level (L1/L2/L3)."""

    @staticmethod
    def determine_level(article: ArticleData, has_ai_summary: bool = False) -> int:
        """Determine content level."""
        # L1: Basic info only
        # L2: Has AI-generated summary
        # L3: Has deep analysis (not implemented in MVP)

        if has_ai_summary:
            return 2
        return 1


class ContentProcessor:
    """Main content processor."""

    def __init__(self):
        self.cleaner = ContentCleaner()
        self.deduplicator = Deduplicator()
        self.leveler = ContentLeveler()

    def process(self, article: ArticleData, existing_articles: List[Article] = None) -> Tuple[bool, Optional[dict]]:
        """
        Process a crawled article.

        Returns:
            (is_valid, processed_data)
        """
        if existing_articles is None:
            existing_articles = []

        # Clean content
        cleaned_title = self.cleaner.clean(article.title)
        cleaned_summary = self.cleaner.clean(article.summary)

        if not cleaned_title:
            logger.warning("Article has no title after cleaning")
            return False, None

        # Check for duplicates
        article.title = cleaned_title
        article.summary = cleaned_summary

        if self.deduplicator.is_duplicate(article, existing_articles):
            logger.info(f"Duplicate article found: {article.url}")
            return False, None

        # Extract keywords
        keywords = self.cleaner.extract_keywords(
            f"{cleaned_title} {cleaned_summary}", max_keywords=5
        )

        # Determine level (will be updated after AI analysis)
        level = self.leveler.determine_level(article, has_ai_summary=False)

        return True, {
            "title": cleaned_title,
            "summary": cleaned_summary,
            "url": article.url,
            "source": article.source,
            "level": level,
            "keywords": keywords,
            "authors": article.authors,
            "published_at": article.published_at,
        }
