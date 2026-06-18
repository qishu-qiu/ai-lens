"""Knowledge base CRUD operations."""
import logging
from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy import or_, and_
from sqlalchemy.orm import Session

from app.models import Article, Snapshot
from app.schemas import ArticleCreate, ArticleSearch

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """Knowledge base operations."""

    def __init__(self, db: Session):
        self.db = db

    def create_article(self, article_data: dict) -> Article:
        """Create a new article."""
        article = Article(**article_data)
        self.db.add(article)
        self.db.commit()
        self.db.refresh(article)
        logger.info(f"Created article: {article.title[:50]}...")
        return article

    def get_article(self, article_id: int) -> Optional[Article]:
        """Get article by ID."""
        return self.db.query(Article).filter(Article.id == article_id).first()

    def get_article_by_url(self, url: str) -> Optional[Article]:
        """Get article by URL."""
        return self.db.query(Article).filter(Article.url == url).first()

    def list_articles(self, skip: int = 0, limit: int = 100) -> List[Article]:
        """List articles with pagination."""
        return self.db.query(Article).order_by(Article.created_at.desc()).offset(skip).limit(limit).all()

    def search_articles(self, search: ArticleSearch) -> List[Article]:
        """Search articles with filters."""
        query = self.db.query(Article)

        if search.query:
            search_pattern = f"%{search.query}%"
            query = query.filter(
                or_(
                    Article.title.ilike(search_pattern),
                    Article.summary.ilike(search_pattern),
                )
            )

        if search.source:
            query = query.filter(Article.source == search.source)

        if search.level:
            query = query.filter(Article.level == search.level)

        if search.date_from:
            query = query.filter(Article.created_at >= search.date_from)

        if search.date_to:
            query = query.filter(Article.created_at <= search.date_to)

        return query.order_by(Article.created_at.desc()).limit(100).all()

    def get_daily_briefing(self, date: Optional[datetime] = None) -> dict:
        """Get daily briefing for a specific date."""
        if date is None:
            date = datetime.utcnow()

        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

        articles = self.db.query(Article).filter(
            and_(
                Article.created_at >= start_of_day,
                Article.created_at < end_of_day,
            )
        ).order_by(Article.created_at.desc()).all()

        total = self.db.query(Article).count()

        return {
            "date": start_of_day.strftime("%Y-%m-%d"),
            "total_articles": total,
            "new_articles": len(articles),
            "updated_articles": 0,  # TODO: track updates
            "articles": articles,
        }

    def get_all_articles(self) -> List[Article]:
        """Get all articles for deduplication."""
        return self.db.query(Article).all()

    def update_article_level(self, article_id: int, level: int) -> Optional[Article]:
        """Update article level after AI analysis."""
        article = self.get_article(article_id)
        if article:
            article.level = level
            article.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(article)
        return article

    def update_article_summary(self, article_id: int, summary: str) -> Optional[Article]:
        """Update article summary after AI analysis."""
        article = self.get_article(article_id)
        if article:
            article.summary = summary
            article.level = 2  # L2 if we have AI summary
            article.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(article)
        return article

    def update_article_keywords(self, article_id: int, keywords: List[str]) -> Optional[Article]:
        """Update article keywords."""
        article = self.get_article(article_id)
        if article:
            article.keywords = keywords
            article.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(article)
        return article

    # Snapshot operations

    def create_snapshot(self, version: str, article_count: int, metadata: dict = None) -> Snapshot:
        """Create a knowledge base snapshot."""
        snapshot = Snapshot(
            version=version,
            article_count=article_count,
            metadata_json=metadata or {},
        )
        self.db.add(snapshot)
        self.db.commit()
        self.db.refresh(snapshot)
        return snapshot

    def get_latest_snapshot(self) -> Optional[Snapshot]:
        """Get latest snapshot."""
        return self.db.query(Snapshot).order_by(Snapshot.created_at.desc()).first()

    def list_snapshots(self) -> List[Snapshot]:
        """List all snapshots."""
        return self.db.query(Snapshot).order_by(Snapshot.created_at.desc()).all()
