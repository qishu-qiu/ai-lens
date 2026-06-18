"""SQLAlchemy database models."""
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.config import DATABASE_URL

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Article(Base):
    """Article model for knowledge base."""
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False)
    summary = Column(Text, nullable=True)
    chinese_summary = Column(Text, nullable=True)  # AI-generated Chinese summary
    url = Column(String(1000), nullable=False, unique=True)
    source = Column(String(100), nullable=False)  # e.g., 'arxiv', 'openai_blog'
    level = Column(Integer, default=1)  # L1=1, L2=2, L3=3
    keywords = Column(JSON, default=list)
    authors = Column(String(500), nullable=True)
    published_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class UpdateLog(Base):
    """Self-update log."""
    __tablename__ = "updates"

    id = Column(Integer, primary_key=True, index=True)
    update_type = Column(String(50), nullable=False)  # 'low_risk', 'medium_risk', 'high_risk'
    description = Column(Text, nullable=False)
    reason = Column(Text, nullable=False)
    status = Column(String(50), default="pending")  # 'pending', 'approved', 'rejected', 'executed'
    diff_summary = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    executed_at = Column(DateTime, nullable=True)


class Config(Base):
    """System configuration."""
    __tablename__ = "configs"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), nullable=False, unique=True)
    value = Column(Text, nullable=False)
    version = Column(Integer, default=1)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Topic(Base):
    """Topic model for article clustering."""
    __tablename__ = "topics"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    keywords = Column(JSON, default=list)
    concept_summary = Column(Text, nullable=True)  # AI-generated concept explanation
    first_seen = Column(DateTime, nullable=True)
    last_updated = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ArticleTopic(Base):
    """Association table for articles and topics."""
    __tablename__ = "article_topics"

    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(Integer, nullable=False, index=True)
    topic_id = Column(Integer, nullable=False, index=True)
    relevance_score = Column(Integer, default=80)  # 0-100


class Snapshot(Base):
    """Knowledge base version snapshot."""
    __tablename__ = "snapshots"

    id = Column(Integer, primary_key=True, index=True)
    version = Column(String(20), nullable=False, unique=True)
    article_count = Column(Integer, default=0)
    topic_count = Column(Integer, default=0)
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
