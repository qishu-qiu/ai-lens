"""Pydantic schemas for API."""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict


class ArticleBase(BaseModel):
    title: str
    summary: Optional[str] = None
    url: str
    source: str
    level: int = 1
    keywords: List[str] = []
    authors: Optional[str] = None
    published_at: Optional[datetime] = None


class ArticleCreate(ArticleBase):
    pass


class ArticleResponse(ArticleBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    updated_at: datetime


class ArticleSearch(BaseModel):
    query: Optional[str] = None
    source: Optional[str] = None
    level: Optional[int] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None


class UpdateLogResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    update_type: str
    description: str
    reason: str
    status: str
    diff_summary: Optional[str] = None
    created_at: datetime
    executed_at: Optional[datetime] = None


class SnapshotResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    version: str
    article_count: int
    metadata_json: dict
    created_at: datetime


class DailyBriefing(BaseModel):
    date: str
    total_articles: int
    new_articles: int
    updated_articles: int
    articles: List[ArticleResponse]


class CrawlRequest(BaseModel):
    url: str
    source: str = "manual"


class CrawlResponse(BaseModel):
    success: bool
    message: str
    article_id: Optional[int] = None
