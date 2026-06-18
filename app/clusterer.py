"""AI Topic Clustering using cloud LLM API."""
import os
import time
import json
import requests
from typing import List, Dict, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from app.models import SessionLocal, Article, Topic, ArticleTopic

# Agnes API configuration (OpenAI-compatible)
LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("AGNES_API_KEY", "sk-thpVPgLvZubGt2neZLRUrYpazpBppXKU5YbQhfOjdYaF33pA"))
LLM_API_URL = os.getenv("LLM_API_URL", "https://apihub.agnes-ai.com/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "agnes-2.0-flash")

MAX_RPS = 2
_last_call_time = 0.0


CLUSTER_PROMPT = """你是一位AI领域专家。请分析以下文章列表，提取5-10个核心研究主题。

要求：
1. 每个主题用中文命名（4-8个字）
2. 每个主题配一段50字以内的中文描述
3. 将每篇文章归类到最相关的1-3个主题
4. 返回严格的JSON格式

文章列表：
{articles}

请返回以下JSON格式（不要加任何markdown标记）：
{{
  "topics": [
    {{
      "name": "主题名称",
      "description": "主题描述",
      "keywords": ["关键词1", "关键词2"]
    }}
  ],
  "assignments": [
    {{
      "article_index": 0,
      "topic_indices": [0, 1],
      "relevance": [0.95, 0.7]
    }}
  ]
}}

注意：article_index对应文章在列表中的索引（从0开始），topic_indices对应topics数组中的索引。"""


def _rate_limit():
    global _last_call_time
    elapsed = time.time() - _last_call_time
    min_interval = 1.0 / MAX_RPS
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_call_time = time.time()


def _call_llm(prompt: str) -> Optional[str]:
    if not LLM_API_KEY:
        return None
    _rate_limit()
    try:
        resp = requests.post(
            LLM_API_URL,
            headers={
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4000,
                "temperature": 0.2,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[Clusterer] API error: {e}")
        return None


def _parse_json(text: str) -> Optional[dict]:
    """Extract JSON from LLM response."""
    # Try to find JSON block
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        text = text[start:end].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end+1])
            except:
                pass
        return None


def cluster_articles(article_ids: Optional[List[int]] = None, limit: int = 80) -> dict:
    """Cluster articles into topics using LLM.

    Returns:
        dict with 'topics_created', 'assignments_created', 'articles_processed'
    """
    db = SessionLocal()
    try:
        query = db.query(Article)
        if article_ids:
            query = query.filter(Article.id.in_(article_ids))
        query = query.order_by(Article.created_at.desc())
        articles = query.limit(limit).all()

        if len(articles) < 5:
            return {"topics_created": 0, "assignments_created": 0, "articles_processed": len(articles), "error": "Too few articles"}

        # Build article list for prompt
        article_texts = []
        for i, a in enumerate(articles):
            text = f"[{i}] {a.title}"
            if a.summary and len(a.summary) > 10:
                text += f" | {a.summary[:200]}"
            if a.keywords:
                text += f" | 关键词: {', '.join(a.keywords[:5])}"
            article_texts.append(text)

        prompt = CLUSTER_PROMPT.format(articles="\n".join(article_texts))

        print(f"[Clusterer] Clustering {len(articles)} articles...")
        response = _call_llm(prompt)
        if not response:
            return {"topics_created": 0, "assignments_created": 0, "articles_processed": len(articles), "error": "API failed"}

        data = _parse_json(response)
        if not data or "topics" not in data or "assignments" not in data:
            print(f"[Clusterer] Invalid response: {response[:500]}")
            return {"topics_created": 0, "assignments_created": 0, "articles_processed": len(articles), "error": "Invalid JSON"}

        # Clear old topics and assignments (simple approach: recreate)
        db.query(ArticleTopic).delete()
        db.query(Topic).delete()
        db.commit()

        # Create topics
        topic_map = {}  # index -> Topic object
        for i, t_data in enumerate(data["topics"]):
            topic = Topic(
                name=t_data.get("name", f"主题{i+1}"),
                description=t_data.get("description", ""),
                keywords=t_data.get("keywords", []),
                first_seen=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )
            db.add(topic)
            db.flush()
            topic_map[i] = topic

        # Create assignments
        assignments_count = 0
        for assign in data.get("assignments", []):
            article_idx = assign.get("article_index", -1)
            topic_indices = assign.get("topic_indices", [])
            relevances = assign.get("relevance", [0.8] * len(topic_indices))

            if article_idx < 0 or article_idx >= len(articles):
                continue

            article = articles[article_idx]
            for ti, rel in zip(topic_indices, relevances):
                if ti not in topic_map:
                    continue
                at = ArticleTopic(
                    article_id=article.id,
                    topic_id=topic_map[ti].id,
                    relevance_score=min(max(float(rel), 0.0), 1.0),
                )
                db.add(at)
                assignments_count += 1

        db.commit()

        # Update topic last_updated
        for topic in topic_map.values():
            topic.last_updated = datetime.utcnow()
        db.commit()

        print(f"[Clusterer] Created {len(topic_map)} topics, {assignments_count} assignments")
        return {
            "topics_created": len(topic_map),
            "assignments_created": assignments_count,
            "articles_processed": len(articles),
        }

    finally:
        db.close()


def get_topic_articles(topic_id: int, db: Session = None) -> List[dict]:
    """Get all articles for a topic with relevance scores."""
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True
    try:
        results = db.query(Article, ArticleTopic.relevance_score).\
            join(ArticleTopic, Article.id == ArticleTopic.article_id).\
            filter(ArticleTopic.topic_id == topic_id).\
            order_by(ArticleTopic.relevance_score.desc()).\
            all()
        return [
            {
                "id": a.id,
                "title": a.title,
                "summary": a.summary,
                "chinese_summary": a.chinese_summary,
                "url": a.url,
                "source": a.source,
                "published_at": a.published_at.isoformat() if a.published_at else None,
                "relevance": rel,
            }
            for a, rel in results
        ]
    finally:
        if close_db:
            db.close()


def get_all_topics(db: Session = None) -> List[dict]:
    """Get all topics with article counts."""
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True
    try:
        topics = db.query(Topic).all()
        result = []
        for t in topics:
            count = db.query(ArticleTopic).filter(ArticleTopic.topic_id == t.id).count()
            result.append({
                "id": t.id,
                "name": t.name,
                "description": t.description,
                "keywords": t.keywords,
                "article_count": count,
                "first_seen": t.first_seen.isoformat() if t.first_seen else None,
                "last_updated": t.last_updated.isoformat() if t.last_updated else None,
            })
        return result
    finally:
        if close_db:
            db.close()
