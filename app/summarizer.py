"""AI Chinese Summary Generator using cloud LLM API."""
import os
import time
import json
import requests
from typing import Optional, List
from sqlalchemy.orm import Session

from app.models import SessionLocal, Article


# Agnes API configuration (OpenAI-compatible)
LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("AGNES_API_KEY", "sk-thpVPgLvZubGt2neZLRUrYpazpBppXKU5YbQhfOjdYaF33pA"))
LLM_API_URL = os.getenv("LLM_API_URL", "https://apihub.agnes-ai.com/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "agnes-2.0-flash")

# Rate limiting
MAX_RPS = 3  # max requests per second
_last_call_time = 0.0


SUMMARY_PROMPT = """请用中文提炼以下AI论文/文章的核心贡献，要求：
1. 200字以内
2. 面向技术从业者，语言专业但易懂
3. 包含：研究问题、核心方法、主要结论
4. 不要翻译原文，而是提炼核心要点

标题：{title}
原文摘要：{summary}

请直接输出中文摘要，不要加任何前缀或解释。"""


def _rate_limit():
    """Ensure we don't exceed rate limit."""
    global _last_call_time
    elapsed = time.time() - _last_call_time
    min_interval = 1.0 / MAX_RPS
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_call_time = time.time()


def generate_chinese_summary(title: str, summary: str) -> Optional[str]:
    """Generate Chinese summary for an article using Agnes API."""
    if not LLM_API_KEY:
        return None

    _rate_limit()

    prompt = SUMMARY_PROMPT.format(title=title, summary=summary[:2000])

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
                "max_tokens": 300,
                "temperature": 0.3,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content.strip()
    except Exception as e:
        print(f"[Summarizer] API error: {e}")
        return None


def batch_summarize(article_ids: Optional[List[int]] = None, limit: int = 50) -> dict:
    """Batch generate Chinese summaries for articles.

    Args:
        article_ids: Specific article IDs to process. If None, processes articles without chinese_summary.
        limit: Max articles to process in one batch.

    Returns:
        dict with 'success', 'failed', 'skipped' counts.
    """
    db = SessionLocal()
    try:
        query = db.query(Article)
        if article_ids:
            query = query.filter(Article.id.in_(article_ids))
        else:
            query = query.filter(
                (Article.chinese_summary == None) | (Article.chinese_summary == "")
            )

        articles = query.limit(limit).all()

        results = {"success": 0, "failed": 0, "skipped": 0}

        for article in articles:
            # Skip if already has summary
            if article.chinese_summary and len(article.chinese_summary) > 10:
                results["skipped"] += 1
                continue

            # Skip if no English summary to work with
            if not article.summary or len(article.summary) < 20:
                results["skipped"] += 1
                continue

            summary = generate_chinese_summary(article.title, article.summary)
            if summary:
                article.chinese_summary = summary
                db.commit()
                results["success"] += 1
                print(f"[Summarizer] OK: {article.title[:50]}...")
            else:
                results["failed"] += 1
                print(f"[Summarizer] FAIL: {article.title[:50]}...")

        return results
    finally:
        db.close()


def summarize_all_pending() -> dict:
    """Process all articles pending Chinese summary."""
    return batch_summarize(limit=100)
