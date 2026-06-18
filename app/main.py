"""FastAPI main application."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Depends, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.config import DEBUG
from app.models import init_db, get_db
from app.crawler import CrawlerManager
from app.content_processor import ContentProcessor
from app.ai_service import AIService
from app.knowledge_base import KnowledgeBase
from app.self_update import UpdateManager
from app.schemas import ArticleSearch, CrawlRequest

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
init_db()

crawler_manager = CrawlerManager()
content_processor = ContentProcessor()
ai_service = AIService()


from app.scheduler import start_scheduler, shutdown_scheduler
from app.summarizer import batch_summarize, summarize_all_pending
from app.clusterer import cluster_articles, get_all_topics, get_topic_articles

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("AI-Self-Evolution starting up...")
    start_scheduler()
    yield
    logger.info("AI-Self-Evolution shutting down...")
    shutdown_scheduler()


app = FastAPI(
    title="AI-Self-Evolution",
    description="Self-evolving AI knowledge base system",
    version="1.0.0",
    debug=DEBUG,
    lifespan=lifespan,
)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ========== API Routes ==========

@app.get("/api/articles")
async def api_list_articles(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List articles."""
    kb = KnowledgeBase(db)
    articles = kb.list_articles(skip=skip, limit=limit)
    return {"articles": articles, "total": len(articles)}


@app.post("/api/articles/search")
async def api_search_articles(search: ArticleSearch, db: Session = Depends(get_db)):
    """Search articles."""
    kb = KnowledgeBase(db)
    articles = kb.search_articles(search)
    return {"articles": articles, "total": len(articles)}


@app.get("/api/articles/{article_id}")
async def api_get_article(article_id: int, db: Session = Depends(get_db)):
    """Get article by ID."""
    kb = KnowledgeBase(db)
    article = kb.get_article(article_id)
    if not article:
        return {"error": "Article not found"}
    return article


@app.post("/api/crawl")
async def api_crawl_manual(request: CrawlRequest, db: Session = Depends(get_db)):
    """Manually crawl a URL."""
    article_data = crawler_manager.crawl_manual(request.url, request.source)
    if not article_data:
        return {"success": False, "message": "Failed to crawl URL"}

    # Process content
    existing = KnowledgeBase(db).get_all_articles()
    is_valid, processed = content_processor.process(article_data, existing)

    if not is_valid:
        return {"success": False, "message": "Content processing failed or duplicate"}

    # Create article
    kb = KnowledgeBase(db)
    article = kb.create_article(processed)

    # Try AI analysis
    try:
        analysis = await ai_service.analyze_article(article.title, article.summary or "")
        if analysis["summary"]:
            kb.update_article_summary(article.id, analysis["summary"])
        if analysis["keywords"]:
            kb.update_article_keywords(article.id, analysis["keywords"])
    except Exception as e:
        logger.error(f"AI analysis failed: {e}")

    return {
        "success": True,
        "message": f"Article crawled: {article.title}",
        "article_id": article.id,
    }


@app.post("/api/crawl/auto")
async def api_crawl_auto(db: Session = Depends(get_db)):
    """Run automatic crawl."""
    articles_data = crawler_manager.crawl_all()
    kb = KnowledgeBase(db)
    existing = kb.get_all_articles()

    created = 0
    failed = 0

    for article_data in articles_data:
        is_valid, processed = content_processor.process(article_data, existing)
        if not is_valid:
            failed += 1
            continue

        article = kb.create_article(processed)
        existing.append(article)
        created += 1

        # Try AI analysis
        try:
            analysis = await ai_service.analyze_article(article.title, article.summary or "")
            if analysis["summary"]:
                kb.update_article_summary(article.id, analysis["summary"])
            if analysis["keywords"]:
                kb.update_article_keywords(article.id, analysis["keywords"])
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")

    return {
        "success": True,
        "created": created,
        "failed": failed,
        "total": len(articles_data),
    }


@app.get("/api/briefing")
async def api_daily_briefing(db: Session = Depends(get_db)):
    """Get daily briefing."""
    kb = KnowledgeBase(db)
    briefing = kb.get_daily_briefing()
    return briefing


@app.get("/api/updates")
async def api_update_history(db: Session = Depends(get_db)):
    """Get self-update history."""
    manager = UpdateManager(db)
    history = manager.updater.get_update_history()
    return {"updates": history}


@app.post("/api/updates/run")
async def api_run_updates(db: Session = Depends(get_db)):
    """Run self-update cycle."""
    manager = UpdateManager(db)
    result = manager.run_update_cycle()
    return result


@app.get("/api/snapshots")
async def api_list_snapshots(db: Session = Depends(get_db)):
    """List knowledge base snapshots."""
    kb = KnowledgeBase(db)
    snapshots = kb.list_snapshots()
    return {"snapshots": snapshots}


@app.post("/api/snapshots/create")
async def api_create_snapshot(db: Session = Depends(get_db)):
    """Create a new snapshot."""
    kb = KnowledgeBase(db)
    total = len(kb.get_all_articles())

    latest = kb.get_latest_snapshot()
    version = f"v1.{(latest.id + 1) if latest else 0}" if latest else "v1.0"

    snapshot = kb.create_snapshot(version=version, article_count=total)
    return {"snapshot": snapshot}


# ========== NEW v2.0 API Routes ==========

@app.post("/api/summarize")
async def api_summarize(limit: int = 50, db: Session = Depends(get_db)):
    """Batch generate Chinese summaries for articles."""
    result = batch_summarize(limit=limit)
    return {"success": True, "result": result}


@app.post("/api/cluster")
async def api_cluster(limit: int = 80, db: Session = Depends(get_db)):
    """Run AI topic clustering on articles."""
    result = cluster_articles(limit=limit)
    return {"success": True, "result": result}


@app.get("/api/topics")
async def api_list_topics(db: Session = Depends(get_db)):
    """List all topics with article counts."""
    topics = get_all_topics(db)
    return {"topics": topics}


@app.get("/api/topics/{topic_id}/articles")
async def api_topic_articles(topic_id: int, db: Session = Depends(get_db)):
    """Get articles for a specific topic."""
    articles = get_topic_articles(topic_id, db)
    return {"topic_id": topic_id, "articles": articles}


# ========== Web UI Routes ==========

@app.get("/", response_class=HTMLResponse)
async def web_index(request: Request, db: Session = Depends(get_db)):
    """Home page - daily briefing."""
    kb = KnowledgeBase(db)
    briefing = kb.get_daily_briefing()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "briefing": briefing,
    }, media_type="text/html")


@app.get("/knowledge", response_class=HTMLResponse)
async def web_knowledge(request: Request, db: Session = Depends(get_db)):
    """Knowledge base page."""
    kb = KnowledgeBase(db)
    articles = kb.list_articles(limit=50)
    return templates.TemplateResponse("knowledge.html", {
        "request": request,
        "articles": articles,
    })


@app.get("/article/{article_id}", response_class=HTMLResponse)
async def web_article(request: Request, article_id: int, db: Session = Depends(get_db)):
    """Article detail page."""
    kb = KnowledgeBase(db)
    article = kb.get_article(article_id)
    return templates.TemplateResponse("article.html", {
        "request": request,
        "article": article,
    })


@app.get("/updates", response_class=HTMLResponse)
async def web_updates(request: Request, db: Session = Depends(get_db)):
    """Self-update history page."""
    manager = UpdateManager(db)
    updates = manager.updater.get_update_history()
    return templates.TemplateResponse("updates.html", {
        "request": request,
        "updates": updates,
    })


@app.get("/snapshots", response_class=HTMLResponse)
async def web_snapshots(request: Request, db: Session = Depends(get_db)):
    """Snapshots page."""
    kb = KnowledgeBase(db)
    snapshots = kb.list_snapshots()
    return templates.TemplateResponse("snapshots.html", {
        "request": request,
        "snapshots": snapshots,
    })


@app.get("/crawl", response_class=HTMLResponse)
async def web_crawl(request: Request):
    """Manual crawl page."""
    return templates.TemplateResponse("crawl.html", {"request": request})


@app.post("/crawl")
async def web_crawl_submit(request: Request, url: str = Form(...), db: Session = Depends(get_db)):
    """Handle manual crawl form submission."""
    article_data = crawler_manager.crawl_manual(url)
    if not article_data:
        return templates.TemplateResponse("crawl.html", {
            "request": request,
            "error": "Failed to crawl URL",
        })

    existing = KnowledgeBase(db).get_all_articles()
    is_valid, processed = content_processor.process(article_data, existing)

    if not is_valid:
        return templates.TemplateResponse("crawl.html", {
            "request": request,
            "error": "Content processing failed or duplicate",
        })

    kb = KnowledgeBase(db)
    article = kb.create_article(processed)

    # Try AI analysis
    try:
        analysis = await ai_service.analyze_article(article.title, article.summary or "")
        if analysis["summary"]:
            kb.update_article_summary(article.id, analysis["summary"])
    except Exception as e:
        logger.error(f"AI analysis failed: {e}")

    return templates.TemplateResponse("crawl.html", {
        "request": request,
        "success": f"Article crawled: {article.title}",
        "article": article,
    })


@app.get("/search", response_class=HTMLResponse)
async def web_search(request: Request, q: str = "", db: Session = Depends(get_db)):
    """Search page."""
    kb = KnowledgeBase(db)
    if q:
        search = ArticleSearch(query=q)
        articles = kb.search_articles(search)
    else:
        articles = []
    return templates.TemplateResponse("search.html", {
        "request": request,
        "query": q,
        "articles": articles,
    })
