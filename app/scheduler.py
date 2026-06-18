"""Scheduled tasks for automatic crawling and self-updates."""
import logging
from datetime import datetime
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy.orm import Session

from app.config import CRAWL_INTERVAL_HOURS
from app.models import SessionLocal
from app.crawler import CrawlerManager
from app.content_processor import ContentProcessor
from app.ai_service import AIService
from app.knowledge_base import KnowledgeBase
from app.self_update import UpdateManager
from app.self_healing import HealthMonitor, DataQualityChecker
from app.knowledge_retirement import KnowledgeRetirementEngine

logger = logging.getLogger(__name__)


class TaskScheduler:
    """Background task scheduler."""

    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.crawler = CrawlerManager()
        self.processor = ContentProcessor()
        self.ai = AIService()

    def start(self):
        """Start the scheduler."""
        # Schedule automatic crawl every N hours
        self.scheduler.add_job(
            self._auto_crawl,
            trigger=CronTrigger(hour=f"*/{CRAWL_INTERVAL_HOURS}"),
            id="auto_crawl",
            name="Automatic article crawling",
            replace_existing=True,
        )

        # Schedule self-update check daily at 3 AM
        self.scheduler.add_job(
            self._self_update,
            trigger=CronTrigger(hour=3, minute=0),
            id="self_update",
            name="Self-update cycle",
            replace_existing=True,
        )

        # Schedule snapshot creation daily at 4 AM
        self.scheduler.add_job(
            self._create_snapshot,
            trigger=CronTrigger(hour=4, minute=0),
            id="create_snapshot",
            name="Create knowledge base snapshot",
            replace_existing=True,
        )

        # --- Self-healing tasks ---

        # Data quality check after each crawl
        self.scheduler.add_job(
            self._post_crawl_quality_check,
            trigger=None,  # triggered manually after crawl
            id="post_crawl_quality_check",
            name="Post-crawl data quality check",
            replace_existing=True,
        )

        # Weekly self-healing cycle: Sunday 5 AM
        self.scheduler.add_job(
            self._weekly_self_healing,
            trigger=CronTrigger(day_of_week="sun", hour=5, minute=0),
            id="weekly_self_healing",
            name="Weekly self-healing cycle",
            replace_existing=True,
        )

        self.scheduler.start()
        logger.info("Task scheduler started")

    def shutdown(self):
        """Shutdown the scheduler."""
        self.scheduler.shutdown()
        logger.info("Task scheduler shutdown")

    def _get_db(self) -> Session:
        """Get database session."""
        return SessionLocal()

    def _auto_crawl(self):
        """Run automatic crawl."""
        logger.info("Starting automatic crawl...")
        db = self._get_db()
        try:
            articles_data = self.crawler.crawl_all()
            kb = KnowledgeBase(db)
            existing = kb.get_all_articles()

            created = 0
            for article_data in articles_data:
                is_valid, processed = self.processor.process(article_data, existing)
                if not is_valid:
                    continue

                article = kb.create_article(processed)
                existing.append(article)
                created += 1

                # Try AI analysis
                try:
                    import asyncio
                    analysis = asyncio.run(self.ai.analyze_article(
                        article.title, article.summary or ""
                    ))
                    if analysis["summary"]:
                        kb.update_article_summary(article.id, analysis["summary"])
                    if analysis["keywords"]:
                        kb.update_article_keywords(article.id, analysis["keywords"])
                except Exception as e:
                    logger.error(f"AI analysis failed: {e}")

            logger.info(f"Auto crawl complete: {created} articles created")

            # Trigger post-crawl quality check
            try:
                self._post_crawl_quality_check()
            except Exception as e:
                logger.error(f"Post-crawl quality check failed: {e}")

        except Exception as e:
            logger.error(f"Auto crawl failed: {e}")
        finally:
            db.close()

    def _self_update(self):
        """Run self-update cycle."""
        logger.info("Starting self-update cycle...")
        db = self._get_db()
        try:
            manager = UpdateManager(db)
            result = manager.run_update_cycle()
            logger.info(f"Self-update complete: {result}")
        except Exception as e:
            logger.error(f"Self-update failed: {e}")
        finally:
            db.close()

    def _create_snapshot(self):
        """Create knowledge base snapshot."""
        logger.info("Creating snapshot...")
        db = self._get_db()
        try:
            kb = KnowledgeBase(db)
            total = len(kb.get_all_articles())

            latest = kb.get_latest_snapshot()
            version = f"v1.{(latest.id + 1) if latest else 0}" if latest else "v1.0"

            kb.create_snapshot(
                version=version,
                article_count=total,
                metadata={"updates": 0, "timestamp": datetime.utcnow().isoformat()}
            )
            logger.info(f"Snapshot created: {version} ({total} articles)")

        except Exception as e:
            logger.error(f"Snapshot creation failed: {e}")
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Self-healing tasks
    # ------------------------------------------------------------------

    def _post_crawl_quality_check(self):
        """Post-crawl data quality check.

        Checks:
        - Empty summary articles (>5 triggers batch_summarize)
        - Duplicate URLs cleanup
        - Logs check results
        """
        logger.info("Starting post-crawl data quality check...")
        db = self._get_db()
        try:
            # Collect article data for quality checking
            articles = [
                {
                    "id": a.id,
                    "title": a.title,
                    "url": a.url,
                    "chinese_summary": a.chinese_summary,
                    "summary": a.summary,
                }
                for a in db.query(
                    self._get_article_model()
                ).all()
            ]

            checker = DataQualityChecker()

            # Check 1: Empty summaries
            empty_issues = checker.check_empty_summary(articles)
            empty_count = len(empty_issues)
            logger.info("Empty summary check: %d articles with empty summary", empty_count)

            if empty_count > 5:
                logger.info(
                    "Empty summary count (%d) exceeds threshold (5), "
                    "triggering batch_summarize...",
                    empty_count,
                )
                try:
                    from app.summarizer import batch_summarize
                    empty_ids = [issue["article_id"] for issue in empty_issues]
                    result = batch_summarize(article_ids=empty_ids, limit=50)
                    logger.info(
                        "batch_summarize result: success=%d, failed=%d, skipped=%d",
                        result["success"], result["failed"], result["skipped"],
                    )
                except Exception as e:
                    logger.error("batch_summarize failed: %s", e)

            # Check 2: Duplicate URLs
            dup_issues = checker.check_duplicate_urls(articles)
            dup_count = len(dup_issues)
            logger.info("Duplicate URL check: %d groups of duplicates", dup_count)

            if dup_count > 0:
                try:
                    self._cleanup_duplicate_urls(db, dup_issues)
                    logger.info("Duplicate URL cleanup completed")
                except Exception as e:
                    logger.error("Duplicate URL cleanup failed: %s", e)

            logger.info(
                "Data quality check complete: empty=%d, duplicates=%d",
                empty_count, dup_count,
            )

        except Exception as e:
            logger.error("Data quality check failed: %s", e)
        finally:
            db.close()

    def _cleanup_duplicate_urls(self, db: Session, dup_issues: list):
        """Clean up duplicate URLs, keeping the earliest record.

        Args:
            db: Database session.
            dup_issues: List of duplicate URL issues from DataQualityChecker.
        """
        from app.models import Article

        for issue in dup_issues:
            ids = issue["article_ids"]
            if len(ids) <= 1:
                continue
            # Keep the first (earliest) record, delete the rest
            keep_id = ids[0]
            for remove_id in ids[1:]:
                article = db.query(Article).filter(Article.id == remove_id).first()
                if article:
                    db.delete(article)
                    logger.info("Removed duplicate article id=%d (url: %s)", remove_id, issue["url"][:80])
        db.commit()

    def _weekly_self_healing(self):
        """Weekly self-healing cycle (Sunday 5 AM).

        Steps:
        1. Run knowledge_retirement cycle
        2. Check if concept entries need updating
        3. Generate weekly briefing
        """
        logger.info("===== Weekly self-healing cycle starting =====")
        cycle_start = datetime.utcnow()

        # Step 1: Knowledge retirement
        try:
            logger.info("Running knowledge retirement cycle...")
            engine = KnowledgeRetirementEngine()
            report = engine.run_retirement_cycle()
            logger.info("Knowledge retirement complete: %s", report.get("summary", {}))
            engine.close()
        except Exception as e:
            logger.error("Knowledge retirement failed: %s", e)

        # Step 2: Check and update concept entries
        try:
            logger.info("Checking concept entries for updates...")
            from app.concept_manager import ConceptManager
            cm = ConceptManager()
            active_concepts = cm.get_active_concepts()
            updated_count = 0
            for concept in active_concepts:
                try:
                    updated = cm.update_concept(concept.id)
                    if updated:
                        updated_count += 1
                except Exception as e:
                    logger.error("Concept update failed (id=%d): %s", concept.id, e)
            logger.info("Concept update check complete: %d/%d updated", updated_count, len(active_concepts))
            cm.close()
        except Exception as e:
            logger.error("Concept update check failed: %s", e)

        # Step 3: Generate weekly briefing
        try:
            logger.info("Generating weekly briefing...")
            # Import and run the weekly briefing generator
            import importlib.util
            briefing_script = importlib.util.spec_from_file_location(
                "gen_weekly_briefing",
                str(Path(__file__).resolve().parent.parent / "gen_weekly_briefing.py"),
            )
            if briefing_script and briefing_script.loader:
                mod = importlib.util.module_from_spec(briefing_script)
                briefing_script.loader.exec_module(mod)
                result = mod.generate_weekly_briefing()
                logger.info("Weekly briefing generated: %s", result.get("status", "unknown"))
        except Exception as e:
            logger.error("Weekly briefing generation failed: %s", e)

        elapsed = (datetime.utcnow() - cycle_start).total_seconds()
        logger.info("===== Weekly self-healing cycle complete (%.1fs) =====", elapsed)

    def _get_article_model(self):
        """Lazy import Article model to avoid circular imports."""
        from app.models import Article
        return Article


# Global scheduler instance
_scheduler = None


def get_scheduler() -> TaskScheduler:
    """Get or create scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = TaskScheduler()
    return _scheduler


def start_scheduler():
    """Start the background scheduler."""
    scheduler = get_scheduler()
    scheduler.start()
    return scheduler


def shutdown_scheduler():
    """Shutdown the background scheduler."""
    global _scheduler
    if _scheduler:
        _scheduler.shutdown()
        _scheduler = None
