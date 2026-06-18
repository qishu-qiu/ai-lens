"""Self-update module - core differentiator."""
import json
import logging
from datetime import datetime
from typing import List, Optional, Dict

from sqlalchemy.orm import Session

from app.config import CONSTITUTION_RULES
from app.models import Article, UpdateLog, Config
from app.ai_service import AIService

logger = logging.getLogger(__name__)


class ConstitutionChecker:
    """Check updates against constitution rules."""

    def __init__(self, rules: dict = None):
        self.rules = rules or CONSTITUTION_RULES

    def check_update(self, update_type: str, description: str) -> tuple[bool, str]:
        """
        Check if update violates constitution.

        Returns:
            (is_allowed, reason)
        """
        # Rule 1: No file should exceed max lines
        if "file" in description.lower() and "lines" in description.lower():
            return False, "Constitution: Cannot modify file line limits"

        # Rule 2: No function should exceed max lines
        if "function" in description.lower() and "lines" in description.lower():
            return False, "Constitution: Cannot modify function line limits"

        # Rule 3: Cannot remove tests
        if "remove" in description.lower() and "test" in description.lower():
            return False, "Constitution: Cannot remove tests"

        # Rule 4: Low risk updates only for auto-execution
        if update_type == "low_risk":
            return True, "Low risk update approved by constitution"

        return True, "Update passes constitution check"


class SelfUpdater:
    """Self-update engine."""

    def __init__(self, db: Session):
        self.db = db
        self.constitution = ConstitutionChecker()
        self.ai = AIService()

    def scan_for_updates(self) -> List[dict]:
        """
        Scan recent articles for potential system updates.

        Returns:
            List of potential updates
        """
        # Get recent articles (last 24 hours)
        recent = self.db.query(Article).filter(
            Article.created_at >= datetime.utcnow().replace(hour=0, minute=0)
        ).all()

        updates = []
        for article in recent:
            # Check for prompt engineering techniques
            if self._is_prompt_engineering(article):
                updates.append({
                    "type": "low_risk",
                    "description": f"New prompt technique found: {article.title}",
                    "reason": f"Article suggests new prompt engineering approach",
                    "article_id": article.id,
                })

            # Check for new frameworks/tools
            if self._is_new_tool(article):
                updates.append({
                    "type": "medium_risk",
                    "description": f"New AI tool/framework: {article.title}",
                    "reason": f"Article introduces new tool that may improve system",
                    "article_id": article.id,
                })

        return updates

    def evaluate_update(self, update: dict) -> dict:
        """
        Evaluate update risk level.

        Returns:
            Updated dict with status
        """
        update_type = update.get("type", "high_risk")

        # Constitution check
        is_allowed, reason = self.constitution.check_update(
            update_type, update["description"]
        )

        if not is_allowed:
            update["status"] = "rejected"
            update["reason"] = reason
            return update

        # Auto-execute low risk
        if update_type == "low_risk":
            update["status"] = "auto_approved"
            update["action"] = "update_prompt_template"
        elif update_type == "medium_risk":
            update["status"] = "pending_review"
            update["action"] = "generate_pr"
        else:
            update["status"] = "requires_decision"
            update["action"] = "manual_review"

        return update

    def execute_update(self, update: dict) -> bool:
        """
        Execute a low-risk update.

        For MVP: Update prompt templates in config.
        """
        if update.get("type") != "low_risk":
            logger.warning(f"Cannot auto-execute {update.get('type')} update")
            return False

        try:
            # Update system prompt template
            config = self.db.query(Config).filter(Config.key == "system_prompt_template").first()
            if not config:
                config = Config(key="system_prompt_template", value=self._default_prompt())
                self.db.add(config)

            # Increment version
            config.version += 1
            config.value = self._enhance_prompt(config.value, update["description"])
            config.updated_at = datetime.utcnow()

            self.db.commit()

            # Log the update
            self._log_update(update, status="executed")

            logger.info(f"Executed update: {update['description'][:50]}...")
            return True

        except Exception as e:
            logger.error(f"Failed to execute update: {e}")
            self._log_update(update, status="failed")
            return False

    def _log_update(self, update: dict, status: str):
        """Log update to database."""
        log = UpdateLog(
            update_type=update.get("type", "unknown"),
            description=update["description"],
            reason=update["reason"],
            status=status,
            diff_summary=update.get("action", ""),
            executed_at=datetime.utcnow() if status == "executed" else None,
        )
        self.db.add(log)
        self.db.commit()

    def get_update_history(self) -> List[UpdateLog]:
        """Get update history."""
        return self.db.query(UpdateLog).order_by(UpdateLog.created_at.desc()).all()

    def get_pending_updates(self) -> List[UpdateLog]:
        """Get pending updates."""
        return self.db.query(UpdateLog).filter(UpdateLog.status == "pending_review").all()

    def _is_prompt_engineering(self, article: Article) -> bool:
        """Check if article is about prompt engineering."""
        keywords = ["prompt", "prompting", "prompt engineering", "chain-of-thought", "few-shot"]
        text = f"{article.title} {article.summary or ''}".lower()
        return any(kw in text for kw in keywords)

    def _is_new_tool(self, article: Article) -> bool:
        """Check if article introduces new tool."""
        keywords = ["framework", "library", "tool", "platform", "release"]
        text = f"{article.title} {article.summary or ''}".lower()
        return any(kw in text for kw in keywords)

    def _default_prompt(self) -> str:
        """Default system prompt template."""
        return """You are an AI research assistant. Analyze the given article and provide:
1. A concise summary
2. Key technical contributions
3. Potential applications"""

    def _enhance_prompt(self, current: str, update_description: str) -> str:
        """Enhance prompt with new technique."""
        enhancement = f"\n\n# Auto-updated: {datetime.utcnow().isoformat()}\n"
        enhancement += f"# New technique incorporated: {update_description[:100]}\n"
        return current + enhancement


class UpdateManager:
    """Manager for self-update workflow."""

    def __init__(self, db: Session):
        self.updater = SelfUpdater(db)

    def run_update_cycle(self) -> dict:
        """
        Run a full update cycle.

        Returns:
            Summary of updates found and executed
        """
        # Scan for updates
        updates = self.updater.scan_for_updates()

        executed = 0
        pending = 0
        rejected = 0

        for update in updates:
            # Evaluate
            evaluated = self.updater.evaluate_update(update)

            if evaluated["status"] == "rejected":
                rejected += 1
                continue

            if evaluated["status"] == "auto_approved":
                if self.updater.execute_update(evaluated):
                    executed += 1
                else:
                    rejected += 1
            else:
                pending += 1

        return {
            "total_found": len(updates),
            "executed": executed,
            "pending": pending,
            "rejected": rejected,
        }
