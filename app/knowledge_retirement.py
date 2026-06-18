"""知识淘汰引擎 - 检测过时知识并执行淘汰。

职责：
- 分析概念间的吸收关系，检测被新概念覆盖的老概念
- 概念被吸收后清理关联文章
- 筛选低热度学术论文并归档
- 执行完整的淘汰周期并生成报告
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict

from sqlalchemy.orm import Session

from app.models import SessionLocal, Article, ArticleTopic, Topic
from app.concept_manager import Concept, ConceptManager, _call_llm, parse_llm_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt 模板
# ---------------------------------------------------------------------------

ABSORPTION_DETECT_PROMPT = """你是一个AI领域知识管理专家。请分析以下活跃概念列表，判断哪些概念之间存在"吸收关系"。

吸收关系定义：概念B吸收了概念A，当且仅当：
- 概念B的内涵完全覆盖概念A
- 概念A可以视为概念B的子集或早期形态
- 概念A不再有独立存在的必要

活跃概念列表：
{concepts_text}

请严格按以下 JSON 格式输出，不要输出任何其他内容。如果没有吸收关系，返回空列表。
{{
    "absorptions": [
        {{
            "absorbed_name": "被吸收的概念名称",
            "absorbed_id": 被吸收的概念ID,
            "absorber_name": "吸收方概念名称",
            "absorber_id": 吸收方概念ID,
            "reason": "判断理由（50字以内）"
        }}
    ]
}}"""


# ---------------------------------------------------------------------------
# KnowledgeRetirementEngine
# ---------------------------------------------------------------------------

class KnowledgeRetirementEngine:
    """知识淘汰引擎。

    负责检测过时知识、执行淘汰操作、筛选低热度论文，
    并生成淘汰报告。依赖 ConceptManager 管理概念状态。
    """

    def __init__(self, db: Optional[Session] = None):
        self._db = db
        self._owns_session = db is None
        self._concept_mgr = ConceptManager(db)

    @property
    def db(self) -> Session:
        if self._db is None:
            self._db = SessionLocal()
        return self._db

    def close(self):
        """关闭会话。"""
        if self._owns_session and self._db is not None:
            self._db.close()
            self._db = None
        if self._concept_mgr._db is not None and self._owns_session:
            self._concept_mgr._db = None

    # ------------------------------------------------------------------
    # 公共方法
    # ------------------------------------------------------------------

    def detect_absorbed_concepts(self) -> List[Dict]:
        """分析概念之间的关系，检测哪些老概念已被新概念吸收。

        调用 Agnes API，输入所有活跃概念列表，让 AI 判断吸收关系。

        Returns:
            建议的吸收关系列表，每项包含 absorbed_id、absorber_id、reason。
        """
        active_concepts = self._concept_mgr.get_active_concepts()
        if len(active_concepts) < 2:
            logger.info("活跃概念不足2个，跳过吸收检测")
            return []

        concepts_text = self._format_concepts_for_prompt(active_concepts)

        prompt = ABSORPTION_DETECT_PROMPT.format(concepts_text=concepts_text)

        raw = _call_llm(prompt, max_tokens=1000, temperature=0.2)
        if not raw:
            logger.error("吸收关系检测 API 调用失败")
            return []

        parsed = parse_llm_json(raw, expected_key="absorptions")
        if not parsed:
            logger.error("解析吸收关系 JSON 失败")
            return []

        absorptions = parsed["absorptions"]

        # 验证返回的概念ID确实存在且为 active
        valid = []
        active_ids = {c.id for c in active_concepts}
        for item in absorptions:
            if item["absorbed_id"] in active_ids and item["absorber_id"] in active_ids:
                if item["absorbed_id"] != item["absorber_id"]:
                    valid.append(item)
                else:
                    logger.warning("跳过自吸收关系: id=%d", item["absorbed_id"])
            else:
                logger.warning(
                    "跳过无效吸收关系: absorbed=%d, absorber=%d",
                    item["absorbed_id"], item["absorber_id"],
                )

        logger.info("检测到 %d 条有效吸收关系", len(valid))
        return valid

    def retire_articles_for_concept(self, concept_id: int) -> Dict:
        """概念被吸收后，清理该概念下的具体文章。

        策略：
        - 保留概念词条（Concept 记录），仅更新状态
        - 将关联文章标记为 archived（不物理删除，保留可追溯性）
        - 概念词条中的一句话解释和详细说明保留不变

        Args:
            concept_id: 被淘汰的概念ID。

        Returns:
            操作结果统计 {archived_count, total_count}。
        """
        concept = self.db.query(Concept).filter(Concept.id == concept_id).first()
        if not concept:
            logger.error("概念不存在: id=%d", concept_id)
            return {"archived_count": 0, "total_count": 0}

        # 获取关联文章
        articles = self._concept_mgr._get_articles_for_concept(concept)
        total_count = len(articles)
        archived_count = 0

        for article in articles:
            # 标记文章为 archived（通过在 keywords 中添加标记）
            # Article 模型没有 status 字段，使用 keywords JSON 字段标记
            keywords = article.keywords if article.keywords else []
            if isinstance(keywords, str):
                try:
                    keywords = json.loads(keywords)
                except (json.JSONDecodeError, TypeError):
                    keywords = []

            if "_archived" not in keywords:
                keywords.append("_archived")
                article.keywords = keywords
                article.updated_at = datetime.utcnow()
                archived_count += 1

        self.db.commit()
        logger.info(
            "概念 '%s' (id=%d) 文章清理完成: 归档 %d / 共 %d 篇",
            concept.name, concept_id, archived_count, total_count,
        )
        return {"archived_count": archived_count, "total_count": total_count}

    def filter_low_heat_papers(self, months: int = 3) -> Dict:
        """学术论文热度筛选。

        策略：每个主题下仅保留最近 N 个月的 arxiv 论文，
        超过时间窗口的旧论文标记为 archived。

        Args:
            months: 保留的时间窗口（月），默认3个月。

        Returns:
            操作结果统计 {archived_count, kept_count, topic_stats}。
        """
        cutoff = datetime.utcnow() - timedelta(days=30 * months)

        # 查询所有 arxiv 来源的文章
        arxiv_articles = (
            self.db.query(Article)
            .filter(Article.source == "arxiv")
            .filter(Article.created_at < cutoff)
            .all()
        )

        if not arxiv_articles:
            logger.info("没有超过 %d 个月的 arxiv 论文需要处理", months)
            return {"archived_count": 0, "kept_count": 0, "topic_stats": {}}

        archived_count = 0
        topic_stats: Dict[str, int] = {}

        for article in arxiv_articles:
            keywords = article.keywords if article.keywords else []
            if isinstance(keywords, str):
                try:
                    keywords = json.loads(keywords)
                except (json.JSONDecodeError, TypeError):
                    keywords = []

            if "_archived" not in keywords:
                keywords.append("_archived")
                article.keywords = keywords
                article.updated_at = datetime.utcnow()
                archived_count += 1

            # 统计按主题
            topic_links = (
                self.db.query(ArticleTopic)
                .filter(ArticleTopic.article_id == article.id)
                .all()
            )
            for link in topic_links:
                topic = self.db.query(Topic).filter(Topic.id == link.topic_id).first()
                if topic:
                    topic_stats[topic.name] = topic_stats.get(topic.name, 0) + 1

        self.db.commit()

        kept_count = (
            self.db.query(Article)
            .filter(Article.source == "arxiv")
            .filter(Article.created_at >= cutoff)
            .count()
        )

        logger.info(
            "arxiv 论文筛选完成: 归档 %d 篇, 保留 %d 篇 (窗口=%d月)",
            archived_count, kept_count, months,
        )
        return {
            "archived_count": archived_count,
            "kept_count": kept_count,
            "topic_stats": topic_stats,
        }

    def run_retirement_cycle(self) -> Dict:
        """执行完整的淘汰周期。

        流程：
        1. 检测吸收关系
        2. 确认并执行概念吸收
        3. 清理被吸收概念的文章
        4. 筛选低热度论文
        5. 生成淘汰报告

        Returns:
            完整的淘汰报告。
        """
        cycle_start = datetime.utcnow()
        logger.info("===== 知识淘汰周期开始 =====")

        report = {
            "cycle_start": cycle_start.isoformat(),
            "absorptions_detected": [],
            "absorptions_executed": [],
            "articles_retired": {},
            "papers_filtered": {},
            "errors": [],
        }

        # Step 1: 检测吸收关系
        try:
            absorptions = self.detect_absorbed_concepts()
            report["absorptions_detected"] = absorptions
        except Exception as e:
            msg = f"吸收关系检测失败: {e}"
            logger.error(msg)
            report["errors"].append(msg)
            absorptions = []

        # Step 2: 确认并执行概念吸收
        for item in absorptions:
            try:
                success = self._concept_mgr.mark_absorbed(
                    item["absorbed_id"],
                    item["absorber_id"],
                )
                if success:
                    report["absorptions_executed"].append(item)
                    logger.info(
                        "已执行吸收: '%s' -> '%s'",
                        item["absorbed_name"],
                        item["absorber_name"],
                    )
            except Exception as e:
                msg = f"执行吸收失败 ({item.get('absorbed_id')}->{item.get('absorber_id')}): {e}"
                logger.error(msg)
                report["errors"].append(msg)

        # Step 3: 清理被吸收概念的文章
        for item in report["absorptions_executed"]:
            try:
                result = self.retire_articles_for_concept(item["absorbed_id"])
                report["articles_retired"][item["absorbed_name"]] = result
            except Exception as e:
                msg = f"清理文章失败 (concept_id={item['absorbed_id']}): {e}"
                logger.error(msg)
                report["errors"].append(msg)

        # Step 4: 筛选低热度论文
        try:
            filter_result = self.filter_low_heat_papers()
            report["papers_filtered"] = filter_result
        except Exception as e:
            msg = f"论文筛选失败: {e}"
            logger.error(msg)
            report["errors"].append(msg)

        # 生成摘要
        cycle_end = datetime.utcnow()
        report["cycle_end"] = cycle_end.isoformat()
        report["duration_seconds"] = (cycle_end - cycle_start).total_seconds()
        report["summary"] = {
            "absorptions_found": len(report["absorptions_detected"]),
            "absorptions_executed": len(report["absorptions_executed"]),
            "total_articles_archived": sum(
                r.get("archived_count", 0) for r in report["articles_retired"].values()
            ),
            "papers_filtered": report["papers_filtered"].get("archived_count", 0),
            "errors_count": len(report["errors"]),
        }

        logger.info("===== 知识淘汰周期完成 =====")
        logger.info("摘要: %s", json.dumps(report["summary"], ensure_ascii=False))

        return report

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _format_concepts_for_prompt(concepts: List[Concept], max_chars: int = 4000) -> str:
        """将概念列表格式化为 prompt 文本。"""
        parts = []
        total = 0
        for c in concepts:
            line = (
                f"- ID:{c.id} | 名称: {c.name} | "
                f"一句话: {c.one_liner or '(无)'} | "
                f"说明: {(c.description or '(无)')[:100]}"
            )
            total += len(line)
            if total > max_chars:
                break
            parts.append(line)
        return "\n".join(parts) if parts else "（无活跃概念）"

