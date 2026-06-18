"""概念词条管理器 - 管理AI概念的词条生命周期。

职责：
- 根据主题下的文章自动生成概念词条
- 更新已有概念词条
- 管理概念状态（active/absorbed/archived）
- 追踪概念演进链（前身→当前→后续）
"""

import json
import logging
import time
from datetime import datetime
from typing import List, Optional, Dict

import requests
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import Session

import re

from app.models import Base, SessionLocal, Article, ArticleTopic, Topic
from app.summarizer import LLM_API_KEY, LLM_API_URL, LLM_MODEL, MAX_RPS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Concept 数据模型（独立定义，避免修改 models.py）
# ---------------------------------------------------------------------------

class Concept(Base):
    """概念词条模型。"""
    __tablename__ = "concepts"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, unique=True)
    one_liner = Column(String(200), nullable=True)
    description = Column(Text, nullable=True)
    use_cases = Column(Text, nullable=True)
    status = Column(String(20), default="active")  # active / absorbed / archived
    predecessor_id = Column(Integer, ForeignKey("concepts.id"), nullable=True)
    successor_id = Column(Integer, ForeignKey("concepts.id"), nullable=True)
    related_topic_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ---------------------------------------------------------------------------
# 公共工具：Agnes API 调用（复用 summarizer.py 的配置）
# ---------------------------------------------------------------------------

_last_call_time = 0.0


def _rate_limit():
    """确保不超过 API 速率限制。"""
    global _last_call_time
    elapsed = time.time() - _last_call_time
    min_interval = 1.0 / MAX_RPS
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_call_time = time.time()


def _call_llm(prompt: str, max_tokens: int = 1000, temperature: float = 0.3) -> Optional[str]:
    """调用 Agnes LLM API，返回生成文本或 None。"""
    if not LLM_API_KEY:
        logger.error("LLM_API_KEY 未配置")
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
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error("Agnes API 调用失败: %s", e)
        return None


def parse_llm_json(raw: str, expected_key: Optional[str] = None) -> Optional[Dict]:
    """从 LLM 返回的原始文本中提取 JSON 对象。

    采用三级回退策略：
    1. 直接解析整个文本
    2. 提取 ```json ... ``` 代码块
    3. 提取第一个 { ... } 花括号块

    Args:
        raw: LLM 返回的原始文本。
        expected_key: 可选，期望 JSON 中包含的顶层 key，用于验证。

    Returns:
        解析后的字典，或 None。
    """
    # 策略1：直接解析
    try:
        data = json.loads(raw)
        if expected_key is None or expected_key in data:
            return data
    except json.JSONDecodeError:
        pass

    # 策略2：提取 ```json ... ``` 代码块
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if expected_key is None or expected_key in data:
                return data
        except json.JSONDecodeError:
            pass

    # 策略3：提取第一个 { ... } 块
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(raw[start:end + 1])
            if expected_key is None or expected_key in data:
                return data
        except json.JSONDecodeError:
            pass

    logger.error("无法从 LLM 输出中解析 JSON: %s", raw[:200])
    return None


# ---------------------------------------------------------------------------
# Prompt 模板
# ---------------------------------------------------------------------------

CONCEPT_GENERATION_PROMPT = """你是一个AI领域知识管理专家。请根据以下主题和文章列表，生成一个概念词条。

主题名称：{topic_name}

相关文章：
{articles_text}

请严格按以下 JSON 格式输出，不要输出任何其他内容：
{{
    "name": "概念名称（简洁专业，2-8个字）",
    "one_liner": "一句话解释（50字以内，面向技术从业者）",
    "description": "详细说明（200-300字，包含核心原理、技术特点、发展脉络）",
    "use_cases": "适用场景（100-200字，列举2-4个典型应用场景）"
}}"""


CONCEPT_UPDATE_PROMPT = """你是一个AI领域知识管理专家。请根据最新文章更新以下概念词条。

当前概念：
名称：{name}
一句话解释：{one_liner}
详细说明：{description}
适用场景：{use_cases}

最新文章：
{articles_text}

请严格按以下 JSON 格式输出更新后的词条，不要输出任何其他内容：
{{
    "name": "概念名称（如有必要可微调）",
    "one_liner": "一句话解释（50字以内）",
    "description": "详细说明（200-300字，融合新信息）",
    "use_cases": "适用场景（100-200字）"
}}"""


# ---------------------------------------------------------------------------
# ConceptManager
# ---------------------------------------------------------------------------

class ConceptManager:
    """概念词条管理器。

    管理AI概念的完整生命周期：生成、更新、吸收、归档、演进链追踪。
    """

    def __init__(self, db: Optional[Session] = None):
        self._db = db
        self._owns_session = db is None

    @property
    def db(self) -> Session:
        if self._db is None:
            self._db = SessionLocal()
        return self._db

    def close(self):
        """关闭会话（仅当管理器自己创建会话时）。"""
        if self._owns_session and self._db is not None:
            self._db.close()
            self._db = None

    # ------------------------------------------------------------------
    # 公共方法
    # ------------------------------------------------------------------

    def generate_concept(self, topic_name: str, articles: List[Article]) -> Optional[Concept]:
        """根据主题下的文章自动生成概念词条。

        Args:
            topic_name: 主题名称。
            articles: 该主题下的文章列表。

        Returns:
            新创建的 Concept 对象，或 None（生成失败）。
        """
        if not articles:
            logger.warning("文章列表为空，无法生成概念词条: %s", topic_name)
            return None

        # 检查是否已存在同名概念
        existing = self.db.query(Concept).filter(Concept.name == topic_name).first()
        if existing:
            logger.info("概念 '%s' 已存在 (id=%d)，跳过生成", topic_name, existing.id)
            return existing

        # 构造文章摘要文本
        articles_text = self._format_articles_for_prompt(articles[:10])

        prompt = CONCEPT_GENERATION_PROMPT.format(
            topic_name=topic_name,
            articles_text=articles_text,
        )

        raw = _call_llm(prompt, max_tokens=800)
        if not raw:
            logger.error("生成概念词条失败: %s", topic_name)
            return None

        parsed = parse_llm_json(raw)
        if not parsed:
            logger.error("解析概念词条 JSON 失败: %s", topic_name)
            return None

        # 查找关联主题
        topic = self.db.query(Topic).filter(Topic.name == topic_name).first()
        related_topic_id = topic.id if topic else None

        concept = Concept(
            name=parsed["name"],
            one_liner=parsed.get("one_liner"),
            description=parsed.get("description"),
            use_cases=parsed.get("use_cases"),
            status="active",
            related_topic_id=related_topic_id,
        )
        self.db.add(concept)
        self.db.commit()
        self.db.refresh(concept)

        logger.info("概念词条已创建: %s (id=%d)", concept.name, concept.id)
        return concept

    def update_concept(self, concept_id: int) -> Optional[Concept]:
        """根据最新文章更新概念词条。

        Args:
            concept_id: 概念ID。

        Returns:
            更新后的 Concept 对象，或 None。
        """
        concept = self.db.query(Concept).filter(Concept.id == concept_id).first()
        if not concept:
            logger.error("概念不存在: id=%d", concept_id)
            return None

        if concept.status != "active":
            logger.warning("概念 %s 状态为 %s，不更新", concept.name, concept.status)
            return None

        # 获取关联主题下的最新文章
        articles = self._get_articles_for_concept(concept)
        if not articles:
            logger.info("概念 '%s' 无新文章可更新", concept.name)
            return concept

        articles_text = self._format_articles_for_prompt(articles[:10])

        prompt = CONCEPT_UPDATE_PROMPT.format(
            name=concept.name,
            one_liner=concept.one_liner or "",
            description=concept.description or "",
            use_cases=concept.use_cases or "",
            articles_text=articles_text,
        )

        raw = _call_llm(prompt, max_tokens=800)
        if not raw:
            logger.error("更新概念词条失败: %s", concept.name)
            return None

        parsed = parse_llm_json(raw)
        if not parsed:
            logger.error("解析更新后概念 JSON 失败: %s", concept.name)
            return None

        concept.name = parsed["name"]
        concept.one_liner = parsed.get("one_liner")
        concept.description = parsed.get("description")
        concept.use_cases = parsed.get("use_cases")
        concept.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(concept)

        logger.info("概念词条已更新: %s (id=%d)", concept.name, concept.id)
        return concept

    def mark_absorbed(self, concept_id: int, absorbed_by_id: int) -> bool:
        """标记概念被另一个概念吸收。

        Args:
            concept_id: 被吸收的概念ID。
            absorbed_by_id: 吸收方概念ID。

        Returns:
            是否成功。
        """
        absorbed = self.db.query(Concept).filter(Concept.id == concept_id).first()
        absorber = self.db.query(Concept).filter(Concept.id == absorbed_by_id).first()

        if not absorbed or not absorber:
            logger.error("概念不存在: absorbed=%s, absorber=%s", concept_id, absorbed_by_id)
            return False

        if absorbed.status != "active":
            logger.warning("概念 %s 状态为 %s，无法标记为 absorbed", absorbed.name, absorbed.status)
            return False

        absorbed.status = "absorbed"
        absorbed.successor_id = absorbed_by_id
        absorbed.updated_at = datetime.utcnow()

        # 同时在吸收方设置 predecessor
        absorber.predecessor_id = concept_id
        absorber.updated_at = datetime.utcnow()

        self.db.commit()
        logger.info(
            "概念 '%s' (id=%d) 已被 '%s' (id=%d) 吸收",
            absorbed.name, concept_id, absorber.name, absorbed_by_id,
        )
        return True

    def mark_archived(self, concept_id: int) -> bool:
        """标记概念归档。

        Args:
            concept_id: 概念ID。

        Returns:
            是否成功。
        """
        concept = self.db.query(Concept).filter(Concept.id == concept_id).first()
        if not concept:
            logger.error("概念不存在: id=%d", concept_id)
            return False

        concept.status = "archived"
        concept.updated_at = datetime.utcnow()
        self.db.commit()

        logger.info("概念 '%s' (id=%d) 已归档", concept.name, concept_id)
        return True

    def get_active_concepts(self) -> List[Concept]:
        """获取所有活跃概念。

        Returns:
            active 状态的概念列表。
        """
        return self.db.query(Concept).filter(Concept.status == "active").all()

    def get_concept_evolution_chain(self, concept_id: int) -> Dict:
        """获取概念的完整演进链。

        沿 predecessor_id 向前追溯所有前身概念，沿 successor_id 向后追溯后续概念。

        Args:
            concept_id: 起始概念ID。

        Returns:
            包含 predecessors、current、successors 的字典。
        """
        concept = self.db.query(Concept).filter(Concept.id == concept_id).first()
        if not concept:
            logger.error("概念不存在: id=%d", concept_id)
            return {"predecessors": [], "current": None, "successors": []}

        # 向前追溯前身
        predecessors = []
        current = concept.predecessor_id
        visited = {concept_id}
        while current and current not in visited:
            visited.add(current)
            pred = self.db.query(Concept).filter(Concept.id == current).first()
            if pred:
                predecessors.append(pred)
                current = pred.predecessor_id
            else:
                break
        predecessors.reverse()  # 从最早到最近

        # 向后追溯后续
        successors = []
        current = concept.successor_id
        visited = {concept_id}
        while current and current not in visited:
            visited.add(current)
            succ = self.db.query(Concept).filter(Concept.id == current).first()
            if succ:
                successors.append(succ)
                current = succ.successor_id
            else:
                break

        return {
            "predecessors": predecessors,
            "current": concept,
            "successors": successors,
        }

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _get_articles_for_concept(self, concept: Concept) -> List[Article]:
        """获取与概念关联的文章。"""
        if concept.related_topic_id:
            article_ids = (
                self.db.query(ArticleTopic.article_id)
                .filter(ArticleTopic.topic_id == concept.related_topic_id)
                .all()
            )
            ids = [aid for (aid,) in article_ids]
            if ids:
                return self.db.query(Article).filter(Article.id.in_(ids)).all()
        return []

    @staticmethod
    def _format_articles_for_prompt(articles: List[Article], max_chars: int = 3000) -> str:
        """将文章列表格式化为 prompt 文本。"""
        parts = []
        total = 0
        for i, a in enumerate(articles, 1):
            line = f"{i}. [{a.title}] {a.summary or ''}"
            if len(line) > 300:
                line = line[:300] + "..."
            total += len(line)
            if total > max_chars:
                break
            parts.append(line)
        return "\n".join(parts) if parts else "（无文章内容）"

