"""系统级自愈架构 - 三层感知/诊断/修复体系.

解决项目历史问题：
- API URL错误导致3次试错浪费
- 批量脚本静默挂起无反馈
- JSON截断未检测
- Token消耗异常未告警

架构分层：
1. 感知层 (Perception): HealthMonitor / TokenUsageTracker / DataQualityChecker
2. 诊断层 (Diagnosis): DiagnosisEngine - 规则驱动的根因分析
3. 修复层 (Remediation): SelfHealingExecutor - 装饰器式自动修复
"""

import functools
import json
import logging
import os
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class ValidationError(ValueError):
    """结果验证失败时抛出的专用异常，与普通 ValueError 区分."""


# ---------------------------------------------------------------------------
# Agnes API 配置 - 复用 summarizer.py 的配置
# ---------------------------------------------------------------------------
API_KEY = os.getenv(
    "LLM_API_KEY",
    os.getenv("AGNES_API_KEY", "sk-thpVPgLvZubGt2neZLRUrYpazpBppXKU5YbQhfOjdYaF33pA"),
)
API_URL = os.getenv("LLM_API_URL", "https://apihub.agnes-ai.com/v1/chat/completions")
MODEL = os.getenv("LLM_MODEL", "agnes-2.0-flash")


# ===========================================================================
#  感知层 (Perception)
# ===========================================================================


@dataclass
class HealthMonitor:
    """监控 API 可用性、响应时间、成功率.

    维护一个滑动窗口，记录最近 N 次请求的状态，
    提供 is_healthy() / get_stats() 等查询接口。
    """

    window_size: int = 50
    _records: deque = field(default_factory=deque)
    _consecutive_failures: int = 0

    def __post_init__(self):
        if not hasattr(self, '_records_initialized'):
            self._records = deque(maxlen=self.window_size)
            self._records_initialized = True

    # --- 记录 ---

    def record_success(self, response_time: float, url: str = "") -> None:
        """记录一次成功请求."""
        self._records.append({
            "ok": True,
            "time": response_time,
            "url": url,
            "ts": time.time(),
        })
        self._consecutive_failures = 0
        logger.debug("HealthMonitor: success recorded (response_time=%.2fs)", response_time)

    def record_failure(self, error: str, url: str = "", status_code: int = 0) -> None:
        """记录一次失败请求."""
        self._records.append({
            "ok": False,
            "error": error,
            "url": url,
            "status_code": status_code,
            "ts": time.time(),
        })
        self._consecutive_failures += 1
        logger.warning(
            "HealthMonitor: failure #%d recorded - %s (status=%d)",
            self._consecutive_failures, error, status_code,
        )

    # --- 查询 ---

    def is_healthy(self) -> bool:
        """判断 API 当前是否健康（连续失败不超过 3 次）."""
        return self._consecutive_failures < 3

    def get_stats(self) -> Dict[str, Any]:
        """返回窗口内的统计摘要."""
        if not self._records:
            return {
                "total": 0,
                "success_rate": 0.0,
                "avg_response_time": 0.0,
                "consecutive_failures": self._consecutive_failures,
                "healthy": self.is_healthy(),
            }

        successes = [r for r in self._records if r["ok"]]
        failures = [r for r in self._records if not r["ok"]]
        response_times = [r["time"] for r in successes if "time" in r]

        return {
            "total": len(self._records),
            "success_count": len(successes),
            "failure_count": len(failures),
            "success_rate": len(successes) / len(self._records),
            "avg_response_time": sum(response_times) / len(response_times) if response_times else 0.0,
            "consecutive_failures": self._consecutive_failures,
            "healthy": self.is_healthy(),
        }

    def get_recent_errors(self, limit: int = 5) -> List[Dict[str, Any]]:
        """返回最近 N 条错误记录."""
        errors = [r for r in self._records if not r["ok"]]
        return list(reversed(errors))[:limit]


@dataclass
class TokenUsageTracker:
    """追踪 token 消耗，检测异常浪费.

    通过滑动窗口统计 token 用量，当单次或累计消耗超出阈值时触发告警。
    """

    window_size: int = 100
    _records: deque = field(default_factory=deque)
    _alert_threshold_single: int = 5000      # 单次请求 token 上限
    _alert_threshold_avg: float = 2000.0      # 窗口平均 token 上限

    def __post_init__(self):
        if not hasattr(self, '_records_initialized'):
            self._records = deque(maxlen=self.window_size)
            self._records_initialized = True

    def record_usage(self, prompt_tokens: int, completion_tokens: int, model: str = "") -> None:
        """记录一次 token 消耗."""
        total = prompt_tokens + completion_tokens
        self._records.append({
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": total,
            "model": model,
            "ts": time.time(),
        })
        if total > self._alert_threshold_single:
            logger.warning(
                "TokenUsageTracker: 单次 token 消耗异常高: %d (prompt=%d, completion=%d)",
                total, prompt_tokens, completion_tokens,
            )

    def is_abnormal(self) -> bool:
        """判断当前窗口内 token 消耗是否异常."""
        if len(self._records) < 3:
            return False
        avg = sum(r["total"] for r in self._records) / len(self._records)
        return avg > self._alert_threshold_avg

    def get_stats(self) -> Dict[str, Any]:
        """返回 token 用量统计."""
        if not self._records:
            return {"total_requests": 0, "total_tokens": 0, "avg_tokens": 0.0, "abnormal": False}

        totals = [r["total"] for r in self._records]
        return {
            "total_requests": len(self._records),
            "total_tokens": sum(totals),
            "avg_tokens": sum(totals) / len(totals),
            "max_tokens": max(totals),
            "abnormal": self.is_abnormal(),
        }


@dataclass
class DataQualityChecker:
    """检查数据完整性：空摘要、重复 URL、截断 JSON 等.

    提供独立的检查方法，可按需调用单项或全量检查。
    """

    def check_empty_summary(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检查空摘要记录."""
        issues = []
        for article in articles:
            summary = article.get("chinese_summary") or article.get("summary") or ""
            if not summary.strip():
                issues.append({
                    "type": "empty_summary",
                    "article_id": article.get("id"),
                    "title": article.get("title", ""),
                    "message": "摘要为空",
                })
        return issues

    def check_duplicate_urls(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检查重复 URL."""
        url_map: Dict[str, List[int]] = defaultdict(list)
        for article in articles:
            url = article.get("url", "").strip()
            if url:
                url_map[url].append(article.get("id"))

        issues = []
        for url, ids in url_map.items():
            if len(ids) > 1:
                issues.append({
                    "type": "duplicate_url",
                    "url": url,
                    "article_ids": ids,
                    "message": f"URL 重复出现 {len(ids)} 次",
                })
        return issues

    def check_truncated_json(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检查 JSON 截断（内容在引号/大括号内突然结束）."""
        issues = []
        for article in articles:
            text = article.get("chinese_summary") or article.get("summary") or ""
            if text and _is_truncated(text):
                issues.append({
                    "type": "truncated_content",
                    "article_id": article.get("id"),
                    "title": article.get("title", ""),
                    "message": "内容疑似截断",
                    "snippet": text[-100:],
                })
        return issues

    def run_all_checks(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """执行全部数据质量检查."""
        all_issues: List[Dict[str, Any]] = []
        all_issues.extend(self.check_empty_summary(articles))
        all_issues.extend(self.check_duplicate_urls(articles))
        all_issues.extend(self.check_truncated_json(articles))
        return all_issues


def _is_truncated(text: str) -> bool:
    """启发式判断文本是否被截断."""
    if not text:
        return False
    # 以未闭合的引号结尾
    stripped = text.rstrip()
    if stripped and stripped[-1] == '"' and stripped.count('"') % 2 != 0:
        return True
    # 以未闭合的大括号/中括号结尾
    for char in ("{", "["):
        if stripped.endswith(char):
            return True
    # 以 "..." 或省略号结尾（模型输出被截断的典型特征）
    if stripped.endswith("...") or stripped.endswith("...\""):
        return True
    return False


# ===========================================================================
#  诊断层 (Diagnosis)
# ===========================================================================


@dataclass
class DiagnosisResult:
    """诊断结果."""
    severity: str          # "critical" / "warning" / "info"
    root_cause: str        # 根因描述
    affected_component: str  # 受影响组件
    suggestions: List[str]  # 修复建议


class DiagnosisEngine:
    """分析异常模式，给出根因和修复建议.

    诊断规则：
    - API 连续失败 -> 检查 URL / 模型名 / Key
    - 响应为空/截断 -> 检查 max_tokens
    - 大量重复 -> 检查去重逻辑
    - Token 消耗异常 -> 检查 prompt 长度
    """

    def diagnose_api_failures(
        self,
        health_monitor: HealthMonitor,
    ) -> List[DiagnosisResult]:
        """诊断 API 连续失败."""
        results: List[DiagnosisResult] = []
        stats = health_monitor.get_stats()

        if stats["consecutive_failures"] >= 3:
            results.append(DiagnosisResult(
                severity="critical",
                root_cause=f"API 连续失败 {stats['consecutive_failures']} 次",
                affected_component="API",
                suggestions=[
                    "检查 API URL 是否正确",
                    "验证 API Key 是否有效",
                    "确认模型名称是否拼写正确",
                    "检查网络连接和 DNS 解析",
                ],
            ))

        if stats["success_rate"] < 0.5 and stats["total"] >= 10:
            results.append(DiagnosisResult(
                severity="warning",
                root_cause=f"API 成功率过低 ({stats['success_rate']:.1%})",
                affected_component="API",
                suggestions=[
                    "检查是否超出速率限制",
                    "确认 API 账户额度是否充足",
                    "查看最近错误详情排查模式",
                ],
            ))

        return results

    def diagnose_empty_or_truncated(
        self,
        issues: List[Dict[str, Any]],
    ) -> List[DiagnosisResult]:
        """诊断空响应或截断问题."""
        results: List[DiagnosisResult] = []

        truncated = [i for i in issues if i["type"] == "truncated_content"]
        empty = [i for i in issues if i["type"] == "empty_summary"]

        if len(truncated) >= 3:
            results.append(DiagnosisResult(
                severity="warning",
                root_cause=f"检测到 {len(truncated)} 条截断内容",
                affected_component="LLM输出",
                suggestions=[
                    "增大 max_tokens 参数（当前建议 >= 500）",
                    "检查 prompt 是否过长占用输出空间",
                    "考虑缩短输入文本长度",
                ],
            ))

        if len(empty) >= 5:
            results.append(DiagnosisResult(
                severity="warning",
                root_cause=f"检测到 {len(empty)} 条空摘要",
                affected_component="数据处理",
                suggestions=[
                    "检查 API 是否返回空内容",
                    "确认输入文章是否有有效摘要字段",
                    "检查内容处理器是否正确提取文本",
                ],
            ))

        return results

    def diagnose_duplicates(
        self,
        issues: List[Dict[str, Any]],
    ) -> List[DiagnosisResult]:
        """诊断重复数据问题."""
        results: List[DiagnosisResult] = []
        dup_issues = [i for i in issues if i["type"] == "duplicate_url"]

        if len(dup_issues) >= 3:
            results.append(DiagnosisResult(
                severity="warning",
                root_cause=f"检测到 {len(dup_issues)} 组重复 URL",
                affected_component="爬虫/去重",
                suggestions=[
                    "检查爬虫去重逻辑是否正常工作",
                    "确认数据库 URL 唯一索引是否存在",
                    "排查 RSS 源是否返回重复条目",
                ],
            ))

        return results

    def diagnose_token_anomaly(
        self,
        tracker: TokenUsageTracker,
    ) -> List[DiagnosisResult]:
        """诊断 Token 消耗异常."""
        results: List[DiagnosisResult] = []

        if tracker.is_abnormal():
            stats = tracker.get_stats()
            results.append(DiagnosisResult(
                severity="warning",
                root_cause=f"Token 平均消耗异常偏高 ({stats['avg_tokens']:.0f}/次)",
                affected_component="Token使用",
                suggestions=[
                    "检查 prompt 模板是否过长",
                    "确认输入文本是否做了合理的截断",
                    "考虑使用更轻量的模型处理简单任务",
                ],
            ))

        return results

    def run_full_diagnosis(
        self,
        health_monitor: Optional[HealthMonitor] = None,
        token_tracker: Optional[TokenUsageTracker] = None,
        data_issues: Optional[List[Dict[str, Any]]] = None,
    ) -> List[DiagnosisResult]:
        """执行完整诊断，汇总所有规则的结果."""
        all_results: List[DiagnosisResult] = []

        if health_monitor:
            all_results.extend(self.diagnose_api_failures(health_monitor))

        if token_tracker:
            all_results.extend(self.diagnose_token_anomaly(token_tracker))

        if data_issues:
            all_results.extend(self.diagnose_empty_or_truncated(data_issues))
            all_results.extend(self.diagnose_duplicates(data_issues))

        # 按严重程度排序
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        all_results.sort(key=lambda r: severity_order.get(r.severity, 99))

        return all_results


# ===========================================================================
#  修复层 (Remediation)
# ===========================================================================


@dataclass
class RetryConfig:
    """重试配置."""
    max_retries: int = 3
    base_delay: float = 1.0       # 基础退避秒数
    max_delay: float = 30.0        # 最大退避秒数
    backoff_factor: float = 2.0    # 退避倍数
    retryable_exceptions: tuple = (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.HTTPError,
        ConnectionError,
        TimeoutError,
    )


class SelfHealingExecutor:
    """修复层核心：装饰器 + 上下文管理器.

    包装任意函数，自动添加：
    - 指数退避重试 (execute_with_retry)
    - 结果验证 (validate_result)
    - 降级策略 (fallback_degrade)
    - 诊断日志 (log_and_alert)

    用法::

        executor = SelfHealingExecutor()

        # 方式1：作为装饰器
        @executor
        def call_api():
            ...

        # 方式2：手动调用
        result = executor.execute_with_retry(call_api, fallback=my_cache)
    """

    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        health_monitor: Optional[HealthMonitor] = None,
        token_tracker: Optional[TokenUsageTracker] = None,
    ):
        self._config = retry_config or RetryConfig()
        self._health_monitor = health_monitor or HealthMonitor()
        self._token_tracker = token_tracker or TokenUsageTracker()

    # --- 装饰器 ---

    def __call__(self, func: Callable) -> Callable:
        """作为装饰器使用."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute_with_retry(func, *args, **kwargs)
        wrapper._self_healing_executor = self  # type: ignore[attr-defined]
        return wrapper

    # --- 核心方法 ---

    def execute_with_retry(
        self,
        func: Callable,
        *args: Any,
        fallback: Optional[Callable] = None,
        validators: Optional[List[Callable[[Any], bool]]] = None,
        **kwargs: Any,
    ) -> Any:
        """带指数退避的重试执行.

        Args:
            func: 要执行的目标函数.
            fallback: 降级函数，当所有重试耗尽时调用.
            validators: 结果验证器列表，任一返回 False 则视为无效结果触发重试.

        Returns:
            函数返回值，或降级函数返回值.
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, self._config.max_retries + 1):
            try:
                result = func(*args, **kwargs)

                # 记录成功
                self._health_monitor.record_success(0.0)

                # 验证结果
                if validators:
                    invalid_validators = [
                        v for v in validators if not v(result)
                    ]
                    if invalid_validators:
                        raise ValidationError(
                            f"结果验证失败 ({len(invalid_validators)}/{len(validators)} 个验证器不通过)"
                        )

                return result

            except self._config.retryable_exceptions as exc:
                last_error = exc
                self._health_monitor.record_failure(str(exc))

                if attempt < self._config.max_retries:
                    delay = min(
                        self._config.base_delay * (self._config.backoff_factor ** (attempt - 1)),
                        self._config.max_delay,
                    )
                    logger.warning(
                        "SelfHealing: 第 %d/%d 次尝试失败，%.1fs 后重试: %s",
                        attempt, self._config.max_retries, delay, exc,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "SelfHealing: %d 次尝试全部失败: %s",
                        self._config.max_retries, exc,
                    )

            except ValidationError as exc:
                # 验证失败 - 也重试
                last_error = exc
                logger.warning(
                    "SelfHealing: 第 %d/%d 次结果验证失败: %s",
                    attempt, self._config.max_retries, exc,
                )
                if attempt < self._config.max_retries:
                    delay = min(
                        self._config.base_delay * (self._config.backoff_factor ** (attempt - 1)),
                        self._config.max_delay,
                    )
                    time.sleep(delay)

            except Exception as exc:
                # 不可重试的异常，直接抛出
                self._health_monitor.record_failure(str(exc))
                logger.error("SelfHealing: 不可重试异常: %s", exc)
                raise

        # 所有重试耗尽，尝试降级
        return self.fallback_degrade(fallback, last_error)

    def validate_result(
        self,
        result: Any,
        validators: Optional[List[Callable[[Any], bool]]] = None,
    ) -> bool:
        """验证输出格式和完整性.

        内置验证：非 None、非空字符串、可解析 JSON（如果是字符串）。
        可额外传入自定义验证器。
        """
        if result is None:
            logger.warning("validate_result: 结果为 None")
            return False

        if isinstance(result, str) and not result.strip():
            logger.warning("validate_result: 结果为空字符串")
            return False

        if isinstance(result, str):
            # 检查是否为截断的 JSON
            if result.strip().startswith("{") and not result.strip().endswith("}"):
                logger.warning("validate_result: JSON 疑似截断")
                return False
            if result.strip().startswith("[") and not result.strip().endswith("]"):
                logger.warning("validate_result: JSON 数组疑似截断")
                return False

        # 自定义验证器
        if validators:
            for i, validator in enumerate(validators):
                if not validator(result):
                    logger.warning("validate_result: 自定义验证器 #%d 不通过", i)
                    return False

        return True

    def fallback_degrade(
        self,
        fallback: Optional[Callable] = None,
        error: Optional[Exception] = None,
    ) -> Any:
        """降级策略：API 失败 -> 缓存 -> 跳过.

        Args:
            fallback: 降级函数，接受 error 参数.
            error: 触发降级的原始异常.

        Returns:
            降级函数的返回值，或 None.
        """
        if fallback:
            try:
                result = fallback(error) if _accepts_error(fallback) else fallback()
                logger.info("SelfHealing: 降级策略已执行")
                return result
            except Exception as exc:
                logger.error("SelfHealing: 降级策略也失败: %s", exc)

        # 无降级函数，记录并返回 None
        self.log_and_alert(
            severity="critical",
            message=f"所有重试和降级均失败: {error}",
        )
        return None

    def log_and_alert(
        self,
        severity: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """记录诊断信息到日志.

        Args:
            severity: "critical" / "warning" / "info".
            message: 描述信息.
            context: 附加上下文数据.
        """
        log_ctx = {"health": self._health_monitor.get_stats()}
        if self._token_tracker.get_stats()["total_requests"] > 0:
            log_ctx["tokens"] = self._token_tracker.get_stats()
        if context:
            log_ctx.update(context)

        log_msg = f"[SelfHealing:{severity.upper()}] {message}"
        if log_ctx:
            log_msg += f" | context={json.dumps(log_ctx, default=str, ensure_ascii=False)}"

        if severity == "critical":
            logger.critical(log_msg)
        elif severity == "warning":
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

    # --- 属性访问 ---

    @property
    def health_monitor(self) -> HealthMonitor:
        return self._health_monitor

    @property
    def token_tracker(self) -> TokenUsageTracker:
        return self._token_tracker


def _accepts_error(func: Callable) -> bool:
    """检查函数是否接受 error 参数（用于降级调用）."""
    import inspect
    sig = inspect.signature(func)
    return "error" in sig.parameters


# ===========================================================================
#  便捷工厂函数
# ===========================================================================


def create_self_healing_executor(
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> SelfHealingExecutor:
    """创建一个预配置的自愈执行器.

    Args:
        max_retries: 最大重试次数.
        base_delay: 基础退避延迟（秒）.

    Returns:
        配置好的 SelfHealingExecutor 实例.
    """
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
    )
    return SelfHealingExecutor(
        retry_config=config,
        health_monitor=HealthMonitor(),
        token_tracker=TokenUsageTracker(),
    )
