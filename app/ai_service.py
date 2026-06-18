"""Ollama AI service for content analysis."""
import json
import logging
from typing import Optional, List

import httpx

from app.config import OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_TIMEOUT

logger = logging.getLogger(__name__)


class OllamaService:
    """Service for Ollama local AI analysis."""

    def __init__(self, host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = OLLAMA_TIMEOUT
        self._available = None

    async def is_available(self) -> bool:
        """Check if Ollama is available."""
        if self._available is not None:
            return self._available

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.host}/api/tags")
                self._available = response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            self._available = False

        return self._available

    async def generate_summary(self, title: str, content: str) -> Optional[str]:
        """Generate a one-sentence summary."""
        if not await self.is_available():
            return None

        prompt = f"""Summarize this AI research article in one sentence (max 100 words):

Title: {title}
Content: {content[:1500]}

Summary:"""

        try:
            result = await self._generate(prompt)
            return result.strip() if result else None
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return None

    async def extract_keywords(self, title: str, content: str) -> List[str]:
        """Extract keywords using AI."""
        if not await self.is_available():
            return []

        prompt = f"""Extract 3-5 key technical keywords from this AI article.
Return ONLY a JSON array of strings, nothing else.

Title: {title}
Content: {content[:1000]}

Keywords (JSON array):"""

        try:
            result = await self._generate(prompt)
            if result:
                # Try to parse JSON
                try:
                    keywords = json.loads(result.strip())
                    if isinstance(keywords, list):
                        return [str(k).strip() for k in keywords[:5]]
                except json.JSONDecodeError:
                    # Fallback: split by comma or newline
                    keywords = [k.strip() for k in result.replace("\n", ",").split(",") if k.strip()]
                    return keywords[:5]
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")

        return []

    async def analyze_content_level(self, title: str, content: str) -> int:
        """Analyze content level (1-3)."""
        if not await self.is_available():
            return 1

        prompt = f"""Rate the technical depth of this AI article on a scale of 1-3:
1 = Introductory/Survey, 2 = Technical/Method, 3 = Advanced/Novel
Return ONLY the number.

Title: {title}
Content: {content[:1000]}

Level (1-3):"""

        try:
            result = await self._generate(prompt)
            if result:
                level = int(result.strip()[0])
                if 1 <= level <= 3:
                    return level
        except Exception as e:
            logger.error(f"Error analyzing level: {e}")

        return 1

    async def _generate(self, prompt: str) -> Optional[str]:
        """Generate text using Ollama."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.host}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "num_predict": 200,
                        }
                    }
                )
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return None


class AIService:
    """High-level AI service facade."""

    def __init__(self):
        self.ollama = OllamaService()

    async def analyze_article(self, title: str, content: str) -> dict:
        """
        Analyze article with AI.

        Returns:
            dict with keys: summary, keywords, level
        """
        summary = await self.ollama.generate_summary(title, content)
        keywords = await self.ollama.extract_keywords(title, content)
        level = await self.ollama.analyze_content_level(title, content)

        return {
            "summary": summary,
            "keywords": keywords,
            "level": level if summary else 1,  # Only L2+ if we got a summary
        }
