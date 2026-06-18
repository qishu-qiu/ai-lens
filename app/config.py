"""Application configuration."""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Database
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/ai_self_evolution.db")

# Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "30"))

# ============================================================
# Crawler - RSS Sources (Tier 1: Auto-crawl via RSS)
# ============================================================

# --- arXiv: AI/ML core categories ---
ARXIV_RSS_URLS = [
    "https://rss.arxiv.org/rss/cs.AI",       # Artificial Intelligence
    "https://rss.arxiv.org/rss/cs.CL",       # Computation and Language (NLP)
    "https://rss.arxiv.org/rss/cs.LG",       # Machine Learning
    "https://rss.arxiv.org/rss/cs.CV",       # Computer Vision
    "https://rss.arxiv.org/rss/cs.RO",       # Robotics
    "https://rss.arxiv.org/rss/cs.MA",       # Multiagent Systems
    "https://rss.arxiv.org/rss/stat.ML",     # Statistics - Machine Learning
]

# --- HuggingFace Daily Papers (community RSS) ---
HF_DAILY_PAPERS_RSS = "https://papers.takara.ai/api/feed"

# --- AI Lab Official Blogs (RSS) ---
AI_LAB_RSS_URLS = {
    "openai": "https://openai.com/blog/rss.xml",
    "anthropic": "https://www.anthropic.com/rss/feed",
    "deepmind": "https://deepmind.google/discover/blog/rss/",
    "meta_ai": "https://ai.meta.com/blog/rss/",
    "microsoft_research": "https://www.microsoft.com/research/blog/feed/",
    "nvidia_research": "https://www.nvidia.com/research/feed/",
    "bair": "https://bair.berkeley.edu/blog/feed.xml",
}

# --- Chinese AI Media (RSS) ---
CN_AI_RSS_URLS = {
    "jiqizhixin": "https://www.jiqizhixin.com/rss/",
    "qbitai": "https://www.qbitai.com/rss/",
}

# --- AI News (RSS) ---
AI_NEWS_RSS_URLS = {
    "techcrunch_ai": "https://techcrunch.com/category/artificial-intelligence/feed/",
    "venturebeat_ai": "https://venturebeat.com/category/ai/feed/",
    "the_gradient": "https://thegradient.pub/feed/",
    "synced": "https://syncedreview.com/feed/",
}

# --- AI变现与商业模式 (RSS) ---
AI_BUSINESS_RSS_URLS = {
    "techcrunch_ai_business": {
        "type": "rss",
        "name": "TechCrunch AI",
        "url": "https://techcrunch.com/category/artificial-intelligence/feed/",
        "category": "ai_business",
    },
    "theverge_ai": {
        "type": "rss",
        "name": "The Verge AI",
        "url": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
        "category": "ai_business",
    },
}

# --- AI工具生态 (RSS + Web) ---
AI_TOOLS_SOURCES = {
    "huggingface_blog": {
        "type": "rss",
        "name": "HuggingFace Blog",
        "url": "https://huggingface.co/blog/feed.xml",
        "category": "ai_tools",
    },
    "openai_blog": {
        "type": "rss",
        "name": "OpenAI Blog",
        "url": "https://openai.com/blog/rss.xml",
        "category": "ai_tools",
    },
    "anthropic_blog": {
        "type": "rss",
        "name": "Anthropic Blog",
        "url": "https://www.anthropic.com/news/rss",
        "category": "ai_tools",
    },
    "cursor_blog": {
        "type": "web",
        "name": "Cursor Blog",
        "url": "https://www.cursor.com/blog",
        "category": "ai_tools",
    },
}

# --- AI开发者社区 (RSS) ---
AI_COMMUNITY_RSS_URLS = {
    "reddit_local_llama": {
        "type": "rss",
        "name": "Reddit r/LocalLLaMA",
        "url": "https://www.reddit.com/r/LocalLLaMA/.rss",
        "category": "ai_community",
    },
    "reddit_artificial": {
        "type": "rss",
        "name": "Reddit r/artificial",
        "url": "https://www.reddit.com/r/artificial/.rss",
        "category": "ai_community",
    },
    "hacker_news_ai": {
        "type": "rss",
        "name": "Hacker News AI",
        "url": "https://hnrss.org/newest?q=AI+agent+LLM+GPT",
        "category": "ai_community",
    },
}

# --- AI应用框架 (RSS + Web) ---
AI_FRAMEWORK_SOURCES = {
    "langchain_blog": {
        "type": "rss",
        "name": "LangChain Blog",
        "url": "https://blog.langchain.dev/rss.xml",
        "category": "ai_framework",
    },
    "crewai_blog": {
        "type": "web",
        "name": "CrewAI Blog",
        "url": "https://www.crewai.com/blog",
        "category": "ai_framework",
    },
}

# ============================================================
# Crawler - Web Sources (Tier 2: Manual/API crawl)
# ============================================================

# --- Reddit communities (RSS via old.reddit.com) ---
REDDIT_RSS_URLS = {
    "r_machinelearning": "https://old.reddit.com/r/MachineLearning/new/.rss",
    "r_localllama": "https://old.reddit.com/r/LocalLLaMA/new/.rss",
    "r_artificial": "https://old.reddit.com/r/artificial/new/.rss",
}

# --- AI Trend / Leaderboard APIs (periodic poll) ---
AI_TREND_APIS = {
    "huggingface_trending": {
        "url": "https://huggingface.co/api/models?sort=trending&limit=20",
        "type": "api",
        "desc": "HuggingFace trending models",
    },
    "hf_daily_papers": {
        "url": "https://huggingface.co/api/daily_papers",
        "type": "api",
        "desc": "HuggingFace daily papers",
    },
    "lmsys_arena": {
        "url": "https://chat.lmsys.org/api/v1/leaderboard",
        "type": "api",
        "desc": "LMSYS Chatbot Arena leaderboard",
    },
    "semantic_scholar_trending": {
        "url": "https://api.semanticscholar.org/graph/v1/papers/search?query=large+language+model&limit=20&fields=title,abstract,url,citationCount,publicationDate",
        "type": "api",
        "desc": "Semantic Scholar trending AI papers",
    },
    "github_trending_ai": {
        "url": "https://api.github.com/search/repositories?q=topic:ai+topic:machine-learning&sort=stars&order=desc&per_page=10",
        "type": "api",
        "desc": "GitHub trending AI repos",
    },
    "artificial_analysis": {
        "url": "https://artificialanalysis.ai/api/models",
        "type": "api",
        "desc": "LLM performance/price comparison",
    },
}

# --- AI Tool Directories (web crawl) ---
AI_TOOL_SITES = {
    "product_hunt_ai": {
        "url": "https://www.producthunt.com/topics/artificial-intelligence",
        "type": "web",
        "desc": "Product Hunt AI topic",
    },
    "theresanaiforthat": {
        "url": "https://theresanaiforthat.com",
        "type": "web",
        "desc": "AI tools directory (7000+ tools)",
    },
    "futurepedia": {
        "url": "https://www.futurepedia.io",
        "type": "web",
        "desc": "AI tools directory (5000+ tools)",
    },
}

# Crawl settings
CRAWL_INTERVAL_HOURS = int(os.getenv("CRAWL_INTERVAL_HOURS", "6"))
MAX_ARTICLES_PER_RUN = int(os.getenv("MAX_ARTICLES_PER_RUN", "50"))

# Self-update
SELF_UPDATE_ENABLED = os.getenv("SELF_UPDATE_ENABLED", "true").lower() == "true"
CONSTITUTION_RULES = {
    "max_file_lines": 500,
    "max_function_lines": 50,
    "max_modules": 20,
    "require_tests": True,
}

# App
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
