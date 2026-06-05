"""
配置文件 — 每日学术论文热点追踪系统
基于 OpenClaw 多智能体协同理念设计

所有敏感信息通过环境变量 / .env 文件注入，不硬编码到代码中。
"""

import os
from pathlib import Path

# ============================================================
# 加载 .env 文件（如果存在）
# ============================================================
_ENV_FILE = Path(__file__).parent / ".env"
if _ENV_FILE.exists():
    with open(_ENV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key not in os.environ:  # 不覆盖已设置的环境变量
                os.environ[key] = val

# ============================================================
# DeepSeek API 配置（OpenAI 兼容接口）
# ============================================================
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# ============================================================
# arXiv API 配置
# ============================================================
ARXIV_CATEGORIES = [
    "cs.AI",
    "cs.CL",
    "cs.LG",
]

MAX_PAPERS_PER_CATEGORY = 20
MAX_PAPERS_TOTAL = 300
LOOKBACK_DAYS = 2

# arXiv API 基础 URL
ARXIV_API_URL = "https://export.arxiv.org/api/query"

# 请求间隔（秒）—— arXiv 官方要求至少 1 秒，我们设 6 秒避免 429
REQUEST_DELAY = float(os.environ.get("ARXIV_DELAY", "2.0"))

# ============================================================
# 数据库配置
# ============================================================
DATABASE_PATH = os.path.join(os.path.dirname(__file__), "data", "papers.db")

# ============================================================
# 日报输出路径
# ============================================================
REPORT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "reports")

# ============================================================
# 定时任务配置
# ============================================================
SCHEDULE_HOUR = int(os.environ.get("SCHEDULE_HOUR", "8"))
SCHEDULE_MINUTE = int(os.environ.get("SCHEDULE_MINUTE", "0"))

# ============================================================
# Web UI 配置
# ============================================================
WEB_HOST = os.environ.get("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.environ.get("WEB_PORT", "7860"))

# ============================================================
# 日志配置
# ============================================================
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
