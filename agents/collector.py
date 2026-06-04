"""
智能体 1: 论文采集智能体 (Collector Agent)
─────────────────────────────────────────
职责：每日从 arXiv API 自动获取 CS/AI 领域最新论文
输出：结构化论文元数据列表（标题、作者、摘要、分类等）

限流策略：
  - 请求间隔 5 秒（可配置）
  - HTTP 429 / 5xx 指数退避重试（最多 3 次）
  - 连接错误自动重试
"""

import requests
import feedparser
import time
import random
from datetime import date, timedelta
from config import (
    ARXIV_API_URL,
    ARXIV_CATEGORIES,
    MAX_PAPERS_PER_CATEGORY,
    MAX_PAPERS_TOTAL,
    LOOKBACK_DAYS,
    REQUEST_DELAY,
)


def _build_query(category: str) -> str:
    """
    构建 arXiv API 查询字符串
    查询指定分类下，最近 N 天的新论文
    """
    today = date.today()
    start = today - timedelta(days=LOOKBACK_DAYS)
    # arXiv 支持的日期格式: YYYYMMDD 或 YYYYMMDDHHMM
    date_range = f"[{start.strftime('%Y%m%d')}0000+TO+{today.strftime('%Y%m%d')}2359]"

    query = (
        f"search_query=cat:{category}"
        f"&sortBy=submittedDate"
        f"&sortOrder=descending"
        f"&max_results={MAX_PAPERS_PER_CATEGORY}"
        f"&start=0"
    )
    return query


def _parse_arxiv_entry(entry: dict) -> dict:
    """
    解析单条 arXiv API 返回的论文条目
    提取结构化字段
    """
    # 作者列表
    authors = [a.get("name", "Unknown") for a in entry.get("authors", [])]

    # 分类标签（arXiv 可能返回多个）
    tags = entry.get("tags", [])
    categories = [t.get("term", "") for t in tags] if tags else []

    # 主分类
    arxiv_primary = entry.get("arxiv_primary_category", {})
    primary_cat = arxiv_primary.get("term", categories[0] if categories else "")

    # 解析 arXiv ID（从 URL 中提取）
    arxiv_id = entry.get("id", "").split("/abs/")[-1]
    # 去除版本号（如 v1, v2）
    if "v" in arxiv_id.split("/")[-1]:
        arxiv_id = arxiv_id.rsplit("v", 1)[0] if not arxiv_id.startswith("http") else arxiv_id

    # 干净的 ID
    clean_id = entry.get("id", "").rsplit("/", 1)[-1] if "/" in entry.get("id", "") else entry.get("id", "")
    # 去掉 URL 前缀
    id_url = entry.get("id", "")
    if "arxiv.org/abs/" in id_url:
        clean_id = id_url.split("arxiv.org/abs/")[-1]

    return {
        "arxiv_id": clean_id,
        "title": entry.get("title", "Untitled").strip().replace("\n", " "),
        "authors": authors,
        "abstract": entry.get("summary", "").strip().replace("\n", " "),
        "categories": categories,
        "primary_cat": primary_cat,
        "published_date": entry.get("published", "")[:10],  # YYYY-MM-DD
        "pdf_url": entry.get("link", ""),
        "collected_date": date.today().isoformat(),
    }


def _fetch_with_retry(url: str, max_retries: int = 3, base_delay: float = 5.0) -> requests.Response | None:
    """
    带指数退避的 HTTP GET 请求

    参数:
        url: 请求 URL
        max_retries: 最大重试次数
        base_delay: 基础等待秒数（每次重试翻倍 + 随机抖动）

    返回:
        Response 对象，或 None（所有重试均失败）
    """
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=30)

            # 成功
            if resp.status_code == 200:
                return resp

            # 限流 — 等更久
            if resp.status_code == 429:
                wait = base_delay * (2 ** attempt) + random.uniform(1, 5)
                print(f"HTTP 429 (限流)，第 {attempt + 1} 次重试，等待 {wait:.1f} 秒...")
                time.sleep(wait)
                continue

            # 服务器错误 — 可重试
            if resp.status_code >= 500:
                wait = base_delay * (2 ** attempt) + random.uniform(0, 3)
                print(f"HTTP {resp.status_code}，第 {attempt + 1} 次重试，等待 {wait:.1f} 秒...")
                time.sleep(wait)
                continue

            # 其他错误码不重试
            print(f"HTTP {resp.status_code}")
            return None

        except requests.exceptions.Timeout:
            wait = base_delay * (2 ** attempt)
            print(f"超时，第 {attempt + 1} 次重试，等待 {wait:.1f} 秒...")
            time.sleep(wait)

        except requests.exceptions.ConnectionError:
            wait = base_delay * (2 ** attempt) + random.uniform(1, 3)
            print(f"连接失败，第 {attempt + 1} 次重试，等待 {wait:.1f} 秒...")
            time.sleep(wait)

        except requests.exceptions.RequestException as e:
            print(f"请求异常: {e}")
            return None

    print(f"已重试 {max_retries} 次，放弃")
    return None


def collect_papers(categories: list[str] = None) -> list[dict]:
    """
    主采集函数 — 从 arXiv 获取各分类最新论文

    参数:
        categories: 要查询的 arXiv 分类列表，默认使用 config 中的配置

    返回:
        list[dict]: 论文列表，每个元素包含完整元数据
    """
    if categories is None:
        categories = ARXIV_CATEGORIES

    all_papers = []
    seen_ids = set()
    fail_count = 0

    print(f"[Collector Agent] 开始采集论文，目标分类: {len(categories)} 个")
    print(f"[Collector Agent] 时间范围: 最近 {LOOKBACK_DAYS} 天")
    print(f"[Collector Agent] 请求间隔: {REQUEST_DELAY} 秒")

    for i, cat in enumerate(categories):
        try:
            query = _build_query(cat)
            url = f"{ARXIV_API_URL}?{query}"

            print(f"  -> 查询分类: {cat} ...", end=" ", flush=True)

            # 每次请求前等待（第一个请求也等，避免被识别为高频）
            if i > 0:
                time.sleep(REQUEST_DELAY)

            response = _fetch_with_retry(url)

            if response is None:
                print("失败")
                fail_count += 1
                continue

            # 解析 Atom XML
            feed = feedparser.parse(response.text)
            entries = feed.entries

            print(f"获取 {len(entries)} 篇")

            for entry in entries:
                paper = _parse_arxiv_entry(entry)
                if paper["arxiv_id"] not in seen_ids:
                    seen_ids.add(paper["arxiv_id"])
                    all_papers.append(paper)

        except Exception as e:
            print(f"解析异常: {e}")
            fail_count += 1

        # 达到总量上限则停止
        if len(all_papers) >= MAX_PAPERS_TOTAL:
            print(f"  → 已达总量上限 {MAX_PAPERS_TOTAL}，停止采集")
            break

    print(f"[Collector Agent] 采集完成: 成功={len(all_papers)} 篇, 失败={fail_count}/{len(categories)} 个分类")

    # 按日期排序取最新
    if all_papers:
        all_papers.sort(key=lambda x: x.get("published_date", ""), reverse=True)

    if len(all_papers) > MAX_PAPERS_TOTAL:
        all_papers = all_papers[:MAX_PAPERS_TOTAL]

    return all_papers


# ============================================================
# 工具函数：格式化输出（方便报告生成）
# ============================================================

def format_paper_brief(paper: dict) -> str:
    """格式化单篇论文的简要信息"""
    authors_str = ", ".join(paper.get("authors", [])[:3])
    if len(paper.get("authors", [])) > 3:
        authors_str += " et al."
    return (
        f"**{paper['title']}**  \n"
        f"  📎 `{paper['arxiv_id']}` | {authors_str}  \n"
        f"  📂 {paper.get('primary_cat', 'N/A')} | 📅 {paper.get('published_date', 'N/A')}  \n"
    )


def format_papers_summary(papers: list[dict]) -> str:
    """格式化论文列表摘要（用于注入 LLM prompt）"""
    lines = []
    for i, p in enumerate(papers[:50], 1):  # 最多 50 篇供 LLM 处理
        lines.append(
            f"{i}. [{p['arxiv_id']}] {p['title']}\n"
            f"   Categories: {', '.join(p.get('categories', []))}\n"
            f"   Abstract: {p.get('abstract', '')[:300]}..."
        )
    return "\n\n".join(lines)


if __name__ == "__main__":
    # 单独测试采集智能体
    papers = collect_papers()
    for p in papers[:5]:
        print(format_paper_brief(p))
        print()
