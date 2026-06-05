"""
智能体 3: 研究趋势分析智能体 (Trend Analyzer Agent)
────────────────────────────────────────────────
职责：分析论文关键词趋势，识别热门研究方向，对比历史数据
使用 jieba 分词 + DeepSeek API 语义分析
"""

import json
import re
import time
from datetime import date, timedelta
from collections import Counter
from openai import OpenAI
import jieba
import jieba.analyse

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from database import get_papers_by_date, get_daily_trend, save_daily_trend

# 初始化 DeepSeek 客户端
_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# ============================================================
# 学术停用词（在关键词提取时过滤）
# ============================================================
ACADEMIC_STOP_WORDS = {
    "提出", "方法", "模型", "实验", "结果", "表明", "基于", "使用",
    "本文", "我们", "一个", "一种", "可以", "能够", "问题", "任务",
    "不同", "新的", "其中", "通过", "进行", "包括", "这个", "它们",
    "以及", "这些", "一些", "所有", "主要", "相关", "之间", "此外",
    "该", "等", "及", "与", "中", "上", "在", "对", "从", "为",
    "approach", "method", "model", "propose", "paper", "result",
    "experiment", "show", "demonstrate", "evaluate",
}

# ============================================================
# Prompt 模板
# ============================================================
TREND_ANALYSIS_PROMPT = """你是一位资深的技术趋势分析师，专注于 AI 和计算机科学领域的前沿动态。

## 今日论文概况
- 总数: {total_papers} 篇
- 子领域分布: {category_dist}

## 今日高频关键词
{keywords}

## 今日热点论文（Top 5）
{hot_papers}

## 昨日趋势（如有）
{yesterday_trend}

## 任务
请基于以上数据，撰写一份今日研究趋势分析报告（中文），包含：

1. **整体态势** (1-2句概括今日研究热点)
2. **热门方向 TOP 3** (每个方向简要说明)
3. **值得关注的新趋势** (今天出现的新的研究方向或方法)
4. **与昨日对比** (如果有历史数据，分析趋势变化)
5. **明日预测** (预测哪些方向可能持续升温)

## 输出格式
请用 Markdown 格式直接输出报告正文。不要输出"好的"、"以下是我"等开场白或角色确认语，直接从"## 整体态势"开始写。"""


def extract_keywords_from_papers(papers: list[dict], top_k: int = 30) -> list[tuple[str, int]]:
    """
    从论文标题和摘要中提取高频关键词
    使用 jieba TF-IDF + 规则过滤
    """
    print(f"[Trend Analyzer] 提取关键词，论文数: {len(papers)}")

    # 合并所有标题和摘要
    all_text = ""
    for p in papers:
        all_text += p.get("title", "") + " "
        all_text += p.get("abstract", "") + " "

    # 清除 LaTeX 和特殊字符
    all_text = re.sub(r"\$[^$]+\$", "", all_text)  # 移除 LaTeX 公式
    all_text = re.sub(r"\\[a-zA-Z]+", "", all_text)  # 移除 LaTeX 命令
    all_text = re.sub(r"[^\w\s一-鿿]", " ", all_text)  # 保留中英文和数字

    # 使用 jieba 的 TF-IDF 提取关键词
    keywords_tfidf = jieba.analyse.extract_tags(
        all_text, topK=top_k * 2, withWeight=True
    )

    # 过滤停用词和过短词
    keywords = []
    for word, weight in keywords_tfidf:
        if (
            len(word) >= 2
            and word not in ACADEMIC_STOP_WORDS
            and word.lower() not in ACADEMIC_STOP_WORDS
            and not word.isdigit()
        ):
            keywords.append((word, int(weight * 1000)))

    keywords = keywords[:top_k]

    # 补充英文关键词（jieba 对英文分词效果一般，手动提取高频英文词组）
    english_keywords = _extract_english_keywords(all_text, top_k=10)
    for ek, count in english_keywords:
        if ek not in [k[0] for k in keywords]:
            keywords.append((ek, count))

    print(f"[Trend Analyzer] 提取到 {len(keywords)} 个关键词")
    return keywords


def _extract_english_keywords(text: str, top_k: int = 10) -> list[tuple[str, int]]:
    """提取英文高频词组（2-3 词组合）"""
    # 提取英文单词序列
    english_words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    # 过滤常见停用词
    en_stop = {"the", "and", "for", "our", "with", "that", "this", "from",
               "which", "have", "been", "also", "its", "not", "but", "are",
               "was", "can", "has", "had", "one", "two", "new", "use",
               "based", "show", "using", "into", "such"}
    filtered = [w for w in english_words if w not in en_stop]
    counter = Counter(filtered)
    return [(w, c) for w, c in counter.most_common(top_k)]


def analyze_trends_with_llm(
    papers: list[dict],
    keywords: list[tuple[str, int]],
    category_dist: dict,
    yesterday_trend: dict = None,
) -> str:
    """
    调用 DeepSeek API 生成趋势分析报告
    """
    if not papers:
        return "今日无论文数据，无法生成趋势分析。"

    print(f"[Trend Analyzer] 调用 LLM 生成趋势分析...")

    # 格式化关键词
    kw_text = "\n".join(f"- {k[0]} (热度: {k[1]})" for k in keywords[:20])

    # 格式化热点论文
    hot = sorted(papers, key=lambda x: x.get("importance", 0), reverse=True)[:5]
    hot_text = "\n\n".join(
        f"**{p['title']}** (评分: {p.get('importance', 0)}/5)\n"
        f"  ID: {p['arxiv_id']} | 子领域: {p.get('subfield', 'N/A')}\n"
        f"  贡献: {p.get('key_contribution', 'N/A')}"
        for p in hot
    )

    # 格式化昨日趋势——只提取关键数字而非完整报告，避免 LLM 混乱
    yesterday_text = "（无昨日数据）"
    if yesterday_trend:
        y_total = yesterday_trend.get("total_papers", "?")
        y_dist = yesterday_trend.get("category_dist", {})
        y_top3 = list(y_dist.items())[:3] if y_dist else []
        y_kw = yesterday_trend.get("keyword_trends", {})
        y_top_kw = list(y_kw.keys())[:5] if y_kw else []
        y_dist_text = "、".join(f"{k}({v}篇)" for k,v in y_top3) if y_top3 else "无"
        y_kw_text = "、".join(y_top_kw) if y_top_kw else "无"
        yesterday_text = (
            f"昨日共采集 {y_total} 篇论文。"
            f"昨日热门方向 TOP 3: {y_dist_text}。"
            f"昨日高频关键词: {y_kw_text}。"
        )

    # 格式化子领域分布
    dist_text = "\n".join(f"- {k}: {v} 篇" for k, v in category_dist.items())

    prompt = TREND_ANALYSIS_PROMPT.format(
        total_papers=len(papers),
        category_dist=dist_text,
        keywords=kw_text,
        hot_papers=hot_text,
        yesterday_trend=yesterday_text,
    )

    try:
        resp = _client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "你是一位资深技术趋势分析师，请用中文输出结构化的 Markdown 报告。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
        )
        summary = resp.choices[0].message.content
        print(f"[Trend Analyzer] 趋势分析完成 ({len(summary)} 字符)")
        return summary
    except Exception as e:
        print(f"[Trend Analyzer] LLM 调用失败: {e}")
        return f"## 趋势分析\n\nLLM 调用失败: {e}\n\n基于规则的关键词 Top 10: {', '.join(k[0] for k in keywords[:10])}"


def run_trend_analysis(report_date: str = None) -> dict:
    """
    执行完整的趋势分析流程

    参数:
        report_date: 分析日期，默认今天

    返回:
        dict: 包含关键词、子领域分布、热点论文、趋势摘要
    """
    if report_date is None:
        report_date = date.today().isoformat()

    # 从数据库获取今日论文
    papers = get_papers_by_date(report_date)
    if not papers:
        print(f"[Trend Analyzer] {report_date} 无论文数据")
        return None

    # 1. 提取关键词
    keywords = extract_keywords_from_papers(papers)

    # 2. 统计子领域分布（来自分类智能体的结果）
    category_dist = Counter(p.get("subfield", "其他") for p in papers)
    category_dist = dict(category_dist.most_common())

    # 3. 热点论文 ID 列表
    hot_papers = sorted(papers, key=lambda x: x.get("importance", 0), reverse=True)
    hot_ids = [p["arxiv_id"] for p in hot_papers[:10]]

    # 4. 获取昨日趋势用于对比
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    yesterday_trend = get_daily_trend(yesterday)

    # 5. LLM 趋势分析
    trend_summary = analyze_trends_with_llm(
        papers, keywords, category_dist, yesterday_trend
    )

    # 6. 持久化
    save_daily_trend(
        report_date=report_date,
        total_papers=len(papers),
        keyword_trends=dict(keywords),
        category_dist=category_dist,
        hot_papers=hot_ids,
        trend_summary=trend_summary,
    )

    return {
        "report_date": report_date,
        "total_papers": len(papers),
        "keywords": keywords,
        "category_dist": category_dist,
        "hot_papers": hot_ids,
        "trend_summary": trend_summary,
    }


# ============================================================
# 历史对比工具
# ============================================================

def compare_with_history(current_keywords: dict, days_back: int = 7) -> str:
    """
    对比今日关键词与历史趋势
    返回对比分析的文本描述
    """
    today = date.today()
    all_historical = []

    for i in range(1, days_back + 1):
        d = (today - timedelta(days=i)).isoformat()
        trend = get_daily_trend(d)
        if trend:
            all_historical.append({
                "date": d,
                "keywords": trend.get("keyword_trends", {}),
            })

    if not all_historical:
        return "无足够历史数据可供对比。"

    lines = ["## 历史趋势对比\n"]
    for h in all_historical:
        top5 = list(h["keywords"].keys())[:5]
        lines.append(f"- **{h['date']}**: {', '.join(top5)}")

    # 找出今日新出现的关键词
    historical_kw_set = set()
    for h in all_historical:
        historical_kw_set.update(h["keywords"].keys())

    today_set = set(current_keywords.keys())
    new_keywords = today_set - historical_kw_set

    if new_keywords:
        lines.append(f"\n🆕 **今日新出现的热词**: {', '.join(list(new_keywords)[:10])}")

    return "\n".join(lines)
