"""
报告汇总智能体 (Report Aggregator Agent)
───────────────────────────────────────
职责：整合三个智能体的分析结果，生成结构化的每日日报（Markdown 格式）
"""

import os
import json
from datetime import date, datetime
from config import REPORT_OUTPUT_DIR
from database import save_daily_report, get_daily_report


def generate_daily_report(
    papers: list[dict],
    trend_result: dict,
    report_date: str = None,
) -> str:
    """
    汇总所有分析结果，生成中文 Markdown 日报

    参数:
        papers: 带分类信息的论文列表
        trend_result: 趋势分析智能体的输出
        report_date: 报告日期

    返回:
        str: Markdown 格式的完整日报
    """
    if report_date is None:
        report_date = date.today().isoformat()

    today_str = date.today().strftime("%Y年%m月%d日")

    sections = []

    # ============ 报告头部 ============
    sections.append(f"# 📰 AI/CS 学术论文日报")
    sections.append(f"**日期**: {today_str}  |  **报告类型**: 自动生成  |  **数据来源**: arXiv API  \n")

    # ============ 一、今日概览 ============
    sections.append("---\n")
    sections.append("## 一、今日概览\n")
    sections.append(f"- 📊 **采集论文总数**: {len(papers)} 篇")
    if trend_result:
        dist = trend_result.get("category_dist", {})
        dist_text = "、".join(f"{k}({v}篇)" for k, v in list(dist.items())[:6])
        sections.append(f"- 📂 **子领域分布**: {dist_text}")
        sections.append(f"- 🔥 **热点论文数**: {len(trend_result.get('hot_papers', []))} 篇\n")

    # ============ 二、热门方向 TOP 3 ============
    if trend_result and trend_result.get("category_dist"):
        dist = trend_result["category_dist"]
        top3_cats = list(dist.keys())[:3]
        sections.append("## 二、今日热门研究方向 TOP 3\n")
        for i, cat in enumerate(top3_cats, 1):
            count = dist[cat]
            medals = ["🥇", "🥈", "🥉"]
            sections.append(f"### {medals[i-1]} {cat}（{count} 篇）\n")

            # 该方向评分最高的一篇代表论文
            cat_papers = [p for p in papers if p.get("subfield") == cat]
            cat_papers.sort(key=lambda x: float(x.get("importance", 0)), reverse=True)
            if cat_papers:
                p = cat_papers[0]
                imp_val = float(p.get("importance", 0))
                stars = "⭐" * int(imp_val) + f" ({imp_val:.1f})"
                sections.append(f"- {stars} **{p['title']}**")
                if p.get("key_contribution"):
                    sections.append(f"  > {p['key_contribution']}")
                sections.append(f"  > `{p['arxiv_id']}` | {', '.join(p.get('authors', [])[:2])}")
                if len(cat_papers) > 1:
                    sections.append(f"  > *该方向另有 {len(cat_papers)-1} 篇相关论文*\n")
                else:
                    sections.append("")

    # ============ 三、今日热点论文 TOP 10（过滤模糊分类+低质量）============
    sections.append("## 三、今日热点论文 TOP 10\n")
    # 只保留有明确分类(不含"其他")且评分>=2.0的论文
    hot_candidates = [
        p for p in papers
        if float(p.get("importance", 0)) >= 2.0
        and "其他" not in (p.get("subfield") or "")
    ]
    hot_papers = sorted(hot_candidates, key=lambda x: float(x.get("importance", 0)), reverse=True)[:10]
    if not hot_papers:
        hot_papers = sorted(papers, key=lambda x: float(x.get("importance", 0)), reverse=True)[:10]
    for i, p in enumerate(hot_papers, 1):
        imp = p.get("importance", 0)
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        sections.append(f"### {medal} {p['title']}\n")
        sections.append(f"| 属性 | 内容 |")
        sections.append(f"|------|------|")
        sections.append(f"| 📎 ID | [{p['arxiv_id']}](https://arxiv.org/abs/{p['arxiv_id']}) |")
        sections.append(f"| ✍️ 作者 | {', '.join(p.get('authors', [])[:4])}{' et al.' if len(p.get('authors', [])) > 4 else ''} |")
        sections.append(f"| 📂 子领域 | {p.get('subfield', 'N/A')} |")
        sections.append(f"| ⭐ 评分 | {imp}/5 |")
        sections.append(f"| 💡 核心贡献 | {p.get('key_contribution', 'N/A')} |\n")
        # 摘要截断
        abstract = p.get("abstract", "N/A")
        if len(abstract) > 300:
            abstract = abstract[:300] + "..."
        sections.append(f"**摘要**: {abstract}\n")

    # ============ 四、趋势分析 ============
    sections.append("## 四、研究趋势分析\n")
    if trend_result and trend_result.get("trend_summary"):
        sections.append(trend_result["trend_summary"])
        sections.append("")
    else:
        sections.append("（趋势分析数据暂缺）\n")

    # ============ 五、关键词云 ============
    sections.append("## 五、今日高频关键词\n")
    if trend_result and trend_result.get("keywords"):
        keywords = trend_result["keywords"]
        # 按热度排序
        kw_sorted = sorted(keywords, key=lambda x: x[1], reverse=True)[:30]
        sections.append("```")
        max_freq = max(k[1] for k in kw_sorted) if kw_sorted else 1
        for kw, freq in kw_sorted:
            bar_len = max(1, int(freq / max_freq * 40))
            bar = "█" * bar_len
            sections.append(f"{kw:<20} {bar} {freq}")
        sections.append("```\n")

    # ============ 六、子领域论文完整列表 ============
    sections.append("## 六、各子领域论文列表\n")
    by_subfield = {}
    for p in papers:
        sf = p.get("subfield", "其他")
        by_subfield.setdefault(sf, []).append(p)

    for sf, sf_papers in sorted(by_subfield.items(), key=lambda x: len(x[1]), reverse=True):
        sections.append(f"### {sf}（{len(sf_papers)} 篇）\n")
        for p in sf_papers:
            authors_short = ", ".join(p.get("authors", [])[:2])
            if len(p.get("authors", [])) > 2:
                authors_short += " et al."
            sections.append(
                f"- [{p['arxiv_id']}] **{p['title']}** — {authors_short}"
            )
        sections.append("")

    # ============ 报告尾部 ============
    sections.append("---\n")
    sections.append(f"*本报告由「每日学术论文热点追踪系统」自动生成*  \n")
    sections.append(f"*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*  \n")
    sections.append(f"*基于 OpenClaw 多智能体协同框架设计*  \n")
    sections.append(f"*数据来源: [arXiv.org](https://arxiv.org)*  \n")

    report = "\n".join(sections)

    # 保存到文件和数据库
    os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(REPORT_OUTPUT_DIR, f"report_{report_date}.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)

    save_daily_report(report_date, report)

    print(f"[Aggregator] 日报已生成: {filepath} ({len(report)} 字符)")
    return report


def load_report(report_date: str = None) -> str:
    """加载已有的日报"""
    if report_date is None:
        report_date = date.today().isoformat()

    # 优先从数据库加载
    db_report = get_daily_report(report_date)
    if db_report:
        return db_report["report_content"]

    # 回退到文件
    filepath = os.path.join(REPORT_OUTPUT_DIR, f"report_{report_date}.md")
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    return None
