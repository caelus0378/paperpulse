"""
可视化模块 — 生成图表用于报告和 Web UI
───────────────────────────────────
使用 matplotlib 生成：
  - 子领域分布柱状图
  - 关键词词云
  - 日度论文趋势折线图
"""

import os
import matplotlib
matplotlib.use("Agg")  # 非交互后端（服务器部署兼容）

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import Counter
from datetime import date, timedelta
from database import get_papers_by_date

# ============================================================
# 中文字体配置
# ============================================================

def _setup_chinese_font():
    """自动查找并设置中文字体"""
    # 尝试常见的 Windows 中文字体
    chinese_fonts = [
        "Microsoft YaHei", "SimHei", "KaiTi", "FangSong",
        "PingFang SC", "Noto Sans CJK SC", "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in chinese_fonts:
        if font in available:
            plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return font
    # 没有任何中文字体时使用英文
    print("[Vis] ⚠ 未找到中文字体，图表将使用英文标签")
    return None


_CHINESE_FONT = _setup_chinese_font()

# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "reports", "charts")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. 子领域分布柱状图
# ============================================================

def plot_category_distribution(papers: list[dict], report_date: str = None):
    """生成子领域论文数量分布柱状图"""
    if report_date is None:
        report_date = date.today().isoformat()

    dist = Counter(p.get("subfield", "其他") for p in papers)
    if not dist:
        return None

    labels = list(dist.keys())
    values = list(dist.values())

    # 颜色映射
    colors = plt.cm.Set3([i % 12 / 12 for i in range(len(labels))])

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.8)

    # 在柱子上标注数值
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=11, fontweight="bold")

    ax.set_xlabel("论文数量" if _CHINESE_FONT else "Paper Count", fontsize=12)
    ax.set_title(f"子领域论文分布 ({report_date})" if _CHINESE_FONT
                 else f"Paper Distribution by Subfield ({report_date})",
                 fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.2)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, f"category_dist_{report_date}.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    return filepath


# ============================================================
# 2. 日度论文数量趋势
# ============================================================

def plot_daily_trend(days: int = 7):
    """生成最近 N 天论文数量趋势折线图"""
    today = date.today()
    dates = []
    counts = []

    for i in range(days - 1, -1, -1):
        d = (today - timedelta(days=i)).isoformat()
        papers = get_papers_by_date(d)
        dates.append(d[-5:])  # MM-DD
        counts.append(len(papers))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, counts, marker="o", linewidth=2.5, markersize=8,
            color="#2196F3", markerfacecolor="#1565C0")

    # 填充区域
    ax.fill_between(range(len(dates)), counts, alpha=0.15, color="#2196F3")

    # 标注数值
    for i, (d, c) in enumerate(zip(dates, counts)):
        ax.annotate(str(c), (i, c), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("日期" if _CHINESE_FONT else "Date", fontsize=12)
    ax.set_ylabel("论文数" if _CHINESE_FONT else "Paper Count", fontsize=12)
    ax.set_title("每日论文采集数量趋势" if _CHINESE_FONT else "Daily Paper Collection Trend",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.3 if counts else 10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "daily_trend.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    return filepath


# ============================================================
# 3. 热点论文评分图
# ============================================================

def plot_top_papers(papers: list[dict], top_k: int = 10, report_date: str = None):
    """生成 TOP-K 论文重要性评分图"""
    if report_date is None:
        report_date = date.today().isoformat()

    sorted_papers = sorted(papers, key=lambda x: x.get("importance", 0), reverse=True)[:top_k]

    titles = [p["title"][:60] + "..." if len(p["title"]) > 60 else p["title"]
              for p in sorted_papers]
    scores = [p.get("importance", 0) for p in sorted_papers]

    # 颜色渐变（高分为暖色）
    colors = plt.cm.RdYlGn([s / 5.0 for s in scores])

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(titles)), scores, color=colors, edgecolor="white")

    ax.set_yticks(range(len(titles)))
    ax.set_yticklabels(titles, fontsize=9)
    ax.set_xlabel("重要性评分 (0-5)" if _CHINESE_FONT else "Importance Score (0-5)", fontsize=12)
    ax.set_title(f"TOP {top_k} 热点论文 ({report_date})" if _CHINESE_FONT
                 else f"TOP {top_k} Hot Papers ({report_date})",
                 fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 5.5)
    ax.grid(axis="x", alpha=0.3)

    # 标注分数
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{score}/5", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, f"top_papers_{report_date}.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    return filepath


# ============================================================
# 批量生成全部图表
# ============================================================

def generate_all_charts(papers: list[dict] = None, report_date: str = None):
    """生成所有可视化图表"""
    if report_date is None:
        report_date = date.today().isoformat()
    if papers is None:
        papers = get_papers_by_date(report_date)

    charts = {}

    print("[Vis] 生成可视化图表...")

    # 1. 子领域分布
    fp = plot_category_distribution(papers, report_date)
    if fp:
        charts["category_dist"] = fp
        print(f"  ✓ 子领域分布图: {fp}")

    # 2. 日度趋势
    fp = plot_daily_trend()
    if fp:
        charts["daily_trend"] = fp
        print(f"  ✓ 日度趋势图: {fp}")

    # 3. 热点论文
    if papers:
        fp = plot_top_papers(papers, report_date=report_date)
        if fp:
            charts["top_papers"] = fp
            print(f"  ✓ 热点论文图: {fp}")

    print(f"[Vis] 共生成 {len(charts)} 张图表")
    return charts
