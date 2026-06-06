"""
Web UI — 基于新拟态 (Neumorphism) 仪表盘风格（中文界面）
适配 Gradio 6.x
"""

import io
import base64
import re
from datetime import date, timedelta
from collections import Counter

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import markdown

from database import (
    get_papers_by_date,
    get_all_report_dates,
    get_daily_trend,
    get_connection,
)
from aggregator import load_report

# ── matplotlib 全局样式 ──────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "PingFang SC", "WenQuanYi Micro Hei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "axes.edgecolor": "#e0e0e0",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.color": "#e8e8e8",
})

# ── 配色 ──────────────────────────────────────────────────
BG   = "#eef1f5"
CARD = "#ffffff"
RED  = "#e74c3c"
BLUE = "#3b82f6"
DARK = "#1e293b"
GRAY = "#64748b"
GREEN = "#22c55e"


# ══════════════════════════════════════════════════════════
# 工具：matplotlib 图表 → base64 HTML <img>
# ══════════════════════════════════════════════════════════

def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=CARD, edgecolor="none", transparent=False)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


def _build_trend_chart(days: int = 7) -> str:
    """论文采集数量趋势 — 7 天折线图"""
    today = date.today()
    xs, ys = [], []
    for i in range(days - 1, -1, -1):
        d = (today - timedelta(days=i))
        papers = get_papers_by_date(d.isoformat())
        ys.append(len(papers))
        xs.append(d.strftime("%a")[:1])

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    fig.patch.set_facecolor(CARD)
    ax.set_facecolor(CARD)

    x_idx = range(len(xs))
    color = BLUE if ys[-1] >= (ys[-2] if len(ys) > 1 else 0) else RED

    ax.fill_between(x_idx, ys, alpha=0.08, color=color)
    ax.plot(x_idx, ys, color=color, linewidth=2.2, marker="o", markersize=7,
            markerfacecolor="white", markeredgewidth=2, markeredgecolor=color, zorder=5)

    if ys:
        ax.annotate(f"{ys[-1]}", (x_idx[-1], ys[-1]),
                    textcoords="offset points", xytext=(0, 14),
                    ha="center", fontsize=13, fontweight="bold", color=color)

    # 中文星期
    weekdays_cn = ["一","二","三","四","五","六","日"]
    start_wday = (today - timedelta(days=days-1)).weekday()
    labels = [weekdays_cn[(start_wday + i) % 7] for i in range(days)]
    ax.set_xticks(x_idx)
    ax.set_xticklabels(labels, fontsize=10, color=GRAY)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.tick_params(axis="y", colors=GRAY, labelsize=9)
    ax.set_ylabel("论文数", fontsize=10, color=GRAY)
    for spine in ax.spines.values():
        spine.set_visible(False)

    return _fig_to_b64(fig)


def _build_category_bars(papers: list[dict]) -> str:
    """子领域分布横向柱状图"""
    if not papers:
        return ""

    dist = Counter(p.get("subfield", "") or "未分类" for p in papers)
    items = dist.most_common(8)  # 显示前 8 个
    labels = [it[0] for it in items]
    vals = [it[1] for it in items]
    labels = [l[:24] + ".." if len(l) > 24 else l for l in labels]

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    fig.patch.set_facecolor(CARD)
    ax.set_facecolor(CARD)

    colors = [BLUE, "#6366f1", "#8b5cf6", RED, "#f59e0b", "#14b8a6", "#ec4899", "#84cc16"][:len(labels)]
    bars = ax.barh(list(reversed(labels)), list(reversed(vals)), color=list(reversed(colors)),
                   height=0.6, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, reversed(vals)):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=10, fontweight="bold", color=DARK)

    ax.set_xlim(0, max(vals) * 1.3 if vals else 10)
    ax.set_xlabel("论文数", fontsize=10, color=GRAY)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", colors=GRAY, labelsize=8)
    ax.tick_params(axis="y", colors=DARK, labelsize=10)

    return _fig_to_b64(fig)


# ══════════════════════════════════════════════════════════
# CSS — 新拟态风格
# ══════════════════════════════════════════════════════════

DASHBOARD_CSS = """
body, .gradio-container {
    background: #eef1f5 !important;
    font-family: 'Inter', 'PingFang SC', 'Microsoft YaHei', system-ui, sans-serif;
}

/* ── 导航栏 ── */
.navbar {
    display: flex; align-items: center; justify-content: space-between;
    background: #ffffff; border-radius: 16px;
    padding: 14px 28px; margin-bottom: 24px;
    box-shadow: 6px 6px 14px rgba(0,0,0,0.04), -4px -4px 10px #ffffff;
}
.nav-brand {
    display: flex; align-items: center; gap: 10px;
    font-size: 22px; font-weight: 800; color: #1e293b;
}
.nav-dot { width: 10px; height: 10px; background: #e74c3c; border-radius: 50%; }
.nav-links { display: flex; gap: 28px; }
.nav-links a {
    color: #64748b; text-decoration: none; font-size: 14px; font-weight: 500;
    transition: color 0.2s;
}
.nav-links a:hover, .nav-links a.active { color: #1e293b; }
.nav-right { display: flex; align-items: center; gap: 16px; }
.nav-avatar {
    width: 36px; height: 36px; border-radius: 50%;
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    display: flex; align-items: center; justify-content: center;
    color: white; font-weight: 700; font-size: 14px;
}

/* ── 卡片通用 ── */
.card {
    background: #ffffff; border-radius: 18px; padding: 24px;
    box-shadow: 8px 8px 18px rgba(0,0,0,0.05), -6px -6px 14px #ffffff;
    transition: box-shadow 0.25s;
}
.card:hover { box-shadow: 10px 10px 22px rgba(0,0,0,0.07), -6px -6px 16px #ffffff; }
.card-title { font-size: 18px; font-weight: 700; color: #1e293b; margin-bottom: 6px; }
.card-subtitle { font-size: 12px; color: #94a3b8; margin-bottom: 16px; }

/* ── 数据 badge ── */
.badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 600;
}
.badge-red  { background: #fef2f2; color: #dc2626; }
.badge-blue { background: #eff6ff; color: #3b82f6; }
.badge-green{ background: #f0fdf4; color: #16a34a; }
.badge-gray { background: #f8fafc; color: #64748b; }

/* ── 涨幅标识 ── */
.growth-up   { color: #22c55e; font-size: 22px; font-weight: 800; }
.growth-down { color: #e74c3c; font-size: 22px; font-weight: 800; }

/* ── 论文列表行 ── */
.paper-row {
    display: flex; align-items: flex-start; gap: 12px;
    padding: 12px 0; border-bottom: 1px solid #f1f5f9;
}
.paper-row:last-child { border-bottom: none; }
.paper-icon {
    width: 40px; height: 40px; border-radius: 10px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-size: 16px;
}
.paper-info { flex: 1; min-width: 0; }
.paper-info .title {
    font-size: 13px; font-weight: 600; color: #1e293b;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.paper-info .meta { font-size: 11px; color: #94a3b8; margin-top: 3px; }

/* ── 统计小卡 ── */
.stat-card {
    background: #f8fafc; border-radius: 14px; padding: 18px; text-align: center;
    box-shadow: inset 2px 2px 6px rgba(0,0,0,0.03), inset -2px -2px 6px #fff;
}
.stat-card .number { font-size: 28px; font-weight: 800; color: #1e293b; }
.stat-card .label  { font-size: 11px; color: #94a3b8; margin-top: 4px; }

/* ── 搜索框 ── */
.search-box {
    width: 100%; border: none; background: #f1f5f9;
    border-radius: 12px; padding: 12px 18px; font-size: 14px; color: #1e293b;
    outline: none; transition: background 0.2s;
}
.search-box:focus { background: #e8ecf1; }

/* ── 布局 ── */
.dashboard-grid {
    display: grid; gap: 22px;
    grid-template-columns: 1fr 1fr;
}
.stats-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }

@media (max-width: 900px) {
    .dashboard-grid { grid-template-columns: 1fr; }
}

footer { display: none !important; }
"""


# ══════════════════════════════════════════════════════════
# HTML 卡片构建器（中文）
# ══════════════════════════════════════════════════════════

def _importance_display(n) -> str:
    """论文重要性显示 — 支持小数精度"""
    try:
        val = float(n)
    except (TypeError, ValueError):
        val = 0.0
    # 颜色分级
    if val >= 4.0:
        color = "#dc2626"
    elif val >= 3.0:
        color = "#f59e0b"
    elif val >= 2.0:
        color = "#3b82f6"
    else:
        color = "#94a3b8"
    return f'<span style="color:{color};font-weight:700;font-size:14px">{val:.1f}</span>'


def _build_navbar() -> str:
    return f"""
    <div class="navbar">
        <div class="nav-brand">
            <div class="nav-dot"></div> PaperPulse
        </div>
        <div class="nav-right">
            <span style="font-size:11px;color:#94a3b8">多智能体协同框架 &middot; 每日学术热点追踪</span>
        </div>
    </div>"""


def _build_paper_trends_card(chart_b64: str, pct_change: float, abs_change: int) -> str:
    arrow = "↑" if pct_change >= 0 else "↓"
    color_cls = "growth-up" if pct_change >= 0 else "growth-down"
    more_less = "多" if abs_change >= 0 else "少"
    return f"""
    <div class="card" style="height:100%">
        <div class="card-title">论文趋势</div>
        <div class="card-subtitle">arXiv 每日论文采集数量（近 7 天）</div>
        <img src="{chart_b64}" style="width:100%; border-radius:8px" alt="趋势图">
        <div style="margin-top:12px; display:flex; align-items:baseline; gap:8px">
            <span class="{color_cls}">{arrow} {abs(pct_change):.0f}%</span>
            <span style="font-size:12px;color:#94a3b8">
                较昨日{more_less} {abs(abs_change)} 篇
            </span>
        </div>
    </div>"""


def _build_top_papers_card(papers: list[dict], top_k: int = 5) -> str:
    icons = ["🔴", "🟠", "🟡", "🔵", "⚪"]
    rows = ""
    # 过滤掉"其他"和模糊论文、低评分论文
    hot_candidates = [
        p for p in papers
        if float(p.get("importance", 0)) >= 1.5
        and "其他" not in (p.get("subfield") or "")
    ]
    sorted_p = sorted(hot_candidates, key=lambda x: float(x.get("importance", 0)), reverse=True)[:top_k]
    if not sorted_p:
        sorted_p = sorted(papers, key=lambda x: float(x.get("importance", 0)), reverse=True)[:top_k]
    if not sorted_p:
        sorted_p = papers[:top_k]
    for i, p in enumerate(sorted_p):
        imp = float(p.get("importance", 0))
        badge_cls = "badge-red" if imp >= 4.0 else ("badge-blue" if imp >= 2.0 else "badge-gray")
        title = (p.get("title") or "无标题")[:65]
        arxiv_id = p.get("arxiv_id", "")
        authors = ", ".join((p.get("authors") or [])[:2])
        if len(p.get("authors") or []) > 2:
            authors += " 等"
        sf = p.get("subfield", "")
        sf_badge = f'<span class="badge badge-green">{sf}</span>' if sf else ""
        imp_display = _importance_display(imp)
        rows += f"""
        <div class="paper-row">
            <div class="paper-icon" style="background:{'#fef2f2' if imp>=4.0 else '#eff6ff' if imp>=2.0 else '#f8fafc'}">
                {icons[min(i,4)]}
            </div>
            <div class="paper-info">
                <div class="title"><a href="https://arxiv.org/abs/{arxiv_id}" target="_blank"
                     style="color:#1e293b;text-decoration:none">{title}</a></div>
                <div class="meta">{imp_display} &nbsp;|&nbsp; {arxiv_id} &nbsp;|&nbsp; {authors} &nbsp; {sf_badge}</div>
            </div>
            <span class="badge {badge_cls}">{imp:.1f}/5.0</span>
        </div>"""
    return f"""
    <div class="card" style="height:100%">
        <div>
            <div class="card-title">今日热点论文</div>
        </div>
        <div class="card-subtitle">AI 评分最高的论文（精确到小数点后一位）</div>
        {rows}
    </div>"""


def _build_quick_stats(papers: list[dict]) -> str:
    total = len(papers)
    hot = len([p for p in papers if float(p.get("importance", 0)) >= 4.0])
    dist = Counter(p.get("subfield", "") or "未分类" for p in papers)
    top_cat = dist.most_common(1)[0][0] if dist else "暂无"

    return f"""
    <div class="card" style="height:100%">
        <div class="card-title">数据概览</div>
        <div class="card-subtitle">今日论文采集概况（共 {total} 篇）</div>
        <div class="stats-row">
            <div class="stat-card">
                <div class="number">{total}</div>
                <div class="label">采集论文</div>
            </div>
            <div class="stat-card">
                <div class="number">{hot}</div>
                <div class="label">突破性工作 (≥4.0)</div>
            </div>
            <div class="stat-card">
                <div class="number">{len(dist)}</div>
                <div class="label">覆盖子领域</div>
            </div>
            <div class="stat-card">
                <div class="number" style="font-size:16px">{top_cat[:14]}</div>
                <div class="label">最热方向</div>
            </div>
        </div>
    </div>"""


def _build_ai_digest(trend: dict) -> str:
    summary = ""
    if trend and trend.get("trend_summary"):
        summary = trend["trend_summary"]
        if len(summary) > 1000:
            summary = summary[:1000] + "..."
    if not summary:
        summary = "暂无 AI 摘要。请先运行每日分析流水线生成趋势报告。"

    # 渲染 Markdown
    summary_html = markdown.markdown(
        summary,
        extensions=['tables', 'fenced_code', 'nl2br'],
        output_format='html',
    )

    return f"""
    <div class="card" style="height:100%">
        <div class="card-title">AI 趋势摘要</div>
        <div class="card-subtitle">由 DeepSeek 大模型自动生成</div>
        <div class="ai-digest-body" style="font-size:12px;color:#475569;line-height:1.7;max-height:240px;overflow-y:auto">
            {summary_html}
        </div>
    </div>
    <style>
    .ai-digest-body h1, .ai-digest-body h2 {{ font-size:15px; font-weight:700; color:#1e293b; margin:8px 0 4px; }}
    .ai-digest-body h3 {{ font-size:13px; font-weight:600; color:#334155; margin:6px 0 3px; }}
    .ai-digest-body strong {{ color:#1e293b; }}
    .ai-digest-body p {{ margin:4px 0; }}
    .ai-digest-body ul, .ai-digest-body ol {{ margin:4px 0; padding-left:18px; }}
    .ai-digest-body li {{ margin:2px 0; }}
    .ai-digest-body code {{ background:#f1f5f9; padding:1px 5px; border-radius:3px; font-size:11px; }}
    </style>"""


def _build_category_card(chart_b64: str, papers: list[dict]) -> str:
    dist = Counter(p.get("subfield", "") or "未分类" for p in papers)
    total = len(papers)
    items = dist.most_common(6)  # 展示前 6 个
    bars_html = ""
    colors = [BLUE, "#6366f1", "#8b5cf6", RED, "#f59e0b", "#14b8a6"]
    max_v = items[0][1] if items else 1
    shown = 0
    for (cat, cnt), clr in zip(items, colors):
        h_pct = int(cnt / max_v * 100)
        shown += cnt
        bars_html += f"""
        <div style="margin-bottom:14px">
            <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                <span style="font-size:12px;font-weight:600;color:#1e293b">{cat}</span>
                <span style="font-size:12px;color:#94a3b8">{cnt} 篇</span>
            </div>
            <div style="background:#f1f5f9;border-radius:8px;height:8px;overflow:hidden">
                <div style="height:100%;width:{h_pct}%;background:{clr};border-radius:8px"></div>
            </div>
        </div>"""
    remaining = total - shown
    note = f"（以上 {shown} 篇，另有 {remaining} 篇在其他方向）" if remaining > 0 else f"（共 {total} 篇）"

    return f"""
    <div class="card" style="height:100%">
        <div class="card-title">领域分布</div>
        <div class="card-subtitle">各研究方向论文数量 {note}</div>
        {bars_html}
        <img src="{chart_b64}" style="width:100%;border-radius:8px;margin-top:8px" alt="领域分布图">
    </div>"""


# ══════════════════════════════════════════════════════════
# 仪表盘组装
# ══════════════════════════════════════════════════════════

def build_dashboard_html() -> str:
    """组装完整仪表盘 HTML"""
    today_str = date.today().isoformat()
    today_papers = get_papers_by_date(today_str)
    yesterday_papers = get_papers_by_date((date.today() - timedelta(days=1)).isoformat())

    chart_b64 = _build_trend_chart(7)

    yest_cnt = len(yesterday_papers)
    today_cnt = len(today_papers)
    if yest_cnt > 0:
        pct_change = (today_cnt - yest_cnt) / yest_cnt * 100
    else:
        pct_change = 0
    abs_change = today_cnt - yest_cnt

    papers = today_papers if today_papers else yesterday_papers
    cat_chart = _build_category_bars(papers) if papers else ""
    trend = get_daily_trend(today_str)

    navbar = _build_navbar()
    row1_left = _build_paper_trends_card(chart_b64, pct_change, abs_change)
    row1_right = _build_top_papers_card(papers)
    row2_left_top = _build_quick_stats(papers)
    row2_left_bottom = _build_ai_digest(trend)
    row2_right = _build_category_card(cat_chart, papers)

    return f"""
    <style>{DASHBOARD_CSS}</style>
    {navbar}
    <div class="dashboard-grid">
        {row1_left}
        {row1_right}
        <div style="display:flex;flex-direction:column;gap:22px">
            {row2_left_top}
            {row2_left_bottom}
        </div>
        {row2_right}
    </div>
    <div style="text-align:center;padding:20px;color:#94a3b8;font-size:11px">
        PaperPulse &middot; 基于 OpenClaw 多智能体协同框架 &middot; 数据来源: arXiv &middot; AI 模型: DeepSeek &middot; {today_str}
    </div>
    """


# ══════════════════════════════════════════════════════════
# 功能页面（中文）
# ══════════════════════════════════════════════════════════

def build_search_html(query: str = "", subfield_filter: str = "全部") -> str:
    """论文搜索页 — 由 Gradio 回调调用"""
    today_papers = get_papers_by_date(date.today().isoformat())

    # ── 语义别名映射：中文/英文/缩写相互扩展 ──
    ALIAS_MAP = {
        "计算机": ["计算机视觉", "计算机", "cv", "cs", "computer"],
        "人工智能": ["人工智能", "ai", "artificial intelligence", "cs.ai", "llm", "机器学习", "深度学习"],
        "ai": ["ai", "artificial intelligence", "cs.ai", "人工智能", "llm", "大语言模型", "大模型"],
        "cv": ["cv", "computer vision", "计算机视觉", "图像"],
        "nlp": ["nlp", "natural language", "自然语言", "文本", "语言模型"],
        "rl": ["rl", "reinforcement", "强化学习"],
        "机器人": ["机器人", "robotics", "cs.ro"],
        "金融": ["金融", "finance", "q-fin", "经济"],
        "生物": ["生物", "bio", "q-bio", "医学", "医疗"],
        "安全": ["安全", "security", "cs.cr", "密码"],
        "信号": ["信号", "signal", "通信", "eess"],
        "数学": ["数学", "math", "stat", "优化", "统计"],
    }

    def _expand_query(q: str) -> list[str]:
        """将用户输入扩展为同义词列表"""
        q_lower = q.lower().strip()
        expanded = set()
        expanded.add(q_lower)
        for key, aliases in ALIAS_MAP.items():
            if key in q_lower or any(a in q_lower for a in aliases):
                expanded.update([a.lower() for a in aliases])
        # 也把原始用户输入加入
        expanded.add(q_lower)
        return list(expanded)

    results_html = ""
    matched = 0
    q = (query or "").lower().strip()

    for p in today_papers:
        title = (p.get("title") or "").lower()
        abstract = (p.get("abstract") or "").lower()
        sf = (p.get("subfield") or "未分类").lower()
        categories = " ".join(p.get("categories", [])).lower()
        primary_cat = (p.get("primary_cat") or "").lower()
        key_contrib = (p.get("key_contribution") or "").lower()

        # 子领域筛选
        if subfield_filter != "全部":
            if (p.get("subfield") or "未分类") != subfield_filter:
                continue

        # 关键词搜索：语义扩展 + 多字段匹配
        if q:
            search_text = f"{title} {abstract} {sf} {categories} {primary_cat} {key_contrib}"
            # 用原始关键词做 AND 切分
            raw_keywords = q.split()
            # 每个原始关键词都做语义扩展，匹配任一别名即通过
            all_match = True
            for rk in raw_keywords:
                expanded = _expand_query(rk)
                if not any(alias in search_text for alias in expanded):
                    all_match = False
                    break
            if not all_match:
                continue

        matched += 1
        imp = float(p.get("importance", 0))
        imp_display = _importance_display(imp)
        authors = ", ".join((p.get("authors") or [])[:3])
        sf_clean = p.get("subfield", "") or "未分类"
        arxiv_cats = ", ".join(p.get("categories", [])[:2])
        results_html += f"""
        <div class="paper-row">
            <div class="paper-icon" style="background:{'#fef2f2' if imp>=4.0 else '#eff6ff' if imp>=2.0 else '#f8fafc'};font-size:14px">
                {'🔥' if imp>=4.0 else '📄' if imp>=2.0 else '📑'}
            </div>
            <div class="paper-info">
                <div class="title">
                    <a href="https://arxiv.org/abs/{p['arxiv_id']}" target="_blank"
                       style="color:#1e293b;text-decoration:none">{p.get('title', '')}</a>
                </div>
                <div class="meta">
                    {imp_display} &nbsp;|&nbsp; {p['arxiv_id']} &nbsp;|&nbsp; {authors}
                </div>
                <div class="meta" style="margin-top:2px">
                    <span class="badge badge-green">{sf_clean}</span>
                    <span class="badge badge-gray" style="margin-left:4px">{arxiv_cats}</span>
                </div>
                <div style="font-size:11px;color:#64748b;margin-top:4px">
                    {(p.get('abstract') or '')[:200]}...
                </div>
            </div>
            <span class="badge {'badge-red' if imp>=4.0 else 'badge-blue' if imp>=2.0 else 'badge-gray'}">{imp:.1f}/5.0</span>
        </div>"""

    return f"""
    <style>{DASHBOARD_CSS}</style>
    {_build_navbar()}
    <div class="card" style="margin-bottom:20px">
        <div class="card-title">论文搜索</div>
        <div class="card-subtitle">在今日采集的 {len(today_papers)} 篇论文中检索。支持标题、摘要、子领域、arXiv分类多字段匹配，空格分隔多关键词（AND逻辑）。</div>
    </div>
    <div class="card">
        <div style="font-size:14px;color:#64748b;margin-bottom:12px">
            {'搜索「' + (query or '') + '」共 ' + str(matched) + ' 条结果' if query else '今日全部 ' + str(len(today_papers)) + ' 篇论文·子领域: ' + (subfield_filter or '全部')}
        </div>
        {results_html if results_html else '<div style="color:#94a3b8;text-align:center;padding:40px">未找到匹配结果。<br>试试换个关键词，如 "LLM"、"扩散模型"、"机器人"、"金融" 等。</div>'}
    </div>"""


def build_weekly_html() -> str:
    """周报页面"""
    from datetime import date, timedelta
    from agents.weekly_digest import generate_weekly_digest, get_weekly_report

    today = date.today()
    # 先尝试加载已有的本周周报
    report = get_weekly_report(today.isoformat())

    if not report:
        # 检查是否有足够的数据
        available = get_all_report_dates()
        if len(available) >= 3:
            try:
                report = generate_weekly_digest(today.isoformat())
            except Exception as e:
                report = f"## 周报生成失败\n\n错误: {e}\n\n请确认已配置 API Key 并有足够的日报数据。"
        else:
            report = "暂无足够数据生成周报（需要至少 3 天的日报数据）。请先运行每日分析流水线。"

    report_html = markdown.markdown(
        report,
        extensions=['tables', 'fenced_code', 'nl2br'],
        output_format='html',
    )

    available_weeks = []
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT end_date, week_label FROM weekly_reports ORDER BY end_date DESC LIMIT 10")
        available_weeks = [(r["end_date"], r["week_label"]) for r in cur.fetchall()]
        conn.close()
    except Exception:
        pass

    week_links = "".join(
        f'<option value="{d}">{label}</option>'
        for d, label in available_weeks
    ) if available_weeks else '<option>暂无历史周报</option>'

    return f"""
    <style>{DASHBOARD_CSS}</style>
    {_build_navbar()}
    <div class="card" style="margin-bottom:20px">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <div>
                <div class="card-title">学术周报</div>
                <div class="card-subtitle">基于 7 天日报数据自动汇总，由 DeepSeek 大模型生成趋势综述</div>
            </div>
        </div>
    </div>
    <div class="card" style="max-height:70vh;overflow-y:auto;padding:24px">
        <div class="markdown-body">{report_html}</div>
    </div>
    <div style="text-align:center;padding:12px;color:#94a3b8;font-size:11px">
        周报由 Weekly Digest Agent 自动生成 &middot; {date.today().isoformat()}
    </div>
    <style>
    .markdown-body {{ font-size:14px; color:#334155; line-height:1.8; }}
    .markdown-body h1 {{ font-size:22px; font-weight:700; color:#1e293b; margin-top:16px; margin-bottom:8px; }}
    .markdown-body h2 {{ font-size:18px; font-weight:700; color:#1e293b; margin-top:14px; margin-bottom:6px; border-bottom:1px solid #e2e8f0; padding-bottom:4px; }}
    .markdown-body h3 {{ font-size:15px; font-weight:600; color:#334155; margin-top:12px; }}
    .markdown-body table {{ border-collapse:collapse; width:100%; margin:10px 0; }}
    .markdown-body th, .markdown-body td {{ border:1px solid #e2e8f0; padding:8px 12px; text-align:left; font-size:13px; }}
    .markdown-body th {{ background:#f8fafc; font-weight:600; }}
    .markdown-body code {{ background:#f1f5f9; padding:2px 6px; border-radius:4px; font-size:12px; }}
    .markdown-body pre {{ background:#f8fafc; padding:16px; border-radius:8px; overflow-x:auto; border:1px solid #e2e8f0; }}
    .markdown-body a {{ color:#3b82f6; }}
    .markdown-body blockquote {{ border-left:3px solid #3b82f6; padding-left:16px; margin:8px 0; color:#64748b; }}
    .markdown-body hr {{ border:none; border-top:1px solid #e2e8f0; margin:20px 0; }}
    </style>"""


def build_history_html(selected_date: str = "") -> str:
    """历史日报页"""
    available = get_all_report_dates()
    date_options = "".join(
        f'<option value="{d}" {"selected" if d==selected_date else ""}>{d}</option>'
        for d in available
    )

    report = None
    if selected_date:
        report = load_report(selected_date)
    elif available:
        selected_date = available[0]
        report = load_report(selected_date)

    # Markdown → HTML 渲染
    if report:
        report_html = markdown.markdown(
            report,
            extensions=['tables', 'fenced_code', 'nl2br'],
            output_format='html',
        )
        report_html = f'<div class="markdown-body">{report_html}</div>'
    else:
        report_html = '<div style="color:#94a3b8;text-align:center;padding:60px">暂无日报数据。请先运行一次分析流水线。</div>'

    return f"""
    <style>{DASHBOARD_CSS}</style>
    <style>
    .markdown-body {{ font-size:14px; color:#334155; line-height:1.8; }}
    .markdown-body h1 {{ font-size:22px; font-weight:700; color:#1e293b; margin-top:16px; margin-bottom:8px; }}
    .markdown-body h2 {{ font-size:18px; font-weight:700; color:#1e293b; margin-top:14px; margin-bottom:6px; border-bottom:1px solid #e2e8f0; padding-bottom:4px; }}
    .markdown-body h3 {{ font-size:15px; font-weight:600; color:#334155; margin-top:12px; }}
    .markdown-body table {{ border-collapse:collapse; width:100%; margin:10px 0; }}
    .markdown-body th, .markdown-body td {{ border:1px solid #e2e8f0; padding:8px 12px; text-align:left; font-size:13px; }}
    .markdown-body th {{ background:#f8fafc; font-weight:600; }}
    .markdown-body code {{ background:#f1f5f9; padding:2px 6px; border-radius:4px; font-size:12px; }}
    .markdown-body pre {{ background:#f8fafc; color:#1e293b; padding:16px; border-radius:8px; overflow-x:auto; border:1px solid #e2e8f0; }}
    .markdown-body pre code {{ background:none; padding:0; color:#1e293b; }}
    .markdown-body a {{ color:#3b82f6; }}
    .markdown-body blockquote {{ border-left:3px solid #3b82f6; padding-left:16px; margin:8px 0; color:#64748b; }}
    .markdown-body hr {{ border:none; border-top:1px solid #e2e8f0; margin:20px 0; }}
    </style>
    {_build_navbar()}
    <div class="card" style="margin-bottom:20px">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <div>
                <div class="card-title">历史日报</div>
                <div class="card-subtitle">浏览往期每日学术热点分析</div>
            </div>
            <select id="history-date-select" style="border:none;background:#f1f5f9;
                    border-radius:12px;padding:10px 16px;font-size:14px;color:#1e293b;outline:none">
                {date_options}
            </select>
        </div>
    </div>
    <div class="card" style="max-height:70vh;overflow-y:auto;padding:24px">
        {report_html}
    </div>"""


# ══════════════════════════════════════════════════════════
# Gradio App 入口
# ══════════════════════════════════════════════════════════

def build_ui():
    """构建 Gradio Blocks 界面"""

    with gr.Blocks(title="PaperPulse - 学术论文日报系统") as app:
        gr.HTML("<style>footer{display:none!important}</style>")

        with gr.Tabs():
            # ── Tab 1: 仪表盘 ──
            with gr.TabItem("仪表盘", id="dashboard"):
                dashboard_html = gr.HTML(value=build_dashboard_html())
                refresh_btn = gr.Button("刷新仪表盘", variant="primary",
                                        elem_id="refresh-dash-btn")
                refresh_btn.click(
                    fn=lambda: build_dashboard_html(),
                    outputs=dashboard_html,
                )

            # ── Tab 2: 搜索 ──
            with gr.TabItem("搜索论文", id="search"):
                with gr.Column(elem_classes="search-container"):
                    with gr.Row():
                        search_input = gr.Textbox(
                            label="关键词",
                            placeholder="Transformer, RAG, 多模态, 强化学习...",
                            scale=3,
                            show_label=False,
                            elem_classes="search-input",
                        )
                        subfield_dropdown = gr.Dropdown(
                            choices=["全部"],
                            value="全部",
                            scale=1,
                            show_label=False,
                            elem_classes="subfield-drop",
                            interactive=True,
                        )
                        search_btn = gr.Button("搜索", variant="primary", scale=1)
                    search_result = gr.HTML(
                        value=build_search_html(),
                        elem_classes="search-result",
                    )

                def _on_search(query, subfield):
                    return build_search_html(query or "", subfield or "全部")

                def _update_dropdown():
                    papers = get_papers_by_date(date.today().isoformat())
                    subs = sorted(set(p.get("subfield", "") or "未分类" for p in papers))
                    return gr.Dropdown(choices=["全部"] + subs, value="全部")

                search_btn.click(_on_search, [search_input, subfield_dropdown], search_result)
                search_input.submit(_on_search, [search_input, subfield_dropdown], search_result)
                # 进入搜索页时刷新下拉列表
                search_result.change(
                    fn=_update_dropdown,
                    outputs=subfield_dropdown,
                )

            # ── Tab 3: 历史日报 ──
            with gr.TabItem("历史日报", id="history"):
                available_dates = get_all_report_dates()
                default_date = available_dates[0] if available_dates else ""

                with gr.Column():
                    history_date = gr.Dropdown(
                        label="选择日期",
                        choices=available_dates,
                        value=default_date,
                        interactive=True,
                    )
                    history_html = gr.HTML(
                        value=build_history_html(default_date),
                    )

                def _on_history_change(d):
                    return build_history_html(d)

                history_date.change(_on_history_change, history_date, history_html)

            # ── Tab 4: 周报 ──
            with gr.TabItem("周报", id="weekly"):
                weekly_html = gr.HTML(value=build_weekly_html())
                refresh_weekly_btn = gr.Button("生成/刷新周报", variant="primary")
                refresh_weekly_btn.click(
                    fn=lambda: build_weekly_html(),
                    outputs=weekly_html,
                )

            # ── Tab 5: 关于系统 ──
            with gr.TabItem("关于系统", id="about"):
                gr.HTML(f"""
                <style>{DASHBOARD_CSS}</style>
                {_build_navbar()}
                <div class="card" style="max-width:700px;margin:40px auto">
                    <div class="card-title">关于 PaperPulse</div>
                    <div class="card-subtitle">每日学术论文热点追踪系统</div>
                    <br>
                    <h3>多智能体架构（7 Agents）</h3>
                    <table style="width:100%;border-collapse:collapse;margin:12px 0">
                        <tr style="border-bottom:1px solid #f1f5f9">
                            <td style="padding:10px;font-weight:600;width:200px">📡 采集智能体</td>
                            <td style="padding:10px;color:#64748b">OpenAlex API 开源学术数据</td>
                        </tr>
                        <tr style="border-bottom:1px solid #f1f5f9">
                            <td style="padding:10px;font-weight:600">🏷️ 分类与摘要智能体</td>
                            <td style="padding:10px;color:#64748b">DeepSeek API 语义理解与分类</td>
                        </tr>
                        <tr style="border-bottom:1px solid #f1f5f9">
                            <td style="padding:10px;font-weight:600">📊 趋势分析智能体</td>
                            <td style="padding:10px;color:#64748b">jieba 分词 + DeepSeek API 趋势预测</td>
                        </tr>
                        <tr style="border-bottom:1px solid #f1f5f9">
                            <td style="padding:10px;font-weight:600">✅ 质量评估智能体</td>
                            <td style="padding:10px;color:#64748b">四维评估（方法论/创新性/可复现性/写作质量）</td>
                        </tr>
                        <tr style="border-bottom:1px solid #f1f5f9">
                            <td style="padding:10px;font-weight:600">🔗 交叉引用智能体</td>
                            <td style="padding:10px;color:#64748b">TF-IDF 语义相似度 + 关联图谱构建</td>
                        </tr>
                        <tr style="border-bottom:1px solid #f1f5f9">
                            <td style="padding:10px;font-weight:600">🌐 双语翻译智能体</td>
                            <td style="padding:10px;color:#64748b">DeepSeek API 学术中英互译</td>
                        </tr>
                        <tr style="border-bottom:1px solid #f1f5f9">
                            <td style="padding:10px;font-weight:600">📝 周报汇总智能体</td>
                            <td style="padding:10px;color:#64748b">7天日报聚合 + LLM 趋势综述</td>
                        </tr>
                        <tr>
                            <td style="padding:10px;font-weight:600">📋 日报汇总器</td>
                            <td style="padding:10px;color:#64748b">Markdown 日报自动生成</td>
                        </tr>
                    </table>
                    <br>
                    <h3>技术栈</h3>
                    <p style="color:#64748b">
                        Python &middot; arXiv API &middot; DeepSeek &middot;
                        Gradio &middot; SQLite &middot; Docker &middot; matplotlib &middot; jieba
                    </p>
                    <br>
                    <h3>数据覆盖范围</h3>
                    <p style="color:#64748b">
                        OpenAlex 开放学术数据库，覆盖 AI、CV、NLP、ML、机器人、强化学习、深度学习、大语言模型 共 8 大热门主题
                    </p>
                    <br>
                    <p style="color:#94a3b8;font-size:12px">
                        基于 OpenClaw 多智能体协同框架 &middot; {date.today().isoformat()}
                    </p>
                </div>
                """)

        # 页面加载时填充仪表盘
        app.load(fn=lambda: build_dashboard_html(), outputs=dashboard_html)

    return app


def launch_ui(host: str = "0.0.0.0", port: int = 7860, share: bool = False):
    """启动 Web UI"""
    app = build_ui()
    print(f"[Web UI] PaperPulse 仪表盘已启动: http://{host}:{port}")
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=False,
    )


if __name__ == "__main__":
    launch_ui()
