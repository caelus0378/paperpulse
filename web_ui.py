"""
Web UI — 学术科技风格仪表盘
设计系统: 深蓝主色 · 无衬线字体 · 卡片系统 · 信息密度控制
适配 Gradio 6.x
"""

import io
import base64
import re
import json
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
from agents.deep_analyzer import get_deep_analysis

# ══════════════════════════════════════════════════════════
# 设计系统 — 配色 / 字体 / 间距
# ══════════════════════════════════════════════════════════

# ── 主色板 (Academic Deep Blue) ──
PRIMARY   = "#1B3A5C"   # 深蓝 — 导航栏、标题
ACCENT    = "#2563EB"   # 亮蓝 — 图表、链接、交互
TEAL      = "#0D9488"   # 青绿 — 正向指标
GOLD      = "#F59E0B"   # 金色 — 中等评分
RED       = "#DC2626"   # 红色 — 高分/警报
DARK      = "#1E293B"   # 深灰 — 正文
GRAY      = "#64748B"   # 中灰 — 辅助文字
LIGHT_GRAY = "#94A3B8"  # 浅灰 — 占位符
BG        = "#F5F7FA"   # 页面底色
CARD_BG   = "#FFFFFF"   # 卡片底色
BORDER    = "#E2E8F0"   # 边框/分割线

# ── matplotlib 全局样式 ──────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "PingFang SC", "WenQuanYi Micro Hei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "axes.edgecolor": BORDER,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.color": "#E8ECF1",
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


# ══════════════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════════════

def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight",
                facecolor=CARD_BG, edgecolor="none", transparent=False)
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

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    fig.patch.set_facecolor(CARD_BG)
    ax.set_facecolor(CARD_BG)

    x_idx = range(len(xs))
    color = ACCENT if ys[-1] >= (ys[-2] if len(ys) > 1 else 0) else RED

    ax.fill_between(x_idx, ys, alpha=0.06, color=color)
    ax.plot(x_idx, ys, color=color, linewidth=2.4, marker="o", markersize=8,
            markerfacecolor="white", markeredgewidth=2.2, markeredgecolor=color, zorder=5)

    if ys:
        ax.annotate(f"{ys[-1]}", (x_idx[-1], ys[-1]),
                    textcoords="offset points", xytext=(0, 16),
                    ha="center", fontsize=14, fontweight="bold", color=color)

    weekdays_cn = ["一","二","三","四","五","六","日"]
    start_wday = (today - timedelta(days=days-1)).weekday()
    labels = [weekdays_cn[(start_wday + i) % 7] for i in range(days)]
    ax.set_xticks(x_idx)
    ax.set_xticklabels(labels, fontsize=10, color=GRAY)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.tick_params(axis="y", colors=GRAY, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    return _fig_to_b64(fig)


def _build_category_bars(papers: list[dict]) -> str:
    """子领域分布横向柱状图"""
    if not papers:
        return ""

    dist = Counter(p.get("subfield", "") or "未分类" for p in papers)
    items = dist.most_common(8)
    labels = [it[0] for it in items]
    vals = [it[1] for it in items]
    labels = [l[:22] + ".." if len(l) > 22 else l for l in labels]

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    fig.patch.set_facecolor(CARD_BG)
    ax.set_facecolor(CARD_BG)

    bar_colors = [ACCENT, "#6366F1", "#8B5CF6", "#EC4899", GOLD, TEAL, "#14B8A6", "#84CC16"]
    bar_colors = bar_colors[:len(labels)]
    bars = ax.barh(list(reversed(labels)), list(reversed(vals)),
                   color=list(reversed(bar_colors)),
                   height=0.58, edgecolor="white", linewidth=0.6)

    for bar, val in zip(bars, reversed(vals)):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=11, fontweight="bold", color=DARK)

    ax.set_xlim(0, max(vals) * 1.35 if vals else 10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", colors=GRAY, labelsize=8)
    ax.tick_params(axis="y", colors=DARK, labelsize=10)

    return _fig_to_b64(fig)


def _importance_display(n) -> str:
    try:
        val = float(n)
    except (TypeError, ValueError):
        val = 0.0
    if val >= 4.0:
        color = RED
    elif val >= 3.0:
        color = GOLD
    elif val >= 2.0:
        color = ACCENT
    else:
        color = LIGHT_GRAY
    return f'<span style="color:{color};font-weight:700;font-size:13px">{val:.1f}</span>'


# ══════════════════════════════════════════════════════════
# CSS — 学术科技风格
# ══════════════════════════════════════════════════════════

DASHBOARD_CSS = f"""
/* ── 全局 ── */
body, .gradio-container {{
    background: {BG} !important;
    font-family: 'Inter', 'PingFang SC', 'Microsoft YaHei', 'Segoe UI', system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
}}

/* ── 导航栏 (深蓝学术风) ── */
.navbar {{
    display: flex; align-items: center; justify-content: space-between;
    background: linear-gradient(135deg, {PRIMARY} 0%, #0F2B47 100%);
    border-radius: 0; padding: 16px 32px; margin: 0 0 28px 0;
    box-shadow: 0 2px 16px rgba(27,58,92,0.15);
    color: #FFFFFF !important;
}}
.nav-brand {{
    display: flex; align-items: center; gap: 12px;
    font-size: 22px; font-weight: 800; color: #FFFFFF !important; letter-spacing: -0.3px;
}}
.nav-dot {{
    width: 10px; height: 10px; background: #FFFFFF; border-radius: 50%;
    box-shadow: 0 0 10px rgba(255,255,255,0.4);
}}
.nav-right {{ display: flex; align-items: center; gap: 16px; }}
.nav-right span {{ font-size: 12px; color: #CBD5E1 !important; }}

/* ── 卡片系统 ── */
.card {{
    background: {CARD_BG}; border-radius: 14px; padding: 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04), 0 4px 14px rgba(0,0,0,0.03);
    border: 1px solid {BORDER};
    transition: box-shadow 0.2s;
    position: relative;
}}
.card:hover {{ box-shadow: 0 2px 8px rgba(0,0,0,0.06), 0 6px 20px rgba(0,0,0,0.04); }}
.card-title {{
    font-size: 17px; font-weight: 700; color: {PRIMARY}; margin-bottom: 4px;
    letter-spacing: -0.2px;
}}
.card-subtitle {{ font-size: 12px; color: {LIGHT_GRAY}; margin-bottom: 18px; }}

/* 带左侧强调线的卡片 */
.card-accent {{
    border-left: 4px solid {ACCENT};
    padding-left: 20px;
}}

/* ── Badge ── */
.badge {{
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 600; letter-spacing: 0.2px;
}}
.badge-red    {{ background: #FEF2F2; color: {RED}; }}
.badge-blue   {{ background: #EFF6FF; color: {ACCENT}; }}
.badge-teal   {{ background: #F0FDF9; color: {TEAL}; }}
.badge-gold   {{ background: #FFFBEB; color: #D97706; }}
.badge-gray   {{ background: #F8FAFC; color: {GRAY}; }}

/* ── 涨跌 ── */
.growth-up   {{ color: {TEAL}; font-size: 22px; font-weight: 800; }}
.growth-down {{ color: {RED}; font-size: 22px; font-weight: 800; }}

/* ── 论文行 ── */
.paper-row {{
    display: flex; align-items: flex-start; gap: 12px;
    padding: 14px 0; border-bottom: 1px solid #F1F5F9;
}}
.paper-row:last-child {{ border-bottom: none; }}
.paper-icon {{
    width: 42px; height: 42px; border-radius: 10px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-size: 16px;
}}
.paper-info {{ flex: 1; min-width: 0; }}
.paper-info .title {{
    font-size: 13px; font-weight: 600; color: {DARK}; line-height: 1.4;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
    overflow: hidden;
}}
.paper-info .title a {{ color: {DARK}; text-decoration: none; }}
.paper-info .title a:hover {{ color: {ACCENT}; }}
.paper-info .meta {{ font-size: 11px; color: {LIGHT_GRAY}; margin-top: 4px; }}

/* ── 统计小卡 ── */
.stats-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
.stat-card {{
    background: {BG}; border-radius: 12px; padding: 18px 14px; text-align: center;
    border: 1px solid {BORDER};
    transition: transform 0.15s;
}}
.stat-card:hover {{ transform: translateY(-2px); }}
.stat-card .number {{ font-size: 28px; font-weight: 800; color: {PRIMARY}; letter-spacing: -0.5px; }}
.stat-card .label  {{ font-size: 11px; color: {LIGHT_GRAY}; margin-top: 6px; }}

/* 统计卡强调色 */
.stat-card-accent {{ border-top: 3px solid {ACCENT}; }}
.stat-card-teal   {{ border-top: 3px solid {TEAL}; }}
.stat-card-gold   {{ border-top: 3px solid {GOLD}; }}
.stat-card-red    {{ border-top: 3px solid {RED}; }}

/* ── 布局 ── */
.dashboard-grid {{
    display: grid; gap: 24px;
    grid-template-columns: 1fr 1fr;
}}
@media (max-width: 960px) {{
    .dashboard-grid {{ grid-template-columns: 1fr; }}
}}

/* ── 进度条 (领域分布) ── */
.progress-bar-bg {{
    background: #F1F5F9; border-radius: 6px; height: 8px; overflow: hidden;
}}
.progress-bar-fill {{
    height: 100%; border-radius: 6px;
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}}

/* ── 搜索框 ── */
.search-box {{
    width: 100%; border: 1.5px solid {BORDER}; background: {CARD_BG};
    border-radius: 10px; padding: 12px 18px; font-size: 14px; color: {DARK};
    outline: none; transition: border-color 0.2s; font-family: inherit;
}}
.search-box:focus {{ border-color: {ACCENT}; box-shadow: 0 0 0 3px rgba(37,99,235,0.1); }}

/* ── Markdown 正文 ── */
.markdown-body {{ font-size: 14px; color: #334155; line-height: 1.85; }}
.markdown-body h1 {{ font-size: 22px; font-weight: 700; color: {PRIMARY}; margin-top: 16px; margin-bottom: 8px; }}
.markdown-body h2 {{ font-size: 18px; font-weight: 700; color: {PRIMARY}; margin-top: 14px; margin-bottom: 6px; border-bottom: 1px solid {BORDER}; padding-bottom: 4px; }}
.markdown-body h3 {{ font-size: 15px; font-weight: 600; color: {DARK}; margin-top: 12px; }}
.markdown-body table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px; }}
.markdown-body th, .markdown-body td {{ border: 1px solid {BORDER}; padding: 8px 14px; text-align: left; }}
.markdown-body th {{ background: {BG}; font-weight: 600; color: {PRIMARY}; }}
.markdown-body code {{ background: #F1F5F9; padding: 2px 6px; border-radius: 4px; font-size: 12px; color: {PRIMARY}; }}
.markdown-body pre {{ background: #F8FAFC; padding: 16px; border-radius: 8px; overflow-x: auto; border: 1px solid {BORDER}; }}
.markdown-body pre code {{ background: none; padding: 0; color: {DARK}; }}
.markdown-body a {{ color: {ACCENT}; text-decoration: none; }}
.markdown-body a:hover {{ text-decoration: underline; }}
.markdown-body blockquote {{ border-left: 3px solid {ACCENT}; padding: 4px 16px; margin: 8px 0; color: {GRAY}; background: {BG}; border-radius: 0 6px 6px 0; }}
.markdown-body hr {{ border: none; border-top: 1px solid {BORDER}; margin: 20px 0; }}

/* ── AI Digest ── */
.ai-digest-body {{
    font-size: 13px; color: #475569; line-height: 1.75;
    max-height: 240px; overflow-y: auto;
}}
.ai-digest-body h1, .ai-digest-body h2 {{ font-size: 15px; font-weight: 700; color: {PRIMARY}; margin: 8px 0 4px; }}
.ai-digest-body h3 {{ font-size: 13px; font-weight: 600; color: #334155; margin: 6px 0 3px; }}
.ai-digest-body strong {{ color: {DARK}; }}
.ai-digest-body p {{ margin: 4px 0; }}
.ai-digest-body ul, .ai-digest-body ol {{ margin: 4px 0; padding-left: 18px; }}
.ai-digest-body li {{ margin: 2px 0; }}
.ai-digest-body code {{ background: #F1F5F9; padding: 1px 5px; border-radius: 3px; font-size: 11px; }}

/* ── Footer ── */
footer {{ display: none !important; }}
.page-footer {{
    text-align: center; padding: 24px;
    color: {LIGHT_GRAY}; font-size: 11px;
    border-top: 1px solid {BORDER}; margin-top: 28px;
}}
.page-footer a {{ color: {ACCENT}; text-decoration: none; }}
"""


# ══════════════════════════════════════════════════════════
# 导航栏
# ══════════════════════════════════════════════════════════

def _build_navbar() -> str:
    return f"""
    <div class="navbar" style="color:#FFFFFF !important">
        <div class="nav-brand" style="color:#FFFFFF !important">
            <div class="nav-dot"></div> <span style="color:#FFFFFF">PaperPulse</span>
        </div>
        <div class="nav-right">
            <span style="color:#CBD5E1">多智能体协同框架 &middot; 每日学术热点追踪</span>
        </div>
    </div>"""


def _build_footer() -> str:
    today_str = date.today().isoformat()
    return f"""
    <div class="page-footer">
        PaperPulse &middot; 基于 OpenClaw 多智能体协同框架 &middot;
        <a href="https://openalex.org/" target="_blank">OpenAlex</a> 数据 &middot;
        AI 模型: DeepSeek &middot; {today_str}
    </div>"""


# ══════════════════════════════════════════════════════════
# 仪表盘 — 卡片构建器
# ══════════════════════════════════════════════════════════

def _build_paper_trends_card(chart_b64: str, pct_change: float, abs_change: int) -> str:
    arrow = "↑" if pct_change >= 0 else "↓"
    color_cls = "growth-up" if pct_change >= 0 else "growth-down"
    more_less = "多" if abs_change >= 0 else "少"
    return f"""
    <div class="card card-accent" style="height:100%">
        <div class="card-title">📈 论文采集趋势</div>
        <div class="card-subtitle">OpenAlex 每日论文采集数量（近 7 天）</div>
        <img src="{chart_b64}" style="width:100%; border-radius:8px" alt="趋势图">
        <div style="margin-top:14px; display:flex; align-items:baseline; gap:8px">
            <span class="{color_cls}">{arrow} {abs(pct_change):.0f}%</span>
            <span style="font-size:12px;color:{LIGHT_GRAY}">
                较昨日{more_less} {abs(abs_change)} 篇
            </span>
        </div>
    </div>"""


def _build_top_papers_card(papers: list[dict], top_k: int = 5) -> str:
    rank_icons = ["🥇", "🥈", "🥉", "④", "⑤"]
    rows = ""
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
        badge_cls = "badge-red" if imp >= 4.0 else ("badge-gold" if imp >= 3.0 else "badge-blue")
        title = (p.get("title") or "无标题")[:70]
        arxiv_id = p.get("arxiv_id", "")
        authors = ", ".join((p.get("authors") or [])[:2])
        if len(p.get("authors") or []) > 2:
            authors += " 等"
        sf = p.get("subfield", "")
        sf_badge = f'<span class="badge badge-teal">{sf[:18]}</span>' if sf else ""
        imp_display = _importance_display(imp)

        rows += f"""
        <div class="paper-row">
            <div class="paper-icon" style="font-size:20px">{rank_icons[min(i,4)]}</div>
            <div class="paper-info">
                <div class="title"><a href="https://doi.org/{arxiv_id}" target="_blank">{title}</a></div>
                <div class="meta">
                    {imp_display} &nbsp;|&nbsp; {arxiv_id[:24]} &nbsp;|&nbsp; {authors} &nbsp; {sf_badge}
                </div>
            </div>
            <span class="badge {badge_cls}">{imp:.1f}</span>
        </div>"""

    return f"""
    <div class="card" style="height:100%">
        <div class="card-title">🔥 今日热点论文</div>
        <div class="card-subtitle">AI 评分最高的论文 · 精确到小数点后一位</div>
        {rows}
    </div>"""


def _build_quick_stats(papers: list[dict]) -> str:
    total = len(papers)
    hot = len([p for p in papers if float(p.get("importance", 0)) >= 4.0])
    dist = Counter(p.get("subfield", "") or "未分类" for p in papers)
    top_cat = dist.most_common(1)[0][0] if dist else "暂无"

    return f"""
    <div class="card" style="height:100%">
        <div class="card-title">📊 数据概览</div>
        <div class="card-subtitle">今日论文采集概况 · 共 {total} 篇</div>
        <div class="stats-row">
            <div class="stat-card stat-card-accent">
                <div class="number">{total}</div>
                <div class="label">采集论文总数</div>
            </div>
            <div class="stat-card stat-card-red">
                <div class="number">{hot}</div>
                <div class="label">突破性工作 ≥4.0</div>
            </div>
            <div class="stat-card stat-card-teal">
                <div class="number">{len(dist)}</div>
                <div class="label">覆盖子领域数</div>
            </div>
            <div class="stat-card stat-card-gold">
                <div class="number" style="font-size:15px">{top_cat[:12]}</div>
                <div class="label">最热研究方向</div>
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

    summary_html = markdown.markdown(
        summary,
        extensions=['tables', 'fenced_code', 'nl2br'],
        output_format='html',
    )

    return f"""
    <div class="card" style="height:100%">
        <div class="card-title">🤖 AI 趋势摘要</div>
        <div class="card-subtitle">由 DeepSeek 大模型自动生成</div>
        <div class="ai-digest-body">{summary_html}</div>
    </div>"""


def _build_category_card(chart_b64: str, papers: list[dict]) -> str:
    dist = Counter(p.get("subfield", "") or "未分类" for p in papers)
    total = len(papers)
    items = dist.most_common(6)
    bars_html = ""
    bar_colors = [ACCENT, "#6366F1", "#8B5CF6", "#EC4899", GOLD, TEAL]
    max_v = items[0][1] if items else 1
    shown = 0
    for (cat, cnt), clr in zip(items, bar_colors):
        h_pct = int(cnt / max_v * 100)
        shown += cnt
        short_cat = cat[:22] + ".." if len(cat) > 22 else cat
        bars_html += f"""
        <div style="margin-bottom:14px">
            <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                <span style="font-size:12px;font-weight:600;color:{DARK}">{short_cat}</span>
                <span style="font-size:12px;color:{LIGHT_GRAY}">{cnt} 篇</span>
            </div>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill" style="width:{h_pct}%;background:{clr}"></div>
            </div>
        </div>"""
    remaining = total - shown
    note = f"以上 {shown} 篇，另有 {remaining} 篇在其他方向" if remaining > 0 else f"共 {total} 篇"

    return f"""
    <div class="card" style="height:100%">
        <div class="card-title">📂 领域分布</div>
        <div class="card-subtitle">各研究方向论文数量 · {note}</div>
        {bars_html}
        <img src="{chart_b64}" style="width:100%;border-radius:8px;margin-top:12px" alt="领域分布图">
    </div>"""


# ══════════════════════════════════════════════════════════
# Agent 协作状态面板
# ══════════════════════════════════════════════════════════

def _build_agent_status_card(today_str: str = None) -> str:
    """构建 Agent 协作状态面板 — 显示每个智能体的工作成果"""
    if today_str is None:
        today_str = date.today().isoformat()

    # 查询各 Agent 的产出
    agents = []

    # Agent 1: Collector
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) as cnt FROM papers WHERE collected_date=?", (today_str,))
        paper_cnt = cur.fetchone()["cnt"]
        conn.close()
        agents.append(("📡", "Collector", f"采集 {paper_cnt} 篇", "done" if paper_cnt > 0 else "pending"))
    except Exception:
        agents.append(("📡", "Collector", "等待运行", "pending"))

    # Agent 2: Classifier
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(DISTINCT subfield) as cnt FROM papers WHERE collected_date=? AND subfield != ''", (today_str,))
        sf_cnt = cur.fetchone()["cnt"]
        conn.close()
        agents.append(("🏷️", "Classifier", f"分类 {sf_cnt} 个子领域" if sf_cnt > 0 else "等待运行", "done" if sf_cnt > 0 else "pending"))
    except Exception:
        agents.append(("🏷️", "Classifier", "等待运行", "pending"))

    # Agent 3: TrendAnalyzer
    try:
        trend = get_daily_trend(today_str)
        if trend and trend.get("keyword_trends"):
            kw_cnt = len(trend["keyword_trends"])
            agents.append(("📊", "TrendAnalyzer", f"识别 {kw_cnt} 个关键词", "done"))
        else:
            agents.append(("📊", "TrendAnalyzer", "等待运行", "pending"))
    except Exception:
        agents.append(("📊", "TrendAnalyzer", "等待运行", "pending"))

    # Agent 4: Quality Assessor
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) as cnt FROM quality_scores WHERE assessed_date=?", (today_str,))
        qs_cnt = cur.fetchone()["cnt"]
        conn.close()
        agents.append(("✅", "Quality Assessor", f"评估 {qs_cnt} 篇" if qs_cnt > 0 else "等待运行", "done" if qs_cnt > 0 else "pending"))
    except Exception:
        agents.append(("✅", "Quality Assessor", "等待运行", "pending"))

    # Agent 5: Cross Referencer
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT graph_json FROM cross_ref_graphs WHERE report_date=?", (today_str,))
        row = cur.fetchone()
        conn.close()
        if row:
            g = json.loads(row["graph_json"]) if isinstance(row["graph_json"], str) else row["graph_json"]
            nodes = len(g.get("nodes", []))
            edges = len(g.get("edges", []))
            agents.append(("🔗", "Cross Referencer", f"图谱: {nodes}节点/{edges}边", "done"))
        else:
            agents.append(("🔗", "Cross Referencer", "等待运行", "pending"))
    except Exception:
        agents.append(("🔗", "Cross Referencer", "等待运行", "pending"))

    # Agent 6: Translator
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) as cnt FROM paper_translations")
        trans_cnt = cur.fetchone()["cnt"]
        conn.close()
        agents.append(("🌐", "Translator", f"翻译 {trans_cnt} 篇" if trans_cnt > 0 else "等待运行", "done" if trans_cnt > 0 else "pending"))
    except Exception:
        agents.append(("🌐", "Translator", "等待运行", "pending"))

    # Agent 7: Deep Analyzer
    try:
        deep = get_deep_analysis(today_str)
        if deep and deep.get("tldr"):
            agents.append(("📝", "Deep Analyzer", f"深度解读 1 篇", "done"))
        else:
            agents.append(("📝", "Deep Analyzer", "等待运行", "pending"))
    except Exception:
        agents.append(("📝", "Deep Analyzer", "等待运行", "pending"))

    # 构建 HTML
    done_count = sum(1 for _, _, _, s in agents if s == "done")
    total = len(agents)
    pct = int(done_count / total * 100) if total > 0 else 0

    agent_rows = ""
    for icon, name, detail, status in agents:
        status_dot = '<span style="color:#22C55E">●</span>' if status == "done" else '<span style="color:#CBD5E1">○</span>'
        agent_rows += f"""
        <div style="display:flex;align-items:center;gap:10px;padding:6px 0;font-size:12px">
            <span style="font-size:14px">{icon}</span>
            <span style="font-weight:600;color:{DARK};width:130px;flex-shrink:0">{name}</span>
            <span style="color:{GRAY};flex:1">{detail}</span>
            {status_dot}
        </div>"""

    return f"""
    <div class="card" style="height:100%">
        <div class="card-title">🤝 Agent 协作状态</div>
        <div class="card-subtitle">
            {done_count}/{total} 个智能体已完成 &middot;
            <span style="color:{ACCENT}">{pct}%</span>
        </div>
        <div class="progress-bar-bg" style="margin-bottom:14px">
            <div class="progress-bar-fill" style="width:{pct}%;background:{ACCENT}"></div>
        </div>
        {agent_rows}
        <div style="margin-top:10px;padding-top:8px;border-top:1px solid {BORDER};font-size:10px;color:{LIGHT_GRAY}">
            🕐 数据截止: {date.today().isoformat()}
        </div>
    </div>"""


def _build_deep_dive_preview(deep: dict = None) -> str:
    """Dashboard 中的深度解读预览卡片"""
    if not deep:
        return f"""
        <div class="card" style="height:100%">
            <div class="card-title">🔬 今日深度解读</div>
            <div class="card-subtitle">AI 深度分析今日最具代表性的论文</div>
            <div style="color:{LIGHT_GRAY};text-align:center;padding:40px 0">
                暂无深度解读数据<br><br>
                <span style="font-size:12px">请先运行今日分析流水线</span>
            </div>
        </div>"""

    tldr = deep.get("tldr", "")
    impact = deep.get("impact_prediction", "medium")
    impact_badge = {"high": '<span class="badge badge-red">🔥 高影响力</span>',
                    "medium": '<span class="badge badge-gold">⭐ 中等影响</span>',
                    "low": '<span class="badge badge-gray">📄 渐进贡献</span>'}.get(impact, '<span class="badge badge-gray">待评估</span>')

    title = ""
    arxiv_id = deep.get("arxiv_id", "")
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT title FROM papers WHERE arxiv_id=?", (arxiv_id,))
        row = cur.fetchone()
        conn.close()
        if row:
            title = row["title"]
    except Exception:
        pass

    audience = deep.get("target_audience", "")

    return f"""
    <div class="card" style="height:100%">
        <div class="card-title">🔬 今日深度解读</div>
        <div class="card-subtitle">AI 深度分析今日最具代表性的论文</div>
        <div style="margin-bottom:14px">{impact_badge}</div>
        <div style="font-size:13px;font-weight:700;color:{DARK};margin-bottom:6px;line-height:1.4">
            {title[:80]}{'...' if len(title) > 80 else ''}
        </div>
        <div style="background:{BG};border-radius:8px;padding:12px;margin-bottom:12px;
                    border-left:3px solid {ACCENT}">
            <div style="font-size:11px;color:{LIGHT_GRAY};margin-bottom:4px">💡 一句话速览</div>
            <div style="font-size:13px;color:{DARK};line-height:1.5">{tldr}</div>
        </div>
        <div style="font-size:11px;color:{GRAY};margin-bottom:12px">
            🎯 适合读者: {audience}
        </div>
    </div>
    <script>
    // 点击预览卡片跳转到深度解读 Tab (发送给 Gradio)
    </script>"""


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
    deep = get_deep_analysis(today_str)

    return f"""
    <style>{DASHBOARD_CSS}</style>
    {_build_navbar()}
    <div class="dashboard-grid">
        {_build_paper_trends_card(chart_b64, pct_change, abs_change)}
        {_build_top_papers_card(papers)}
        <div style="display:flex;flex-direction:column;gap:24px">
            {_build_quick_stats(papers)}
            {_build_agent_status_card(today_str)}
            {_build_ai_digest(trend)}
        </div>
        {_build_category_card(cat_chart, papers)}
        {_build_deep_dive_preview(deep)}
    </div>
    {_build_footer()}
    """


# ══════════════════════════════════════════════════════════
# 论文搜索页
# ══════════════════════════════════════════════════════════

def build_search_html(query: str = "", subfield_filter: str = "全部") -> str:
    """论文搜索页 — 语义增强检索"""
    today_papers = get_papers_by_date(date.today().isoformat())

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
        q_lower = q.lower().strip()
        expanded = set([q_lower])
        for key, aliases in ALIAS_MAP.items():
            if key in q_lower or any(a in q_lower for a in aliases):
                expanded.update([a.lower() for a in aliases])
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

        if subfield_filter != "全部":
            if (p.get("subfield") or "未分类") != subfield_filter:
                continue

        if q:
            search_text = f"{title} {abstract} {sf} {categories} {primary_cat} {key_contrib}"
            raw_keywords = q.split()
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
        badge_cls = "badge-red" if imp >= 4.0 else ("badge-gold" if imp >= 3.0 else "badge-blue")

        results_html += f"""
        <div class="paper-row">
            <div class="paper-icon" style="background:{'#FEF2F2' if imp>=4.0 else '#FFFBEB' if imp>=3.0 else '#EFF6FF'};font-size:14px">
                {'🔥' if imp>=4.0 else '⭐' if imp>=3.0 else '📄'}
            </div>
            <div class="paper-info">
                <div class="title">
                    <a href="https://doi.org/{p['arxiv_id']}" target="_blank">{p.get('title', '')}</a>
                </div>
                <div class="meta">
                    {imp_display} &nbsp;|&nbsp; {p['arxiv_id'][:28]} &nbsp;|&nbsp; {authors}
                </div>
                <div class="meta" style="margin-top:2px">
                    <span class="badge badge-teal">{sf_clean}</span>
                    <span class="badge badge-gray" style="margin-left:4px">{arxiv_cats}</span>
                </div>
                <div style="font-size:11px;color:{GRAY};margin-top:4px;line-height:1.5">
                    {(p.get('abstract') or '')[:200]}...
                </div>
            </div>
            <span class="badge {badge_cls}">{imp:.1f}</span>
        </div>"""

    return f"""
    <style>{DASHBOARD_CSS}</style>
    {_build_navbar()}
    <div class="card" style="margin-bottom:20px">
        <div class="card-title">🔍 论文搜索</div>
        <div class="card-subtitle">
            在今日采集的 {len(today_papers)} 篇论文中检索 · 支持标题、摘要、子领域、分类多字段匹配 ·
            空格分隔为 AND 逻辑 · 中英文+缩写自动扩展
        </div>
    </div>
    <div class="card">
        <div style="font-size:13px;color:{GRAY};margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid {BORDER}">
            {'搜索「' + (query or '') + '」共 <b>' + str(matched) + '</b> 条结果' if query
             else '今日全部 <b>' + str(len(today_papers)) + '</b> 篇论文 · 子领域: ' + (subfield_filter or '全部')}
        </div>
        {results_html if results_html else '<div style="color:{LIGHT_GRAY};text-align:center;padding:50px">未找到匹配结果。<br><br>💡 试试: LLM · 扩散模型 · 机器人 · 联邦学习 · 金融</div>'}
    </div>
    {_build_footer()}
    """


# ══════════════════════════════════════════════════════════
# 历史日报页
# ══════════════════════════════════════════════════════════

def build_history_html(selected_date: str = "") -> str:
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

    if report:
        report_html = markdown.markdown(
            report,
            extensions=['tables', 'fenced_code', 'nl2br'],
            output_format='html',
        )
        report_html = f'<div class="markdown-body">{report_html}</div>'
    else:
        report_html = '<div style="color:{LIGHT_GRAY};text-align:center;padding:60px">暂无日报数据。<br><br>请先运行一次分析流水线后刷新页面。</div>'

    return f"""
    <style>{DASHBOARD_CSS}</style>
    {_build_navbar()}
    <div class="card" style="margin-bottom:20px">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <div>
                <div class="card-title">📅 历史日报</div>
                <div class="card-subtitle">浏览往期每日学术热点分析报告</div>
            </div>
            <select id="history-date-select" style="border:1.5px solid {BORDER};background:{CARD_BG};
                    border-radius:10px;padding:10px 18px;font-size:14px;color:{DARK};outline:none;
                    cursor:pointer;font-family:inherit">
                {date_options}
            </select>
        </div>
    </div>
    <div class="card" style="max-height:72vh;overflow-y:auto;padding:28px">
        {report_html}
    </div>
    {_build_footer()}
    """


# ══════════════════════════════════════════════════════════
# 学术周报页
# ══════════════════════════════════════════════════════════

def build_weekly_html() -> str:
    from agents.weekly_digest import generate_weekly_digest, get_weekly_report

    today = date.today()
    report = get_weekly_report(today.isoformat())

    if not report:
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

    return f"""
    <style>{DASHBOARD_CSS}</style>
    {_build_navbar()}
    <div class="card" style="margin-bottom:20px">
        <div class="card-title">📝 学术周报</div>
        <div class="card-subtitle">基于 7 天日报数据自动汇总 · 由 DeepSeek 大模型生成趋势综述</div>
    </div>
    <div class="card" style="max-height:70vh;overflow-y:auto;padding:28px">
        <div class="markdown-body">{report_html}</div>
    </div>
    {_build_footer()}
    """


# ══════════════════════════════════════════════════════════
# 今日深度解读页
# ══════════════════════════════════════════════════════════

def build_deep_dive_html(selected_date: str = "") -> str:
    """今日深度解读 — 每日精选一篇论文做全方位分析"""
    if not selected_date:
        selected_date = date.today().isoformat()

    deep = get_deep_analysis(selected_date)

    if not deep:
        return f"""
        <style>{DASHBOARD_CSS}</style>
        {_build_navbar()}
        <div class="card" style="max-width:760px;margin:60px auto;text-align:center">
            <div class="card-title">🔬 今日深度解读</div>
            <div class="card-subtitle">AI 深度分析每日最具代表性的论文</div>
            <div style="color:{LIGHT_GRAY};padding:60px 0">
                <div style="font-size:48px;margin-bottom:16px">📭</div>
                <div>暂无 {selected_date} 的深度解读数据</div>
                <div style="font-size:12px;margin-top:8px">请先运行今日分析流水线（包含 Deep Analyzer 阶段）</div>
            </div>
        </div>
        {_build_footer()}
        """

    # 获取论文详情
    arxiv_id = deep.get("arxiv_id", "")
    paper = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM papers WHERE arxiv_id=?", (arxiv_id,))
        row = cur.fetchone()
        conn.close()
        if row:
            paper = dict(row)
            for f in ["authors", "categories"]:
                if f in paper and isinstance(paper[f], str):
                    try:
                        paper[f] = json.loads(paper[f])
                    except Exception:
                        pass
    except Exception:
        pass

    if not paper:
        paper = {"title": "未知论文", "authors": [], "subfield": "", "importance": 0}

    # 评分
    importance = float(paper.get("importance", 0))
    imp_stars = "⭐" * min(int(importance), 5) + f" {importance:.1f}/5.0"

    # 影响力 badge
    impact = deep.get("impact_prediction", "medium")
    impact_config = {
        "high": ("badge-red", "🔥 高影响力 — 可能成为领域重要工作"),
        "medium": ("badge-gold", "⭐ 中等影响 — 有价值的贡献"),
        "low": ("badge-gray", "📄 渐进贡献 — 扎实的增量工作"),
    }
    impact_cls, impact_label = impact_config.get(impact, impact_config["medium"])

    # 创新点列表
    innovations = deep.get("innovations", [])
    if isinstance(innovations, str):
        try:
            innovations = json.loads(innovations)
        except Exception:
            innovations = [innovations]
    innovations_html = "".join(
        f'<li style="margin-bottom:6px;color:{DARK}">{inv}</li>'
        for inv in innovations
    ) if innovations else '<li style="color:{LIGHT_GRAY}">暂无</li>'

    # 完整 Markdown 正文
    full_md = deep.get("full_analysis", "")
    if full_md:
        analysis_html = markdown.markdown(
            full_md,
            extensions=['tables', 'fenced_code', 'nl2br'],
            output_format='html',
        )
    else:
        analysis_html = '<p style="color:{LIGHT_GRAY}">暂无完整分析</p>'

    authors_str = ", ".join((paper.get("authors") or [])[:4])
    if len(paper.get("authors") or []) > 4:
        authors_str += " 等"

    return f"""
    <style>{DASHBOARD_CSS}</style>
    {_build_navbar()}
    <div class="card" style="margin-bottom:20px">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:20px">
            <div style="flex:1">
                <div class="card-title">🔬 今日深度解读</div>
                <div class="card-subtitle">
                    OpenAI Deep Analyzer Agent 每日精选最具代表性论文进行全方位解读
                </div>
            </div>
            <span class="badge {impact_cls}" style="font-size:13px;padding:6px 14px">{impact_label}</span>
        </div>
    </div>

    <div style="display:grid;grid-template-columns:1fr 360px;gap:24px">
        <!-- 左侧: 完整分析报告 -->
        <div class="card" style="padding:28px">
            <h1 style="font-size:20px;font-weight:700;color:{PRIMARY};line-height:1.4;margin-bottom:8px">
                {paper.get('title', '')}
            </h1>
            <div style="font-size:12px;color:{GRAY};margin-bottom:20px;padding-bottom:16px;border-bottom:1px solid {BORDER}">
                {authors_str} &nbsp;|&nbsp;
                <a href="https://doi.org/{arxiv_id}" target="_blank" style="color:{ACCENT}">{arxiv_id}</a> &nbsp;|&nbsp;
                <span class="badge badge-teal">{paper.get('subfield', '')}</span> &nbsp;|&nbsp;
                {imp_stars}
            </div>
            <div class="markdown-body">{analysis_html}</div>
        </div>

        <!-- 右侧: 信息卡片 -->
        <div style="display:flex;flex-direction:column;gap:20px">
            <!-- TL;DR -->
            <div class="card card-accent">
                <div style="font-size:11px;color:{LIGHT_GRAY};margin-bottom:6px">💡 一句话速览</div>
                <div style="font-size:14px;font-weight:600;color:{DARK};line-height:1.5">{deep.get('tldr', '')}</div>
            </div>

            <!-- 核心问题 -->
            <div class="card">
                <div class="card-title" style="font-size:14px">🎯 要解决的问题</div>
                <div style="font-size:12px;color:{GRAY};line-height:1.6;margin-top:8px">{deep.get('problem', '')}</div>
            </div>

            <!-- 关键创新 -->
            <div class="card">
                <div class="card-title" style="font-size:14px">💎 关键创新点</div>
                <ul style="font-size:12px;padding-left:18px;line-height:1.6;margin-top:8px;color:{GRAY}">
                    {innovations_html}
                </ul>
            </div>

            <!-- 局限性 -->
            <div class="card">
                <div class="card-title" style="font-size:14px">⚠️ 局限性</div>
                <div style="font-size:12px;color:{GRAY};line-height:1.6;margin-top:8px">{deep.get('limitations', '')}</div>
            </div>

            <!-- 适合读者 -->
            <div class="card">
                <div class="card-title" style="font-size:14px">🎓 适合读者</div>
                <div style="font-size:12px;color:{GRAY};line-height:1.6;margin-top:8px">{deep.get('target_audience', '')}</div>
            </div>

            <!-- 影响力预判 -->
            <div class="card">
                <div class="card-title" style="font-size:14px">📊 影响力预判</div>
                <div style="font-size:12px;color:{GRAY};line-height:1.6;margin-top:8px">{deep.get('impact_reason', '')}</div>
            </div>
        </div>
    </div>
    {_build_footer()}
    """


# ══════════════════════════════════════════════════════════
# Gradio App 入口
# ══════════════════════════════════════════════════════════

def build_ui():
    with gr.Blocks(title="PaperPulse · 学术论文日报系统") as app:
        gr.HTML("<style>footer{display:none!important}</style>")

        with gr.Tabs():
            # ── Tab 1: 仪表盘 ──
            with gr.TabItem("📊 仪表盘", id="dashboard"):
                dashboard_html = gr.HTML(value=build_dashboard_html())
                refresh_btn = gr.Button("↻ 刷新仪表盘", variant="primary",
                                        elem_id="refresh-dash-btn")
                refresh_btn.click(
                    fn=lambda: build_dashboard_html(),
                    outputs=dashboard_html,
                )

            # ── Tab 2: 搜索 ──
            with gr.TabItem("🔍 搜索论文", id="search"):
                with gr.Column():
                    with gr.Row():
                        search_input = gr.Textbox(
                            label="",
                            placeholder="输入关键词: Transformer · RAG · 多模态 · 强化学习...",
                            scale=3,
                            show_label=False,
                        )
                        subfield_dropdown = gr.Dropdown(
                            choices=["全部"],
                            value="全部",
                            scale=1,
                            show_label=False,
                            interactive=True,
                        )
                        search_btn = gr.Button("搜索", variant="primary", scale=1)
                    search_result = gr.HTML(value=build_search_html())

                def _on_search(query, subfield):
                    return build_search_html(query or "", subfield or "全部")

                def _update_dropdown():
                    papers = get_papers_by_date(date.today().isoformat())
                    subs = sorted(set(p.get("subfield", "") or "未分类" for p in papers))
                    return gr.Dropdown(choices=["全部"] + subs, value="全部")

                search_btn.click(_on_search, [search_input, subfield_dropdown], search_result)
                search_input.submit(_on_search, [search_input, subfield_dropdown], search_result)
                search_result.change(fn=_update_dropdown, outputs=subfield_dropdown)

            # ── Tab 3: 历史日报 ──
            with gr.TabItem("📅 历史日报", id="history"):
                available_dates = get_all_report_dates()
                default_date = available_dates[0] if available_dates else ""

                with gr.Column():
                    history_date = gr.Dropdown(
                        label="选择日期（点击下拉框自动刷新列表）",
                        choices=available_dates,
                        value=default_date,
                        interactive=True,
                    )
                    history_html = gr.HTML(value=build_history_html(default_date))

                def _on_history_change(d):
                    dates = get_all_report_dates()
                    if not d and dates:
                        d = dates[0]
                    html = build_history_html(d) if d else ""
                    return html, gr.Dropdown(choices=dates, value=d)

                history_date.change(
                    fn=_on_history_change,
                    inputs=history_date,
                    outputs=[history_html, history_date],
                )

            # ── Tab 4: 深度解读 ──
            with gr.TabItem("🔬 深度解读", id="deep-dive"):
                deep_dive_html = gr.HTML(value=build_deep_dive_html())

                def _refresh_deep_dive():
                    return build_deep_dive_html()

                refresh_deep_btn = gr.Button("↻ 刷新解读", variant="primary")
                refresh_deep_btn.click(
                    fn=_refresh_deep_dive,
                    outputs=deep_dive_html,
                )

            # ── Tab 5: 周报 ──
            with gr.TabItem("📝 学术周报", id="weekly"):
                weekly_html = gr.HTML(value=build_weekly_html())
                refresh_weekly_btn = gr.Button("↻ 生成/刷新周报", variant="primary")
                refresh_weekly_btn.click(
                    fn=lambda: build_weekly_html(),
                    outputs=weekly_html,
                )

            # ── Tab 6: 关于系统 ──
            with gr.TabItem("ℹ️ 关于系统", id="about"):
                gr.HTML(f"""
                <style>{DASHBOARD_CSS}</style>
                {_build_navbar()}
                <div class="card" style="max-width:760px;margin:40px auto">
                    <div class="card-title">关于 PaperPulse</div>
                    <div class="card-subtitle">每日学术论文热点追踪系统 · 7 Agent 多智能体协同</div>
                    <br>
                    <table style="width:100%;border-collapse:collapse;margin:12px 0;font-size:13px">
                        <tr style="border-bottom:1px solid {BORDER}">
                            <td style="padding:10px 12px;font-weight:600;color:{PRIMARY};width:200px">📡 采集智能体</td>
                            <td style="padding:10px 12px;color:{GRAY}">OpenAlex API 开源学术数据</td>
                        </tr>
                        <tr style="border-bottom:1px solid {BORDER}">
                            <td style="padding:10px 12px;font-weight:600;color:{PRIMARY}">🏷️ 分类与摘要智能体</td>
                            <td style="padding:10px 12px;color:{GRAY}">DeepSeek API · 22 子领域 · 0-5 评分</td>
                        </tr>
                        <tr style="border-bottom:1px solid {BORDER}">
                            <td style="padding:10px 12px;font-weight:600;color:{PRIMARY}">📊 趋势分析智能体</td>
                            <td style="padding:10px 12px;color:{GRAY}">jieba TF-IDF + DeepSeek 趋势预测</td>
                        </tr>
                        <tr style="border-bottom:1px solid {BORDER}">
                            <td style="padding:10px 12px;font-weight:600;color:{PRIMARY}">✅ 质量评估智能体</td>
                            <td style="padding:10px 12px;color:{GRAY}">方法论 · 创新性 · 可复现性 · 写作质量</td>
                        </tr>
                        <tr style="border-bottom:1px solid {BORDER}">
                            <td style="padding:10px 12px;font-weight:600;color:{PRIMARY}">🔗 交叉引用智能体</td>
                            <td style="padding:10px 12px;color:{GRAY}">TF-IDF 语义相似度 · 关联图谱 · 聚类</td>
                        </tr>
                        <tr style="border-bottom:1px solid {BORDER}">
                            <td style="padding:10px 12px;font-weight:600;color:{PRIMARY}">🌐 双语翻译智能体</td>
                            <td style="padding:10px 12px;color:{GRAY}">DeepSeek API 学术中英互译</td>
                        </tr>
                        <tr style="border-bottom:1px solid {BORDER}">
                            <td style="padding:10px 12px;font-weight:600;color:{PRIMARY}">📝 周报汇总智能体</td>
                            <td style="padding:10px 12px;color:{GRAY}">7 天日报聚合 + LLM 趋势综述</td>
                        </tr>
                        <tr>
                            <td style="padding:10px 12px;font-weight:600;color:{PRIMARY}">🔬 深度解读智能体</td>
                            <td style="padding:10px 12px;color:{GRAY}">每日精选论文全方位分析 · TL;DR · 方法拆解 · 影响力预判</td>
                        </tr>
                    </table>
                    <br>
                    <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:8px">
                        <span class="badge badge-blue">Python 3.11</span>
                        <span class="badge badge-blue">DeepSeek API</span>
                        <span class="badge badge-teal">OpenAlex</span>
                        <span class="badge badge-teal">Gradio 6.x</span>
                        <span class="badge badge-gray">SQLite</span>
                        <span class="badge badge-gray">Docker</span>
                        <span class="badge badge-gray">matplotlib</span>
                        <span class="badge badge-gray">jieba</span>
                    </div>
                    <br>
                    <p style="color:{LIGHT_GRAY};font-size:12px;margin-top:16px;padding-top:12px;border-top:1px solid {BORDER}">
                        基于 OpenClaw 多智能体协同框架设计 · 浙江大学人工智能基础 A 课程项目 · {date.today().isoformat()}
                    </p>
                </div>
                {_build_footer()}
                """)

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
