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

/* ── 关于系统 — 架构展示动画 ── */
@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(24px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes pulse {{ 0%,100% {{ box-shadow: 0 0 0 0 rgba(37,99,235,0.3); }} 50% {{ box-shadow: 0 0 0 8px rgba(37,99,235,0); }} }}
@keyframes flowRight {{
    from {{ width: 0; }}
    to   {{ width: 100%; }}
}}
.animate-in {{ animation: fadeInUp 0.6s cubic-bezier(0.22, 0.61, 0.36, 1) both; }}
.animate-in.d1 {{ animation-delay: 0.05s; }} .animate-in.d2 {{ animation-delay: 0.12s; }}
.animate-in.d3 {{ animation-delay: 0.19s; }} .animate-in.d4 {{ animation-delay: 0.26s; }}
.animate-in.d5 {{ animation-delay: 0.33s; }} .animate-in.d6 {{ animation-delay: 0.40s; }}
.animate-in.d7 {{ animation-delay: 0.47s; }} .animate-in.d8 {{ animation-delay: 0.54s; }}

/* Agent 卡片 */
.agent-card {{
    background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 14px;
    padding: 20px 16px; text-align: center; position: relative; overflow: hidden;
    cursor: pointer; transition: all 0.35s cubic-bezier(0.22, 0.61, 0.36, 1);
    min-height: 100px; display: flex; flex-direction: column; align-items: center;
    justify-content: center; gap: 6px;
}}
.agent-card:hover {{
    transform: translateY(-6px); border-color: {ACCENT};
    box-shadow: 0 12px 28px rgba(27,58,92,0.12), 0 0 0 1px {ACCENT} inset;
}}
.agent-card .agent-icon {{ font-size: 32px; transition: transform 0.3s; }}
.agent-card:hover .agent-icon {{ transform: scale(1.2); }}
.agent-card .agent-name {{ font-size: 14px; font-weight: 700; color: {PRIMARY}; }}
.agent-card .agent-role {{ font-size: 11px; color: {LIGHT_GRAY}; max-height: 0; overflow: hidden; transition: max-height 0.4s, margin 0.3s; }}
.agent-card:hover .agent-role {{ max-height: 80px; margin-top: 4px; }}
.agent-card .agent-tech {{ font-size: 10px; color: {GRAY}; max-height: 0; overflow: hidden; transition: max-height 0.4s; }}
.agent-card:hover .agent-tech {{ max-height: 40px; }}
.agent-card .agent-io {{ display: flex; gap: 6px; margin-top: 8px; opacity: 0; transition: opacity 0.3s; }}
.agent-card:hover .agent-io {{ opacity: 1; }}

/* 流水线箭头 */
.pipeline-arrow {{
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; color: {ACCENT}; font-weight: 700;
    animation: pulse 2s infinite;
}}

/* 架构层卡片 */
.arch-layer {{
    background: {CARD_BG}; border: 1.5px solid {BORDER}; border-radius: 14px;
    padding: 24px 20px; text-align: center;
    transition: box-shadow 0.3s, border-color 0.3s;
}}
.arch-layer:hover {{ border-color: {ACCENT}; box-shadow: 0 6px 20px rgba(27,58,92,0.08); }}
.arch-layer .layer-title {{ font-size: 16px; font-weight: 700; color: {PRIMARY}; margin-bottom: 10px; }}
.arch-layer .layer-detail {{ font-size: 12px; color: {GRAY}; line-height: 1.6; }}

/* 团队成员卡片 */
.team-card {{
    background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 16px;
    padding: 28px 20px; text-align: center;
    transition: transform 0.3s, box-shadow 0.3s;
}}
.team-card:hover {{ transform: translateY(-4px); box-shadow: 0 8px 24px rgba(0,0,0,0.08); }}
.team-card .avatar {{
    width: 64px; height: 64px; border-radius: 50%; margin: 0 auto 16px;
    display: flex; align-items: center; justify-content: center;
    font-size: 28px; font-weight: 800; color: #FFFFFF;
}}
.team-card .member-name {{ font-size: 16px; font-weight: 700; color: {PRIMARY}; margin-bottom: 4px; }}
.team-card .member-label {{ font-size: 11px; color: {LIGHT_GRAY}; margin-bottom: 14px; }}
.team-card .agent-list {{ font-size: 12px; color: {GRAY}; line-height: 1.9; }}

/* 对比表 */
.compare-table {{
    width: 100%; border-collapse: collapse; font-size: 13px;
}}
.compare-table th {{
    background: {PRIMARY}; color: #FFFFFF; padding: 14px 16px; text-align: center;
    font-weight: 700; font-size: 14px;
}}
.compare-table th:first-child {{ border-radius: 10px 0 0 0; }}
.compare-table th:last-child {{ border-radius: 0 10px 0 0; }}
.compare-table td {{
    padding: 12px 16px; text-align: center; border-bottom: 1px solid {BORDER};
}}
.compare-table tr:last-child td:first-child {{ border-radius: 0 0 0 10px; }}
.compare-table tr:last-child td:last-child {{ border-radius: 0 0 10px 0; }}
.compare-table tr.winner td {{ background: #F0F9FF; }}
.compare-table tr.winner td:last-child {{ font-weight: 700; color: {ACCENT}; }}

/* 技术标签组 */
.tech-group {{ margin-bottom: 16px; }}
.tech-group .tech-label {{ font-size: 12px; font-weight: 700; color: {PRIMARY}; margin-bottom: 8px; }}
.tech-tag {{
    display: inline-block; padding: 5px 12px; border-radius: 20px;
    margin: 3px 4px; font-size: 12px; font-weight: 500;
    transition: transform 0.15s;
}}
.tech-tag:hover {{ transform: scale(1.06); }}

/* Hero 横幅 */
.about-hero {{
    background: linear-gradient(135deg, {PRIMARY} 0%, #0F2B47 100%);
    border-radius: 16px; padding: 48px 40px; margin-bottom: 32px;
    text-align: center; position: relative; overflow: hidden;
}}
.about-hero::after {{
    content: ""; position: absolute; top: -50%; right: -20%;
    width: 400px; height: 400px; border-radius: 50%;
    background: rgba(37,99,235,0.08);
}}
.about-hero h1 {{ color: #FFFFFF; font-size: 36px; font-weight: 800; margin: 0; position: relative; z-index: 1; }}
.about-hero p {{ color: #CBD5E1; font-size: 15px; margin-top: 8px; position: relative; z-index: 1; }}

/* 统计数字横幅 */
.stat-banner {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;
    margin: 32px 0;
}}
.stat-item {{
    text-align: center; padding: 20px 12px;
    border-radius: 12px; background: {BG};
    border: 1px solid {BORDER};
}}
.stat-item .stat-num {{ font-size: 28px; font-weight: 800; color: {PRIMARY}; }}
.stat-item .stat-desc {{ font-size: 11px; color: {LIGHT_GRAY}; margin-top: 6px; }}
@media (max-width: 768px) {{ .stat-banner {{ grid-template-columns: repeat(2, 1fr); }} }}
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
    items = dist.most_common(8)
    shown = sum(c for _, c in items)

    # 顶部方向统计（紧凑版）
    top_tags = ""
    tag_colors = [ACCENT, "#6366F1", "#8B5CF6", "#EC4899", GOLD, TEAL, "#14B8A6", "#84CC16"]
    for (cat, cnt), clr in zip(items, tag_colors):
        short = cat[:16] + ".." if len(cat) > 16 else cat
        top_tags += f'<span style="display:inline-block;margin:2px 4px;font-size:11px;color:{clr}"><b>{short}</b> {cnt}</span>'

    remaining = total - shown
    note = f"另有 {remaining} 篇在其他方向" if remaining > 0 else f"共 {total} 篇"

    return f"""
    <div class="card">
        <div class="card-title">📂 领域分布</div>
        <div class="card-subtitle">{total} 篇论文 · {len(items)} 个方向 · {note}</div>
        <div style="margin-bottom:8px;line-height:2">{top_tags}</div>
        <img src="{chart_b64}" style="width:100%;border-radius:8px" alt="领域分布图">
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
        {_build_quick_stats(papers)}
        {_build_category_card(cat_chart, papers)}
        {_build_agent_status_card(today_str)}
        {_build_deep_dive_preview(deep)}
        <div style="grid-column:1/-1">
            {_build_ai_digest(trend)}
        </div>
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


# ══════════════════════════════════════════════════════════
# 跨天对比分析
# ══════════════════════════════════════════════════════════

def _build_comparison_chart(date1: str, date2: str, metric: str = "count") -> str:
    """构建双日子领域对比图"""
    from collections import Counter

    p1 = get_papers_by_date(date1)
    p2 = get_papers_by_date(date2)

    if metric == "count":
        d1 = Counter((x.get("subfield") or "未分类" for x in p1)).most_common(6)
        d2 = Counter((x.get("subfield") or "未分类" for x in p2)).most_common(6)
    else:  # avg_score
        by_sf1 = {}
        for x in p1:
            sf = x.get("subfield") or "未分类"
            by_sf1.setdefault(sf, []).append(float(x.get("importance", 0)))
        d1 = sorted([(k, sum(v)/len(v)) for k,v in by_sf1.items() if len(v)>=3], key=lambda x:x[1], reverse=True)[:6]
        by_sf2 = {}
        for x in p2:
            sf = x.get("subfield") or "未分类"
            by_sf2.setdefault(sf, []).append(float(x.get("importance", 0)))
        d2 = sorted([(k, sum(v)/len(v)) for k,v in by_sf2.items() if len(v)>=3], key=lambda x:x[1], reverse=True)[:6]

    # Combine all categories
    all_cats = sorted(set([k for k,_ in d1] + [k for k,_ in d2]))
    # limit to top 8
    all_cats = all_cats[:8]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6), sharey=False)
    fig.patch.set_facecolor(CARD_BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(CARD_BG)

    colors1 = [ACCENT, "#6366F1", "#8B5CF6", "#EC4899", GOLD, TEAL, "#14B8A6", "#84CC16"]
    colors2 = [TEAL, "#14B8A6", ACCENT, GOLD, "#6366F1", "#EC4899", "#8B5CF6", "#84CC16"]

    for ax, data, title, colors, d_str in [
        (ax1, d1, date1, colors1, date1),
        (ax2, d2, date2, colors2, date2)
    ]:
        if not data:
            ax.text(0.5, 0.5, "无数据", transform=ax.transAxes, ha="center", va="center", color=LIGHT_GRAY, fontsize=14)
            ax.set_title(d_str, fontsize=12, fontweight="bold", color=PRIMARY)
            for spine in ax.spines.values():
                spine.set_visible(False)
            continue
        labels = [it[0][:16] for it in data]
        vals = [it[1] for it in data]
        bar_colors = colors[:len(labels)]

        if metric == "score":
            ax.barh(list(reversed(labels)), list(reversed(vals)), color=list(reversed(bar_colors)),
                    height=0.55, edgecolor="white", linewidth=0.5)
            for bar, val in zip([], []): pass
            ax.set_xlabel("均分", fontsize=10, color=GRAY)
            ax.set_xlim(0, 5.0)
        else:
            bars = ax.barh(list(reversed(labels)), list(reversed(vals)), color=list(reversed(bar_colors)),
                          height=0.55, edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, reversed(vals)):
                ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                        str(val), va="center", fontsize=10, fontweight="bold", color=DARK)

        ax.set_title(d_str, fontsize=12, fontweight="bold", color=PRIMARY)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="x", colors=GRAY, labelsize=8)
        ax.tick_params(axis="y", colors=DARK, labelsize=9)

    plt.tight_layout()
    return _fig_to_b64(fig)


def _build_compare_score_dist(date1: str, date2: str) -> str:
    """双日评分分布叠加图"""
    p1 = get_papers_by_date(date1)
    p2 = get_papers_by_date(date2)

    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig.patch.set_facecolor(CARD_BG)
    ax.set_facecolor(CARD_BG)

    s1 = [float(x.get("importance", 0)) for x in p1 if float(x.get("importance", 0)) > 0]
    s2 = [float(x.get("importance", 0)) for x in p2 if float(x.get("importance", 0)) > 0]

    bins = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    kw = dict(alpha=0.55, bins=bins, edgecolor="white", linewidth=0.5)

    ax.hist(s1, label=date1, color=ACCENT, **kw)
    ax.hist(s2, label=date2, color=TEAL, alpha=0.45, bins=bins, edgecolor="white", linewidth=0.5)

    ax.axvline(x=sum(s1)/len(s1) if s1 else 0, color=ACCENT, linestyle="--", linewidth=2,
               label=f'{date1} 均分 {sum(s1)/len(s1):.1f}' if s1 else '')
    ax.axvline(x=sum(s2)/len(s2) if s2 else 0, color=TEAL, linestyle="--", linewidth=2,
               label=f'{date2} 均分 {sum(s2)/len(s2):.1f}' if s2 else '')

    ax.legend(fontsize=9, loc="upper right", framealpha=0.8, edgecolor=BORDER)
    ax.set_xlabel("重要性评分", fontsize=10, color=GRAY)
    ax.set_ylabel("论文数", fontsize=10, color=GRAY)
    for spine in ax.spines.values():
        spine.set_visible(False)

    return _fig_to_b64(fig)


def build_compare_html(date1: str = "", date2: str = "") -> str:
    """跨天对比分析页"""
    available = get_all_report_dates()

    # 默认取最后两天
    if not date1 and len(available) >= 2:
        date1 = available[1]  # yesterday
    elif not date1 and available:
        date1 = available[0]
    if not date2 and available:
        date2 = available[0]  # today/latest

    p1 = get_papers_by_date(date1) if date1 else []
    p2 = get_papers_by_date(date2) if date2 else []

    cnt1, cnt2 = len(p1), len(p2)
    avg1 = sum(float(x.get("importance", 0)) for x in p1) / cnt1 if cnt1 else 0
    avg2 = sum(float(x.get("importance", 0)) for x in p2) / cnt2 if cnt2 else 0

    # 高频关键词变化
    from collections import Counter
    kw1 = Counter()
    for x in p1:
        for w in (x.get("categories") or []):
            kw1[w] += 1
    kw2 = Counter()
    for x in p2:
        for w in (x.get("categories") or []):
            kw2[w] += 1

    top_kw1 = kw1.most_common(5)
    top_kw2 = kw2.most_common(5)

    # 新出现/消失的关键词
    kw1_set = set(kw1.keys())
    kw2_set = set(kw2.keys())
    new_kw = kw2_set - kw1_set
    gone_kw = kw1_set - kw2_set

    new_kw_html = "".join(f'<span class="badge badge-blue" style="margin:2px">{k}</span> ' for k in list(new_kw)[:6]) if new_kw else '<span style="color:{LIGHT_GRAY};font-size:12px">无</span>'
    gone_kw_html = "".join(f'<span class="badge badge-gray" style="margin:2px">{k}</span> ' for k in list(gone_kw)[:6]) if gone_kw else '<span style="color:{LIGHT_GRAY};font-size:12px">无</span>'

    # 评分变化最大的论文
    def _top_k(papers, k=5):
        return sorted(papers, key=lambda x: float(x.get("importance", 0)), reverse=True)[:k]

    top1 = _top_k(p1, 3)
    top2 = _top_k(p2, 3)

    def _render_paper_row(p, rank):
        imp = float(p.get("importance", 0))
        badge = "badge-red" if imp >= 4.0 else "badge-gold" if imp >= 3.0 else "badge-blue"
        title = (p.get("title") or "")[:60]
        return f"""
        <div class="paper-row">
            <span style="font-size:18px;width:24px">{rank}</span>
            <div class="paper-info">
                <div class="title"><a href="https://doi.org/{p['arxiv_id']}" target="_blank">{title}</a></div>
                <div class="meta">{_importance_display(imp)} &nbsp;|&nbsp; {p.get('subfield','')[:18]}</div>
            </div>
            <span class="badge {badge}">{imp:.1f}</span>
        </div>"""

    chart_score = _build_comparison_chart(date1, date2, "count") if (p1 or p2) else ""
    chart_dist = _build_compare_score_dist(date1, date2) if (p1 or p2) else ""

    # Change indicators
    cnt_delta = cnt2 - cnt1
    cnt_delta_str = f'<span style="color:{TEAL if cnt_delta>=0 else RED};font-weight:700">{"+" if cnt_delta>=0 else ""}{cnt_delta}</span>'
    avg_delta = avg2 - avg1
    avg_delta_str = f'<span style="color:{TEAL if avg_delta>=0 else RED};font-weight:700">{"+" if avg_delta>=0 else ""}{avg_delta:.1f}</span>'

    return f"""
    <style>{DASHBOARD_CSS}</style>
    {_build_navbar()}

    <div class="card" style="margin-bottom:20px">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:14px">
            <div>
                <div class="card-title">📊 跨天对比分析</div>
                <div class="card-subtitle">并排对比两天的论文数据 · 发现趋势变化和关键差异</div>
            </div>
        </div>
    </div>

    <!-- KPI 对比卡 -->
    <div class="stats-row" style="margin-bottom:24px;grid-template-columns:repeat(4,1fr)">
        <div class="stat-card stat-card-accent">
            <div class="number">{cnt1}</div>
            <div class="label">{date1}<br>论文数</div>
        </div>
        <div class="stat-card stat-card-teal">
            <div class="number">{cnt2}</div>
            <div class="label">{date2}<br>论文数 &nbsp; {cnt_delta_str}</div>
        </div>
        <div class="stat-card stat-card-gold">
            <div class="number">{avg1:.1f}</div>
            <div class="label">{date1}<br>均分</div>
        </div>
        <div class="stat-card stat-card-red">
            <div class="number">{avg2:.1f}</div>
            <div class="label">{date2}<br>均分 &nbsp; {avg_delta_str}</div>
        </div>
    </div>

    <div class="dashboard-grid">
        <!-- 左: 子领域分布对比 -->
        <div class="card" style="grid-column:1/-1">
            <div class="card-title">📂 子领域分布对比</div>
            <div class="card-subtitle">{date1}（左） vs {date2}（右）</div>
            <img src="{chart_score}" style="width:100%;border-radius:8px" alt="子领域对比">
        </div>

        <!-- 评分分布 -->
        <div class="card">
            <div class="card-title">📊 评分分布对比</div>
            <div class="card-subtitle">垂直虚线 = 该日均分</div>
            <img src="{chart_dist}" style="width:100%;border-radius:8px" alt="评分分布">
        </div>

        <!-- 关键词变化 -->
        <div class="card">
            <div class="card-title">🏷️ 分类关键词变化</div>
            <div class="card-subtitle">新兴 vs 消退的分类标签</div>
            <div style="margin-top:12px">
                <div style="font-size:12px;font-weight:600;color:{PRIMARY};margin-bottom:6px">
                    🆕 {date2} 新出现 ({len(new_kw)} 个)
                </div>
                <div style="margin-bottom:14px">{new_kw_html}</div>
                <div style="font-size:12px;font-weight:600;color:{GRAY};margin-bottom:6px">
                    📭 {date1} 有过但 {date2} 消退 ({len(gone_kw)} 个)
                </div>
                <div>{gone_kw_html}</div>
            </div>
        </div>

        <!-- Top 3 对比 -->
        <div class="card">
            <div class="card-title">🏆 {date1} TOP 3</div>
            <div class="card-subtitle">当日评分最高的论文</div>
            {"".join(_render_paper_row(p, medals[i]) for i,p in enumerate(top1) if p) if top1 else '<div style="color:{LIGHT_GRAY};padding:20px">无数据</div>'}
        </div>

        <div class="card">
            <div class="card-title">🏆 {date2} TOP 3</div>
            <div class="card-subtitle">当日评分最高的论文</div>
            {"".join(_render_paper_row(p, medals[i]) for i,p in enumerate(top2) if p) if top2 else '<div style="color:{LIGHT_GRAY};padding:20px">无数据</div>'}
        </div>
    </div>
    {_build_footer()}
    """


medals = {0: "🥇", 1: "🥈", 2: "🥉"}


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


# ══════════════════════════════════════════════════════════
# 关于系统 — 结课展示用架构总览
# ══════════════════════════════════════════════════════════

def _get_system_stats() -> dict:
    """实时获取系统运行统计"""
    try:
        from database import get_connection
        c = get_connection()
        cur = c.execute("SELECT COUNT(*) FROM papers")
        tp = cur.fetchone()[0]
        cur = c.execute("SELECT COUNT(DISTINCT collected_date) FROM papers")
        td = cur.fetchone()[0]
        cur = c.execute("SELECT COUNT(DISTINCT subfield) FROM papers WHERE subfield IS NOT NULL AND subfield!=''")
        sf = cur.fetchone()[0]
        c.close()
        return {"total_papers": tp, "total_days": td, "subfields": sf}
    except Exception:
        return {"total_papers": 0, "total_days": 0, "subfields": 0}


def build_about_html() -> str:
    stats = _get_system_stats()

    # ======= Agent 卡片数据 =======
    agents = [
        {"icon": "📡", "name": "Collector", "code": "collector.py",
         "role": "从 OpenAlex API 采集 8 大 AI 主题最新论文",
         "tech": "OpenAlex API · requests", "io": "输入: 主题列表 → 输出: 论文元数据"},
        {"icon": "🏷️", "name": "Classifier", "code": "classifier.py",
         "role": "22 子领域 + 中文摘要 + 0-5 精细评分",
         "tech": "DeepSeek API · JSON 结构化", "io": "输入: 论文摘要 → 输出: 分类+评分"},
        {"icon": "📊", "name": "TrendAnalyzer", "code": "trend_analyzer.py",
         "role": "关键词提取 + 趋势预测 + 历史对比",
         "tech": "jieba TF-IDF · DeepSeek API", "io": "输入: 全部论文 → 输出: 趋势报告"},
        {"icon": "✅", "name": "Quality Assessor", "code": "quality_assessor.py",
         "role": "四维质量评估: 方法论·创新性·可复现性·写作",
         "tech": "DeepSeek API · 加权评分", "io": "输入: 论文 → 输出: 四维分数组"},
        {"icon": "🔗", "name": "Cross Referencer", "code": "cross_referencer.py",
         "role": "TF-IDF 语义相似度 + 关联图谱 + 聚类分析",
         "tech": "scikit-learn · TF-IDF", "io": "输入: 论文 → 输出: 关联图谱"},
        {"icon": "🌐", "name": "Translator", "code": "translator.py",
         "role": "Top 20 论文标题+摘要中英互译",
         "tech": "DeepSeek API", "io": "输入: 英文论文 → 输出: 中文译本"},
        {"icon": "📝", "name": "Weekly Digest", "code": "weekly_digest.py",
         "role": "7 天日报聚合 + 持续热点识别 + 突破性工作",
         "tech": "DeepSeek API · 数据库聚合", "io": "输入: 7天日报 → 输出: 周报"},
        {"icon": "🔬", "name": "Deep Analyzer", "code": "deep_analyzer.py",
         "role": "每日精选论文全方位解读: TL;DR + 方法拆解 + 影响力预判",
         "tech": "DeepSeek API · 8 维分析", "io": "输入: Top论文 → 输出: 深度报告"},
    ]

    agent_cards = ""
    for i, a in enumerate(agents):
        agent_cards += f"""
        <div class="agent-card animate-in d{i+1}">
            <div class="agent-icon">{a['icon']}</div>
            <div class="agent-name">{a['name']}</div>
            <div class="agent-role">{a['role']}</div>
            <div class="agent-tech">🔧 {a['tech']}</div>
            <div class="agent-io">
                <span class="badge badge-blue">{a['code']}</span>
            </div>
        </div>"""

    # ======= 对比表行 =======
    compare_rows = [
        ("语义分类准确率", "~60% (仅标签)", "> 90%", True),
        ("评分区分度 (标准差)", "N/A", "σ = 0.7", True),
        ("每日 Token 消耗", "~120K", "~50K–80K", True),
        ("报告完整性", "低 (纯统计)", "六章结构化日报", True),
        ("模块可扩展性", "差 (重写Prompt)", "即插即用 新增Agent", True),
        ("单模块故障影响", "全部崩溃", "仅影响该模块", True),
    ]
    compare_html = ""
    for label, single, multi, winner in compare_rows:
        cls = ' class="winner"' if winner else ""
        compare_html += f'<tr{cls}><td style="text-align:left;font-weight:600;padding-left:20px">{label}</td><td>{single}</td><td style="font-weight:{";700" if winner else "400"}">{multi}</td></tr>'

    return f"""
    <style>{DASHBOARD_CSS}</style>
    {_build_navbar()}

    <!-- ═══ ① Hero 横幅 ═══ -->
    <div class="about-hero animate-in d1">
        <h1>PaperPulse</h1>
        <p>每日学术论文热点追踪系统 · 基于 OpenClaw 的多智能体协同分析平台</p>
    </div>

    <!-- ═══ ② 架构总览 ═══ -->
    <div class="card animate-in d2" style="margin-bottom:32px">
        <div class="card-title">🏗️ 系统架构总览</div>
        <div class="card-subtitle">三层解耦 + 8 Agent 松耦合协同 · 数据单向流动</div>
        <div style="display:grid;grid-template-columns:1fr auto 1fr auto 1fr;gap:12px;align-items:center;margin:24px 0">
            <div class="arch-layer">
                <div class="layer-title">📥 数据采集层</div>
                <div class="layer-detail">
                    OpenAlex 免费学术 API<br>
                    8 大 AI 热门主题<br>
                    每日 7 天窗口检索<br>
                    指数退避 + 去重
                </div>
            </div>
            <div class="pipeline-arrow" style="font-size:28px">→</div>
            <div class="arch-layer" style="border-color:{ACCENT};border-width:2px">
                <div class="layer-title" style="color:{ACCENT}">🧠 智能体协同层</div>
                <div class="layer-detail">
                    8 个专用 Agent 分工协作<br>
                    DeepSeek API 语义分析<br>
                    SQLite 松耦合数据共享<br>
                    批次处理 + 重试容错
                </div>
            </div>
            <div class="pipeline-arrow" style="font-size:28px">→</div>
            <div class="arch-layer">
                <div class="layer-title">📤 应用展示层</div>
                <div class="layer-detail">
                    Gradio Web 仪表盘<br>
                    日报 · 周报 · 搜索<br>
                    深度解读 · 跨天对比<br>
                    Docker 容器化部署
                </div>
            </div>
        </div>
    </div>

    <!-- ═══ ③ Agent 全景矩阵 ═══ -->
    <div class="card animate-in d3" style="margin-bottom:32px">
        <div class="card-title">🤖 智能体全景矩阵</div>
        <div class="card-subtitle">8 个独立 Agent 分两阶段流水线 · 悬停卡片查看详情</div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:20px 0">
            {agent_cards}
        </div>
        <div style="text-align:center;margin-top:16px">
            <span style="font-size:12px;color:{LIGHT_GRAY}">💡 每个 Agent 卡片悬停展示职责、技术栈和输入输出</span>
        </div>
    </div>

    <!-- ═══ ④ 设计哲学 ═══ -->
    <div class="card animate-in d4" style="margin-bottom:32px">
        <div class="card-title">🎯 设计哲学: 为什么选择多智能体？</div>
        <div class="card-subtitle">我们对比了三种方案，最终选择了多智能体协同架构</div>
        <div style="overflow-x:auto;margin-top:16px">
            <table class="compare-table">
                <tr>
                    <th>评价维度</th>
                    <th>方案 A: 单 Agent 大模型</th>
                    <th>方案 B: 多 Agent 协同 (本系统)</th>
                </tr>
                {compare_html}
            </table>
        </div>
    </div>

    <!-- ═══ ⑤ 技术栈 ═══ -->
    <div class="card animate-in d5" style="margin-bottom:32px">
        <div class="card-title">🛠️ 技术栈全景</div>
        <div class="card-subtitle">从数据获取到云部署的完整技术选型</div>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin-top:16px">
            <div class="tech-group">
                <div class="tech-label">🤖 AI 模型</div>
                <span class="tech-tag badge-blue">DeepSeek V3</span>
                <span class="tech-tag badge-blue">OpenAI SDK</span>
                <span class="tech-tag badge-blue">Prompt Engineering</span>
            </div>
            <div class="tech-group">
                <div class="tech-label">📡 数据获取</div>
                <span class="tech-tag badge-teal">OpenAlex API</span>
                <span class="tech-tag badge-teal">requests</span>
                <span class="tech-tag badge-teal">指数退避重试</span>
            </div>
            <div class="tech-group">
                <div class="tech-label">🔤 中文 NLP</div>
                <span class="tech-tag badge-teal">jieba 分词</span>
                <span class="tech-tag badge-teal">TF-IDF 提取</span>
                <span class="tech-tag badge-teal">学术停用词表</span>
            </div>
            <div class="tech-group">
                <div class="tech-label">📊 可视化</div>
                <span class="tech-tag badge-gold">matplotlib</span>
                <span class="tech-tag badge-gold">markdown 渲染</span>
                <span class="tech-tag badge-gold">CSS Grid 布局</span>
            </div>
            <div class="tech-group">
                <div class="tech-label">🌐 前端</div>
                <span class="tech-tag badge-blue">Gradio 6.x</span>
                <span class="tech-tag badge-blue">HTML5/CSS3</span>
                <span class="tech-tag badge-blue">响应式设计</span>
            </div>
            <div class="tech-group">
                <div class="tech-label">💾 存储</div>
                <span class="tech-tag badge-gray">SQLite</span>
                <span class="tech-tag badge-gray">JSON 序列化</span>
                <span class="tech-tag badge-gray">文件系统</span>
            </div>
            <div class="tech-group">
                <div class="tech-label">🐳 部署运维</div>
                <span class="tech-tag badge-gray">Docker</span>
                <span class="tech-tag badge-gray">Docker Compose</span>
                <span class="tech-tag badge-gray">阿里云 ECS</span>
            </div>
            <div class="tech-group">
                <div class="tech-label">⏰ 调度</div>
                <span class="tech-tag badge-gray">schedule 库</span>
                <span class="tech-tag badge-gray">每日 8:00 CST</span>
                <span class="tech-tag badge-gray">异常自动恢复</span>
            </div>
        </div>
    </div>

    <!-- ═══ ⑥ 团队分工 ═══ -->
    <div class="card animate-in d6" style="margin-bottom:32px">
        <div class="card-title">👥 团队分工</div>
        <div class="card-subtitle">三人协作，基于 OpenClaw 框架分工开发 8 个智能体</div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin-top:20px">
            <div class="team-card">
                <div class="avatar" style="background:linear-gradient(135deg,{ACCENT},#6366F1)">A</div>
                <div class="member-name">架构与编排</div>
                <div class="member-label">System Architecture & Orchestration</div>
                <div class="agent-list">
                    📡 Collector — 论文采集<br>
                    📊 TrendAnalyzer — 趋势分析<br>
                    🔗 Cross Referencer — 关联图谱<br>
                    📝 Weekly Digest — 周报汇总<br>
                    🏗️ main.py — 流水线编排
                </div>
            </div>
            <div class="team-card">
                <div class="avatar" style="background:linear-gradient(135deg,{TEAL},#14B8A6)">B</div>
                <div class="member-name">智能分析</div>
                <div class="member-label">AI Analysis & Deep Insight</div>
                <div class="agent-list">
                    🏷️ Classifier — 分类与评分<br>
                    ✅ Quality Assessor — 质量评估<br>
                    🔬 Deep Analyzer — 深度解读<br>
                    ✍️ Prompt 工程 — 全系统提示词设计
                </div>
            </div>
            <div class="team-card">
                <div class="avatar" style="background:linear-gradient(135deg,{GOLD},#D97706)">C</div>
                <div class="member-name">工程实现</div>
                <div class="member-label">Engineering & Infrastructure</div>
                <div class="agent-list">
                    🌐 Translator — 双语翻译<br>
                    🖥️ Web UI — Gradio 全栈开发<br>
                    💾 Database — SQLite 设计<br>
                    🐳 DevOps — Docker + 云部署
                </div>
            </div>
        </div>
    </div>

    <!-- ═══ ⑦ 实时统计 ═══ -->
    <div class="card animate-in d7" style="margin-bottom:0">
        <div class="card-title">📊 系统运行数据</div>
        <div class="card-subtitle">来自真实数据库的实时统计</div>
        <div class="stat-banner">
            <div class="stat-item stat-card-accent">
                <div class="stat-num">{stats['total_papers']}</div>
                <div class="stat-desc">累计采集论文</div>
            </div>
            <div class="stat-item stat-card-teal">
                <div class="stat-num">{stats['total_days']}</div>
                <div class="stat-desc">覆盖天数</div>
            </div>
            <div class="stat-item stat-card-gold">
                <div class="stat-num">{stats['subfields']}</div>
                <div class="stat-desc">子领域覆盖</div>
            </div>
            <div class="stat-item stat-card-red">
                <div class="stat-num">8</div>
                <div class="stat-desc">智能体数量</div>
            </div>
        </div>
    </div>

    <div style="text-align:center;padding:28px 0 12px;color:{LIGHT_GRAY};font-size:11px">
        基于 OpenClaw 多智能体协同框架设计 · 浙江大学人工智能基础 A 课程项目 · {date.today().isoformat()}
    </div>
    {_build_footer()}
    """


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

            # ── Tab 5: 跨天对比 ──
            with gr.TabItem("📊 跨天对比", id="compare"):
                available_dates = get_all_report_dates()
                d1 = available_dates[1] if len(available_dates) >= 2 else (available_dates[0] if available_dates else "")
                d2 = available_dates[0] if available_dates else ""

                with gr.Column():
                    with gr.Row():
                        compare_date1 = gr.Dropdown(
                            label="日期 A",
                            choices=available_dates,
                            value=d1,
                            interactive=True,
                        )
                        compare_date2 = gr.Dropdown(
                            label="日期 B",
                            choices=available_dates,
                            value=d2,
                            interactive=True,
                        )
                    compare_html = gr.HTML(value=build_compare_html(d1, d2))

                def _on_compare_change(d1, d2):
                    return build_compare_html(d1, d2)

                compare_date1.change(_on_compare_change, [compare_date1, compare_date2], compare_html)
                compare_date2.change(_on_compare_change, [compare_date1, compare_date2], compare_html)

            # ── Tab 6: 周报 ──
            with gr.TabItem("📝 学术周报", id="weekly"):
                weekly_html = gr.HTML(value=build_weekly_html())
                refresh_weekly_btn = gr.Button("↻ 生成/刷新周报", variant="primary")
                refresh_weekly_btn.click(
                    fn=lambda: build_weekly_html(),
                    outputs=weekly_html,
                )

            # ── Tab 7: 关于系统 ──
            with gr.TabItem("🏛️ 关于系统", id="about"):
                about_html = gr.HTML(value=build_about_html())
                refresh_about_btn = gr.Button("↻ 刷新", variant="primary")
                refresh_about_btn.click(
                    fn=lambda: build_about_html(),
                    outputs=about_html,
                )

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
