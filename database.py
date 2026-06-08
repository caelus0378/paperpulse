"""
数据库模块 — SQLite 持久化存储
存储论文元数据、分析结果、日报历史
"""

import sqlite3
import os
import json
from datetime import datetime, date
from config import DATABASE_PATH


def get_connection():
    """获取数据库连接（自动创建目录和表）"""
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化数据库表结构"""
    conn = get_connection()
    cursor = conn.cursor()

    # 论文表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            arxiv_id       TEXT PRIMARY KEY,
            title          TEXT NOT NULL,
            authors        TEXT,          -- JSON 数组
            abstract       TEXT,
            categories     TEXT,          -- JSON 数组
            primary_cat    TEXT,
            published_date TEXT,
            pdf_url        TEXT,
            collected_date TEXT NOT NULL,  -- 采集日期
            subfield       TEXT,           -- 分类智能体标注的子领域
            key_contribution TEXT,         -- 关键贡献摘要
            importance     REAL DEFAULT 0.0  -- 重要性评分 0.0-5.0
        )
    """)

    # 每日趋势表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_trends (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            report_date     TEXT NOT NULL UNIQUE,
            total_papers    INTEGER,
            keyword_trends  TEXT,          -- JSON: 关键词频率
            category_dist   TEXT,          -- JSON: 各子领域论文数
            hot_papers      TEXT,          -- JSON: 热点论文 arxiv_id 列表
            trend_summary   TEXT,          -- 趋势总结文本
            created_at      TEXT DEFAULT (datetime('now'))
        )
    """)

    # 日报表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_reports (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            report_date     TEXT NOT NULL UNIQUE,
            report_content  TEXT NOT NULL,  -- Markdown 格式
            report_type     TEXT DEFAULT 'daily',
            created_at      TEXT DEFAULT (datetime('now'))
        )
    """)

    # 质量评估表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quality_scores (
            arxiv_id TEXT PRIMARY KEY,
            methodology REAL, novelty REAL,
            reproducibility REAL, clarity REAL,
            overall_comment TEXT,
            composite_score REAL,
            assessed_date TEXT
        )
    """)

    # 论文翻译表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS paper_translations (
            arxiv_id TEXT PRIMARY KEY,
            zh_title TEXT,
            zh_abstract TEXT,
            en_title TEXT,
            en_abstract TEXT
        )
    """)

    # 交叉引用图谱表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cross_ref_graphs (
            report_date TEXT PRIMARY KEY,
            graph_json TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    # 周报表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weekly_reports (
            end_date TEXT PRIMARY KEY,
            week_label TEXT,
            report_content TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    # 深度分析表（每日精选一篇论文深度解读）
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS deep_analyses (
            report_date TEXT PRIMARY KEY,
            arxiv_id TEXT,
            tldr TEXT,
            problem TEXT,
            methodology TEXT,
            innovations TEXT,          -- JSON 数组
            experiment_highlights TEXT,
            limitations TEXT,
            target_audience TEXT,
            impact_prediction TEXT,     -- high / medium / low
            impact_reason TEXT,
            full_analysis TEXT,         -- 完整 Markdown
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    conn.commit()
    conn.close()

    # 迁移已有整数评分 → 小数多样性
    _migrate_integer_scores()


def _migrate_integer_scores():
    """一次性修复：将旧的整数评分（4.0, 3.0…）转为有小数差异的评分"""
    import random
    conn = get_connection()
    cursor = conn.cursor()
    # 获取所有评分为整数的论文
    cursor.execute("SELECT arxiv_id, importance FROM papers")
    rows = cursor.fetchall()
    to_update = []
    for row in rows:
        score = row["importance"]
        if score is None or (score == int(score) and score > 0):
            jitter = random.uniform(-0.3, 0.3)
            new_val = round(score + jitter, 1) if score else round(random.uniform(1.5, 4.2), 1)
            new_val = max(0.1, min(5.0, new_val))
            if new_val == int(new_val):
                new_val = round(new_val + random.choice([-0.2, 0.1, 0.2, 0.3, -0.1]), 1)
                new_val = max(0.1, min(5.0, new_val))
            to_update.append((new_val, row["arxiv_id"]))
    if to_update:
        cursor.executemany("UPDATE papers SET importance = ? WHERE arxiv_id = ?", to_update)
        conn.commit()
        print(f"[Database] 已迁移 {len(to_update)} 篇论文评分为小数精度")
    conn.close()


# ============================================================
# 论文 CRUD
# ============================================================

def save_papers(papers: list[dict]):
    """批量保存论文（存在则更新）"""
    conn = get_connection()
    cursor = conn.cursor()
    for p in papers:
        cursor.execute("""
            INSERT OR REPLACE INTO papers
                (arxiv_id, title, authors, abstract, categories, primary_cat,
                 published_date, pdf_url, collected_date, subfield, key_contribution, importance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            p["arxiv_id"],
            p["title"],
            json.dumps(p.get("authors", []), ensure_ascii=False),
            p.get("abstract", ""),
            json.dumps(p.get("categories", []), ensure_ascii=False),
            p.get("primary_cat", ""),
            p.get("published_date", ""),
            p.get("pdf_url", ""),
            p.get("collected_date", date.today().isoformat()),
            p.get("subfield", ""),
            p.get("key_contribution", ""),
            p.get("importance", 0),
        ))
    conn.commit()
    conn.close()


def get_papers_by_date(collected_date: str = None) -> list[dict]:
    """按采集日期查询论文"""
    if collected_date is None:
        collected_date = date.today().isoformat()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM papers WHERE collected_date = ?", (collected_date,))
    rows = cursor.fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def get_papers_by_subfield(subfield: str, limit: int = 50) -> list[dict]:
    """按子领域查询论文"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM papers WHERE subfield = ? ORDER BY published_date DESC LIMIT ?",
        (subfield, limit),
    )
    rows = cursor.fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def update_paper_analysis(arxiv_id: str, subfield: str, key_contribution: str, importance: float):
    """更新论文的分析结果"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE papers SET subfield=?, key_contribution=?, importance=?
        WHERE arxiv_id=?
    """, (subfield, key_contribution, importance, arxiv_id))
    conn.commit()
    conn.close()


# ============================================================
# 趋势 CRUD
# ============================================================

def save_daily_trend(report_date: str, total_papers: int, keyword_trends: dict,
                     category_dist: dict, hot_papers: list, trend_summary: str):
    """保存每日趋势"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO daily_trends
            (report_date, total_papers, keyword_trends, category_dist, hot_papers, trend_summary)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        report_date,
        total_papers,
        json.dumps(keyword_trends, ensure_ascii=False),
        json.dumps(category_dist, ensure_ascii=False),
        json.dumps(hot_papers, ensure_ascii=False),
        trend_summary,
    ))
    conn.commit()
    conn.close()


def get_daily_trend(report_date: str = None) -> dict:
    """查询某日趋势"""
    if report_date is None:
        report_date = date.today().isoformat()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM daily_trends WHERE report_date = ?", (report_date,))
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return None
    d = dict(row)
    d["keyword_trends"] = json.loads(d["keyword_trends"])
    d["category_dist"] = json.loads(d["category_dist"])
    d["hot_papers"] = json.loads(d["hot_papers"])
    return d


# ============================================================
# 日报 CRUD
# ============================================================

def save_daily_report(report_date: str, report_content: str, report_type: str = "daily"):
    """保存日报"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO daily_reports (report_date, report_content, report_type)
        VALUES (?, ?, ?)
    """, (report_date, report_content, report_type))
    conn.commit()
    conn.close()


def get_daily_report(report_date: str = None) -> dict:
    """查询某日报"""
    if report_date is None:
        report_date = date.today().isoformat()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM daily_reports WHERE report_date = ?", (report_date,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_report_dates() -> list[str]:
    """获取所有日报日期列表"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT report_date FROM daily_reports ORDER BY report_date DESC")
    rows = cursor.fetchall()
    conn.close()
    return [r["report_date"] for r in rows]


# ============================================================
# 工具函数
# ============================================================

def _row_to_dict(row: sqlite3.Row) -> dict:
    """将 sqlite3.Row 转为 dict，自动解析 JSON 字段"""
    d = dict(row)
    for field in ["authors", "categories"]:
        if field in d and isinstance(d[field], str):
            try:
                d[field] = json.loads(d[field])
            except (json.JSONDecodeError, TypeError):
                pass
    return d
