"""
Agent 7: Weekly Digest Agent
-----------------------------
Generates a weekly synthesis report by aggregating 7 days of
daily reports, identifying persistent trends, and highlighting
the week's most impactful papers.
"""

import os
import json
from datetime import date, timedelta, datetime
from collections import Counter, defaultdict
from openai import OpenAI

from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, REPORT_OUTPUT_DIR,
)
from database import (
    get_connection, get_papers_by_date, get_daily_report, get_daily_trend,
    get_all_report_dates,
)

_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

WEEKLY_PROMPT = """You are a research director summarizing a week of AI/CS progress.

## Daily Data
{daily_summaries}

## Task
Write a weekly research digest (Markdown, in Chinese) covering:

1. **本周概览** — total papers, daily trend, most active day
2. **持续热点方向 TOP 5** — themes that appeared consistently across multiple days
3. **本周突破性工作 TOP 5** — highest-rated papers of the week with brief commentary
4. **趋势变化** — themes rising/falling compared to prior week patterns
5. **下周展望** — what to watch for next week

Output directly in Markdown starting from "## 本周概览". No preamble."""


class WeeklyDigest:
    """Aggregates daily reports into a weekly synthesis."""

    def __init__(self):
        self.weekly_dir = os.path.join(REPORT_OUTPUT_DIR, "weekly")
        os.makedirs(self.weekly_dir, exist_ok=True)

    # -- public API --
    def generate(self, end_date: str = None) -> str:
        """Generate a weekly digest for the 7 days ending on end_date."""
        if end_date is None:
            end_date = date.today().isoformat()

        end = date.fromisoformat(end_date)
        start = end - timedelta(days=6)
        week_label = f"{start.isoformat()} ~ {end.isoformat()}"

        print(f"[Weekly Agent] generating digest: {week_label}")

        # collect daily data
        daily_data = self._gather_daily_data(start, end)
        if not daily_data:
            return f"# 周报 {week_label}\n\n本周暂无足够数据。\n"

        # build prompt and call LLM
        summary_text = self._format_daily_summaries(daily_data)
        try:
            digest_body = self._call_llm(
                WEEKLY_PROMPT.format(daily_summaries=summary_text)
            )
        except Exception as e:
            print(f"[Weekly Agent] LLM call failed: {e}")
            digest_body = self._fallback_report(daily_data)

        # assemble full report
        report = self._assemble_report(week_label, daily_data, digest_body)

        # save
        filepath = os.path.join(self.weekly_dir, f"weekly_{end_date}.md")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)
        self._persist(week_label, end_date, report)

        print(f"[Weekly Agent] saved: {filepath} ({len(report)} chars)")
        return report

    # -- internal --
    def _gather_daily_data(self, start: date, end: date) -> list[dict]:
        """Collect paper counts, top papers, and trend summaries for each day."""
        data = []
        d = start
        while d <= end:
            ds = d.isoformat()
            papers = get_papers_by_date(ds)
            trend = get_daily_trend(ds)

            if papers:
                top3 = sorted(papers, key=lambda x: float(x.get("importance", 0)),
                              reverse=True)[:3]
                dist = Counter(p.get("subfield", "") or "未分类" for p in papers)
                data.append({
                    "date": ds,
                    "weekday": ["周一","周二","周三","周四","周五","周六","周日"][d.weekday()],
                    "total": len(papers),
                    "top_papers": [
                        {"title": p.get("title",""), "importance": float(p.get("importance",0)),
                         "id": p["arxiv_id"]}
                        for p in top3
                    ],
                    "top_categories": dist.most_common(5),
                    "trend_summary": (trend.get("trend_summary", "") or "")[:400] if trend else "",
                })
            d += timedelta(days=1)
        return data

    def _format_daily_summaries(self, data: list[dict]) -> str:
        lines = []
        for day in data:
            lines.append(f"### {day['date']} ({day['weekday']}) — {day['total']}篇")
            lines.append(f"TOP方向: {', '.join(f'{c}({n}篇)' for c,n in day['top_categories'][:3])}")
            lines.append(f"TOP论文: {'; '.join(p['title'][:50] for p in day['top_papers'][:2])}")
            if day.get("trend_summary"):
                lines.append(f"趋势摘要: {day['trend_summary'][:200]}")
            lines.append("")
        return "\n".join(lines)

    def _call_llm(self, prompt, retries=3):
        import time
        for attempt in range(retries):
            try:
                resp = _client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[
                        {"role": "system", "content": "你是一位资深研究主管，请用中文输出结构化周报。"},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=3072,
                )
                return resp.choices[0].message.content
            except Exception as e:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)

    def _assemble_report(self, week_label, daily_data, body):
        total_papers = sum(d["total"] for d in daily_data)
        all_papers_list = []
        for d in daily_data:
            all_papers_list.extend(d["top_papers"])

        # collect all categories
        cat_counter = Counter()
        for d in daily_data:
            for cat, cnt in d["top_categories"]:
                cat_counter[cat] += cnt

        header = f"""# 📊 学术论文周报

**周期**: {week_label}
**总论文数**: {total_papers} 篇
**日均**: {total_papers // max(len(daily_data), 1)} 篇
**活跃天数**: {len(daily_data)} 天

---

"""
        return header + body + f"""

---

*本报告由 PaperPulse 周报汇总智能体自动生成*
*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*数据来源: OpenAlex | AI 模型: DeepSeek*
"""

    def _fallback_report(self, data):
        total = sum(d["total"] for d in data)
        cats = Counter()
        for d in data:
            for c, n in d["top_categories"]:
                cats[c] += n
        top_cats = cats.most_common(5)

        lines = [
            "## 本周概览",
            f"本周共采集 {total} 篇论文，覆盖 {len(data)} 天。",
            "",
            "## 持续热点方向",
        ]
        for cat, cnt in top_cats:
            lines.append(f"- **{cat}**: 累计 {cnt} 篇")
        lines.append("")
        lines.append("## 说明")
        lines.append("LLM 分析暂时不可用，以上为基于规则统计的周报摘要。")
        return "\n".join(lines)

    def _persist(self, week_label, end_date, report):
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS weekly_reports (
                    end_date TEXT PRIMARY KEY,
                    week_label TEXT,
                    report_content TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            cur.execute(
                "INSERT OR REPLACE INTO weekly_reports (end_date, week_label, report_content) VALUES (?, ?, ?)",
                (end_date, week_label, report),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[Weekly Agent] persist failed: {e}")


# -- module-level convenience --
_digest_engine = None

def generate_weekly_digest(end_date: str = None) -> str:
    """Generate a weekly digest report."""
    global _digest_engine
    if _digest_engine is None:
        _digest_engine = WeeklyDigest()
    return _digest_engine.generate(end_date)


def get_weekly_report(end_date: str = None) -> str:
    """Load an existing weekly report from DB."""
    if end_date is None:
        end_date = date.today().isoformat()
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT report_content FROM weekly_reports WHERE end_date = ?", (end_date,))
        row = cur.fetchone()
        conn.close()
        return row["report_content"] if row else None
    except Exception:
        return None
