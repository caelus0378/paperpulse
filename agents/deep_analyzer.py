"""
Agent 8: 论文深度解读智能体 (Deep Analyzer Agent)
------------------------------------------------
职责：每日选取最具代表性的一篇论文，进行全方位深度分析。
模拟资深研究员审读论文的过程——从问题背景、方法拆解、创新
亮点、局限性到影响力预判，生成一份可发表的深度解读报告。

使用 DeepSeek API 进行语义理解与结构化输出。
"""

import json
import time
from datetime import date
from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from database import get_connection

_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

DEEP_ANALYSIS_PROMPT = """你是一位资深 AI 研究员和学术论文审稿人。请对以下今日最具影响力的论文进行全方位深度解读。

## 论文信息
**标题**: {title}
**作者**: {authors}
**摘要**: {abstract}
**子领域**: {subfield}
**AI 评分**: {importance}/5.0
**质量评估**: 方法论 {q_methodology}/5, 创新性 {q_novelty}/5, 可复现性 {q_repro}/5, 写作 {q_clarity}/5

## 解读要求

请以 JSON 格式输出，包含以下字段：

```json
{{
  "tldr": "一句话速览（30字以内中文）",
  "problem": "这篇论文要解决什么核心问题？为什么重要？（100-150字）",
  "methodology": "方法拆解——用通俗中文解释技术路线，就像给研究生讲解一样。分步骤描述，每步解释'为什么这样做'。（200-300字）",
  "innovations": ["创新1（一句话）", "创新2（一句话）", "创新3（一句话，可选）"],
  "experiment_highlights": "实验中最值得关注的数据或发现（80-120字）",
  "limitations": "论文的局限性——包括作者自己提到的和你作为审稿人观察到的（60-100字）",
  "target_audience": "推荐哪些方向的研究者阅读？列举2-3个具体方向",
  "impact_prediction": "high/medium/low",
  "impact_reason": "给出影响力预判的理由（40-60字）",
  "full_analysis": "完整的深度解读 Markdown（500-800字），包含以上所有内容的结构化呈现，使用中文"
}}
```

## 注意事项
- 基于摘要和质量评估信息做分析，但可以结合你的领域知识做合理推断
- 语气专业但不枯燥，像资深研究员在实验室讨论
- 只输出 JSON，不要有任何其他内容
- full_analysis 字段用 Markdown 格式，包含 ## 标题层级"""


def analyze_paper(paper: dict, quality: dict = None) -> dict:
    """
    对单篇论文进行深度分析

    参数:
        paper: 论文字典（含 title, authors, abstract, subfield, importance）
        quality: 质量评估结果字典（可选，含 methodology, novelty, reproducibility, clarity）

    返回:
        dict: 深度分析结果
    """
    title = paper.get("title", "")
    authors = ", ".join((paper.get("authors") or [])[:5])
    abstract = paper.get("abstract", "")
    subfield = paper.get("subfield", "未知")
    importance = paper.get("importance", 0)

    # 质量评估数据（如果有）
    if quality:
        q_methodology = quality.get("methodology", "-")
        q_novelty = quality.get("novelty", "-")
        q_repro = quality.get("reproducibility", "-")
        q_clarity = quality.get("clarity", "-")
    else:
        q_methodology = q_novelty = q_repro = q_clarity = "-"

    prompt = DEEP_ANALYSIS_PROMPT.format(
        title=title,
        authors=authors,
        abstract=abstract if len(abstract) < 1500 else abstract[:1500] + "...",
        subfield=subfield,
        importance=importance,
        q_methodology=q_methodology,
        q_novelty=q_novelty,
        q_repro=q_repro,
        q_clarity=q_clarity,
    )

    print(f"[Deep Analyzer] 正在深度解读: {title[:60]}...")

    for attempt in range(3):
        try:
            resp = _client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": "你是一位严谨的 AI 研究员，擅长对学术论文进行深度解读。请严格按 JSON 格式输出。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=3072,
            )
            content = resp.choices[0].message.content
            result = _parse_json_response(content)
            print(f"[Deep Analyzer] ✅ 解读完成 ({len(result.get('full_analysis', ''))} 字)")
            return result
        except Exception as e:
            print(f"[Deep Analyzer] LLM 调用失败 (attempt {attempt+1}/3): {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise


def _parse_json_response(response: str) -> dict:
    """解析 LLM 返回的 JSON"""
    clean = response.strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    if clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]

    try:
        return json.loads(clean.strip())
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError("无法从 LLM 输出中提取有效 JSON")


def get_quality_for_paper(arxiv_id: str) -> dict:
    """从数据库获取某篇论文的质量评估"""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT methodology, novelty, reproducibility, clarity FROM quality_scores WHERE arxiv_id=?",
            (arxiv_id,),
        )
        row = cur.fetchone()
        conn.close()
        if row:
            return {
                "methodology": row["methodology"],
                "novelty": row["novelty"],
                "reproducibility": row["reproducibility"],
                "clarity": row["clarity"],
            }
    except Exception:
        pass
    return None


def run_deep_analysis(papers: list[dict], report_date: str = None) -> dict:
    """
    执行每日深度解读：选取今日评分最高的论文进行全方位分析

    参数:
        papers: 今日论文列表
        report_date: 分析日期

    返回:
        dict: 深度分析结果
    """
    if not papers:
        print("[Deep Analyzer] 无论文数据，跳过深度解读")
        return None

    if report_date is None:
        report_date = date.today().isoformat()

    # 选评分最高的论文（至少 2.5 分以上），如果全部低于 2.5 则选最高的
    scored = [p for p in papers if float(p.get("importance", 0)) >= 2.5]
    if not scored:
        scored = papers
    top_paper = max(scored, key=lambda x: float(x.get("importance", 0)))

    # 尝试获取质量评估数据
    quality = get_quality_for_paper(top_paper["arxiv_id"])

    print(f"[Deep Analyzer] 今日深度解读论文: {top_paper['title'][:60]}...")
    print(f"[Deep Analyzer] 评分: {top_paper.get('importance', 0)} | 子领域: {top_paper.get('subfield', '?')}")

    # 深度分析
    result = analyze_paper(top_paper, quality)

    # 持久化
    save_deep_analysis(report_date, top_paper["arxiv_id"], result)

    return result


# ============================================================
# 数据库 CRUD
# ============================================================

def save_deep_analysis(report_date: str, arxiv_id: str, analysis: dict):
    """保存深度分析结果"""
    innovations = analysis.get("innovations", [])
    if isinstance(innovations, list):
        innovations = json.dumps(innovations, ensure_ascii=False)
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO deep_analyses
            (report_date, arxiv_id, tldr, problem, methodology,
             innovations, experiment_highlights, limitations,
             target_audience, impact_prediction, impact_reason, full_analysis)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        report_date,
        arxiv_id,
        analysis.get("tldr", ""),
        analysis.get("problem", ""),
        analysis.get("methodology", ""),
        innovations,
        analysis.get("experiment_highlights", ""),
        analysis.get("limitations", ""),
        analysis.get("target_audience", ""),
        analysis.get("impact_prediction", "medium"),
        analysis.get("impact_reason", ""),
        analysis.get("full_analysis", ""),
    ))
    conn.commit()
    conn.close()
    print(f"[Deep Analyzer] 深度分析已入库: {report_date}")


def get_deep_analysis(report_date: str = None) -> dict:
    """获取某日的深度分析"""
    if report_date is None:
        report_date = date.today().isoformat()
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM deep_analyses WHERE report_date=?", (report_date,))
        row = cur.fetchone()
        conn.close()
        if row:
            d = dict(row)
            try:
                d["innovations"] = json.loads(d.get("innovations", "[]"))
            except (json.JSONDecodeError, TypeError):
                d["innovations"] = []
            return d
    except Exception:
        pass
    return None
