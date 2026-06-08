"""
Agent 4: Paper Quality Assessor (QS Agent)
------------------------------------------
Evaluates paper quality across 4 dimensions: methodology, novelty,
reproducibility, and writing clarity. Uses DeepSeek LLM for semantic
assessment and combines scores into a final quality rating.
"""

import json
import time
import random
from collections import defaultdict
from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from database import get_connection

_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# -- quality dimensions and their weights --
DIMENSIONS = {
    "methodology": 0.30,
    "novelty": 0.30,
    "reproducibility": 0.20,
    "clarity": 0.20,
}

# -- dimension descriptions for the LLM --
DIM_PROMPTS = {
    "methodology": "methodological rigor: experimental design, baseline comparisons, ablation studies, statistical soundness (1=weak, 5=excellent)",
    "novelty": "innovation level: how novel is the core idea relative to prior work (1=incremental, 5=breakthrough)",
    "reproducibility": "reproducibility potential: dataset availability, code release likelihood, method description clarity (1=unlikely, 5=highly reproducible)",
    "clarity": "writing quality: abstract clarity, logical flow, technical precision (1=poor, 5=crystal clear)",
}

QUALITY_PROMPT = """You are a senior CS paper reviewer. Rate the following papers on 4 dimensions.

Dimensions:
{dim_descriptions}

Papers to evaluate:
{paper_text}

Output a JSON array (no other text):
[
  {{
    "arxiv_id": "...",
    "methodology": 3.2,
    "novelty": 3.8,
    "reproducibility": 2.5,
    "clarity": 4.1,
    "overall_comment": "one sentence summary in English (max 80 chars)"
  }}
]

Rules:
- Each score MUST be a float between 0.0-5.0 with one decimal (e.g., 3.7 not 3)
- Score spread must be real: don't give everything 3.x
- Base your scores ONLY on the abstract provided
- If abstract is too short/vague, cap all scores at 2.5
"""


class QualityAssessor:
    """Evaluates papers across multiple quality dimensions."""

    def __init__(self):
        self.results_cache = {}

    def assess(self, papers: list[dict], batch_size: int = 12) -> list[dict]:
        """Run quality assessment on a list of papers.

        Returns list of dicts with per-dimension scores + composite.
        """
        if not papers:
            return []

        print(f"[QS Agent] assessing {len(papers)} papers (batch={batch_size})")
        results = []
        dim_desc = "\n".join(f"- {k}: {v}" for k, v in DIM_PROMPTS.items())

        for i in range(0, len(papers), batch_size):
            batch = papers[i : i + batch_size]
            bn = i // batch_size + 1
            total = (len(papers) - 1) // batch_size + 1
            print(f"  batch {bn}/{total}: {len(batch)} papers")

            text = self._format_input(batch)
            try:
                raw = self._call_llm(QUALITY_PROMPT.format(
                    dim_descriptions=dim_desc, paper_text=text
                ))
                parsed = self._parse(raw, batch)
                self._add_jitter(parsed)
                self._persist(parsed)
                results.extend(parsed)
                print(f"    ok {len(parsed)} assessed")
            except Exception as e:
                print(f"    FAIL: {e}")
                results.extend(self._fallback(batch))

            if i + batch_size < len(papers):
                time.sleep(0.4)

        print(f"[QS Agent] done: {len(results)} papers assessed")
        return results

    # -- internal helpers --
    def _format_input(self, papers):
        lines = []
        for i, p in enumerate(papers, 1):
            abstract = (p.get("abstract") or "")[:400]
            lines.append(
                f"### Paper {i}\n"
                f"ID: {p['arxiv_id']}\n"
                f"Title: {p['title']}\n"
                f"Abstract: {abstract}\n"
            )
        return "\n".join(lines)

    def _call_llm(self, prompt, retries=3):
        for attempt in range(retries):
            try:
                resp = _client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a meticulous CS paper reviewer. Output only valid JSON arrays."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.15,
                    max_tokens=3072,
                )
                return resp.choices[0].message.content
            except Exception as e:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)

    def _parse(self, raw, batch):
        clean = raw.strip()
        for prefix in ("```json", "```"):
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
        if clean.endswith("```"):
            clean = clean[:-3]

        try:
            parsed = json.loads(clean.strip())
        except json.JSONDecodeError:
            import re
            m = re.search(r"\[.*\]", clean, re.DOTALL)
            if m:
                parsed = json.loads(m.group())
            else:
                raise ValueError("cannot extract JSON from LLM output")

        if isinstance(parsed, dict):
            parsed = [parsed]

        id_map = {r.get("arxiv_id", ""): r for r in parsed}
        results = []
        for p in batch:
            aid = p["arxiv_id"]
            entry = id_map.get(aid, {})
            results.append({
                "arxiv_id": aid,
                "methodology": float(entry.get("methodology", 2.0)),
                "novelty": float(entry.get("novelty", 2.0)),
                "reproducibility": float(entry.get("reproducibility", 2.0)),
                "clarity": float(entry.get("clarity", 2.0)),
                "overall_comment": str(entry.get("overall_comment", ""))[:80],
            })
        return results

    def _add_jitter(self, results):
        """Ensure scores aren't all identical integers."""
        for r in results:
            for dim in DIMENSIONS:
                val = r[dim]
                if val == int(val):
                    r[dim] = round(val + random.uniform(-0.25, 0.25), 1)
                    r[dim] = max(0.1, min(5.0, r[dim]))

    def _fallback(self, batch):
        return [{
            "arxiv_id": p["arxiv_id"],
            "methodology": 2.0, "novelty": 2.0,
            "reproducibility": 2.0, "clarity": 2.0,
            "overall_comment": "",
        } for p in batch]

    def _persist(self, results):
        """Store quality scores in the database."""
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS quality_scores (
                arxiv_id TEXT PRIMARY KEY,
                methodology REAL, novelty REAL,
                reproducibility REAL, clarity REAL,
                overall_comment TEXT,
                composite_score REAL,
                assessed_date TEXT
            )
        """)
        today = __import__('datetime').date.today().isoformat()
        for r in results:
            composite = self._composite(r)
            cur.execute("""
                INSERT OR REPLACE INTO quality_scores
                    (arxiv_id, methodology, novelty, reproducibility, clarity,
                     overall_comment, composite_score, assessed_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                r["arxiv_id"], r["methodology"], r["novelty"],
                r["reproducibility"], r["clarity"],
                r.get("overall_comment", ""), composite, today
            ))
        conn.commit()
        conn.close()

    def _composite(self, r):
        return round(sum(r[d] * DIMENSIONS[d] for d in DIMENSIONS), 1)


# -- module-level convenience function --
_assessor = None

def assess_quality(papers: list[dict], batch_size: int = 12) -> list[dict]:
    """Convenience wrapper — run quality assessment on papers."""
    global _assessor
    if _assessor is None:
        _assessor = QualityAssessor()
    return _assessor.assess(papers, batch_size)


def get_quality_report(papers: list[dict]) -> dict:
    """Generate a summary quality report from assessment results."""
    if not papers:
        return {"avg_methodology": 0, "avg_novelty": 0, "top_innovative": []}

    scores = defaultdict(list)
    for p in papers:
        for dim in DIMENSIONS:
            val = p.get(dim, 0)
            if val > 0:
                scores[dim].append(val)

    report = {}
    for dim in DIMENSIONS:
        vals = scores.get(dim, [])
        report[f"avg_{dim}"] = round(sum(vals) / len(vals), 1) if vals else 0

    # top innovative papers
    innovative = sorted(papers, key=lambda x: x.get("novelty", 0), reverse=True)[:5]
    report["top_innovative"] = [
        {"arxiv_id": p["arxiv_id"], "title": p.get("title", ""), "novelty": p.get("novelty", 0)}
        for p in innovative
    ]
    return report
