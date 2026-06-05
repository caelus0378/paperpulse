"""
智能体 1: 论文采集智能体 (Collector Agent)
数据源: OpenAlex API — 完全开放免费，不限量，覆盖全学科
"""
import requests
import time
from datetime import date

# OpenAlex 概念 ID（研究领域）
SEARCH_TOPICS = [
    ("Artificial Intelligence", "C154945302"),
    ("Computer Vision", "C31972630"),
    ("Natural Language Processing", "C204321447"),
    ("Machine Learning", "C119857082"),
    ("Robotics", "C127413603"),
    ("Reinforcement Learning", "C2780451532"),
    ("Deep Learning", "C108583219"),
    ("Large Language Models", "C2776919584"),
]

OPENALEX_API = "https://api.openalex.org/works"
MAX_PAGE = 30  # max results per page
# Fields we want: title, authors, abstract, DOI, date, concepts, open access URL
FIELDS = "title,authorships,abstract_inverted_index,doi,publication_date,concepts,primary_location,id"


def _reconstruct_abstract(inverted_index: dict | None) -> str:
    """OpenAlex 返回倒排索引格式的摘要，需重建"""
    if not inverted_index:
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort(key=lambda x: x[0])
    return " ".join(w for _, w in word_positions)


def _fetch_topic(topic: str, concept_id: str, limit: int = 25) -> list[dict]:
    """从 OpenAlex 获取指定主题的最新论文（只取最近 7 天）"""
    from datetime import timedelta
    papers = []
    max_attempts = 3
    from datetime import timedelta
    today = date.today()
    week_ago = today - timedelta(days=7)
    date_filter = (
        f"concepts.id:{concept_id},"
        f"from_publication_date:{week_ago.isoformat()},"
        f"to_publication_date:{today.isoformat()},"
        f"has_abstract:true"
    )

    for page in range(1, 5):
        if len(papers) >= limit:
            break

        for attempt in range(max_attempts):
            try:
                resp = requests.get(
                    OPENALEX_API,
                    params={
                        "filter": date_filter,
                        "sort": "publication_date:desc",
                        "per_page": min(MAX_PAGE, limit - len(papers)),
                        "page": page,
                        "select": FIELDS,
                    },
                    timeout=30,
                    headers={"User-Agent": "mailto:paperpulse@zju.edu.cn"},
                )
                if resp.status_code == 200:
                    break
                if resp.status_code == 429:
                    time.sleep(2)
                else:
                    print(f"    HTTP {resp.status_code}")
                    break
            except requests.RequestException:
                if attempt == max_attempts - 1:
                    return papers
                time.sleep(2)

        if resp.status_code != 200:
            continue

        data = resp.json()
        results = data.get("results", [])
        if not results:
            break

        for hit in results:
            pub_date = hit.get("publication_date") or ""
            # 跳过未来日期和太旧的
            if pub_date > today.isoformat() or pub_date < week_ago.isoformat():
                continue

            aid = hit.get("doi", "") or hit.get("id", "").split("/")[-1]
            authors = [
                a.get("author", {}).get("display_name", "Unknown")
                for a in (hit.get("authorships") or [])
            ]
            concepts = [
                c.get("display_name", "")
                for c in (hit.get("concepts") or [])
                if c.get("level") == 0
            ][:4]

            pdf_url = ""
            loc = hit.get("primary_location") or {}
            if loc:
                pdf_url = loc.get("landing_page_url", "") or ""

            papers.append({
                "arxiv_id": str(aid).replace("https://doi.org/", ""),
                "title": (hit.get("title") or "Untitled").strip(),
                "authors": authors,
                "abstract": _reconstruct_abstract(hit.get("abstract_inverted_index")),
                "categories": concepts,
                "primary_cat": concepts[0] if concepts else topic,
                "published_date": pub_date,
                "pdf_url": pdf_url,
                "collected_date": today.isoformat(),
            })

        time.sleep(0.5)

    return papers[:limit]


def collect_papers(topics: list = None) -> list[dict]:
    """主采集函数"""
    if topics is None:
        topics = SEARCH_TOPICS

    all_papers = []
    seen_ids = set()

    print(f"[Collector Agent] 数据源: OpenAlex (完全免费，不限量)")
    print(f"[Collector Agent] 搜索主题: {len(topics)} 个")

    for topic_name, concept_id in topics:
        print(f"  -> {topic_name} ...", end=" ", flush=True)
        try:
            batch = _fetch_topic(topic_name, concept_id, limit=25)
            added = 0
            for p in batch:
                if p["arxiv_id"] not in seen_ids:
                    seen_ids.add(p["arxiv_id"])
                    all_papers.append(p)
                    added += 1
            print(f"获取 {len(batch)} 篇 (新增 {added})")
        except Exception as e:
            print(f"失败: {e}")

        if len(all_papers) >= 200:
            print(f"  → 已达 200 篇上限")
            break

    print(f"[Collector Agent] 采集完成: {len(all_papers)} 篇")

    if all_papers:
        all_papers.sort(key=lambda x: x.get("published_date", ""), reverse=True)

    return all_papers


def format_paper_brief(paper: dict) -> str:
    authors_str = ", ".join(paper.get("authors", [])[:3])
    if len(paper.get("authors", [])) > 3:
        authors_str += " et al."
    return (
        f"**{paper['title']}**  \n"
        f"  📎 `{paper['arxiv_id']}` | {authors_str}  \n"
        f"  📂 {paper.get('primary_cat', 'N/A')} | 📅 {paper.get('published_date', 'N/A')}  \n"
    )


def format_papers_summary(papers: list[dict]) -> str:
    lines = []
    for i, p in enumerate(papers[:50], 1):
        lines.append(
            f"{i}. [{p['arxiv_id']}] {p['title']}\n"
            f"   Categories: {', '.join(p.get('categories', []))}\n"
            f"   Abstract: {p.get('abstract', '')[:300]}..."
        )
    return "\n\n".join(lines)


if __name__ == "__main__":
    p = collect_papers()
    for x in p[:5]:
        print(format_paper_brief(x))
        print()
