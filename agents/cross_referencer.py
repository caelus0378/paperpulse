"""
Agent 5: Cross-Reference & Relation Mining Agent
------------------------------------------------
职责：构建论文之间的关联图谱，发现跨领域研究热点
通过关键词共现 + 语义相似度挖掘论文间的隐含联系
"""

import re
import time
import json
import math
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Callable

import numpy as np

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from database import get_papers_by_date, get_connection

## ============================================================
## 文本预处理工具（函数式风格）
## ============================================================

STOP_WORDS_EN = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "shall",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after",
    "this", "that", "these", "those", "it", "its", "they",
    "them", "their", "we", "our", "and", "but", "or", "not",
    "no", "nor", "so", "if", "then", "than", "too", "very",
    "just", "also", "now", "here", "there", "when", "where",
    "how", "all", "both", "each", "few", "more", "most",
    "other", "some", "such", "only", "own", "same", "new",
    "well", "however", "therefore", "thus", "yet", "still",
    "using", "used", "based", "show", "shows", "shown",
    "propose", "proposed", "method", "model", "approach",
    "paper", "result", "results", "work", "study", "data",
    "training", "test", "performance", "state",
}


def tokenize(text: str) -> list[str]:
    """英文分词 + 清洗"""
    text = re.sub(r"\$[^$]+\$", " ", text)        # 去除 LaTeX 公式
    text = re.sub(r"\\[a-zA-Z]+", " ", text)       # 去除 LaTeX 命令
    text = re.sub(r"[^a-zA-Z\s]", " ", text)       # 只保留英文字母
    tokens = text.lower().split()
    return [t for t in tokens if len(t) >= 3 and t not in STOP_WORDS_EN]


def extract_ngrams(tokens: list[str], n: int = 2) -> list[str]:
    """提取 n-gram 词组"""
    if len(tokens) < n:
        return []
    return ["_".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def build_tfidf(documents: list[list[str]]) -> dict:
    """
    构建 TF-IDF 词频矩阵（精简实现，无 scikit-learn 依赖）
    返回 {term: [doc_scores]} 和 idf 值
    """
    N = len(documents)
    if N == 0:
        return {}

    # 文档频率
    df = Counter()
    for doc in documents:
        unique_terms = set(doc)
        for term in unique_terms:
            df[term] += 1

    # IDF
    idf = {term: math.log((N + 1) / (df[term] + 1)) + 1 for term in df}

    # TF-IDF 矩阵
    tfidf_matrix = {}
    for term in df:
        tfidf_matrix[term] = []
        for doc in documents:
            tf = doc.count(term) / max(len(doc), 1)
            tfidf_matrix[term].append(tf * idf[term])

    return {"matrix": tfidf_matrix, "idf": idf, "doc_count": N}


## ============================================================
## 装饰器：性能监控
## ============================================================

def timed(func: Callable) -> Callable:
    """记录函数执行时间"""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"  [timing] {func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper


## ============================================================
## 论文相似度计算
## ============================================================

@timed
def compute_similarity_matrix(papers: list[dict]) -> np.ndarray:
    """
    基于 TF-IDF 的余弦相似度矩阵
    每篇论文用其 title + abstract 的 token 表示
    """
    n = len(papers)
    if n <= 1:
        return np.zeros((n, n))

    # 构建文档词袋
    docs = []
    for p in papers:
        title_tokens = tokenize(p.get("title", ""))
        abstract_tokens = tokenize(p.get("abstract", ""))
        # 同时提取 unigrams 和 bigrams
        all_tokens = title_tokens * 2 + abstract_tokens  # title 加权
        all_tokens += extract_ngrams(title_tokens, 2) + extract_ngrams(abstract_tokens, 2)
        docs.append(all_tokens)

    # 构建词汇表（取前5000高频词限制内存）
    word_freq = Counter()
    for doc in docs:
        word_freq.update(doc)
    vocab = [w for w, _ in word_freq.most_common(5000)]
    vocab_idx = {w: i for i, w in enumerate(vocab)}

    # 构建稀疏向量
    vectors = np.zeros((n, len(vocab)))
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab_idx:
                vectors[i, vocab_idx[token]] += 1

    # TF-IDF 加权
    df = np.sum(vectors > 0, axis=0)
    idf = np.log((n + 1) / (df + 1)) + 1
    vectors = vectors * idf

    # L2 归一化
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors = vectors / norms

    # 余弦相似度
    sim_matrix = vectors @ vectors.T
    np.fill_diagonal(sim_matrix, 0)  # 去除自相似

    return sim_matrix


## ============================================================
## 关联图谱构建
## ============================================================

@lru_cache(maxsize=128)
def _get_cached_papers(date_str: str) -> tuple:
    """缓存论文查询结果（不可变 tuple 可哈希）"""
    papers = get_papers_by_date(date_str)
    return tuple(papers)


def build_relation_graph(papers: list[dict], top_k: int = 5) -> dict:
    """
    构建论文关联图谱

    Returns:
        {
            "nodes": [{id, title, subfield, importance}],
            "edges": [{source, target, weight, relation_type}],
            "clusters": [{name, members, description}]
        }
    """
    if len(papers) < 2:
        return {"nodes": [], "edges": [], "clusters": []}

    print(f"[XR Agent] building relation graph for {len(papers)} papers")

    # 节点
    nodes = [
        {
            "id": p["arxiv_id"],
            "title": (p.get("title") or "Untitled")[:80],
            "subfield": p.get("subfield", "未分类"),
            "importance": float(p.get("importance", 0)),
        }
        for p in papers
    ]

    # 计算相似度矩阵
    sim = compute_similarity_matrix(papers)

    # 提取 Top-K 边
    edges = []
    n = len(papers)
    for i in range(n):
        # 每篇论文找 top_k 最相似的
        row = sim[i]
        top_indices = np.argsort(row)[::-1][:top_k]
        for j in top_indices:
            if row[j] > 0.15:  # 相似度阈值
                # 判断关系类型
                same_field = (
                    papers[i].get("subfield") == papers[j].get("subfield")
                )
                rel_type = "同领域" if same_field else "跨领域关联"
                edges.append({
                    "source": papers[i]["arxiv_id"],
                    "target": papers[j]["arxiv_id"],
                    "weight": round(float(row[j]), 3),
                    "relation_type": rel_type,
                })

    # 简单的社区发现：按子领域聚类
    clusters = _detect_clusters(papers)

    result = {"nodes": nodes, "edges": edges, "clusters": clusters}
    print(f"[XR Agent] graph: {len(nodes)} nodes, {len(edges)} edges, {len(clusters)} clusters")
    return result


def _detect_clusters(papers: list[dict]) -> list[dict]:
    """按子领域 + 关键词共现进行简单聚类"""
    by_field = defaultdict(list)
    for p in papers:
        sf = p.get("subfield", "未分类")
        by_field[sf].append(p)

    clusters = []
    for field, field_papers in by_field.items():
        if len(field_papers) >= 3:
            # 提取该领域的关键主题
            all_kw = []
            for p in field_papers:
                all_kw.extend(tokenize(p.get("abstract", "")))
            top_kw = [kw for kw, _ in Counter(all_kw).most_common(5)]

            clusters.append({
                "name": field,
                "member_count": len(field_papers),
                "members": [p["arxiv_id"] for p in field_papers[:10]],
                "keywords": top_kw,
            })

    clusters.sort(key=lambda x: x["member_count"], reverse=True)
    return clusters


## ============================================================
## 论文推荐（基于关联图）
## ============================================================

def recommend_related(
    target_arxiv_id: str,
    papers: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    给定一篇论文 ID，推荐最相关的 top_k 篇论文
    基于 TF-IDF 余弦相似度
    """
    # 找到目标论文索引
    target_idx = None
    for i, p in enumerate(papers):
        if p["arxiv_id"] == target_arxiv_id:
            target_idx = i
            break

    if target_idx is None:
        return []

    sim = compute_similarity_matrix(papers)
    row = sim[target_idx]
    top_indices = np.argsort(row)[::-1][:top_k]

    recommendations = []
    for idx in top_indices:
        if row[idx] > 0.1:
            recommendations.append({
                "arxiv_id": papers[idx]["arxiv_id"],
                "title": papers[idx].get("title", ""),
                "similarity": round(float(row[idx]), 3),
                "subfield": papers[idx].get("subfield", ""),
            })

    return recommendations


## ============================================================
## 交叉引用分析报告
## ============================================================

def generate_cross_ref_report(papers: list[dict]) -> str:
    """
    生成交叉引用分析报告（Markdown 格式）
    包含：领域聚类、跨领域热点、论文关联网络统计
    """
    if not papers:
        return "## 交叉引用分析\n\n暂无足够数据。\n"

    graph = build_relation_graph(papers)

    sections = ["## 六、论文关联分析\n"]

    # 聚类概览
    sections.append("### 研究领域聚类\n")
    for c in graph["clusters"][:6]:
        kw_str = "、".join(c["keywords"][:3])
        sections.append(
            f"- **{c['name']}**（{c['member_count']} 篇）→ 热点关键词: {kw_str}"
        )
    sections.append("")

    # 跨领域关联
    cross_field_edges = [e for e in graph["edges"] if "跨领域" in e.get("relation_type", "")]
    if cross_field_edges:
        sections.append("### 跨领域关联发现\n")
        # 按权重排序
        cross_field_edges.sort(key=lambda x: x["weight"], reverse=True)
        seen_pairs = set()
        for e in cross_field_edges[:5]:
            pair = tuple(sorted([e["source"], e["target"]]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                sections.append(
                    f"- `{e['source'][:20]}...` ↔ `{e['target'][:20]}...` "
                    f"（相似度: {e['weight']:.3f}）"
                )
        sections.append("")

    # 网络统计
    sections.append("### 关联网络统计\n")
    sections.append(f"- 论文节点: {len(graph['nodes'])}")
    sections.append(f"- 关联边数: {len(graph['edges'])}")
    sections.append(f"- 研究聚类: {len(graph['clusters'])}")
    cross_count = len([e for e in graph["edges"] if "跨领域" in e.get("relation_type", "")])
    sections.append(f"- 跨领域链接: {cross_count}")

    if graph["edges"]:
        avg_weight = sum(e["weight"] for e in graph["edges"]) / len(graph["edges"])
        sections.append(f"- 平均关联强度: {avg_weight:.3f}")

    sections.append("")
    return "\n".join(sections)


## ============================================================
## 主入口（集成到流水线）
## ============================================================

def run_cross_reference(papers: list[dict], report_date: str = None) -> dict:
    """
    执行完整的交叉引用分析

    Returns:
        dict with graph data and report
    """
    if report_date is None:
        from datetime import date
        report_date = date.today().isoformat()

    print(f"[XR Agent] starting cross-reference analysis for {report_date}")

    if not papers:
        papers_list = list(_get_cached_papers(report_date))
        if not papers_list:
            print("[XR Agent] no papers found")
            return {"graph": None, "report": ""}
    else:
        papers_list = papers

    graph = build_relation_graph(papers_list)
    report = generate_cross_ref_report(papers_list)

    # persist graph to DB
    _save_graph(report_date, graph)

    print(f"[XR Agent] analysis complete")
    return {"graph": graph, "report": report, "date": report_date}


def _save_graph(date_str: str, graph: dict):
    """持久化关联图谱到数据库"""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS cross_ref_graphs (
                report_date TEXT PRIMARY KEY,
                graph_json TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        cur.execute(
            "INSERT OR REPLACE INTO cross_ref_graphs (report_date, graph_json) VALUES (?, ?)",
            (date_str, json.dumps(graph, ensure_ascii=False)),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[XR Agent] failed to save graph: {e}")
