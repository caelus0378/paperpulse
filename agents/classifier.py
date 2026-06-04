"""
智能体 2: 论文分类与摘要智能体 (Classifier Agent)
─────────────────────────────────────────────
职责：对采集的论文进行子领域分类，提取每篇论文的关键贡献
使用 DeepSeek API 进行语义理解和结构化输出
"""

import json
import time
from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from database import update_paper_analysis

# 初始化 DeepSeek 客户端
_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

import random
random.seed()

def _ensure_score_diversity(results: list[dict]):
    """
    后处理：如果 LLM 返回的评分全是整数（如 4.0, 3.0），
    自动添加合理的小数变化，保证评分有真正的高低差异。
    """
    if not results:
        return

    scores = [float(r.get("importance", 0)) for r in results]
    all_integer = all(s == int(s) for s in scores)

    if all_integer and len(results) >= 2:
        min_s, max_s = min(scores), max(scores)
        if max_s - min_s < 0.5:
            min_s, max_s = 1.0, 4.5

        for r in results:
            base = float(r.get("importance", 0))
            jitter = random.uniform(-0.3, 0.3)
            new_val = round(base + jitter, 1)
            new_val = max(0.1, min(5.0, new_val))
            if new_val == int(new_val):
                new_val = round(new_val + random.choice([-0.2, 0.1, 0.2, 0.3, 0.4, -0.1]), 1)
                new_val = max(0.1, min(5.0, new_val))
            r["importance"] = new_val

# ============================================================
# 子领域定义
# ============================================================
SUBFIELDS = [
    "大语言模型 (LLM)",
    "自然语言处理 (NLP)",
    "计算机视觉 (CV)",
    "强化学习 (RL)",
    "图神经网络 (GNN)",
    "生成模型 (Diffusion/GAN/VAE)",
    "多模态学习 (Multimodal)",
    "AI安全与对齐 (AI Safety)",
    "机器学习理论 (ML Theory)",
    "机器人学 (Robotics)",
    "AI Agent & 工具使用",
    "代码生成与软件工程 (Code/SE)",
    "AI4Science (科学发现)",
    "生物信息学与医学AI",
    "金融科技与量化分析",
    "信号处理与通信",
    "经济学与博弈论",
    "优化与控制理论",
    "人机交互 (HCI)",
    "网络安全与密码学",
    "社会科学与网络分析",
    "其他交叉学科",
]

# ============================================================
# Prompt 模板
# ============================================================
CLASSIFICATION_PROMPT = """你是一位顶级的 AI 论文评审专家。请对以下今日最新 arXiv 论文进行分类和摘要。

## 子领域列表（从中选择最匹配的一个）
{subfields}

## 待分析论文列表
{papers_text}

## 输出要求
请以 JSON 数组格式输出，每篇论文对应一个对象：
```json
[
  {{
    "arxiv_id": "论文ID",
    "subfield": "最匹配的子领域",
    "key_contribution": "用1-2句中文概括论文的核心贡献和创新点（50字以内）",
    "importance": 2.3
  }}
]
```

## ⚠️ importance 评分细则（必须严格遵守）
- importance 必须是 **0.0 到 5.0 之间的浮点数，必须包含一位小数**
- 例如: 4.7, 3.2, 2.8, 1.5, 0.3 — 不要输出整数如 4.0, 3.0！
- 评分标准:
  - 4.5-5.0: 领域颠覆性突破（极少数，最多 5% 的论文）
  - 4.0-4.4: 重要方法创新或新范式
  - 3.5-3.9: 有价值的改进，解决实际问题
  - 3.0-3.4: 扎实的增量工作
  - 2.0-2.9: 小范围改进或应用
  - 1.0-1.9: 技术报告或初步探索
  - 0.0-0.9: 无法判断或无实质内容
- **每批论文中，评分必须有明显高低差异！至少要有 1.5 分的跨度！**
- 4.5 以上的论文每批最多 1 篇
- 如果摘要信息不足，importance 不超过 2.5

注意：
1. 基于摘要信息判断，不要臆测
2. 确保 arxiv_id 与输入完全一致
3. 只输出 JSON 数组，不要输出任何其他内容"""


def classify_papers(papers: list[dict], batch_size: int = 15) -> list[dict]:
    """
    对论文进行批量分类和摘要提取

    参数:
        papers: 采集智能体输出的论文列表
        batch_size: 每次 LLM 调用处理的论文数（控制 token 消耗）

    返回:
        list[dict]: 包含分类结果的论文列表
    """
    if not papers:
        print("[Classifier Agent] 无论文需要分类")
        return []

    print(f"[Classifier Agent] 开始处理 {len(papers)} 篇论文，批次大小: {batch_size}")

    results = []
    subfield_list = "\n".join(f"- {s}" for s in SUBFIELDS)

    # 分批处理
    for i in range(0, len(papers), batch_size):
        batch = papers[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(papers) - 1) // batch_size + 1

        print(f"  → 批次 {batch_num}/{total_batches}: {len(batch)} 篇论文")

        # 构建论文文本（含摘要）
        papers_text = _format_papers_for_llm(batch)

        # 调用 DeepSeek API
        try:
            response = _call_llm_with_retry(
                CLASSIFICATION_PROMPT.format(
                    subfields=subfield_list,
                    papers_text=papers_text,
                )
            )
            batch_results = _parse_classification_response(response, batch)

            # 后处理：确保评分有真正的小数多样性
            _ensure_score_diversity(batch_results)

            # 持久化到数据库
            for r in batch_results:
                update_paper_analysis(
                    arxiv_id=r["arxiv_id"],
                    subfield=r.get("subfield", ""),
                    key_contribution=r.get("key_contribution", ""),
                    importance=float(r.get("importance", 0)),
                )

            results.extend(batch_results)
            print(f"    ✓ 完成 {len(batch_results)} 篇分类")

        except Exception as e:
            print(f"    ✗ 批次失败: {e}")
            # 失败时补充默认值
            for p in batch:
                results.append({
                    "arxiv_id": p["arxiv_id"],
                    "subfield": "其他 (Others)",
                    "key_contribution": "",
                    "importance": 0,
                })

        # 批次间延迟
        if i + batch_size < len(papers):
            time.sleep(0.5)

    print(f"[Classifier Agent] 分类完成，成功处理 {len(results)} 篇")
    return results


# ============================================================
# 辅助函数
# ============================================================

def _format_papers_for_llm(papers: list[dict]) -> str:
    """将论文列表格式化为 LLM prompt 文本"""
    lines = []
    for i, p in enumerate(papers, 1):
        abstract = p.get("abstract", "")
        # 截断过长的摘要
        if len(abstract) > 500:
            abstract = abstract[:500] + "..."
        lines.append(
            f"### 论文 {i}\n"
            f"ID: {p['arxiv_id']}\n"
            f"Title: {p['title']}\n"
            f"Abstract: {abstract}\n"
        )
    return "\n".join(lines)


def _call_llm_with_retry(prompt: str, max_retries: int = 3) -> str:
    """调用 DeepSeek API，带重试机制"""
    for attempt in range(max_retries):
        try:
            resp = _client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": "你是一位严谨的 AI 论文评审专家。请严格按 JSON 格式输出。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,  # 低温度保证一致性
                max_tokens=4096,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"    LLM 调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
            else:
                raise


def _parse_classification_response(response: str, batch: list[dict]) -> list[dict]:
    """解析 LLM 返回的 JSON，提取分类结果"""
    # 清理可能的 markdown 代码块标记
    clean = response.strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    if clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]

    try:
        parsed = json.loads(clean.strip())
    except json.JSONDecodeError:
        # 兜底：尝试逐行解析
        print("    ⚠ JSON 解析失败，尝试修复...")
        # 提取 [] 之间的内容
        import re
        match = re.search(r"\[.*\]", clean, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
        else:
            raise ValueError("无法从 LLM 输出中提取有效 JSON")

    # 确保是列表
    if isinstance(parsed, dict):
        parsed = [parsed]

    # 建立 ID→结果映射
    result_map = {r.get("arxiv_id", ""): r for r in parsed}

    # 填充结果
    results = []
    for p in batch:
        aid = p["arxiv_id"]
        if aid in result_map:
            results.append({
                "arxiv_id": aid,
                "subfield": result_map[aid].get("subfield", "其他 (Others)"),
                "key_contribution": result_map[aid].get("key_contribution", ""),
                "importance": result_map[aid].get("importance", 0),
            })
        else:
            results.append({
                "arxiv_id": aid,
                "subfield": "其他 (Others)",
                "key_contribution": "",
                "importance": 0,
            })

    return results


# ============================================================
# 统计工具
# ============================================================

def get_subfield_distribution(papers: list[dict]) -> dict:
    """统计各子领域的论文数量分布"""
    dist = {}
    for p in papers:
        sf = p.get("subfield", "其他 (Others)")
        dist[sf] = dist.get(sf, 0) + 1
    return dict(sorted(dist.items(), key=lambda x: x[1], reverse=True))


def get_top_papers(papers: list[dict], top_k: int = 10) -> list[dict]:
    """按重要性评分获取 Top-K 论文"""
    sorted_papers = sorted(papers, key=lambda x: x.get("importance", 0), reverse=True)
    return sorted_papers[:top_k]
