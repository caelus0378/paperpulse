"""
智能体 6: 双语翻译智能体 (Bilingual Translator Agent)
───────────────────────────────────────────────────
职责：将论文标题和摘要进行中英互译，让报告支持双语展示
使用 DeepSeek API 进行高质量学术翻译
"""

import json
import time
from typing import Optional
from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from database import get_connection

# ============================================================
# 初始化 DeepSeek 客户端
# ============================================================
_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# ============================================================
# 翻译 Prompt 模板
# ============================================================

TRANSLATION_PROMPT = """你是一位专业的学术翻译专家，精通中英文学术用语。

## 翻译规则
- 学术术语必须准确，使用领域内通用译法
- 保持原文的学术严谨性，不要口语化
- 英文人名、机构名、算法名保持原文不翻译
- 专业缩写（如 LLM, RAG, GAN, BERT 等）保持大写

## 待翻译内容
{text}

## 翻译方向
{direction}

请输出 JSON 格式：
```json
{{
  "original": "原文前30字...",
  "translated": "完整译文",
  "detected_language": "en/zh"
}}
```

只输出 JSON，不要输出其他内容。"""


def translate_text(text: str, target_lang: str = "zh") -> str:
    """
    翻译单段文本（带重试机制）

    参数:
        text: 待翻译的文本
        target_lang: 目标语言 "zh"（中文）或 "en"（英文）

    返回:
        str: 翻译后的文本，失败时返回原文
    """
    if not text or len(text.strip()) < 10:
        return text

    direction = "英文 → 中文" if target_lang == "zh" else "中文 → 英文"

    for attempt in range(3):
        try:
            resp = _client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": "你是专业学术翻译专家，只输出 JSON。"},
                    {"role": "user", "content": TRANSLATION_PROMPT.format(
                        text=text, direction=direction
                    )},
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            content = resp.choices[0].message.content.strip()

            # 清理 markdown 标记
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            result = json.loads(content.strip())
            return result.get("translated", text)

        except (json.JSONDecodeError, Exception) as e:
            if attempt == 2:
                print(f"[Translator] 翻译失败: {e}")
                return text
            time.sleep(1)

    return text


def translate_paper_batch(
    papers: list[dict],
    translate_title: bool = True,
    translate_abstract: bool = True,
    batch_size: int = 10,
) -> list[dict]:
    """
    批量翻译论文的标题和摘要

    参数:
        papers: 论文列表
        translate_title: 是否翻译标题
        translate_abstract: 是否翻译摘要
        batch_size: 每批处理数量（控制 API 调用频率）

    返回:
        list[dict]: 添加了 zh_title, zh_abstract, en_title, en_abstract 字段的论文列表
    """
    if not papers:
        print("[Translator Agent] 无论文需要翻译")
        return []

    total = len(papers)
    print(f"[Translator Agent] 开始翻译 {total} 篇论文")
    print(f"  → 翻译标题: {'是' if translate_title else '否'}")
    print(f"  → 翻译摘要: {'是' if translate_abstract else '否'}")

    translated_count = 0

    for i, p in enumerate(papers):
        try:
            # 翻译标题
            if translate_title:
                title = p.get("title", "")
                if title:
                    # 检测是否含中文来判断翻译方向
                    has_chinese = any('一' <= c <= '鿿' for c in title)
                    if has_chinese:
                        p["en_title"] = title  # 原标题即中文/含中文
                        p["zh_title"] = translate_text(title, "zh")
                    else:
                        p["zh_title"] = translate_text(title, "zh")
                        p["en_title"] = title  # 原标题即英文

            # 翻译摘要
            if translate_abstract:
                abstract = p.get("abstract", "")
                if abstract and len(abstract) > 20:
                    has_chinese = any('一' <= c <= '鿿' for c in abstract)
                    if has_chinese:
                        p["en_abstract"] = abstract
                        p["zh_abstract"] = translate_text(abstract, "zh")
                    else:
                        p["zh_abstract"] = translate_text(abstract, "zh")
                        p["en_abstract"] = abstract

            translated_count += 1

            # 进度提示
            if (i + 1) % 10 == 0:
                print(f"  → 进度: {i + 1}/{total}")

            # API 调用间隔
            time.sleep(0.3)

        except Exception as e:
            print(f"  [FAIL] 翻译失败 ({p.get('arxiv_id', '?')}): {e}")
            continue

    # 持久化翻译结果
    _persist_translations(papers)

    print(f"[Translator Agent] 翻译完成: {translated_count}/{total} 篇")
    return papers


def _persist_translations(papers: list[dict]):
    """将翻译结果存储到数据库"""
    try:
        conn = get_connection()
        cur = conn.cursor()

        # 确保翻译字段存在
        cur.execute("""
            CREATE TABLE IF NOT EXISTS paper_translations (
                arxiv_id TEXT PRIMARY KEY,
                zh_title TEXT,
                zh_abstract TEXT,
                en_title TEXT,
                en_abstract TEXT
            )
        """)

        for p in papers:
            cur.execute("""
                INSERT OR REPLACE INTO paper_translations
                    (arxiv_id, zh_title, zh_abstract, en_title, en_abstract)
                VALUES (?, ?, ?, ?, ?)
            """, (
                p["arxiv_id"],
                p.get("zh_title", ""),
                p.get("zh_abstract", ""),
                p.get("en_title", p.get("title", "")),
                p.get("en_abstract", p.get("abstract", "")),
            ))

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Translator] 持久化翻译失败: {e}")


def get_translation(arxiv_id: str) -> Optional[dict]:
    """查询某篇论文的翻译"""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM paper_translations WHERE arxiv_id = ?",
            (arxiv_id,),
        )
        row = cur.fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception:
        return None


# ============================================================
# 快捷函数
# ============================================================

def translate_single(text: str, to_chinese: bool = True) -> str:
    """
    快速翻译单段文本（供其他模块调用）

    参数:
        text: 待翻译文本
        to_chinese: True=译为中文, False=译为英文
    """
    return translate_text(text, "zh" if to_chinese else "en")
