"""
主入口 — 每日学术论文热点追踪系统
──────────────────────────────
基于 OpenClaw 多智能体协同框架设计

八个智能体分工协作：
  智能体 1 (Collector)      → 论文采集
  智能体 2 (Classifier)     → 分类与摘要
  智能体 3 (TrendAnalyzer)  → 趋势分析
  智能体 4 (QualityAssessor)→ 质量评估
  智能体 5 (CrossReferencer)→ 交叉引用与关联
  智能体 6 (Translator)     → 双语翻译
  智能体 7 (WeeklyDigest)   → 周报汇总
  智能体 8 (DeepAnalyzer)   → 深度解读

用法:
  # 运行一次完整分析
  python main.py

  # 启动 Web UI
  python main.py --web

  # 启动定时任务
  python main.py --schedule

  # 一键启动（Web UI + 定时任务）
  python main.py --all
"""

import sys
import os
import argparse
import io
from datetime import date

# 修复 Windows GBK 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(__file__))

from config import DEEPSEEK_API_KEY
from database import init_db, save_papers, get_papers_by_date


def run_full_pipeline():
    """
    完整分析流水线（按顺序调用各智能体）
    这是整个系统的核心编排逻辑
    """
    report_date = date.today().isoformat()

    print("=" * 60)
    print(f"  每日学术论文热点追踪系统")
    print(f"  日期: {report_date}")
    print(f"  基于 OpenClaw 多智能体协同框架")
    print("=" * 60)

    # ================================================
    # 阶段 1: 论文采集（智能体 1）
    # ================================================
    print("\n" + "-" * 40)
    print(" 阶段 1/4: 论文采集智能体 (Collector Agent)")
    print("-" * 40)
    from agents.collector import collect_papers
    papers = collect_papers()

    if not papers:
        print("[Main] [WARN] 未采集到论文，流水线终止")
        return None

    # 保存原始论文到数据库
    save_papers(papers)
    print(f"[Main] [OK] 论文已保存到数据库: {len(papers)} 篇")

    # ================================================
    # 阶段 2: 分类与摘要（智能体 2）
    # ================================================
    print("\n" + "-" * 40)
    print(" 阶段 2/4: 分类与摘要智能体 (Classifier Agent)")
    print("-" * 40)

    if _check_api_key():
        from agents.classifier import classify_papers, get_subfield_distribution, get_top_papers
        classified_results = classify_papers(papers)

        # 将分类结果合并回论文列表
        result_map = {r["arxiv_id"]: r for r in classified_results}
        for p in papers:
            if p["arxiv_id"] in result_map:
                p["subfield"] = result_map[p["arxiv_id"]]["subfield"]
                p["key_contribution"] = result_map[p["arxiv_id"]]["key_contribution"]
                p["importance"] = result_map[p["arxiv_id"]]["importance"]

        # 打印分类摘要
        dist = get_subfield_distribution(papers)
        print(f"[Main] [OK] 分类完成，子领域分布:")
        for sf, count in list(dist.items())[:5]:
            print(f"    {sf}: {count} 篇")
    else:
        print("[Main] [WARN] 未配置 API Key，跳过分类阶段")

    # ================================================
    # 阶段 3: 趋势分析（智能体 3）
    # ================================================
    print("\n" + "-" * 40)
    print(" 阶段 3/4: 趋势分析智能体 (Trend Analyzer Agent)")
    print("-" * 40)

    trend_result = None
    if _check_api_key():
        from agents.trend_analyzer import run_trend_analysis
        trend_result = run_trend_analysis(report_date)
        if trend_result:
            print(f"[Main] [OK] 趋势分析完成")
            print(f"    总论文: {trend_result['total_papers']} 篇")
            print(f"    热点论文: {len(trend_result['hot_papers'])} 篇")
        else:
            print("[Main] [WARN] 趋势分析返回空结果")
    else:
        print("[Main] [WARN] 未配置 API Key，跳过趋势分析")

    # ================================================
    # 阶段 4: 报告汇总（Aggregator）
    # ================================================
    print("\n" + "-" * 40)
    print(" 阶段 4/4: 报告汇总智能体 (Aggregator Agent)")
    print("-" * 40)

    from aggregator import generate_daily_report
    report = generate_daily_report(papers, trend_result, report_date)
    print(f"[Main] [OK] 日报生成完成")

    # ================================================
    # 阶段 5: 质量评估（智能体 4）
    # ================================================
    print("\n" + "-" * 40)
    print(" 阶段 5/7: 质量评估智能体 (Quality Assessor Agent)")
    print("-" * 40)

    quality_results = None
    if _check_api_key():
        from agents.quality_assessor import assess_quality, get_quality_report
        try:
            quality_results = assess_quality(papers)
            quality_report = get_quality_report(quality_results)
            print(f"[Main] [OK] 质量评估完成")
            print(f"    方法论均分: {quality_report.get('avg_methodology', 0)}")
            print(f"    创新性均分: {quality_report.get('avg_novelty', 0)}")
        except Exception as e:
            print(f"[Main] [WARN] 质量评估失败: {e}")
    else:
        print("[Main] [WARN] 未配置 API Key，跳过质量评估")

    # ================================================
    # 阶段 6: 交叉引用与关联分析（智能体 5）
    # ================================================
    print("\n" + "-" * 40)
    print(" 阶段 6/7: 交叉引用智能体 (Cross Referencer Agent)")
    print("-" * 40)

    cross_ref_result = None
    try:
        from agents.cross_referencer import run_cross_reference
        cross_ref_result = run_cross_reference(papers, report_date)
        if cross_ref_result and cross_ref_result.get("graph"):
            graph = cross_ref_result["graph"]
            print(f"[Main] [OK] 关联图谱构建完成")
            print(f"    节点: {len(graph.get('nodes', []))}")
            print(f"    边: {len(graph.get('edges', []))}")
            print(f"    聚类: {len(graph.get('clusters', []))}")
    except Exception as e:
        print(f"[Main] [WARN] 交叉引用分析失败: {e}")

    # ================================================
    # 阶段 7: 双语翻译（智能体 6，默认只翻译 Top 论文）
    # ================================================
    print("\n" + "-" * 40)
    print(" 阶段 7/8: 双语翻译智能体 (Translator Agent)")
    print("-" * 40)

    if _check_api_key():
        from agents.translator import translate_paper_batch
        try:
            # 只翻译 Top 20 高评分论文以节省 API 调用
            top_for_translation = sorted(
                papers, key=lambda x: float(x.get("importance", 0)), reverse=True
            )[:20]
            translate_paper_batch(top_for_translation)
            print(f"[Main] [OK] 翻译完成 (Top {len(top_for_translation)} 篇)")
        except Exception as e:
            print(f"[Main] [WARN] 翻译失败: {e}")
    else:
        print("[Main] [WARN] 未配置 API Key，跳过翻译")

    # ================================================
    # 阶段 8: 深度解读（智能体 8，选最高分论文做全方位分析）
    # ================================================
    print("\n" + "-" * 40)
    print(" 阶段 8/8: 深度解读智能体 (Deep Analyzer Agent)")
    print("-" * 40)

    if _check_api_key() and papers:
        from agents.deep_analyzer import run_deep_analysis
        try:
            deep_result = run_deep_analysis(papers, report_date)
            if deep_result:
                print(f"[Main] [OK] 深度解读完成")
                print(f"    论文: {deep_result.get('arxiv_id', '?')}")
                tldr = deep_result.get('tldr', '')
                print(f"    TL;DR: {tldr[:60]}...")
            else:
                print("[Main] [WARN] 深度解读返回空结果")
        except Exception as e:
            print(f"[Main] [WARN] 深度解读失败: {e}")
    elif not papers:
        print("[Main] [WARN] 无论文数据，跳过深度解读")
    else:
        print("[Main] [WARN] 未配置 API Key，跳过深度解读")

    print("\n" + "=" * 60)
    print(f"  全流程完成！8 个智能体全部就绪")
    print(f"  论文数: {len(papers)} | 日报: reports/report_{report_date}.md")
    print("=" * 60)

    return report


def _check_api_key() -> bool:
    """检查 API Key 是否已配置"""
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY.startswith("sk-your-"):
        return False
    return True


# ============================================================
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="每日学术论文热点追踪系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py              运行一次完整分析
  python main.py --web         启动 Web UI (http://localhost:7860)
  python main.py --schedule    启动定时任务 (每天 8:00 执行)
  python main.py --all         同时启动 Web UI 和定时任务
  python main.py --port 8080   指定 Web UI 端口
        """,
    )
    parser.add_argument("--web", action="store_true", help="启动 Gradio Web UI")
    parser.add_argument("--schedule", action="store_true", help="启动定时任务调度器")
    parser.add_argument("--all", action="store_true", help="同时启动 Web UI 和定时任务")
    parser.add_argument("--port", type=int, default=7860, help="Web UI 端口 (默认 7860)")
    parser.add_argument("--share", action="store_true", help="Gradio 生成公网分享链接")

    args = parser.parse_args()

    # 初始化数据库
    init_db()
    print("[Main] 数据库初始化完成")

    if args.all:
        # 先运行一次分析，再启动 Web 和定时任务
        print("[Main] 先运行一次初始分析...")
        run_full_pipeline()
        print("\n[Main] 启动 Web UI + 定时任务...")
        import threading
        t = threading.Thread(target=lambda: __import__("scheduler").start_scheduler(), daemon=True)
        t.start()
        from web_ui import launch_ui
        launch_ui(port=args.port, share=args.share)

    elif args.schedule:
        print("[Main] 先运行一次初始分析...")
        run_full_pipeline()
        from scheduler import start_scheduler
        start_scheduler()

    elif args.web:
        from web_ui import launch_ui

        # 检查是否有今天的数据，没有的话后台跑分析
        today_papers = get_papers_by_date(date.today().isoformat())
        if not today_papers:
            print("[Main] 今日尚无数据，后台执行初始分析...")
            import threading
            t = threading.Thread(target=run_full_pipeline, daemon=True)
            t.start()

        launch_ui(port=args.port, share=args.share)

    else:
        # 默认：运行一次完整分析
        if not _check_api_key():
            print("\n" + "!" * 60)
            print("  [WARN] 未配置 DeepSeek API Key！")
            print("  请设置环境变量: export DEEPSEEK_API_KEY='your-key'")
            print("  或在 config.py 中直接填写")
            print("  获取 API Key: https://platform.deepseek.com/api_keys")
            print("!" * 60 + "\n")

        run_full_pipeline()


if __name__ == "__main__":
    main()
