"""
定时任务模块 — 每天自动执行完整分析流水线
─────────────────────────────────────
作为独立进程运行（Docker Compose 中独立服务）
使用 schedule 库实现轻量级定时调度

特性:
  - 启动时立即执行一次分析（确保有数据）
  - 每日定时触发
  - 异常自动恢复（不会因一次失败退出）
"""

import sys
import os
import time
import signal
import traceback
from datetime import datetime

import schedule

from config import SCHEDULE_HOUR, SCHEDULE_MINUTE, LOG_LEVEL
from main import run_full_pipeline


# ── 优雅退出 ──────────────────────────────────────────
_shutdown = False


def _on_shutdown(signum, frame):
    global _shutdown
    print(f"\n[Scheduler] 收到退出信号 (signal={signum})，等待当前任务完成...")
    _shutdown = True


signal.signal(signal.SIGTERM, _on_shutdown)
signal.signal(signal.SIGINT, _on_shutdown)


# ── 主逻辑 ────────────────────────────────────────────

def run_daily_pipeline():
    """每日分析流水线（异常安全）"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}")
    print(f"[Scheduler] 定时任务触发 — {ts}")
    print(f"{'='*60}")

    try:
        run_full_pipeline()
        print(f"[Scheduler] ✅ 每日分析完成 — {datetime.now().strftime('%H:%M:%S')}")
    except Exception:
        print(f"[Scheduler] ❌ 每日分析失败:")
        traceback.print_exc()
        # 单次失败不退出，继续等待下次触发


def start_scheduler(hour: int = None, minute: int = None):
    """
    启动定时调度器（阻塞运行，适合作为常驻进程）

    参数:
        hour: 每天执行的小时（24小时制），默认使用 config
        minute: 每天执行的分钟，默认使用 config
    """
    if hour is None:
        hour = SCHEDULE_HOUR
    if minute is None:
        minute = SCHEDULE_MINUTE

    schedule_time = f"{hour:02d}:{minute:02d}"
    schedule.every().day.at(schedule_time).do(run_daily_pipeline)

    print(f"[Scheduler] ⏰ 定时任务已设置: 每天 {schedule_time} 自动执行分析")
    print(f"[Scheduler] 📅 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[Scheduler] 💡 启动后立即执行一次分析...")

    # 启动时立即跑一次（确保部署后有数据）
    run_daily_pipeline()

    print(f"\n[Scheduler] 🔄 进入待机模式，每分钟检查一次...")

    while not _shutdown:
        try:
            schedule.run_pending()
            time.sleep(60)
        except KeyboardInterrupt:
            break
        except Exception:
            print(f"[Scheduler] ⚠ 调度循环异常（忽略）: {traceback.format_exc()}")
            time.sleep(30)

    print("[Scheduler] 🛑 已停止")


if __name__ == "__main__":
    start_scheduler()
