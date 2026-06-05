"""
本地采集同步脚本
在你的电脑上运行，采集论文后将数据库和日报同步到服务器
用法: python local_sync.py
"""
import subprocess
import sys
import os

SERVER = "root@47.83.202.19"
SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519")
PROJECT_DIR = os.path.dirname(__file__)

def run(cmd, shell=False):
    result = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] {result.stderr[:300]}")
    return result

print("=" * 50)
print("  本地采集 → 服务器同步")
print("=" * 50)

# 1. 运行完整分析（采集 + 分类 + 趋势 + 日报）
print("\n[1/3] 运行分析流水线...")
sys.path.insert(0, PROJECT_DIR)
from main import run_full_pipeline
run_full_pipeline()

# 2. 上传数据库
print("\n[2/3] 上传数据库...")
db_path = os.path.join(PROJECT_DIR, "data", "papers.db")
run([
    "scp", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no",
    db_path, f"{SERVER}:/app/data/papers.db"
])

# 3. 上传日报文件
print("\n[3/3] 上传日报...")
reports_dir = os.path.join(PROJECT_DIR, "reports")
import glob
for f in glob.glob(os.path.join(reports_dir, "*.md")):
    run([
        "scp", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no",
        f, f"{SERVER}:/app/reports/"
    ])

print("\n✅ 同步完成！刷新 http://47.83.202.19:7860 查看最新数据")
