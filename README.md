# 🎓 每日学术论文热点追踪系统

基于 **OpenClaw 多智能体协同框架** 设计，自动从 arXiv 采集最新 AI/CS 论文，通过三个分工明确的智能体完成分类、分析和趋势预测，最终生成结构化的每日日报。

---

## 系统架构

```
┌───────────────────────────────────────────────────────┐
│                    系统编排层 (main.py)                 │
├───────────┬────────────────┬────────────────┬─────────┤
│  智能体 1  │   智能体 2     │    智能体 3    │ 汇总器  │
│ Collector │  Classifier    │ TrendAnalyzer  │ Agregat │
│ 论文采集   │  分类与摘要     │   趋势分析      │ 日报生成 │
├───────────┴────────────────┴────────────────┴─────────┤
│                    基础设施层                           │
│  arXiv API  │  DeepSeek API  │  SQLite  │  Gradio     │
└───────────────────────────────────────────────────────┘
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 填入你的 DeepSeek API Key
# 获取 Key: https://platform.deepseek.com/api_keys
```

### 3. 运行

```bash
# 运行一次完整分析 + 启动 Web UI
python main.py --web

# 打开浏览器访问 http://localhost:7860
```

---

## ☁️ 云服务器部署（Docker）

```bash
# 1. 服务器上克隆项目（或通过 GitHub 推送）
git clone <你的仓库地址> && cd arxiv-analysis-system

# 2. 配置 API Key
cp .env.example .env
nano .env   # 填入 DEEPSEEK_API_KEY

# 3. 一键部署
bash deploy.sh

# 部署后访问: http://<服务器公网IP>:7860
```

### 常用运维命令

```bash
docker compose logs -f          # 查看日志
docker compose restart          # 重启服务
docker compose down             # 停止服务
docker compose exec web python main.py   # 手动触发一次分析
```

### 防火墙注意

确保云服务器安全组/防火墙已放行 **7860** 端口。

```bash
# 方式一：环境变量
export DEEPSEEK_API_KEY="your-api-key"

# 方式二：直接编辑 config.py
# DEEPSEEK_API_KEY = "your-api-key"
```

> 免费获取 API Key: [https://platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys)
>
> DeepSeek 价格极低，每天分析 200 篇论文约消耗 0.5 元

### 3. 运行

```bash
# 运行一次完整分析
python main.py

# 启动 Web UI（含分析）
python main.py --web

# 启动定时任务（每天 8:00 自动执行）
python main.py --schedule

# 一键启动全部
python main.py --all
```

Web UI 默认地址: **http://localhost:7860**

## 项目结构

```
arxiv-analysis-system/
├── main.py                  # 主入口 + 流水线编排
├── config.py                # 全局配置
├── database.py              # SQLite 数据持久化
├── aggregator.py            # 报告汇总智能体
├── visualization.py         # 图表生成（matplotlib）
├── web_ui.py                # Gradio Web 界面
├── scheduler.py             # 定时任务调度
├── requirements.txt         # Python 依赖
├── README.md                # 项目说明
├── Dockerfile               # Docker 部署
├── agents/
│   ├── __init__.py
│   ├── collector.py         # 智能体 1: 论文采集
│   ├── classifier.py        # 智能体 2: 分类与摘要
│   └── trend_analyzer.py    # 智能体 3: 趋势分析
├── data/                    # SQLite 数据库（自动创建）
└── reports/                 # 日报 + 图表（自动创建）
    ├── charts/
    └── report_YYYY-MM-DD.md
```

## 三个智能体分工

| 智能体 | 文件 | 职责 | 依赖 |
|--------|------|------|------|
| 📡 Collector | `agents/collector.py` | 从 arXiv 采集最新论文，解析元数据 | arXiv API, feedparser |
| 🏷️ Classifier | `agents/classifier.py` | 子领域分类，提取关键贡献，重要性评分 | DeepSeek API |
| 📊 TrendAnalyzer | `agents/trend_analyzer.py` | 关键词提取，趋势预测，历史对比 | jieba, DeepSeek API |
| 📝 Aggregator | `aggregator.py` | 整合三智能体输出，生成 Markdown 日报 | - |

## 数据来源

- **arXiv API** — 免费学术预印本平台
- 覆盖分类: `cs.AI`, `cs.CL`, `cs.CV`, `cs.LG`, `cs.NE`, `cs.RO`, `cs.IR`, `cs.SE`
- 每日采集上限: 200 篇（可配置）

## 技术栈

| 层级 | 技术 |
|------|------|
| AI 模型 | DeepSeek API (deepseek-chat) |
| 数据采集 | arXiv API, requests, feedparser |
| 数据分析 | pandas, numpy |
| 中文处理 | jieba |
| 可视化 | matplotlib |
| Web UI | Gradio |
| 数据库 | SQLite |
| 定时任务 | schedule |
| 部署 | Docker, Nginx, Supervisor |

## 云服务器部署

```bash
# 1. 构建 Docker 镜像
docker build -t arxiv-daily .

# 2. 运行容器
docker run -d \
  -e DEEPSEEK_API_KEY=your-key \
  -p 7860:7860 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/reports:/app/reports \
  --name arxiv-daily \
  --restart always \
  arxiv-daily

# 3. 访问 Web UI
# http://your-server-ip:7860
```

## License

本项目为浙江大学《人工智能基础A》课程大作业。
