# ============================================================
# PaperPulse — Docker 镜像
# 每日学术论文热点追踪系统
# ============================================================
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（中文字体 + curl 用于健康检查）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    fonts-wqy-microhei \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 创建数据和报告目录
RUN mkdir -p /app/data /app/reports/charts

# 设置时区
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 暴露端口
EXPOSE 7860

# 检查 .env 是否存在并运行相应命令
CMD if [ -f .env ]; then \
        python main.py --web --port 7860; \
    else \
        echo "WARNING: .env not found, running without API key"; \
        python main.py --web --port 7860; \
    fi
