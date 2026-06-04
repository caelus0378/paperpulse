#!/bin/bash
# ============================================================
# PaperPulse 一键部署脚本（云服务器端）
# 在服务器上执行:
#   bash deploy.sh
# ============================================================

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}"
echo "============================================"
echo "  PaperPulse — 一键部署脚本"
echo "  每日学术论文热点追踪系统"
echo "============================================"
echo -e "${NC}"

# 1. 检查 Docker
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}[1/4] Docker 未安装，正在安装...${NC}"
    curl -fsSL https://get.docker.com | bash
    sudo systemctl enable docker
    sudo systemctl start docker
    sudo usermod -aG docker $USER
    echo -e "${GREEN}Docker 安装完成，请重新登录使权限生效。${NC}"
else
    echo -e "${GREEN}[1/4] Docker 已安装 ✓${NC}"
fi

# 2. 创建 .env（如果没有）
if [ ! -f .env ]; then
    echo -e "${YELLOW}[2/4] 创建 .env 文件...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}请编辑 .env 文件填入你的 DEEPSEEK_API_KEY，然后重新运行本脚本。${NC}"
    echo -e "${YELLOW}  nano .env${NC}"
    exit 1
else
    echo -e "${GREEN}[2/4] .env 已存在 ✓${NC}"
fi

# 3. 创建数据目录
echo -e "${BLUE}[3/4] 创建数据目录...${NC}"
mkdir -p data reports reports/charts

# 4. 启动服务
echo -e "${BLUE}[4/4] 启动 Docker Compose...${NC}"
docker compose up -d --build

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  PaperPulse 部署成功！🎉${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo -e "  Web UI:  http://$(curl -s ifconfig.me):7860"
echo -e "  日志:    docker compose logs -f"
echo -e "  停止:    docker compose down"
echo -e "  重启:    docker compose restart"
echo ""
echo -e "${YELLOW}  首次部署后建议先手动触发一次分析:${NC}"
echo -e "  docker compose exec web python main.py"
echo ""
