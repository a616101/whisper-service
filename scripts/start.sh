#!/bin/bash
# WhisperX Subtitle Service 啟動腳本
# 支援 Linux (NVIDIA GPU) 和 Mac (Apple Silicon)

set -e

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== WhisperX Subtitle Service ===${NC}"

# 偵測作業系統和架構
OS=$(uname -s)
ARCH=$(uname -m)

echo -e "${BLUE}偵測到: ${OS} / ${ARCH}${NC}"

# 檢查 .env 檔案
if [ ! -f .env ]; then
    echo -e "${YELLOW}警告：找不到 .env 檔案${NC}"
    echo "正在從 .env.example 複製..."
    cp .env.example .env
    echo -e "${RED}請編輯 .env 檔案並填入 HF_TOKEN${NC}"
    exit 1
fi

# 載入環境變數
source .env

# 檢查 HF_TOKEN
if [ -z "$HF_TOKEN" ] || [ "$HF_TOKEN" = "your_huggingface_token_here" ]; then
    echo -e "${YELLOW}警告：HF_TOKEN 未設定，diarization 功能將不可用${NC}"
    echo "獲取方式：https://huggingface.co/settings/tokens"
    echo ""
fi

# 創建快取目錄
mkdir -p cache/huggingface cache/torch cache/app

# 檢查 Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}錯誤：找不到 Docker${NC}"
    exit 1
fi

# 判斷是否為 Mac
is_mac() {
    [[ "$OS" == "Darwin" ]]
}

# 判斷是否有 NVIDIA GPU
has_nvidia_gpu() {
    if is_mac; then
        return 1
    fi
    docker info 2>/dev/null | grep -q "Runtimes.*nvidia"
}

# 解析參數
MODE=${1:-auto}

# 自動偵測模式
if [ "$MODE" = "auto" ]; then
    if is_mac; then
        MODE="mac"
        echo -e "${BLUE}自動選擇: Mac 模式 (CPU)${NC}"
    elif has_nvidia_gpu; then
        MODE="dev"
        echo -e "${BLUE}自動選擇: GPU 開發模式${NC}"
    else
        MODE="mac"
        echo -e "${BLUE}自動選擇: CPU 模式（無 NVIDIA GPU）${NC}"
    fi
fi

case $MODE in
    mac)
        echo -e "${GREEN}啟動 Mac 模式（CPU）${NC}"
        echo -e "${YELLOW}提示：Docker 內只能用 CPU，如需 MPS 加速請使用本地執行${NC}"
        docker compose -f docker-compose.mac.yml up -d --build
        ;;
    dev)
        if ! has_nvidia_gpu; then
            echo -e "${YELLOW}警告：找不到 nvidia-docker runtime，切換到 CPU 模式${NC}"
            docker compose -f docker-compose.mac.yml up -d --build
        else
            echo -e "${GREEN}啟動 GPU 開發模式（單機）${NC}"
            docker compose -f docker-compose.dev.yml up -d --build
        fi
        ;;
    prod)
        echo -e "${GREEN}啟動生產模式（單 GPU + Redis）${NC}"
        # 檢查是否有 nvidia runtime
        if docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
            docker compose up -d --build
        else
            echo -e "${YELLOW}未偵測到 nvidia runtime，使用 DGX 兼容模式${NC}"
            docker compose -f docker-compose.dgx.yml up -d --build
        fi
        ;;
    dgx)
        echo -e "${GREEN}啟動 DGX Spark 模式${NC}"
        docker compose -f docker-compose.dgx.yml up -d --build
        ;;
    multi-gpu)
        echo -e "${GREEN}啟動多 GPU 模式${NC}"
        docker compose --profile multi-gpu up -d --build
        ;;
    monitoring)
        echo -e "${GREEN}啟動監控模式（含 Flower）${NC}"
        docker compose --profile monitoring up -d --build
        ;;
    full)
        echo -e "${GREEN}啟動完整模式（多 GPU + 監控）${NC}"
        docker compose --profile multi-gpu --profile monitoring up -d --build
        ;;
    local)
        echo -e "${GREEN}本地執行模式（不使用 Docker）${NC}"
        echo "請參考 docs/LOCAL_SETUP.md 進行本地安裝"
        echo ""
        echo "快速開始："
        echo "  1. 建立虛擬環境: python -m venv venv && source venv/bin/activate"
        echo "  2. 安裝依賴: pip install torch torchaudio && pip install -r requirements.mac.txt"
        echo "  3. 啟動服務: uvicorn app.main:app --reload"
        exit 0
        ;;
    stop)
        echo -e "${YELLOW}停止所有服務${NC}"
        docker compose down 2>/dev/null || true
        docker compose -f docker-compose.dev.yml down 2>/dev/null || true
        docker compose -f docker-compose.mac.yml down 2>/dev/null || true
        docker compose -f docker-compose.dgx.yml down 2>/dev/null || true
        echo -e "${GREEN}已停止所有服務${NC}"
        exit 0
        ;;
    logs)
        if is_mac; then
            docker compose -f docker-compose.mac.yml logs -f
        else
            docker compose logs -f
        fi
        exit 0
        ;;
    *)
        echo "用法: $0 {auto|mac|dev|prod|dgx|multi-gpu|monitoring|full|local|stop|logs}"
        echo ""
        echo "模式說明："
        echo "  auto        - 自動偵測環境選擇最佳模式"
        echo "  mac         - Mac 模式（Docker CPU）"
        echo "  dev         - GPU 開發模式（單機，需要 NVIDIA）"
        echo "  prod        - 生產模式（單 GPU + Redis）"
        echo "  dgx         - DGX Spark 模式（不依賴 nvidia runtime）"
        echo "  multi-gpu   - 多 GPU 模式"
        echo "  monitoring  - 含 Flower 監控"
        echo "  full        - 完整模式（多 GPU + 監控）"
        echo "  local       - 顯示本地執行指南（MPS 加速）"
        echo "  stop        - 停止所有服務"
        echo "  logs        - 查看日誌"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}服務啟動中...${NC}"
echo ""

# 等待服務啟動
echo "等待服務就緒..."
sleep 5

# 檢查服務狀態
for i in {1..30}; do
    if curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
        echo ""
        echo -e "${GREEN}✓ 服務已就緒！${NC}"
        echo ""
        echo "API 文件：http://localhost:8000/docs"
        echo "健康檢查：http://localhost:8000/api/v1/health"

        if [ "$MODE" = "monitoring" ] || [ "$MODE" = "full" ]; then
            echo "Flower 監控：http://localhost:5555"
        fi
        exit 0
    fi
    echo -n "."
    sleep 2
done

echo ""
echo -e "${YELLOW}服務尚未就緒，請檢查日誌：${NC}"
echo "  $0 logs"
