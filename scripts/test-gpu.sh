#!/bin/bash
# 測試 DGX Spark GPU 是否正確識別

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== DGX Spark GPU 測試 ===${NC}"
echo ""

# 1. 檢查 host 上的 nvidia-smi
echo -e "${YELLOW}1. Host nvidia-smi:${NC}"
nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv || echo "無法執行 nvidia-smi"
echo ""

# 2. 檢查 NVIDIA Container Toolkit
echo -e "${YELLOW}2. NVIDIA Container Toolkit:${NC}"
if command -v nvidia-ctk &> /dev/null; then
    nvidia-ctk --version
else
    echo "nvidia-ctk 未安裝"
fi
echo ""

# 3. 檢查 Docker runtime
echo -e "${YELLOW}3. Docker GPU 支援:${NC}"
docker info 2>/dev/null | grep -E "(Runtimes|Default Runtime)" || echo "無法取得 Docker runtime 資訊"
echo ""

# 4. 測試 Docker GPU 訪問（使用最新 NGC 容器）
echo -e "${YELLOW}4. 測試 Docker 容器 GPU 訪問:${NC}"
echo "拉取測試映像..."
docker pull nvcr.io/nvidia/pytorch:24.09-py3 2>/dev/null || true

echo "執行 GPU 測試..."
docker run --rm \
    --gpus all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    nvcr.io/nvidia/pytorch:24.09-py3 \
    python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        props = torch.cuda.get_device_properties(i)
        print(f'    Memory: {props.total_memory / 1024**3:.1f} GB')
        print(f'    Compute capability: {props.major}.{props.minor}')
else:
    print('ERROR: No GPU detected!')
"

echo ""
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ GPU 測試通過！${NC}"
else
    echo -e "${RED}✗ GPU 測試失敗${NC}"
    echo ""
    echo "可能的解決方案："
    echo "1. 確認 NVIDIA Container Toolkit 已安裝："
    echo "   sudo apt-get install -y nvidia-container-toolkit"
    echo "   sudo systemctl restart docker"
    echo ""
    echo "2. 配置 Docker daemon："
    echo "   sudo nvidia-ctk runtime configure --runtime=docker"
    echo "   sudo systemctl restart docker"
fi
