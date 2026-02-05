# 部署指南

## 系統需求

### 硬體需求

| 組件 | 最低需求 | 建議配置 |
|------|---------|---------|
| GPU | RTX 3060 (8GB) | RTX 3090 / A100 |
| RAM | 16GB | 32GB+ |
| 儲存 | 50GB SSD | 100GB+ NVMe |
| CPU | 4 cores | 8+ cores |

### 軟體需求

- Docker 24.0+
- Docker Compose v2
- NVIDIA Driver 525+
- NVIDIA Container Toolkit

---

## 快速部署

### 1. 安裝 NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 2. 配置環境

```bash
# 克隆專案
cd whisper-service

# 配置環境變數
cp .env.example .env

# 編輯 .env，填入 HF_TOKEN
nano .env
```

### 3. 獲取 HuggingFace Token

1. 前往 https://huggingface.co/settings/tokens
2. 建立新的 Access Token
3. 接受 pyannote 模型條款：
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

### 4. 啟動服務

```bash
# 開發模式（單機）
./scripts/start.sh dev

# 生產模式（單 GPU + Redis）
./scripts/start.sh prod

# 多 GPU 模式
./scripts/start.sh multi-gpu

# 完整模式（多 GPU + 監控）
./scripts/start.sh full
```

### 5. 驗證部署

```bash
# 健康檢查
curl http://localhost:8000/api/v1/health

# 測試轉寫
curl -X POST http://localhost:8000/api/v1/transcribe \
  -F "file=@test.wav" \
  -F "diarize=true"
```

---

## 部署架構

### 單機部署（開發/小規模）

```
┌─────────────────────────────────────┐
│           Docker Host               │
│  ┌─────────────────────────────┐   │
│  │     whisper-api (GPU 0)     │   │
│  │     FastAPI + WhisperX      │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

### 生產部署（多 GPU）

```
┌─────────────────────────────────────────────────────┐
│                    Docker Host                       │
│  ┌───────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │   Redis   │  │   API    │  │   Celery Workers │  │
│  │  (Queue)  │──│  Server  │──│  (GPU 0, 1, ...) │  │
│  └───────────┘  └──────────┘  └──────────────────┘  │
│                       │                              │
│                 ┌─────┴─────┐                        │
│                 │  Flower   │                        │
│                 │ (Monitor) │                        │
│                 └───────────┘                        │
└─────────────────────────────────────────────────────┘
```

---

## 多 GPU 配置

### 自動 GPU 分配

每個 Worker 會自動綁定到不同的 GPU：

```yaml
# docker-compose.yml 中的 Worker 配置
whisper-worker-0:
  environment:
    - NVIDIA_VISIBLE_DEVICES=0
    - CUDA_VISIBLE_DEVICES=0

whisper-worker-1:
  environment:
    - NVIDIA_VISIBLE_DEVICES=1
    - CUDA_VISIBLE_DEVICES=0
```

### DGX 環境（8 GPU）

創建 `docker-compose.dgx.yml`：

```yaml
services:
  whisper-worker-0:
    extends:
      file: docker-compose.yml
      service: whisper-worker-0
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - device_ids: ['0']

  # 重複 1-7...
```

啟動：
```bash
docker compose -f docker-compose.yml -f docker-compose.dgx.yml up -d
```

---

## 效能調優

### 1. 模型選擇

| 模型 | VRAM | 速度 | 品質 |
|------|------|------|------|
| large-v3 | ~10GB | 1x | ★★★★★ |
| medium | ~5GB | 2x | ★★★★ |
| small | ~2GB | 4x | ★★★ |
| base | ~1GB | 8x | ★★ |

### 2. 批次大小

根據 GPU 記憶體調整：

| GPU | 建議 BATCH_SIZE |
|-----|----------------|
| 8GB | 8-12 |
| 16GB | 16-24 |
| 24GB | 24-32 |
| 80GB (A100) | 32-64 |

### 3. 計算精度

```bash
# float16（推薦，平衡速度和精度）
COMPUTE_TYPE=float16

# int8（省記憶體，稍微降低精度）
COMPUTE_TYPE=int8_float16

# float32（最高精度，最慢）
COMPUTE_TYPE=float32
```

---

## 監控和日誌

### Flower 監控

```bash
# 啟用 Flower
./scripts/start.sh monitoring

# 訪問監控面板
open http://localhost:5555
```

### 日誌查看

```bash
# 查看所有日誌
docker compose logs -f

# 只看 API 日誌
docker compose logs -f whisper-api

# 只看 Worker 日誌
docker compose logs -f whisper-worker-0
```

### Prometheus 指標（可選）

在 `config/settings.py` 啟用：
```python
enable_metrics: bool = True
metrics_port: int = 9090
```

---

## 故障排除

### 常見問題

**1. GPU 無法使用**
```bash
# 檢查 nvidia-docker
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

**2. HF_TOKEN 無效**
```bash
# 確認 token 有效
curl -H "Authorization: Bearer $HF_TOKEN" \
  https://huggingface.co/api/whoami
```

**3. 記憶體不足**
```bash
# 減小批次大小
BATCH_SIZE=8

# 使用較小模型
MODEL_SIZE=medium
```

**4. Diarization 失敗**
- 檢查 HF_TOKEN 是否接受了 pyannote 模型條款
- 服務會自動降級為無說話者標籤模式

---

## 備份和恢復

### 備份模型快取

```bash
# 備份
tar -czvf whisper-cache-backup.tar.gz cache/

# 恢復
tar -xzvf whisper-cache-backup.tar.gz
```

### 備份 Redis 資料

```bash
# 進入 Redis 容器
docker exec -it whisper-redis redis-cli BGSAVE

# 複製 dump.rdb
docker cp whisper-redis:/data/dump.rdb ./backup/
```
