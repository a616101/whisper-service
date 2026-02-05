# WhisperX Subtitle Service
# 生產級 Dockerfile（GPU 支援）

# 明確指定平台為 linux/amd64
FROM --platform=linux/amd64 pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

LABEL maintainer="whisper-service"
LABEL description="WhisperX Subtitle Service with GPU support"

# 設置工作目錄
WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 複製依賴檔案
COPY requirements.txt .

# 安裝 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式碼
COPY config/ ./config/
COPY utils/ ./utils/
COPY app/ ./app/
COPY workers/ ./workers/

# 創建快取目錄
RUN mkdir -p /app/cache/models /app/cache/huggingface

# 設置環境變數
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000

# 預設暴露端口
EXPOSE 8000

# 健康檢查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/v1/health || exit 1

# 預設啟動命令（API 服務）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
