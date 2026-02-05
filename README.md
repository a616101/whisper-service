# WhisperX 字幕服務 - 生產級部署方案

## 🎯 功能概覽

| 功能 | 描述 |
|------|------|
| **ASR 轉寫** | WhisperX large-v3，支援中/英/中英混 |
| **字詞時間軸** | Word-level timestamps（精確到 word） |
| **說話者分離** | pyannote diarization（SPEAKER_00/01...） |
| **輸出格式** | SRT / VTT / JSON（video.js 相容） |
| **長音檔支援** | 自動分段處理（支援 > 1 小時） |
| **高並發** | Redis 任務佇列 + 多 GPU Worker |
| **品質保證** | VAD 前處理、音訊規範化、中英混斷句優化 |

---

## 📁 專案結構

```
whisper-service/
├── docker-compose.yml          # 生產級 compose（含 Redis、多 worker）
├── docker-compose.dev.yml      # 開發用單機版
├── Dockerfile                  # GPU 運行環境
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI 入口
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py           # API 路由
│   │   └── schemas.py          # Pydantic schemas
│   ├── core/
│   │   ├── __init__.py
│   │   ├── transcriber.py      # WhisperX 核心邏輯
│   │   ├── diarizer.py         # 說話者分離
│   │   ├── aligner.py          # 強制對齊
│   │   └── preprocessor.py     # VAD + 音訊前處理
│   └── services/
│       ├── __init__.py
│       ├── subtitle.py         # 字幕生成（SRT/VTT/JSON）
│       ├── correction.py       # 錯字校正
│       └── chunker.py          # 長音檔分段
├── workers/
│   ├── __init__.py
│   ├── celery_app.py           # Celery 配置
│   └── tasks.py                # 異步任務定義
├── utils/
│   ├── __init__.py
│   ├── audio.py                # 音訊工具
│   ├── formatting.py           # 時間格式化
│   └── text.py                 # 文字處理（中英混斷句）
├── config/
│   ├── __init__.py
│   └── settings.py             # 環境配置
└── tests/
    └── ...
```

---

## 🚀 快速開始

### 1. 環境準備

```bash
# 設定 HuggingFace Token（pyannote 需要）
export HF_TOKEN="your_huggingface_token"

# 開發模式（單機）
docker compose -f docker-compose.dev.yml up -d --build

# 生產模式（多 worker + Redis）
docker compose up -d --build
```

### 2. API 使用

```bash
# 基本轉寫（JSON 輸出）
curl -X POST http://localhost:8000/api/v1/transcribe \
  -F "file=@video.mp4" \
  -F "diarize=true"

# 字幕輸出（SRT）
curl -X POST http://localhost:8000/api/v1/subtitle \
  -F "file=@video.mp4" \
  -F "format=srt" \
  -F "diarize=true" > output.srt

# 異步任務（長音檔推薦）
curl -X POST http://localhost:8000/api/v1/transcribe/async \
  -F "file=@long_video.mp4" \
  -F "webhook_url=https://your-server/callback"
```

---

## 📊 API 端點

| 端點 | 方法 | 描述 |
|------|------|------|
| `/api/v1/transcribe` | POST | 同步轉寫（< 15 分鐘音檔） |
| `/api/v1/transcribe/async` | POST | 異步轉寫（長音檔） |
| `/api/v1/subtitle` | POST | 生成字幕檔（SRT/VTT/JSON） |
| `/api/v1/correct` | POST | 文字校正 |
| `/api/v1/align` | POST | 校正後重新對齊 |
| `/api/v1/task/{task_id}` | GET | 查詢異步任務狀態 |
| `/health` | GET | 健康檢查 |

---

## 🔧 配置說明

### 環境變數

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `HF_TOKEN` | - | HuggingFace token（必填） |
| `MODEL_SIZE` | large-v3 | Whisper 模型大小 |
| `COMPUTE_TYPE` | float16 | 計算精度 |
| `BATCH_SIZE` | 16 | 批次大小 |
| `CHUNK_LENGTH_S` | 600 | 長音檔分段秒數 |
| `MAX_CONCURRENT_TASKS` | 4 | 每 GPU 最大並發任務 |
| `REDIS_URL` | redis://redis:6379/0 | Redis 連線 |

---

## 📦 JSON Schema（video.js 相容）

```json
{
  "language": "zh",
  "duration": 3600.5,
  "segments": [
    {
      "id": 1,
      "start": 0.0,
      "end": 3.5,
      "text": "大家好，歡迎來到今天的節目",
      "speaker": "SPEAKER_00",
      "words": [
        {"start": 0.0, "end": 0.3, "word": "大家", "confidence": 0.98},
        {"start": 0.3, "end": 0.5, "word": "好", "confidence": 0.99}
      ]
    }
  ],
  "speakers": ["SPEAKER_00", "SPEAKER_01"]
}
```

---

## ⚡ 效能優化建議

### GPU 記憶體使用

| 模型 | VRAM 需求 | 建議 GPU |
|------|-----------|----------|
| large-v3 | ~10GB | RTX 3090 / A100 |
| medium | ~5GB | RTX 3080 |
| small | ~2GB | RTX 3060 |

### 並發策略

- **單 GPU**：建議 1-2 個並發任務
- **多 GPU**：每 GPU 配置獨立 worker，使用 `CUDA_VISIBLE_DEVICES` 隔離
- **DGX 環境**：可配置 8 個 worker，每個綁定一張 GPU

---

## 🛡️ 穩定性機制

1. **Diarization 降級**：HF_TOKEN 無效或 diarization 失敗時，自動降級為無說話者標籤模式
2. **長音檔保護**：自動分段處理，避免 OOM
3. **重試機制**：網路/模型載入失敗時自動重試 3 次
4. **健康檢查**：定期檢測 GPU 狀態和模型載入

---

## 📝 開發指南

詳見 `docs/` 目錄：
- `docs/API.md` - 完整 API 文件
- `docs/DEPLOYMENT.md` - 部署指南
- `docs/TUNING.md` - 效能調優
