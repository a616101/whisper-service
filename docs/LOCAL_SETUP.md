# Mac 本地開發指南（MPS 加速）

在 Mac 上本地執行可以使用 Metal Performance Shaders (MPS) 進行 GPU 加速，效能比 Docker CPU 模式快很多。

## 系統需求

- macOS 12.3+（Monterey 或更新）
- Apple Silicon (M1/M2/M3/M4) 或 Intel Mac
- Python 3.10 或 3.11
- Homebrew

## 快速安裝

### 1. 安裝系統依賴

```bash
# 安裝 ffmpeg
brew install ffmpeg

# 確認 Python 版本
python3 --version  # 需要 3.10 或 3.11
```

### 2. 建立專案環境

```bash
cd whisper-service

# 建立虛擬環境
python3 -m venv venv
source venv/bin/activate

# 升級 pip
pip install --upgrade pip
```

### 3. 安裝 PyTorch（MPS 支援）

```bash
# Apple Silicon Mac（M1/M2/M3/M4）
pip install torch torchaudio

# 驗證 MPS 可用
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### 4. 安裝專案依賴

```bash
pip install -r requirements.mac.txt
```

### 5. 配置環境變數

```bash
# 複製範例配置
cp .env.example .env

# 編輯配置
nano .env
```

重要設定：
```bash
# HuggingFace Token（可選，用於 diarization）
HF_TOKEN=your_token_here

# Mac 本地建議設定
MODEL_SIZE=base        # 小模型較快，large-v3 更準確
COMPUTE_TYPE=float32   # MPS 支援 float32
DEVICE=mps             # 使用 MPS 加速
BATCH_SIZE=8
```

### 6. 啟動服務

```bash
# 開發模式（自動重載）
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 或使用 Python 直接執行
python -m app.main
```

### 7. 測試服務

```bash
# 健康檢查
curl http://localhost:8000/api/v1/health

# 測試轉寫（需要準備音訊檔案）
curl -X POST http://localhost:8000/api/v1/transcribe \
  -F "file=@test.wav" \
  -F "diarize=false"
```

---

## 效能比較

| 環境 | 設備 | 1 分鐘音訊處理時間 |
|------|------|-------------------|
| Docker CPU | M4 Max | ~60-90 秒 |
| 本地 MPS | M4 Max | ~10-15 秒 |
| 本地 MPS | M1 Pro | ~20-30 秒 |

**建議**：開發和測試時使用本地 MPS 模式，生產環境使用 Docker + GPU。

---

## 配置修改

### 使用 MPS 加速

修改 `config/settings.py`：

```python
# 預設使用 MPS（如果可用）
device: str = "mps"  # 改為 "mps"
```

或透過環境變數：
```bash
export DEVICE=mps
```

### 模型選擇建議

| 模型 | 記憶體需求 | 速度 | 品質 | 建議場景 |
|------|-----------|------|------|---------|
| tiny | ~1GB | 最快 | ★★ | 快速測試 |
| base | ~1GB | 快 | ★★★ | 開發測試 |
| small | ~2GB | 中 | ★★★★ | 一般使用 |
| medium | ~5GB | 慢 | ★★★★ | 高品質需求 |
| large-v3 | ~10GB | 最慢 | ★★★★★ | 最高品質 |

Mac M4 Max (128GB) 可以輕鬆使用 large-v3。

---

## 常見問題

### MPS 相關

**Q: `torch.backends.mps.is_available()` 返回 False**
```bash
# 確認 macOS 版本
sw_vers

# 升級 PyTorch
pip install --upgrade torch torchaudio
```

**Q: MPS 記憶體不足**
```bash
# 減小批次大小
export BATCH_SIZE=4

# 使用較小模型
export MODEL_SIZE=small
```

### Diarization 相關

**Q: pyannote 模型下載失敗**
1. 確認 HF_TOKEN 已設定且有效
2. 前往 HuggingFace 接受模型條款：
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

**Q: 不需要 diarization**
```bash
# 轉寫時設定 diarize=false
curl -X POST http://localhost:8000/api/v1/transcribe \
  -F "file=@audio.wav" \
  -F "diarize=false"
```

---

## 開發工具

### 安裝開發依賴

```bash
pip install pytest pytest-asyncio black isort mypy
```

### 程式碼格式化

```bash
black app/ utils/ config/
isort app/ utils/ config/
```

### 執行測試

```bash
pytest tests/ -v
```

---

## 完整啟動指令

```bash
# 一鍵啟動（本地 MPS 模式）
cd whisper-service
source venv/bin/activate
export DEVICE=mps
export MODEL_SIZE=base
uvicorn app.main:app --reload
```
