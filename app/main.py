"""
FastAPI 應用入口

重要：PyTorch 相容性 patch 必須在最開頭執行
"""
# === 最優先：PyTorch 相容性設定 ===
# 必須在任何可能 import torch 的模組之前執行
import functools

def _apply_torch_patch():
    """立即 patch torch.load，在任何其他 import 之前"""
    try:
        import torch
        if hasattr(torch, '_whisperx_patched'):
            return  # 已經 patch 過

        _original = torch.load

        @functools.wraps(_original)
        def _patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return _original(*args, **kwargs)

        torch.load = _patched_load
        torch._whisperx_patched = True
        print(f"[torch_compat] Patched torch.load (PyTorch {torch.__version__})")
    except Exception as e:
        print(f"[torch_compat] Warning: {e}")

_apply_torch_patch()
# === 結束 PyTorch patch ===

import logging
from contextlib import asynccontextmanager

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from app.api.routes import router as api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用生命週期管理"""
    # 啟動時預載入模型（可選）
    logger.info("Starting WhisperX Subtitle Service")

    # === GPU 診斷資訊 ===
    import torch
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.warning("No GPU detected! Running on CPU (slow)")
        # 顯示可能的原因
        import os
        logger.info(f"  NVIDIA_VISIBLE_DEVICES: {os.environ.get('NVIDIA_VISIBLE_DEVICES', 'not set')}")
        logger.info(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    actual_device = settings.get_device()
    logger.info(f"Model: {settings.model_size}, Device: {actual_device}")

    if settings.debug:
        logger.info("Debug mode enabled")

    # 可以在這裡預載入模型以加快首次請求
    # from app.core.transcriber import get_transcriber
    # get_transcriber()  # 觸發模型載入

    yield

    # 關閉時清理
    logger.info("Shutting down WhisperX Subtitle Service")


# 創建應用
app = FastAPI(
    title="WhisperX Subtitle Service",
    description="""
    影片轉字幕服務 API

    功能：
    - ASR 語音轉文字（支援中/英/中英混）
    - 說話者分離（Speaker Diarization）
    - 字幕生成（SRT / VTT / JSON）
    - 文字校正與重新對齊
    - 長音檔異步處理
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生產環境請限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 註冊路由
app.include_router(api_router, prefix="/api/v1", tags=["transcription"])


# 根路徑
@app.get("/")
async def root():
    return {
        "service": "WhisperX Subtitle Service",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
