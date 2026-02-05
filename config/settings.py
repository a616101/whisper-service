"""
應用配置管理
使用 pydantic-settings 進行環境變數管理
"""
import os
from typing import Optional, Literal
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """應用設定"""

    # === 基本設定 ===
    app_name: str = "WhisperX Subtitle Service"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # === HuggingFace ===
    hf_token: Optional[str] = None

    # === Whisper 模型設定 ===
    model_size: Literal["tiny", "base", "small", "medium", "large-v2", "large-v3"] = "large-v3"
    compute_type: Literal["float16", "float32", "int8", "int8_float16"] = "float16"
    device: str = "auto"  # auto, cuda, mps, cpu
    batch_size: int = 16

    def get_device(self) -> str:
        """
        自動偵測最佳設備

        注意：WhisperX 使用 faster-whisper (CTranslate2)，
        只支援 CUDA 和 CPU，不支援 MPS。
        Mac 用戶必須使用 CPU 模式。
        """
        # 如果明確指定了設備
        if self.device not in ("auto", "mps"):
            return self.device

        # MPS 不被 faster-whisper 支援，強制使用 CPU
        if self.device == "mps":
            import logging
            logging.getLogger(__name__).warning(
                "MPS is not supported by faster-whisper/CTranslate2. "
                "Falling back to CPU mode."
            )
            return "cpu"

        # 自動偵測
        import torch
        if torch.cuda.is_available():
            return "cuda"
        else:
            # Mac (MPS) 和其他都用 CPU
            return "cpu"

    def get_compute_type(self) -> str:
        """根據設備選擇計算類型"""
        device = self.get_device()
        if device == "cpu":
            return "int8"  # CPU 用 int8 較快
        else:
            return self.compute_type

    # === 長音檔處理 ===
    chunk_length_s: int = 600  # 10 分鐘一段
    chunk_overlap_s: int = 2   # 分段重疊秒數
    max_audio_length_s: int = 14400  # 最大 4 小時

    # === VAD 設定 ===
    vad_enabled: bool = True
    vad_threshold: float = 0.5
    min_silence_duration_ms: int = 500
    min_speech_duration_ms: int = 250

    # === Diarization 設定 ===
    diarization_enabled: bool = True
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None

    # === 音訊前處理 ===
    target_sample_rate: int = 16000
    normalize_audio: bool = True
    target_loudness_lufs: float = -23.0

    # === 字幕設定 ===
    max_chars_per_line: int = 42  # 單行最大字數
    max_lines_per_segment: int = 2  # 每段最多行數
    min_segment_duration_s: float = 1.0
    max_segment_duration_s: float = 7.0

    # === Redis / Celery ===
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: Optional[str] = None
    celery_result_backend: Optional[str] = None

    # === 並發控制 ===
    max_concurrent_tasks: int = 4
    task_timeout_s: int = 3600  # 1 小時超時

    # === 快取 ===
    cache_dir: str = "/app/cache"
    model_cache_dir: str = "/app/cache/models"

    # === 監控 ===
    enable_metrics: bool = True
    metrics_port: int = 9090

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def celery_broker(self) -> str:
        return self.celery_broker_url or self.redis_url

    @property
    def celery_backend(self) -> str:
        return self.celery_result_backend or self.redis_url


@lru_cache()
def get_settings() -> Settings:
    """獲取設定單例"""
    return Settings()


# 方便直接 import
settings = get_settings()
