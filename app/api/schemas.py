"""
API Request/Response Schemas
"""
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field


# === Request Schemas ===

class TranscribeRequest(BaseModel):
    """轉寫請求參數"""
    language: Optional[str] = Field(
        None,
        description="語言代碼 (zh/en/ja 等)，None 為自動偵測"
    )
    diarize: bool = Field(
        True,
        description="是否進行說話者分離"
    )
    min_speakers: Optional[int] = Field(
        None,
        description="最少說話者數"
    )
    max_speakers: Optional[int] = Field(
        None,
        description="最多說話者數"
    )


class SubtitleRequest(BaseModel):
    """字幕生成請求參數"""
    language: Optional[str] = Field(None, description="語言代碼")
    diarize: bool = Field(True, description="是否進行說話者分離")
    format: Literal["srt", "vtt", "json"] = Field("srt", description="輸出格式")
    include_speaker: bool = Field(True, description="是否包含說話者標籤")
    speaker_format: Literal["prefix", "tag", "none"] = Field(
        "prefix",
        description="說話者標籤格式"
    )
    max_chars_per_line: int = Field(42, description="每行最大字數")


class CorrectionRequest(BaseModel):
    """校正請求"""
    text: str = Field(..., description="要校正的文字")
    glossary: Optional[Dict[str, str]] = Field(
        None,
        description="額外詞彙表 {'錯誤': '正確'}"
    )
    use_llm: bool = Field(False, description="是否使用 LLM 校正")


class AlignRequest(BaseModel):
    """重新對齊請求"""
    corrected_text: str = Field(..., description="校正後的文字")
    language: Optional[str] = Field(None, description="語言代碼")


class AsyncTranscribeRequest(BaseModel):
    """異步轉寫請求"""
    language: Optional[str] = None
    diarize: bool = True
    webhook_url: Optional[str] = Field(
        None,
        description="完成時回調的 webhook URL"
    )
    callback_headers: Optional[Dict[str, str]] = Field(
        None,
        description="webhook 請求的額外 headers"
    )


# === Response Schemas ===

class WordInfo(BaseModel):
    """單詞資訊"""
    start: float
    end: float
    word: str
    confidence: float = 1.0
    speaker: Optional[str] = None


class SegmentInfo(BaseModel):
    """段落資訊"""
    id: int
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    words: List[WordInfo] = []


class TranscriptionResponse(BaseModel):
    """轉寫結果"""
    language: str
    duration: float
    segments: List[SegmentInfo]
    speakers: List[str] = []


class VideoJsSubtitleResponse(BaseModel):
    """video.js 相容的字幕 JSON"""
    version: str = "1.0"
    language: str
    duration: float
    segments: List[SegmentInfo]
    speakers: List[str] = []
    metadata: Dict[str, Any] = {}


class CorrectionResponse(BaseModel):
    """校正結果"""
    original_text: str
    corrected_text: str
    changes: List[Dict[str, Any]] = []
    applied_rules: List[str] = []


class TaskStatus(BaseModel):
    """異步任務狀態"""
    task_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    progress: float = 0.0
    result: Optional[TranscriptionResponse] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


class HealthResponse(BaseModel):
    """健康檢查回應"""
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    gpu_available: bool
    model_loaded: bool
    diarization_available: bool
    details: Dict[str, Any] = {}


class ErrorResponse(BaseModel):
    """錯誤回應"""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
