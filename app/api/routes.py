"""
API 路由定義
"""
import os
import json
import logging
import tempfile
from typing import Optional, Literal
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse, JSONResponse

from app.core.transcriber import get_transcriber
from app.services.subtitle import SubtitleGenerator, SubtitleOptions, SegmentationConfig
from app.services.correction import TextCorrector
from app.api.schemas import (
    TranscriptionResponse,
    CorrectionResponse,
    TaskStatus,
    HealthResponse,
)
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# === 健康檢查 ===

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """健康檢查端點"""
    import torch

    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0

    # 檢查模型是否已載入
    transcriber = get_transcriber()
    model_loaded = transcriber._asr_model is not None
    diarization_available = settings.hf_token is not None

    status = "healthy"
    if not gpu_available:
        status = "degraded"
    if not model_loaded:
        status = "degraded"

    return HealthResponse(
        status=status,
        version="1.0.0",
        gpu_available=gpu_available,
        model_loaded=model_loaded,
        diarization_available=diarization_available,
        details={
            "gpu_count": gpu_count,
            "model_size": settings.model_size,
            "compute_type": settings.compute_type,
        }
    )


# === 同步轉寫 ===

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(..., description="音訊或影片檔案"),
    language: Optional[str] = Form(None, description="語言代碼 (zh/en/ja 等)"),
    diarize: bool = Form(True, description="是否進行說話者分離"),
    min_speakers: Optional[int] = Form(None, description="最少說話者數"),
    max_speakers: Optional[int] = Form(None, description="最多說話者數"),
):
    """
    同步轉寫

    適用於 < 15 分鐘的音訊檔案。
    長音檔請使用 /transcribe/async 端點。
    """
    # 儲存上傳的檔案
    suffix = os.path.splitext(file.filename or "audio")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        audio_path = tmp.name

    try:
        transcriber = get_transcriber()
        result = transcriber.transcribe(
            audio_path,
            language=language,
            diarize=diarize,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )

        return TranscriptionResponse(
            language=result.language,
            duration=result.duration,
            segments=result.segments,
            speakers=result.speakers or []
        )

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 清理臨時檔案
        try:
            os.remove(audio_path)
        except:
            pass


# === 字幕生成 ===

@router.post("/subtitle")
async def generate_subtitle(
    file: UploadFile = File(..., description="音訊或影片檔案"),
    language: Optional[str] = Form(None),
    diarize: bool = Form(True),
    format: Literal["srt", "vtt", "json"] = Form("srt"),
    include_speaker: bool = Form(True),
    speaker_format: Literal["prefix", "tag", "none"] = Form("prefix"),
    max_chars_per_line: int = Form(42),
    # 專業分段參數（參考 Turboscribe / stable-ts）
    max_segment_duration: float = Form(5.0, description="每段字幕最長秒數"),
    min_segment_duration: float = Form(1.0, description="每段字幕最短秒數"),
    max_chars_per_segment: int = Form(30, description="每段最大字數（中文建議 20-35）"),
    target_cps: float = Form(12.0, description="目標 CPS（每秒字數，中文建議 10-14）"),
    pause_threshold: float = Form(0.3, description="停頓分段閾值（秒）"),
    split_on_speaker_change: bool = Form(True, description="不同說話者是否分段"),
    convert_to_traditional: bool = Form(True, description="是否轉換為繁體中文"),
):
    """
    生成字幕檔案

    專業字幕分段參數說明：
    - max_segment_duration: 最長時長，業界標準 7 秒
    - min_segment_duration: 最短時長，1.2 秒讓眼睛有時間定位
    - pause_threshold: 語音停頓超過此值會觸發分段
    - target_cps: 目標每秒字數，12-17 為最佳閱讀速度

    支援格式：
    - srt: SubRip 字幕格式
    - vtt: WebVTT 格式
    - json: video.js 相容 JSON
    """
    # 儲存上傳的檔案
    suffix = os.path.splitext(file.filename or "audio")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        audio_path = tmp.name

    try:
        # 轉寫
        transcriber = get_transcriber()
        result = transcriber.transcribe(
            audio_path,
            language=language,
            diarize=diarize
        )

        # 配置專業字幕分段（參考 Turboscribe / stable-ts）
        seg_config = SegmentationConfig(
            target_cps=target_cps,
            max_duration=max_segment_duration,
            min_duration=min_segment_duration,
            pause_threshold=pause_threshold,
            max_chars_per_line=max_chars_per_segment,
            split_on_speaker_change=split_on_speaker_change
        )

        options = SubtitleOptions(
            include_speaker=include_speaker,
            max_chars_per_line=max_chars_per_segment,
            speaker_format=speaker_format,
            segmentation=seg_config
        )
        generator = SubtitleGenerator(options)

        # 專業字幕分段：基於 word-level timestamps
        segments = generator.resegment(
            result.segments,
            max_duration=max_segment_duration,
            min_duration=min_segment_duration,
            pause_threshold=pause_threshold,
            split_on_speaker_change=split_on_speaker_change,
            convert_to_traditional=convert_to_traditional
        )

        subtitle = generator.generate(
            segments,
            format=format,
            language=result.language,
            duration=result.duration,
            speakers=result.speakers
        )

        # 返回對應格式
        if format == "json":
            return JSONResponse(content=subtitle)
        elif format == "vtt":
            return PlainTextResponse(
                content=subtitle,
                media_type="text/vtt",
                headers={"Content-Disposition": "attachment; filename=subtitle.vtt"}
            )
        else:  # srt
            return PlainTextResponse(
                content=subtitle,
                media_type="text/plain",
                headers={"Content-Disposition": "attachment; filename=subtitle.srt"}
            )

    except Exception as e:
        logger.error(f"Subtitle generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        try:
            os.remove(audio_path)
        except:
            pass


# === 文字校正 ===

@router.post("/correct", response_model=CorrectionResponse)
async def correct_text(
    text: str = Form(..., description="要校正的文字"),
    glossary_json: Optional[str] = Form(None, description="JSON 格式的詞彙表"),
    use_llm: bool = Form(False, description="是否使用 LLM 校正"),
):
    """
    文字校正

    支援：
    - Glossary 詞彙替換
    - 常見錯誤修正
    - 數字格式化
    - (可選) LLM 校正
    """
    glossary = None
    if glossary_json:
        try:
            glossary = json.loads(glossary_json)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid glossary JSON")

    corrector = TextCorrector()
    result = corrector.correct(text, additional_glossary=glossary, use_llm=use_llm)

    return CorrectionResponse(
        original_text=result.original_text,
        corrected_text=result.corrected_text,
        changes=result.changes,
        applied_rules=result.applied_rules
    )


# === 重新對齊 ===

@router.post("/align")
async def realign(
    file: UploadFile = File(..., description="原始音訊檔案"),
    corrected_text: str = Form(..., description="校正後的文字"),
    language: Optional[str] = Form(None),
):
    """
    校正後重新對齊

    當你對轉寫文字做了修改後，使用此端點重新產生準確的時間軸。
    """
    import whisperx

    # 儲存上傳的檔案
    suffix = os.path.splitext(file.filename or "audio")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        audio_path = tmp.name

    try:
        transcriber = get_transcriber()

        # 載入音訊
        audio = whisperx.load_audio(audio_path)
        duration = len(audio) / 16000.0

        # 確定語言
        if not language:
            # 做一次快速轉寫來偵測語言
            quick_result = transcriber.asr_model.transcribe(
                audio,
                batch_size=1,
                language=None
            )
            language = quick_result.get("language", "zh")

        # 進行對齊
        align_model, metadata = transcriber.get_align_model(language)

        # 將校正後的文字作為單一 segment
        fake_segments = [{
            "start": 0.0,
            "end": duration,
            "text": corrected_text
        }]

        aligned = whisperx.align(
            fake_segments,
            align_model,
            metadata,
            audio,
            settings.device,
            return_char_alignments=False
        )

        # 格式化結果
        segments = transcriber._format_segments(aligned.get("segments", []))

        return JSONResponse({
            "language": language,
            "duration": duration,
            "segments": segments
        })

    except Exception as e:
        logger.error(f"Alignment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        try:
            os.remove(audio_path)
        except:
            pass


# === 異步任務相關（需要 Celery/Redis）===

# 任務狀態存儲（簡易版，生產環境用 Redis）
_task_store = {}


@router.post("/transcribe/async")
async def transcribe_async(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    diarize: bool = Form(True),
    webhook_url: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None,
):
    """
    異步轉寫（適合長音檔）

    返回 task_id，可通過 /task/{task_id} 查詢狀態。
    可選設定 webhook_url，完成時會回調。
    """
    import uuid

    task_id = str(uuid.uuid4())

    # 儲存檔案
    suffix = os.path.splitext(file.filename or "audio")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        audio_path = tmp.name

    # 初始化任務狀態
    _task_store[task_id] = {
        "status": "pending",
        "progress": 0.0,
        "result": None,
        "error": None,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }

    # 啟動背景任務
    # 注意：這是簡易版，生產環境應使用 Celery
    if background_tasks:
        background_tasks.add_task(
            _process_async_transcription,
            task_id,
            audio_path,
            language,
            diarize,
            webhook_url
        )

    return JSONResponse({
        "task_id": task_id,
        "status": "pending",
        "message": "Task created. Use /task/{task_id} to check status."
    })


async def _process_async_transcription(
    task_id: str,
    audio_path: str,
    language: Optional[str],
    diarize: bool,
    webhook_url: Optional[str]
):
    """處理異步轉寫（背景任務）"""
    try:
        _task_store[task_id]["status"] = "processing"
        _task_store[task_id]["updated_at"] = datetime.utcnow().isoformat()

        transcriber = get_transcriber()
        result = transcriber.transcribe(
            audio_path,
            language=language,
            diarize=diarize
        )

        _task_store[task_id]["status"] = "completed"
        _task_store[task_id]["progress"] = 1.0
        _task_store[task_id]["result"] = {
            "language": result.language,
            "duration": result.duration,
            "segments": result.segments,
            "speakers": result.speakers or []
        }

        # Webhook 回調
        if webhook_url:
            import httpx
            async with httpx.AsyncClient() as client:
                await client.post(webhook_url, json=_task_store[task_id])

    except Exception as e:
        logger.error(f"Async transcription failed: {e}")
        _task_store[task_id]["status"] = "failed"
        _task_store[task_id]["error"] = str(e)

    finally:
        _task_store[task_id]["updated_at"] = datetime.utcnow().isoformat()
        try:
            os.remove(audio_path)
        except:
            pass


@router.get("/task/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """查詢異步任務狀態"""
    if task_id not in _task_store:
        raise HTTPException(status_code=404, detail="Task not found")

    task = _task_store[task_id]

    return TaskStatus(
        task_id=task_id,
        status=task["status"],
        progress=task["progress"],
        result=task.get("result"),
        error=task.get("error"),
        created_at=task["created_at"],
        updated_at=task["updated_at"]
    )
