"""
Celery 任務定義
"""
import os
import logging
import tempfile
from typing import Optional, Dict, Any

from celery import Task
import httpx

from workers.celery_app import celery_app
from app.core.transcriber import get_transcriber, TranscriptionResult
from app.services.subtitle import SubtitleGenerator, SubtitleOptions
from app.services.correction import TextCorrector

logger = logging.getLogger(__name__)


class TranscriptionTask(Task):
    """
    轉寫任務基類

    提供模型懶載入和錯誤處理
    """
    _transcriber = None

    @property
    def transcriber(self):
        if self._transcriber is None:
            self._transcriber = get_transcriber()
        return self._transcriber

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """任務失敗處理"""
        logger.error(f"Task {task_id} failed: {exc}")

        # 如果有 webhook，通知失敗
        webhook_url = kwargs.get("webhook_url")
        if webhook_url:
            self._send_webhook(webhook_url, {
                "task_id": task_id,
                "status": "failed",
                "error": str(exc)
            })

    def _send_webhook(self, url: str, data: dict):
        """發送 webhook 通知"""
        try:
            with httpx.Client() as client:
                client.post(url, json=data, timeout=30)
        except Exception as e:
            logger.error(f"Webhook failed: {e}")


@celery_app.task(
    bind=True,
    base=TranscriptionTask,
    name="workers.tasks.transcribe_task",
    max_retries=3,
    default_retry_delay=60
)
def transcribe_task(
    self,
    audio_path: str,
    language: Optional[str] = None,
    diarize: bool = True,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    webhook_url: Optional[str] = None,
    output_format: str = "json"
) -> Dict[str, Any]:
    """
    異步轉寫任務

    Args:
        audio_path: 音訊檔案路徑
        language: 語言代碼
        diarize: 是否進行說話者分離
        min_speakers: 最少說話者數
        max_speakers: 最多說話者數
        webhook_url: 完成回調 URL
        output_format: 輸出格式 (json/srt/vtt)

    Returns:
        轉寫結果
    """
    try:
        # 更新任務狀態
        self.update_state(state="PROCESSING", meta={"progress": 0.1})

        # 執行轉寫
        result = self.transcriber.transcribe(
            audio_path,
            language=language,
            diarize=diarize,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )

        self.update_state(state="PROCESSING", meta={"progress": 0.8})

        # 格式化輸出
        output = {
            "language": result.language,
            "duration": result.duration,
            "segments": result.segments,
            "speakers": result.speakers or []
        }

        # 如果需要字幕格式
        if output_format in ("srt", "vtt"):
            generator = SubtitleGenerator()
            subtitle = generator.generate(
                result.segments,
                format=output_format,
                language=result.language,
                duration=result.duration,
                speakers=result.speakers
            )
            output["subtitle"] = subtitle

        self.update_state(state="SUCCESS", meta={"progress": 1.0})

        # Webhook 回調
        if webhook_url:
            self._send_webhook(webhook_url, {
                "task_id": self.request.id,
                "status": "completed",
                "result": output
            })

        return output

    except Exception as e:
        logger.error(f"Transcription failed: {e}")

        # 重試
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)

        raise

    finally:
        # 清理臨時檔案
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except:
            pass


@celery_app.task(
    bind=True,
    base=TranscriptionTask,
    name="workers.tasks.subtitle_task"
)
def subtitle_task(
    self,
    audio_path: str,
    format: str = "srt",
    language: Optional[str] = None,
    diarize: bool = True,
    include_speaker: bool = True,
    webhook_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    字幕生成任務
    """
    try:
        self.update_state(state="PROCESSING", meta={"progress": 0.1})

        # 轉寫
        result = self.transcriber.transcribe(
            audio_path,
            language=language,
            diarize=diarize
        )

        self.update_state(state="PROCESSING", meta={"progress": 0.7})

        # 生成字幕
        options = SubtitleOptions(include_speaker=include_speaker)
        generator = SubtitleGenerator(options)

        subtitle = generator.generate(
            result.segments,
            format=format,
            language=result.language,
            duration=result.duration,
            speakers=result.speakers
        )

        output = {
            "format": format,
            "language": result.language,
            "duration": result.duration,
            "subtitle": subtitle if isinstance(subtitle, str) else subtitle
        }

        if webhook_url:
            self._send_webhook(webhook_url, {
                "task_id": self.request.id,
                "status": "completed",
                "result": output
            })

        return output

    finally:
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except:
            pass


@celery_app.task(name="workers.tasks.correct_and_realign_task")
def correct_and_realign_task(
    audio_path: str,
    segments: list,
    glossary: Optional[Dict[str, str]] = None,
    language: Optional[str] = None
) -> Dict[str, Any]:
    """
    校正並重新對齊任務

    1. 對 segments 文字進行校正
    2. 重新對齊以更新時間軸
    """
    import whisperx
    from config.settings import settings

    try:
        # 校正文字
        corrector = TextCorrector()
        corrected_segments = corrector.correct_segments(segments, glossary)

        # 合併文字進行重新對齊
        full_text = " ".join(seg["text"] for seg in corrected_segments)

        # 載入音訊和對齊模型
        audio = whisperx.load_audio(audio_path)
        duration = len(audio) / 16000.0

        if not language:
            language = "zh"

        transcriber = get_transcriber()
        align_model, metadata = transcriber.get_align_model(language)

        # 重新對齊
        fake_segments = [{"start": 0.0, "end": duration, "text": full_text}]
        aligned = whisperx.align(
            fake_segments,
            align_model,
            metadata,
            audio,
            settings.device,
            return_char_alignments=False
        )

        return {
            "language": language,
            "duration": duration,
            "segments": transcriber._format_segments(aligned.get("segments", [])),
            "correction_applied": True
        }

    finally:
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except:
            pass


# === 批量處理任務 ===

@celery_app.task(name="workers.tasks.batch_transcribe_task")
def batch_transcribe_task(
    audio_paths: list,
    language: Optional[str] = None,
    diarize: bool = True,
    webhook_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    批量轉寫任務

    處理多個音訊檔案
    """
    results = []
    transcriber = get_transcriber()

    for i, audio_path in enumerate(audio_paths):
        try:
            result = transcriber.transcribe(
                audio_path,
                language=language,
                diarize=diarize
            )

            results.append({
                "file": os.path.basename(audio_path),
                "status": "success",
                "language": result.language,
                "duration": result.duration,
                "segments": result.segments,
                "speakers": result.speakers or []
            })

        except Exception as e:
            results.append({
                "file": os.path.basename(audio_path),
                "status": "failed",
                "error": str(e)
            })

        finally:
            try:
                os.remove(audio_path)
            except:
                pass

    output = {
        "total": len(audio_paths),
        "success": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "results": results
    }

    if webhook_url:
        with httpx.Client() as client:
            client.post(webhook_url, json=output, timeout=30)

    return output
