"""
WhisperX 核心轉寫模組
包含 ASR、對齊、長音檔分段處理
"""
# === PyTorch patch（必須在 import whisperx 之前）===
import functools
def _ensure_torch_patched():
    try:
        import torch
        if hasattr(torch, '_whisperx_patched'):
            return
        _orig = torch.load
        @functools.wraps(_orig)
        def _p(*a, **kw):
            kw['weights_only'] = False
            return _orig(*a, **kw)
        torch.load = _p
        torch._whisperx_patched = True
    except: pass
_ensure_torch_patched()
# === 結束 patch ===

import os
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import tempfile

import whisperx
import torch

from config.settings import settings
from utils.audio import (
    get_audio_info,
    convert_to_wav,
    normalize_loudness,
    split_audio,
)

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """轉寫結果"""
    language: str
    duration: float
    segments: List[Dict[str, Any]]
    word_segments: Optional[List[Dict[str, Any]]] = None
    speakers: Optional[List[str]] = None


@dataclass
class WordInfo:
    """單詞資訊"""
    word: str
    start: float
    end: float
    confidence: float = 1.0
    speaker: Optional[str] = None


@dataclass
class SegmentInfo:
    """段落資訊"""
    id: int
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    words: List[WordInfo] = field(default_factory=list)


class WhisperXTranscriber:
    """
    WhisperX 轉寫器
    支援：
    - 長音檔自動分段
    - VAD 前處理
    - 音訊規範化
    - 說話者分離（可選降級）
    - 自動設備偵測（CUDA/MPS/CPU）
    """

    def __init__(self):
        self._asr_model = None
        self._align_models = {}  # 按語言快取對齊模型
        self._diarize_pipeline = None

        # 自動偵測設備
        self._device = settings.get_device()
        self._compute_type = settings.get_compute_type()

        logger.info(f"Initialized with device: {self._device}, compute_type: {self._compute_type}")

    @property
    def asr_model(self):
        """懶載入 ASR 模型"""
        if self._asr_model is None:
            logger.info(f"Loading ASR model: {settings.model_size} on {self._device}")
            self._asr_model = whisperx.load_model(
                settings.model_size,
                self._device,
                compute_type=self._compute_type,
            )
            logger.info("ASR model loaded successfully")
        return self._asr_model

    def get_align_model(self, language_code: str):
        """獲取對齊模型（按語言快取）"""
        if language_code not in self._align_models:
            logger.info(f"Loading align model for: {language_code}")
            model, metadata = whisperx.load_align_model(
                language_code=language_code,
                device=self._device
            )
            self._align_models[language_code] = (model, metadata)
            logger.info(f"Align model loaded for: {language_code}")
        return self._align_models[language_code]

    @property
    def diarize_pipeline(self):
        """懶載入 Diarization pipeline"""
        if self._diarize_pipeline is None:
            if not settings.hf_token:
                logger.warning("HF_TOKEN not set, diarization unavailable")
                return None

            try:
                logger.info("Loading diarization pipeline")
                self._diarize_pipeline = whisperx.DiarizationPipeline(
                    use_auth_token=settings.hf_token,
                    device=self._device
                )
                logger.info("Diarization pipeline loaded")
            except Exception as e:
                logger.error(f"Failed to load diarization: {e}")
                return None

        return self._diarize_pipeline

    def preprocess_audio(self, audio_path: str) -> str:
        """
        音訊前處理
        - 轉換為 16kHz mono WAV
        - 可選響度標準化
        """
        # 取得音訊資訊
        info = get_audio_info(audio_path)
        logger.info(f"Audio info: duration={info['duration']:.1f}s, sr={info['sample_rate']}")

        # 檢查長度限制
        if info['duration'] > settings.max_audio_length_s:
            raise ValueError(
                f"Audio too long: {info['duration']:.1f}s > {settings.max_audio_length_s}s"
            )

        # 轉換格式
        wav_path = convert_to_wav(
            audio_path,
            sample_rate=settings.target_sample_rate,
            mono=True
        )

        # 響度標準化
        if settings.normalize_audio:
            wav_path = normalize_loudness(
                wav_path,
                target_lufs=settings.target_loudness_lufs
            )

        return wav_path

    def transcribe_chunk(
        self,
        audio_path: str,
        language: Optional[str] = None,
        time_offset: float = 0.0
    ) -> Dict[str, Any]:
        """
        轉寫單個音訊片段

        Args:
            audio_path: 音訊檔案路徑
            language: 語言代碼（None 為自動偵測）
            time_offset: 時間偏移量（用於長音檔分段）

        Returns:
            {
                "language": str,
                "segments": [...],
                "word_segments": [...]
            }
        """
        audio = whisperx.load_audio(audio_path)

        # 1. ASR 轉寫
        result = self.asr_model.transcribe(
            audio,
            batch_size=settings.batch_size,
            language=language
        )

        detected_lang = result.get("language") or language or "zh"
        logger.info(f"Detected language: {detected_lang}")

        # 2. 字詞對齊
        try:
            align_model, metadata = self.get_align_model(detected_lang)
            result = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                audio,
                self._device,
                return_char_alignments=False
            )
        except Exception as e:
            logger.warning(f"Alignment failed: {e}, using coarse timestamps")

        # 3. 應用時間偏移
        if time_offset > 0:
            for seg in result.get("segments", []):
                seg["start"] = seg.get("start", 0) + time_offset
                seg["end"] = seg.get("end", 0) + time_offset
                for word in seg.get("words", []):
                    word["start"] = word.get("start", 0) + time_offset
                    word["end"] = word.get("end", 0) + time_offset

        return {
            "language": detected_lang,
            "segments": result.get("segments", []),
            "word_segments": result.get("word_segments", [])
        }

    def apply_diarization(
        self,
        audio_path: str,
        result: Dict[str, Any],
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        應用說話者分離

        Args:
            audio_path: 音訊檔案
            result: 轉寫結果
            min_speakers: 最少說話者數
            max_speakers: 最多說話者數

        Returns:
            更新後的結果（含 speaker 標籤）
        """
        pipeline = self.diarize_pipeline
        if pipeline is None:
            logger.warning("Diarization unavailable, skipping")
            return result

        try:
            audio = whisperx.load_audio(audio_path)

            # 執行 diarization
            diarize_result = pipeline(
                audio,
                min_speakers=min_speakers or settings.min_speakers,
                max_speakers=max_speakers or settings.max_speakers
            )

            # 指派說話者到 word/segment
            result = whisperx.assign_word_speakers(
                diarize_result,
                result
            )

            logger.info("Diarization applied successfully")

        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            # 降級：不標記說話者

        return result

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        diarize: bool = True,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> TranscriptionResult:
        """
        完整轉寫流程

        Args:
            audio_path: 音訊/影片檔案路徑
            language: 語言代碼（None 自動偵測）
            diarize: 是否進行說話者分離
            min_speakers: 最少說話者數
            max_speakers: 最多說話者數

        Returns:
            TranscriptionResult
        """
        # 前處理
        wav_path = self.preprocess_audio(audio_path)
        info = get_audio_info(wav_path)
        duration = info["duration"]

        # 判斷是否需要分段
        if duration > settings.chunk_length_s:
            logger.info(f"Long audio ({duration:.1f}s), splitting into chunks")
            result = self._transcribe_long_audio(
                wav_path,
                language=language,
                diarize=diarize,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
        else:
            # 短音檔直接處理
            result = self.transcribe_chunk(wav_path, language=language)

            if diarize:
                result = self.apply_diarization(
                    wav_path,
                    result,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers
                )

        # 格式化結果
        segments = self._format_segments(result.get("segments", []))
        speakers = self._extract_speakers(segments)

        return TranscriptionResult(
            language=result.get("language", "zh"),
            duration=duration,
            segments=segments,
            speakers=speakers
        )

    def _transcribe_long_audio(
        self,
        audio_path: str,
        language: Optional[str] = None,
        diarize: bool = True,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        處理長音檔（分段轉寫再合併）
        """
        # 分割音訊
        chunks = split_audio(
            audio_path,
            chunk_length_s=settings.chunk_length_s,
            overlap_s=settings.chunk_overlap_s
        )

        logger.info(f"Split into {len(chunks)} chunks")

        all_segments = []
        detected_lang = language

        for chunk_path, start_time, end_time in chunks:
            logger.info(f"Processing chunk: {start_time:.1f}s - {end_time:.1f}s")

            chunk_result = self.transcribe_chunk(
                chunk_path,
                language=detected_lang,
                time_offset=start_time
            )

            # 使用第一個 chunk 偵測到的語言
            if detected_lang is None:
                detected_lang = chunk_result.get("language")

            all_segments.extend(chunk_result.get("segments", []))

            # 清理臨時檔案
            try:
                os.remove(chunk_path)
            except:
                pass

        # 合併重疊區域的 segments
        merged_segments = self._merge_overlapping_segments(
            all_segments,
            overlap_s=settings.chunk_overlap_s
        )

        result = {
            "language": detected_lang,
            "segments": merged_segments
        }

        # 對完整音檔做 diarization（如果啟用）
        if diarize:
            result = self.apply_diarization(
                audio_path,
                result,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )

        return result

    def _merge_overlapping_segments(
        self,
        segments: List[Dict],
        overlap_s: float
    ) -> List[Dict]:
        """
        合併分段處理產生的重疊 segments
        """
        if not segments:
            return []

        # 按開始時間排序
        segments = sorted(segments, key=lambda x: x.get("start", 0))

        merged = []
        for seg in segments:
            if not merged:
                merged.append(seg)
                continue

            prev = merged[-1]
            prev_end = prev.get("end", 0)
            curr_start = seg.get("start", 0)

            # 如果有重疊，選擇其中一個
            if curr_start < prev_end:
                # 重疊區域：保留前一個 segment 的結尾
                if curr_start < prev_end - overlap_s / 2:
                    # 這個 segment 大部分與前一個重疊，跳過
                    continue
                else:
                    # 調整當前 segment 的開始時間
                    seg["start"] = prev_end

            merged.append(seg)

        return merged

    def _format_segments(self, segments: List[Dict]) -> List[Dict]:
        """格式化 segments 為標準輸出格式"""
        formatted = []

        for i, seg in enumerate(segments):
            formatted_seg = {
                "id": i + 1,
                "start": float(seg.get("start", 0)),
                "end": float(seg.get("end", 0)),
                "text": seg.get("text", "").strip(),
                "speaker": seg.get("speaker"),
                "words": []
            }

            for word in seg.get("words", []):
                formatted_seg["words"].append({
                    "start": float(word.get("start", 0)),
                    "end": float(word.get("end", 0)),
                    "word": word.get("word", ""),
                    "confidence": float(word.get("score", word.get("confidence", 1.0))),
                    "speaker": word.get("speaker")
                })

            formatted.append(formatted_seg)

        return formatted

    def _extract_speakers(self, segments: List[Dict]) -> List[str]:
        """提取所有說話者列表"""
        speakers = set()
        for seg in segments:
            if seg.get("speaker"):
                speakers.add(seg["speaker"])
        return sorted(list(speakers))


# 全域單例
_transcriber: Optional[WhisperXTranscriber] = None


def get_transcriber() -> WhisperXTranscriber:
    """獲取轉寫器單例"""
    global _transcriber
    if _transcriber is None:
        _transcriber = WhisperXTranscriber()
    return _transcriber
