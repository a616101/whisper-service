"""
字幕生成服務
支援 SRT / VTT / JSON (video.js 相容)

專業字幕分段策略（參考 Turboscribe / stable-ts）：
1. 基於 word-level timestamps - 使用 WhisperX 的 word timestamps 確保時間準確
2. Speaker 感知分段 - 不同說話者強制分段
3. 簡繁轉換 - 使用 OpenCC 轉換為繁體中文
4. 分段優先級：speaker 變化 > 停頓 > 句尾標點 > 子句標點 > 最大字數
"""
import re
import logging
from typing import List, Dict, Any, Optional, Literal, Tuple
from dataclasses import dataclass, field
from enum import Enum

from config.settings import settings
from utils.formatting import (
    seconds_to_srt_timestamp,
    seconds_to_vtt_timestamp,
)
from utils.text import (
    wrap_segment_lines,
    segment_text_for_subtitle,
)

logger = logging.getLogger(__name__)

# ============================================================
# 簡繁轉換工具
# ============================================================

try:
    from opencc import OpenCC
    _converter = OpenCC('s2twp')  # 簡體到繁體（台灣用語）
    HAS_OPENCC = True
except ImportError:
    HAS_OPENCC = False
    _converter = None
    logger.warning("OpenCC not installed, Traditional Chinese conversion disabled")


def to_traditional_chinese(text: str) -> str:
    """將簡體中文轉換為繁體中文"""
    if not text or not HAS_OPENCC:
        return text
    try:
        return _converter.convert(text)
    except Exception as e:
        logger.warning(f"OpenCC conversion failed: {e}")
        return text


# ============================================================
# 標點符號定義
# ============================================================

# 句尾標點（強制分段）
SENTENCE_END_PUNCTS = '。！？.!?'

# 子句標點（次要分段點）
CLAUSE_END_PUNCTS = '，、；：,;:'

# 所有標點
ALL_PUNCTS = SENTENCE_END_PUNCTS + CLAUSE_END_PUNCTS + '""''「」『』【】（）()'


@dataclass
class SegmentationConfig:
    """
    專業字幕分段配置

    基於業界標準：
    - CPS (Characters Per Second): 衡量閱讀速度
    - 最佳 CPS: 12-17（觀眾 50% 時間閱讀，50% 看畫面）
    - 中文建議 CPS: 10-14（字符密度較高）
    """
    # CPS 控制（核心指標）
    target_cps: float = 14.0       # 目標 CPS
    max_cps: float = 18.0          # 最大 CPS（超過會強制分段）
    min_cps: float = 8.0           # 最小 CPS（太慢會合併）

    # 時長控制
    min_duration: float = 1.2      # 最短時長（眼睛定位需要時間）
    max_duration: float = 7.0      # 最長時長

    # 停頓偵測（VAD）
    pause_threshold: float = 0.3   # 語音停頓閾值（秒）
    long_pause_threshold: float = 0.6  # 長停頓（強制分段）

    # 字數控制
    max_chars_per_line: int = 42   # 每行最大字數
    max_lines: int = 2             # 最多行數

    # 行為控制
    split_on_speaker_change: bool = True  # 說話者變化時分段
    prefer_sentence_boundary: bool = True  # 優先在句子邊界分段


@dataclass
class SubtitleOptions:
    """字幕生成選項"""
    include_speaker: bool = True
    max_chars_per_line: int = 42
    max_lines_per_segment: int = 2
    speaker_format: Literal["prefix", "tag", "none"] = "prefix"
    # prefix: "SPEAKER_00: 文字"
    # tag: "[SPEAKER_00] 文字"
    # none: 不顯示說話者

    # 分段配置
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)


def generate_srt(
    segments: List[Dict[str, Any]],
    options: Optional[SubtitleOptions] = None
) -> str:
    """
    生成 SRT 格式字幕

    Args:
        segments: 轉寫結果的 segments
        options: 字幕選項

    Returns:
        SRT 格式字串
    """
    if options is None:
        options = SubtitleOptions()

    lines = []

    for i, seg in enumerate(segments, start=1):
        start = seconds_to_srt_timestamp(seg["start"])
        end = seconds_to_srt_timestamp(seg["end"])
        text = seg.get("text", "").strip()

        # 處理說話者標籤
        if options.include_speaker and seg.get("speaker"):
            speaker = seg["speaker"]
            if options.speaker_format == "prefix":
                text = f"{speaker}: {text}"
            elif options.speaker_format == "tag":
                text = f"[{speaker}] {text}"

        # 換行處理
        text = wrap_segment_lines(
            text,
            max_chars_per_line=options.max_chars_per_line,
            max_lines=options.max_lines_per_segment
        )

        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")  # 空行分隔

    return "\n".join(lines).strip() + "\n"


def generate_vtt(
    segments: List[Dict[str, Any]],
    options: Optional[SubtitleOptions] = None
) -> str:
    """
    生成 WebVTT 格式字幕

    Args:
        segments: 轉寫結果的 segments
        options: 字幕選項

    Returns:
        VTT 格式字串
    """
    if options is None:
        options = SubtitleOptions()

    lines = ["WEBVTT", ""]

    for seg in segments:
        start = seconds_to_vtt_timestamp(seg["start"])
        end = seconds_to_vtt_timestamp(seg["end"])
        text = seg.get("text", "").strip()

        # 處理說話者標籤
        speaker = seg.get("speaker")
        if options.include_speaker and speaker:
            if options.speaker_format == "prefix":
                text = f"{speaker}: {text}"
            elif options.speaker_format == "tag":
                text = f"[{speaker}] {text}"

        # 換行處理
        text = wrap_segment_lines(
            text,
            max_chars_per_line=options.max_chars_per_line,
            max_lines=options.max_lines_per_segment
        )

        # VTT 可以加入 cue settings
        cue_line = f"{start} --> {end}"

        lines.append(cue_line)
        lines.append(text)
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def generate_json_videojs(
    segments: List[Dict[str, Any]],
    language: str = "zh",
    duration: float = 0.0,
    speakers: Optional[List[str]] = None,
    include_words: bool = True
) -> Dict[str, Any]:
    """
    生成 video.js 相容的 JSON 格式

    video.js 字幕 JSON schema:
    {
        "language": "zh",
        "duration": 3600.5,
        "segments": [
            {
                "id": 1,
                "start": 0.0,
                "end": 3.5,
                "text": "文字內容",
                "speaker": "SPEAKER_00",
                "words": [...]
            }
        ],
        "speakers": ["SPEAKER_00", "SPEAKER_01"],
        "metadata": {...}
    }
    """
    result = {
        "version": "1.0",
        "language": language,
        "duration": duration,
        "segments": [],
        "speakers": speakers or [],
        "metadata": {
            "generator": "WhisperX Subtitle Service",
            "model": settings.model_size,
        }
    }

    for seg in segments:
        segment_data = {
            "id": seg.get("id", 0),
            "start": seg["start"],
            "end": seg["end"],
            "text": seg.get("text", ""),
            "speaker": seg.get("speaker"),
        }

        if include_words and seg.get("words"):
            segment_data["words"] = seg["words"]

        result["segments"].append(segment_data)

    return result


def generate_json_transcript(
    segments: List[Dict[str, Any]],
    language: str = "zh",
    duration: float = 0.0
) -> Dict[str, Any]:
    """
    生成簡化的文字稿 JSON（適合後處理/校正）

    {
        "language": "zh",
        "duration": 3600.5,
        "full_text": "完整文字...",
        "segments": [
            {
                "start": 0.0,
                "end": 3.5,
                "text": "文字",
                "speaker": "SPEAKER_00"
            }
        ]
    }
    """
    full_text = " ".join(seg.get("text", "") for seg in segments)

    return {
        "language": language,
        "duration": duration,
        "full_text": full_text.strip(),
        "segments": [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg.get("text", ""),
                "speaker": seg.get("speaker")
            }
            for seg in segments
        ]
    }


class SubtitleGenerator:
    """字幕生成器"""

    def __init__(self, options: Optional[SubtitleOptions] = None):
        self.options = options or SubtitleOptions()

    def generate(
        self,
        segments: List[Dict[str, Any]],
        format: Literal["srt", "vtt", "json", "json_transcript"] = "srt",
        language: str = "zh",
        duration: float = 0.0,
        speakers: Optional[List[str]] = None
    ) -> str | Dict[str, Any]:
        """
        生成指定格式的字幕

        Args:
            segments: 轉寫 segments
            format: 輸出格式
            language: 語言代碼
            duration: 總時長
            speakers: 說話者列表

        Returns:
            字幕內容（字串或 dict）
        """
        if format == "srt":
            return generate_srt(segments, self.options)

        elif format == "vtt":
            return generate_vtt(segments, self.options)

        elif format == "json":
            return generate_json_videojs(
                segments,
                language=language,
                duration=duration,
                speakers=speakers
            )

        elif format == "json_transcript":
            return generate_json_transcript(
                segments,
                language=language,
                duration=duration
            )

        else:
            raise ValueError(f"Unknown format: {format}")

    def resegment(
        self,
        segments: List[Dict[str, Any]],
        max_duration: float = 5.0,
        min_duration: float = 1.0,
        pause_threshold: float = 0.3,
        split_on_speaker_change: bool = True,
        convert_to_traditional: bool = True
    ) -> List[Dict[str, Any]]:
        """
        專業字幕分段（參考 Turboscribe / stable-ts）

        核心策略：**基於 word-level timestamps 分段，確保時間準確**

        分段優先級：
        1. Speaker 變化 → 強制分段
        2. 停頓 (>pause_threshold) → 強制分段
        3. 句尾標點 (。！？) → 優先分段點
        4. 子句標點 (，、；) → 次要分段點
        5. 最大字數/時長 → 兜底分段

        Args:
            segments: WhisperX 轉寫結果（包含 words）
            max_duration: 每段最長時長（秒）
            min_duration: 每段最短時長（秒）
            pause_threshold: 停頓閾值（秒）
            split_on_speaker_change: 說話者變化時是否分段
            convert_to_traditional: 是否轉換為繁體中文

        Returns:
            重新分段後的 segments
        """
        if not segments:
            return []

        config = self.options.segmentation
        max_chars = config.max_chars_per_line

        # 收集所有 words 並標記 speaker
        all_words = self._collect_all_words(segments)

        if not all_words:
            # 沒有 word-level 資料，使用 fallback
            logger.warning("No word-level timestamps available, using fallback segmentation")
            return self._fallback_resegment(segments, max_duration, min_duration, convert_to_traditional)

        # 基於 words 進行智慧分段
        result = []
        current_words = []
        current_text = ""
        current_start = None
        current_speaker = None
        current_id = 1

        for i, word in enumerate(all_words):
            word_text = word.get("word", "")
            word_start = word.get("start")
            word_end = word.get("end")
            word_speaker = word.get("speaker")

            # 計算停頓
            pause_before = 0
            if current_words and word_start is not None:
                last_end = current_words[-1].get("end")
                if last_end is not None:
                    pause_before = word_start - last_end

            # 判斷是否需要分段
            should_split = False
            split_reason = ""

            # 1. Speaker 變化
            if split_on_speaker_change and current_speaker and word_speaker:
                if word_speaker != current_speaker and current_words:
                    should_split = True
                    split_reason = "speaker_change"

            # 2. 長停頓
            if not should_split and pause_before >= pause_threshold and current_words:
                should_split = True
                split_reason = "pause"

            # 3. 當前段落以句尾標點結束 + 有停頓
            if not should_split and current_text:
                if current_text[-1] in SENTENCE_END_PUNCTS and pause_before >= pause_threshold * 0.3:
                    if len(current_text) >= 5:  # 最少 5 個字
                        should_split = True
                        split_reason = "sentence_end"

            # 4. 超過最大時長
            if not should_split and current_start is not None and word_end is not None:
                current_duration = word_end - current_start
                if current_duration >= max_duration and current_words:
                    should_split = True
                    split_reason = "max_duration"

            # 5. 超過最大字數
            if not should_split and len(current_text) + len(word_text) > max_chars:
                if current_words:
                    should_split = True
                    split_reason = "max_chars"

            # 執行分段
            if should_split and current_words:
                # 創建新 segment
                seg_start = current_start if current_start is not None else current_words[0].get("start", 0)
                seg_end = current_words[-1].get("end") if current_words[-1].get("end") is not None else seg_start + 1

                text = current_text.strip()
                if convert_to_traditional:
                    text = to_traditional_chinese(text)

                result.append({
                    "id": current_id,
                    "start": round(seg_start, 3),
                    "end": round(seg_end, 3),
                    "text": text,
                    "speaker": current_speaker,
                    "words": current_words.copy()
                })
                current_id += 1

                # 重置
                current_words = []
                current_text = ""
                current_start = word_start
                current_speaker = word_speaker

            # 加入當前 word
            current_words.append(word)
            current_text += word_text
            if current_start is None:
                current_start = word_start
            if current_speaker is None:
                current_speaker = word_speaker

        # 處理最後一段
        if current_words:
            seg_start = current_start if current_start is not None else current_words[0].get("start", 0)
            seg_end = current_words[-1].get("end") if current_words[-1].get("end") is not None else seg_start + 1

            text = current_text.strip()
            if convert_to_traditional:
                text = to_traditional_chinese(text)

            result.append({
                "id": current_id,
                "start": round(seg_start, 3),
                "end": round(seg_end, 3),
                "text": text,
                "speaker": current_speaker,
                "words": current_words
            })

        # 後處理：合併過短的段落
        result = self._merge_short_segments(result, config)

        # 重新編號
        for i, seg in enumerate(result, 1):
            seg["id"] = i

        logger.info(f"Resegmented {len(segments)} segments into {len(result)} segments")
        return result

    def _collect_all_words(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        從所有 segments 收集 words，並確保每個 word 都有 speaker 標記

        WhisperX 的 assign_word_speakers 會給每個 word 標記 speaker
        """
        all_words = []

        for seg in segments:
            seg_speaker = seg.get("speaker")
            words = seg.get("words", [])

            for word in words:
                word_copy = dict(word)
                # 確保每個 word 都有 speaker（從 segment 繼承）
                if not word_copy.get("speaker"):
                    word_copy["speaker"] = seg_speaker
                all_words.append(word_copy)

        return all_words

    def _fallback_resegment(
        self,
        segments: List[Dict[str, Any]],
        max_duration: float,
        min_duration: float,
        convert_to_traditional: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Fallback 分段（當沒有 word-level timestamps 時）

        使用文字分割 + 時間插值
        """
        result = []
        current_id = 1
        max_chars = self.options.segmentation.max_chars_per_line

        for seg in segments:
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            text = seg.get("text", "").strip()
            speaker = seg.get("speaker")
            duration = end - start

            if not text or duration <= 0:
                continue

            # 轉換為繁體
            if convert_to_traditional:
                text = to_traditional_chinese(text)

            # 分句
            sentences = self._split_text_to_sentences(text, max_chars)

            if len(sentences) <= 1:
                result.append({
                    "id": current_id,
                    "start": start,
                    "end": end,
                    "text": text,
                    "speaker": speaker,
                    "words": []
                })
                current_id += 1
            else:
                # 按字數比例分配時間
                total_chars = sum(len(s) for s in sentences)
                current_time = start

                for i, sent in enumerate(sentences):
                    char_ratio = len(sent) / total_chars if total_chars > 0 else 1 / len(sentences)
                    sent_duration = duration * char_ratio
                    sent_end = current_time + sent_duration

                    if i == len(sentences) - 1:
                        sent_end = end

                    result.append({
                        "id": current_id,
                        "start": round(current_time, 3),
                        "end": round(sent_end, 3),
                        "text": sent,
                        "speaker": speaker,
                        "words": []
                    })
                    current_id += 1
                    current_time = sent_end

        return result

    def _split_text_to_sentences(self, text: str, max_chars: int = 30) -> List[str]:
        """將文字分割為適合字幕的句子"""
        if not text or len(text) <= max_chars:
            return [text] if text else []

        result = []

        # 先按句尾標點分割
        pattern = f'([{re.escape(SENTENCE_END_PUNCTS)}]+)'
        parts = re.split(pattern, text)

        sentences = []
        current = ""
        for part in parts:
            if re.match(pattern, part):
                current += part
                if current.strip():
                    sentences.append(current.strip())
                current = ""
            else:
                current += part
        if current.strip():
            sentences.append(current.strip())

        # 處理過長的句子
        for sent in sentences:
            if len(sent) <= max_chars:
                result.append(sent)
            else:
                # 按子句標點分割
                sub_result = self._split_by_clause(sent, max_chars)
                result.extend(sub_result)

        return result

    def _split_by_clause(self, text: str, max_chars: int) -> List[str]:
        """按子句標點分割過長的文字"""
        if len(text) <= max_chars:
            return [text]

        pattern = f'([{re.escape(CLAUSE_END_PUNCTS)}]+)'
        parts = re.split(pattern, text)

        result = []
        current = ""

        for part in parts:
            if len(current) + len(part) <= max_chars:
                current += part
            else:
                if current.strip():
                    result.append(current.strip())
                current = part

        if current.strip():
            result.append(current.strip())

        # 如果還是太長，強制分割
        final_result = []
        for seg in result:
            if len(seg) <= max_chars:
                final_result.append(seg)
            else:
                # 強制按固定長度分割
                while len(seg) > max_chars:
                    final_result.append(seg[:max_chars])
                    seg = seg[max_chars:]
                if seg:
                    final_result.append(seg)

        return final_result


    def _merge_short_segments(
        self,
        segments: List[Dict[str, Any]],
        config: SegmentationConfig
    ) -> List[Dict[str, Any]]:
        """
        合併過短的段落

        策略：
        1. 只合併同一說話者的段落
        2. 合併後不能超過 max_duration
        3. 優先向前合併（和前一段合併）
        """
        if not segments or len(segments) < 2:
            return segments

        result = []

        for seg in segments:
            duration = seg.get("end", 0) - seg.get("start", 0)
            text_len = len(seg.get("text", ""))

            # 判斷是否過短（時長或字數）
            is_too_short = duration < config.min_duration or text_len < 3

            # 如果太短且可以合併到前一段
            if is_too_short and result:
                prev = result[-1]
                prev_speaker = prev.get("speaker")
                curr_speaker = seg.get("speaker")
                prev_duration = prev.get("end", 0) - prev.get("start", 0)

                # 同一說話者才合併
                same_speaker = prev_speaker == curr_speaker or not prev_speaker or not curr_speaker

                if same_speaker:
                    combined_duration = seg["end"] - prev["start"]
                    combined_text_len = len(prev.get("text", "")) + text_len

                    # 檢查合併後是否超過限制
                    if combined_duration <= config.max_duration and combined_text_len <= 50:
                        # 合併到前一段
                        prev["end"] = seg["end"]
                        prev["text"] = prev.get("text", "") + seg.get("text", "")
                        prev["words"] = prev.get("words", []) + seg.get("words", [])
                        continue

            result.append(seg)

        return result

    def _is_sentence_end(self, text: str) -> bool:
        """檢查是否為句子結尾"""
        text = text.strip()
        if not text:
            return False
        # 中英文句尾標點
        sentence_puncts = '。！？.!?'
        return text[-1] in sentence_puncts

    def _is_clause_end(self, text: str) -> bool:
        """檢查是否為子句結尾"""
        text = text.strip()
        if not text:
            return False
        # 中英文子句標點
        clause_puncts = '，、；：,;:'
        return text[-1] in clause_puncts

    def _simple_resegment(
        self,
        segments: List[Dict[str, Any]],
        max_duration: float,
        min_duration: float
    ) -> List[Dict[str, Any]]:
        """簡單分段（當沒有 word-level 資料時使用）"""
        result = []
        current_id = 1

        for seg in segments:
            duration = seg["end"] - seg["start"]
            text = seg.get("text", "")

            if duration <= max_duration:
                new_seg = dict(seg)
                new_seg["id"] = current_id
                result.append(new_seg)
                current_id += 1
            else:
                # 按時間平均分割
                num_parts = int(duration / max_duration) + 1
                sub_texts = segment_text_for_subtitle(
                    text,
                    max_chars=self.options.max_chars_per_line,
                    max_lines=self.options.max_lines_per_segment
                )

                # 確保分段數量合理
                num_parts = max(num_parts, len(sub_texts))
                seg_duration = duration / num_parts
                start = seg["start"]

                for sub_text in sub_texts:
                    result.append({
                        "id": current_id,
                        "start": start,
                        "end": min(start + seg_duration, seg["end"]),
                        "text": sub_text,
                        "speaker": seg.get("speaker"),
                        "words": []
                    })
                    start += seg_duration
                    current_id += 1

        return result

    def _split_by_words(
        self,
        segment: Dict[str, Any],
        max_duration: float,
        min_duration: float
    ) -> List[Dict[str, Any]]:
        """根據 word timestamps 分段（舊版，保留相容性）"""
        return self.resegment([segment], max_duration, min_duration)
