"""
音訊處理工具
"""
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np

from config.settings import settings


def get_audio_info(audio_path: str) -> dict:
    """
    獲取音訊資訊

    Returns:
        {
            "duration": float,  # 秒
            "sample_rate": int,
            "channels": int,
            "codec": str,
            "bitrate": int
        }
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        audio_path
    ]

    import json
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)

    # 找音訊流
    audio_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "audio":
            audio_stream = stream
            break

    if not audio_stream:
        raise ValueError("No audio stream found")

    format_info = data.get("format", {})

    return {
        "duration": float(format_info.get("duration", 0)),
        "sample_rate": int(audio_stream.get("sample_rate", 0)),
        "channels": int(audio_stream.get("channels", 0)),
        "codec": audio_stream.get("codec_name", "unknown"),
        "bitrate": int(format_info.get("bit_rate", 0)),
    }


def convert_to_wav(
    input_path: str,
    output_path: Optional[str] = None,
    sample_rate: int = 16000,
    mono: bool = True
) -> str:
    """
    轉換為 WAV 格式

    Args:
        input_path: 輸入檔案路徑
        output_path: 輸出檔案路徑（不指定則自動產生）
        sample_rate: 採樣率
        mono: 是否轉為單聲道

    Returns:
        輸出檔案路徑
    """
    if output_path is None:
        suffix = ".wav"
        fd, output_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)

    cmd = [
        "ffmpeg",
        "-y",  # 覆蓋輸出
        "-i", input_path,
        "-ar", str(sample_rate),
    ]

    if mono:
        cmd.extend(["-ac", "1"])

    cmd.extend([
        "-c:a", "pcm_s16le",
        output_path
    ])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")

    return output_path


def normalize_loudness(
    input_path: str,
    output_path: Optional[str] = None,
    target_lufs: float = -23.0,
    target_tp: float = -1.0
) -> str:
    """
    響度標準化（EBU R128）

    Args:
        input_path: 輸入檔案
        output_path: 輸出檔案
        target_lufs: 目標響度（LUFS）
        target_tp: 目標 True Peak（dBTP）

    Returns:
        輸出檔案路徑
    """
    if output_path is None:
        suffix = Path(input_path).suffix
        fd, output_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)

    # 兩階段響度標準化
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-af", f"loudnorm=I={target_lufs}:TP={target_tp}:LRA=11",
        "-ar", str(settings.target_sample_rate),
        "-ac", "1",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Loudness normalization failed: {result.stderr}")

    return output_path


def split_audio(
    input_path: str,
    chunk_length_s: int,
    overlap_s: int = 2,
    output_dir: Optional[str] = None
) -> List[Tuple[str, float, float]]:
    """
    將長音檔分割為多個片段

    Args:
        input_path: 輸入檔案
        chunk_length_s: 每段長度（秒）
        overlap_s: 重疊秒數
        output_dir: 輸出目錄

    Returns:
        [(chunk_path, start_time, end_time), ...]
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    info = get_audio_info(input_path)
    duration = info["duration"]

    chunks = []
    start = 0.0
    chunk_idx = 0

    while start < duration:
        end = min(start + chunk_length_s, duration)

        output_path = os.path.join(
            output_dir,
            f"chunk_{chunk_idx:04d}.wav"
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-ss", str(start),
            "-t", str(end - start),
            "-ar", str(settings.target_sample_rate),
            "-ac", "1",
            "-c:a", "pcm_s16le",
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Audio split failed: {result.stderr}")

        chunks.append((output_path, start, end))

        # 下一段開始位置（考慮重疊）
        start = end - overlap_s
        chunk_idx += 1

    return chunks


def extract_audio_from_video(
    video_path: str,
    output_path: Optional[str] = None,
    sample_rate: int = 16000
) -> str:
    """
    從影片中提取音訊

    Args:
        video_path: 影片檔案路徑
        output_path: 音訊輸出路徑
        sample_rate: 採樣率

    Returns:
        音訊檔案路徑
    """
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",  # 不要視訊
        "-ar", str(sample_rate),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed: {result.stderr}")

    return output_path


def detect_silence(
    audio_path: str,
    threshold_db: float = -40,
    min_silence_duration: float = 0.5
) -> List[Tuple[float, float]]:
    """
    偵測靜音片段

    Args:
        audio_path: 音訊檔案
        threshold_db: 靜音閾值（dB）
        min_silence_duration: 最小靜音時長（秒）

    Returns:
        [(start, end), ...] 靜音片段列表
    """
    cmd = [
        "ffmpeg",
        "-i", audio_path,
        "-af", f"silencedetect=noise={threshold_db}dB:d={min_silence_duration}",
        "-f", "null",
        "-"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # 解析 silencedetect 輸出
    import re
    silence_starts = re.findall(
        r"silence_start: ([\d.]+)",
        result.stderr
    )
    silence_ends = re.findall(
        r"silence_end: ([\d.]+)",
        result.stderr
    )

    silences = []
    for start, end in zip(silence_starts, silence_ends):
        silences.append((float(start), float(end)))

    return silences
