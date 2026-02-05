"""
時間格式化工具
"""
from typing import Optional


def seconds_to_srt_timestamp(seconds: float) -> str:
    """
    秒數轉 SRT 時間格式
    格式: HH:MM:SS,mmm
    """
    if seconds < 0:
        seconds = 0

    ms = int(round(seconds * 1000))
    hours = ms // 3600000
    ms %= 3600000
    minutes = ms // 60000
    ms %= 60000
    secs = ms // 1000
    ms %= 1000

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def seconds_to_vtt_timestamp(seconds: float) -> str:
    """
    秒數轉 VTT 時間格式
    格式: HH:MM:SS.mmm
    """
    if seconds < 0:
        seconds = 0

    ms = int(round(seconds * 1000))
    hours = ms // 3600000
    ms %= 3600000
    minutes = ms // 60000
    ms %= 60000
    secs = ms // 1000
    ms %= 1000

    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def parse_srt_timestamp(timestamp: str) -> float:
    """
    解析 SRT 時間格式為秒數
    """
    parts = timestamp.replace(",", ":").split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = int(parts[2])
    milliseconds = int(parts[3])

    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def format_duration(seconds: float) -> str:
    """
    格式化時長為人類可讀格式
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"
