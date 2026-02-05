"""
文字校正服務
支援 Glossary、規則校正、可選 LLM 校正
"""
import re
import json
import logging
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field

from utils.text import (
    normalize_text,
    normalize_punctuation,
    apply_glossary,
    apply_common_fixes,
)

logger = logging.getLogger(__name__)


@dataclass
class CorrectionResult:
    """校正結果"""
    original_text: str
    corrected_text: str
    changes: List[Dict[str, Any]] = field(default_factory=list)
    applied_rules: List[str] = field(default_factory=list)


# === 預設 Glossary（常見技術術語）===
DEFAULT_GLOSSARY = {
    # AI/ML 術語
    "Whisper": "Whisper",
    "Whishper": "Whisper",
    "威斯博": "Whisper",
    "GPT": "GPT",
    "Chat GPT": "ChatGPT",
    "ChatGPT": "ChatGPT",
    "LLM": "LLM",
    "AI": "AI",
    "A.I.": "AI",
    "Machine Learning": "Machine Learning",
    "Deep Learning": "Deep Learning",

    # 常見品牌
    "NVIDIA": "NVIDIA",
    "Nvidia": "NVIDIA",
    "DGX": "DGX",
    "Tesla": "Tesla",
    "PyTorch": "PyTorch",
    "TensorFlow": "TensorFlow",

    # 中文常見錯誤
    "的的": "的",
    "了了": "了",
    "是是": "是",
    "在在": "在",
}


class TextCorrector:
    """
    文字校正器

    校正流程:
    1. 基礎正規化（空白、標點）
    2. Glossary 替換
    3. 規則校正（數字、日期等）
    4. 常見錯誤修正
    5. (可選) LLM 校正
    """

    def __init__(
        self,
        glossary: Optional[Dict[str, str]] = None,
        use_default_glossary: bool = True,
        llm_endpoint: Optional[str] = None
    ):
        self.glossary = {}

        if use_default_glossary:
            self.glossary.update(DEFAULT_GLOSSARY)

        if glossary:
            self.glossary.update(glossary)

        self.llm_endpoint = llm_endpoint

    def correct(
        self,
        text: str,
        additional_glossary: Optional[Dict[str, str]] = None,
        use_llm: bool = False
    ) -> CorrectionResult:
        """
        執行校正

        Args:
            text: 原始文字
            additional_glossary: 額外詞彙表（臨時使用）
            use_llm: 是否使用 LLM 校正

        Returns:
            CorrectionResult
        """
        original = text
        changes = []
        applied_rules = []

        # 1. 基礎正規化
        text = normalize_text(text)
        if text != original:
            applied_rules.append("normalize_text")

        # 2. 標點正規化
        prev = text
        text = normalize_punctuation(text, use_fullwidth=True)
        if text != prev:
            applied_rules.append("normalize_punctuation")

        # 3. Glossary 替換
        glossary = dict(self.glossary)
        if additional_glossary:
            glossary.update(additional_glossary)

        for wrong, correct in glossary.items():
            if wrong in text:
                text = text.replace(wrong, correct)
                changes.append({
                    "type": "glossary",
                    "from": wrong,
                    "to": correct
                })

        if changes:
            applied_rules.append("glossary")

        # 4. 常見錯誤修正
        prev = text
        text = apply_common_fixes(text)
        if text != prev:
            applied_rules.append("common_fixes")

        # 5. 數字格式化
        prev = text
        text = self._fix_numbers(text)
        if text != prev:
            applied_rules.append("number_format")

        # 6. (可選) LLM 校正
        if use_llm and self.llm_endpoint:
            prev = text
            text = self._llm_correct(text)
            if text != prev:
                applied_rules.append("llm")

        return CorrectionResult(
            original_text=original,
            corrected_text=text,
            changes=changes,
            applied_rules=applied_rules
        )

    def _fix_numbers(self, text: str) -> str:
        """
        數字格式修正
        - 移除數字之間的空格
        - 統一數字格式
        """
        # 移除數字間的空格 (如 "1 0 0" -> "100")
        text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)

        # 年份格式 (如 "2 0 2 4" -> "2024")
        text = re.sub(r'(\d)\s+(\d)\s+(\d)\s+(\d)', r'\1\2\3\4', text)

        return text

    def _llm_correct(self, text: str) -> str:
        """
        使用 LLM 進行校正
        （需要外部 LLM 服務）
        """
        if not self.llm_endpoint:
            return text

        # 這裡可以接入外部 LLM API
        # 例如 OpenAI、Claude 等
        # 目前返回原文
        logger.warning("LLM correction not implemented")
        return text

    def correct_segments(
        self,
        segments: List[Dict[str, Any]],
        additional_glossary: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        校正整個 segments 列表

        Args:
            segments: 轉寫結果的 segments
            additional_glossary: 額外詞彙表

        Returns:
            校正後的 segments
        """
        corrected = []

        for seg in segments:
            text = seg.get("text", "")
            result = self.correct(text, additional_glossary)

            new_seg = dict(seg)
            new_seg["text"] = result.corrected_text
            new_seg["_correction"] = {
                "original": result.original_text,
                "changes": result.changes,
                "rules": result.applied_rules
            }

            corrected.append(new_seg)

        return corrected


def load_glossary_from_file(file_path: str) -> Dict[str, str]:
    """
    從檔案載入 glossary

    支援格式:
    - JSON: {"wrong": "correct", ...}
    - TXT: wrong=correct (每行一對)
    """
    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    elif file_path.endswith(".txt"):
        glossary = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    wrong, correct = line.split("=", 1)
                    glossary[wrong.strip()] = correct.strip()
        return glossary

    else:
        raise ValueError(f"Unsupported glossary format: {file_path}")
