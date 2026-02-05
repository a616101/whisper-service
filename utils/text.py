"""
文字處理工具 - 中英混合斷句優化
"""
import re
import unicodedata
from typing import List, Tuple, Optional


# === 字元分類 ===

def is_cjk(char: str) -> bool:
    """判斷是否為 CJK 字元（中日韓）"""
    if len(char) != 1:
        return False
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF or      # CJK Unified Ideographs
        0x3400 <= code <= 0x4DBF or      # CJK Unified Ideographs Extension A
        0x20000 <= code <= 0x2A6DF or    # CJK Unified Ideographs Extension B
        0xF900 <= code <= 0xFAFF or      # CJK Compatibility Ideographs
        0x2F800 <= code <= 0x2FA1F       # CJK Compatibility Ideographs Supplement
    )


def is_english_word_char(char: str) -> bool:
    """判斷是否為英文單詞字元"""
    return char.isalpha() and not is_cjk(char)


def is_punctuation(char: str) -> bool:
    """判斷是否為標點符號"""
    return unicodedata.category(char).startswith('P')


def is_sentence_end_punct(char: str) -> bool:
    """判斷是否為句尾標點"""
    return char in '。！？.!?'


def is_clause_punct(char: str) -> bool:
    """判斷是否為分句標點（可斷行）"""
    return char in '，、；：,;:'


# === 文字清理 ===

def normalize_text(text: str) -> str:
    """
    正規化文字：
    - 移除多餘空白
    - 統一全形/半形標點（可選）
    - 移除不可見字元
    """
    # 移除多餘空白
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    text = text.strip()

    return text


def normalize_punctuation(text: str, use_fullwidth: bool = True) -> str:
    """
    統一標點符號風格

    Args:
        text: 輸入文字
        use_fullwidth: True 使用全形標點，False 使用半形
    """
    if use_fullwidth:
        # 半形轉全形
        mapping = {
            ',': '，',
            '.': '。',
            '!': '！',
            '?': '？',
            ':': '：',
            ';': '；',
            '(': '（',
            ')': '）',
        }
    else:
        # 全形轉半形
        mapping = {
            '，': ',',
            '。': '.',
            '！': '!',
            '？': '?',
            '：': ':',
            '；': ';',
            '（': '(',
            '）': ')',
        }

    for src, dst in mapping.items():
        text = text.replace(src, dst)

    return text


# === 中英混合斷句 ===

def segment_text_for_subtitle(
    text: str,
    max_chars: int = 42,
    max_lines: int = 2,
    prefer_natural_break: bool = True
) -> List[str]:
    """
    將文字切分為適合字幕顯示的多段

    Args:
        text: 輸入文字
        max_chars: 每行最大字數（中文字算 1，英文單詞平均算 5）
        max_lines: 每段最大行數
        prefer_natural_break: 優先在自然斷點（標點）處斷行

    Returns:
        切分後的文字列表
    """
    text = normalize_text(text)
    if not text:
        return []

    # 計算有效字數（中文=1，英文單詞≈5字元）
    def effective_length(s: str) -> int:
        count = 0
        i = 0
        while i < len(s):
            char = s[i]
            if is_cjk(char):
                count += 1
                i += 1
            elif is_english_word_char(char):
                # 計算整個英文單詞
                word_start = i
                while i < len(s) and (is_english_word_char(s[i]) or s[i] == "'"):
                    i += 1
                word_len = i - word_start
                count += max(1, word_len // 3)  # 英文每3字元算1個單位
            else:
                i += 1

        return count

    # 找到所有可能的斷點
    def find_break_points(s: str) -> List[Tuple[int, int]]:
        """
        返回 (位置, 優先級) 列表
        優先級: 1=句尾, 2=分句標點, 3=空格, 4=CJK邊界
        """
        breaks = []
        for i, char in enumerate(s):
            if is_sentence_end_punct(char):
                breaks.append((i + 1, 1))
            elif is_clause_punct(char):
                breaks.append((i + 1, 2))
            elif char == ' ':
                breaks.append((i + 1, 3))
            elif i > 0 and is_cjk(char) != is_cjk(s[i-1]):
                # 中英文邊界
                breaks.append((i, 4))

        return breaks

    # 切分邏輯
    segments = []
    remaining = text

    while remaining:
        if effective_length(remaining) <= max_chars * max_lines:
            # 剩餘部分夠短，直接作為一段
            segments.append(remaining.strip())
            break

        # 需要切分
        target_len = max_chars * max_lines
        breaks = find_break_points(remaining)

        best_break = None
        best_priority = 999

        # 找最佳斷點
        for pos, priority in breaks:
            if effective_length(remaining[:pos]) <= target_len:
                if priority < best_priority:
                    best_priority = priority
                    best_break = pos
                elif priority == best_priority and pos > (best_break or 0):
                    best_break = pos

        if best_break and best_break > 0:
            segments.append(remaining[:best_break].strip())
            remaining = remaining[best_break:].strip()
        else:
            # 沒有好的斷點，強制切分
            cut_pos = min(len(remaining), max_chars * max_lines)
            segments.append(remaining[:cut_pos].strip())
            remaining = remaining[cut_pos:].strip()

    return segments


def wrap_segment_lines(
    text: str,
    max_chars_per_line: int = 42,
    max_lines: int = 2
) -> str:
    """
    將單段文字換行（不切分為多段）

    Returns:
        換行後的文字（用 \n 分隔）
    """
    if not text:
        return ""

    # 簡化的換行邏輯
    lines = []
    current_line = ""

    words = re.split(r'(\s+)', text)

    for word in words:
        if not word:
            continue

        # 計算加入這個詞後的長度
        test_line = current_line + word if current_line else word

        if len(test_line) <= max_chars_per_line:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line.strip())
                if len(lines) >= max_lines:
                    break
            current_line = word

    if current_line and len(lines) < max_lines:
        lines.append(current_line.strip())

    return '\n'.join(lines)


# === Glossary 校正 ===

def apply_glossary(text: str, glossary: dict) -> str:
    """
    應用詞彙表校正

    Args:
        text: 輸入文字
        glossary: 詞彙對照表 {"錯誤詞": "正確詞"}

    Returns:
        校正後的文字
    """
    # 按長度排序，先替換長詞
    sorted_terms = sorted(glossary.keys(), key=len, reverse=True)

    for wrong_term in sorted_terms:
        correct_term = glossary[wrong_term]
        # 使用詞邊界匹配（對英文有效）
        pattern = re.compile(
            r'\b' + re.escape(wrong_term) + r'\b',
            re.IGNORECASE
        )
        text = pattern.sub(correct_term, text)

        # 對中文直接替換
        text = text.replace(wrong_term, correct_term)

    return text


# === 常見錯誤修正規則 ===

COMMON_FIXES = {
    # Whisper 常見中文錯誤
    '的的': '的',
    '了了': '了',
    '是是': '是',
    # 常見英文錯誤
    'A I ': 'AI ',
    'A.I. ': 'AI ',
    'G P T': 'GPT',
    'G.P.T': 'GPT',
    'Chat G P T': 'ChatGPT',
}


def apply_common_fixes(text: str) -> str:
    """應用常見錯誤修正"""
    for wrong, correct in COMMON_FIXES.items():
        text = text.replace(wrong, correct)
    return text


# === 數字格式化 ===

def normalize_numbers(text: str) -> str:
    """
    統一數字格式
    - 年份保持4位數
    - 金額加入千分位（可選）
    """
    # 這裡可以根據需求擴展
    return text
