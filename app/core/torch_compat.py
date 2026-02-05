"""
PyTorch 相容性設定
解決 PyTorch 2.6+ 的 weights_only 安全限制問題

重要：此模組必須在 import whisperx 之前被 import
"""
import logging
import os
import functools

logger = logging.getLogger(__name__)

_patched = False


def patch_torch_load():
    """
    Monkey patch torch.load 以支援舊模型

    這是最可靠的方法，強制所有 torch.load 調用使用 weights_only=False
    """
    global _patched
    if _patched:
        return

    try:
        import torch

        # 保存原始函數
        _original_torch_load = torch.load

        @functools.wraps(_original_torch_load)
        def patched_load(*args, **kwargs):
            # 強制設定 weights_only=False
            kwargs["weights_only"] = False
            return _original_torch_load(*args, **kwargs)

        # 替換 torch.load
        torch.load = patched_load
        _patched = True
        logger.info("✓ Patched torch.load to allow model loading (weights_only=False)")

    except Exception as e:
        logger.error(f"Failed to patch torch.load: {e}")


def setup_torch_compatibility():
    """
    設定 PyTorch 相容性

    直接使用 monkey patch 方式，最可靠
    """
    try:
        import torch
        pytorch_version = torch.__version__
        logger.info(f"PyTorch version: {pytorch_version}")

        # 解析版本
        version_parts = pytorch_version.split(".")
        major = int(version_parts[0])
        minor = int(version_parts[1].split("+")[0].split("a")[0].split("b")[0].split("rc")[0])

        if major >= 2 and minor >= 6:
            logger.info("PyTorch 2.6+ detected, applying compatibility patch...")
            patch_torch_load()
        else:
            logger.info(f"PyTorch {major}.{minor} < 2.6, no patch needed")

    except Exception as e:
        logger.warning(f"Could not setup PyTorch compatibility: {e}")
        # 嘗試 patch 作為備用
        patch_torch_load()


# === 立即執行 ===
# 必須在 import 時就執行，確保在 whisperx import torch 之前完成
setup_torch_compatibility()
