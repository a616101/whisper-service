"""
Celery 應用配置
用於多 GPU 併發任務處理
"""
import os
from celery import Celery

from config.settings import settings

# 創建 Celery 應用
celery_app = Celery(
    "whisper_worker",
    broker=settings.celery_broker,
    backend=settings.celery_backend,
    include=["workers.tasks"]
)

# Celery 配置
celery_app.conf.update(
    # 任務序列化
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    # 時區
    timezone="UTC",
    enable_utc=True,

    # 任務配置
    task_track_started=True,
    task_time_limit=settings.task_timeout_s,
    task_soft_time_limit=settings.task_timeout_s - 60,

    # Worker 配置
    worker_prefetch_multiplier=1,  # 每次只取一個任務（GPU 任務較重）
    worker_concurrency=settings.max_concurrent_tasks,

    # 結果配置
    result_expires=3600 * 24,  # 結果保留 24 小時

    # 重試配置
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # 路由配置
    task_routes={
        "workers.tasks.transcribe_task": {"queue": "transcription"},
        "workers.tasks.diarize_task": {"queue": "diarization"},
    },

    # 優先級佇列
    task_default_queue="default",
    task_queues={
        "transcription": {
            "exchange": "transcription",
            "routing_key": "transcription",
        },
        "diarization": {
            "exchange": "diarization",
            "routing_key": "diarization",
        },
        "default": {
            "exchange": "default",
            "routing_key": "default",
        },
    },
)


# GPU 綁定配置（用於多 GPU 環境）
def get_gpu_id() -> int:
    """
    獲取當前 worker 應使用的 GPU ID

    在多 GPU 環境中，每個 worker 綁定不同的 GPU。
    可通過環境變數 CUDA_VISIBLE_DEVICES 控制。
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpus = cuda_visible.split(",")
    worker_id = int(os.environ.get("CELERY_WORKER_ID", "0"))
    return int(gpus[worker_id % len(gpus)])


# Worker 啟動時的信號處理
from celery.signals import worker_process_init


@worker_process_init.connect
def setup_worker(**kwargs):
    """
    Worker 進程初始化

    在這裡設置 GPU 綁定和預載入模型
    """
    import torch

    # 獲取分配的 GPU
    gpu_id = get_gpu_id()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 驗證 GPU 可用
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"Worker initialized with GPU {gpu_id}: {device_name}")
    else:
        print("Warning: No GPU available, running on CPU")
