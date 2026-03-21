import logging
import json
import sys
import os
from datetime import datetime


class JsonFormatter(logging.Formatter):
    """
    Formats logs as structured JSON.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        extra = getattr(record, "extra", None)
        if isinstance(extra, dict):
            payload.update(extra)

        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> None:
    """
    Initialize root logger with JSON formatter.
    Must be called exactly once at application startup.
    """
    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    # Сбрасываем старые handlers (важно при повторных запусках)
    root.handlers = [handler]


# ============================
# MEMORY TELEMETRY (GPU + RAM)
# ============================

def _get_ram_usage_mb() -> int:
    """
    Resident Set Size (RSS) of current process in MB.
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return int(process.memory_info().rss / 1024 / 1024)
    except Exception:
        return -1


def _get_gpu_usage_mb():
    """
    Returns (used_mb, total_mb, free_mb) or (-1, -1, -1) if unavailable.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return -1, -1, -1

        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)

        total = props.total_memory
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)

        used_mb = int(allocated / 1024 / 1024)
        total_mb = int(total / 1024 / 1024)
        free_mb = int((total - reserved) / 1024 / 1024)

        return used_mb, total_mb, free_mb

    except Exception:
        return -1, -1, -1


def log_memory(stage: str, component: str, job_id: str | None = None) -> None:
    """
    Emit structured memory snapshot to logs.
    """
    logger = logging.getLogger("telemetry")

    ram_mb = _get_ram_usage_mb()
    gpu_used_mb, gpu_total_mb, gpu_free_mb = _get_gpu_usage_mb()

    logger.info(
        "memory_snapshot",
        extra={
            "extra": {
                "event": "memory_snapshot",
                "component": component,
                "stage": stage,
                "job_id": job_id,
                "ram_used_mb": ram_mb,
                "gpu_used_mb": gpu_used_mb,
                "gpu_total_mb": gpu_total_mb,
                "gpu_free_mb": gpu_free_mb,
            }
        },
    )
