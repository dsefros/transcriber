"""Lightweight environment loading helpers for the canonical runtime."""

from __future__ import annotations

import os
from pathlib import Path


def load_env_file_if_present(env_path: str = ".env") -> bool:
    """Populate ``os.environ`` from a local env file without overriding exports."""
    env_file = Path(env_path)
    if not env_file.exists():
        return False

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        cleaned_value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), cleaned_value)

    return True
