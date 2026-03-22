from __future__ import annotations

import os
import sys
import textwrap
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

# Keep only process-level variables that pytest/Python need to execute reliably.
_ENV_ALLOWLIST = {
    'HOME',
    'PATH',
    'PWD',
    'LANG',
    'LC_ALL',
    'LC_CTYPE',
    'SHELL',
    'SHLVL',
    'SYSTEMROOT',
    'TERM',
    'TMP',
    'TEMP',
    'TMPDIR',
    'USER',
    'USERNAME',
    'VIRTUAL_ENV',
    'PYTEST_CURRENT_TEST',
}
_PREFIX_ALLOWLIST = ('PYTHON', 'PYTEST_')


@pytest.fixture(autouse=True)
def isolated_test_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Prevent accidental leakage from a developer shell or local .env exports."""
    for key in tuple(os.environ):
        if key in _ENV_ALLOWLIST or key.startswith(_PREFIX_ALLOWLIST):
            continue
        monkeypatch.delenv(key, raising=False)
    yield


@pytest.fixture
def runtime_env(monkeypatch: pytest.MonkeyPatch) -> Callable[..., dict[str, str]]:
    """Explicitly opt tests into the runtime env they require."""

    def _apply(**overrides: str) -> dict[str, str]:
        values = {'DATABASE_URL': 'postgresql://test-db'}
        values.update(overrides)
        for key, value in values.items():
            monkeypatch.setenv(key, value)
        return values

    return _apply


@pytest.fixture
def models_config_factory(tmp_path: Path) -> Callable[..., Path]:
    """Create local, minimal models.yaml fixtures for each test."""

    def _create(*, default_model: str = 'primary', profiles: dict[str, dict[str, Any]] | None = None) -> Path:
        profile_map = profiles or {
            'primary': {
                'backend': 'ollama',
                'name': 'llama3',
                'description': 'Primary runtime model',
                'params': {'temperature': 0.1},
            },
            'fallback': {
                'backend': 'llama_cpp',
                'path': '/tmp/model.gguf',
                'description': 'Fallback runtime model',
                'params': {'max_tokens': 512},
            },
        }
        lines = [f'default_model: {default_model}', 'profiles:']
        for profile_name, config in profile_map.items():
            lines.append(f'  {profile_name}:')
            for key, value in config.items():
                if isinstance(value, dict):
                    lines.append(f'    {key}:')
                    for nested_key, nested_value in value.items():
                        lines.append(f'      {nested_key}: {nested_value}')
                else:
                    lines.append(f'    {key}: {value}')

        path = tmp_path / 'models.yaml'
        path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        return path

    return _create


@pytest.fixture
def env_file_factory(tmp_path: Path) -> Callable[[str], Path]:
    def _create(contents: str) -> Path:
        path = tmp_path / '.env'
        path.write_text(textwrap.dedent(contents).lstrip(), encoding='utf-8')
        return path

    return _create


@pytest.fixture
def isolated_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


sys.modules.setdefault('dotenv', types.SimpleNamespace(load_dotenv=lambda: None))
sys.modules.setdefault('qdrant_client', types.SimpleNamespace(QdrantClient=object))
sys.modules.setdefault(
    'qdrant_client.http.models',
    types.SimpleNamespace(Distance=types.SimpleNamespace(COSINE='cosine'), VectorParams=object),
)
sys.modules.setdefault('llama_cpp', types.SimpleNamespace(Llama=object))
