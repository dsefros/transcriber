from unittest.mock import patch

import pytest

from src.infrastructure.llm.adapter import LLMAdapter

pytestmark = pytest.mark.unit


def test_close_releases_initialized_backend_deterministically(models_config_factory):
    config_path = models_config_factory(
        profiles={
            'primary': {
                'backend': 'llama_cpp',
                'path': '/tmp/model.gguf',
                'description': 'Test profile',
            }
        }
    )
    adapter = LLMAdapter(models_config_path=str(config_path))

    class FakeBackend:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    backend = FakeBackend()
    adapter._backend = backend

    adapter.close()
    adapter.close()

    assert backend.closed == 1
    assert adapter._backend is None


def test_close_is_noop_when_backend_was_never_initialized(models_config_factory):
    config_path = models_config_factory(
        profiles={
            'primary': {
                'backend': 'llama_cpp',
                'path': '/tmp/model.gguf',
                'description': 'Test profile',
            }
        }
    )
    adapter = LLMAdapter(models_config_path=str(config_path))

    with patch('gc.collect') as gc_collect:
        adapter.close()

    gc_collect.assert_not_called()
