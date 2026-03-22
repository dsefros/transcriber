import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from src.infrastructure.llm.adapter import LLMAdapter


class LLMAdapterLifecycleTests(unittest.TestCase):
    def write_models_yaml(self, directory: str) -> str:
        path = Path(directory) / "models.yaml"
        path.write_text(
            textwrap.dedent(
                """
                default_model: primary
                profiles:
                  primary:
                    backend: llama_cpp
                    path: /tmp/model.gguf
                    description: Test profile
                """
            ).strip(),
            encoding="utf-8",
        )
        return str(path)

    def test_close_releases_initialized_backend_deterministically(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self.write_models_yaml(tmp_dir)
            adapter = LLMAdapter(models_config_path=config_path)

            class FakeBackend:
                def __init__(self):
                    self.closed = 0

                def close(self):
                    self.closed += 1

            backend = FakeBackend()
            adapter._backend = backend

            adapter.close()
            adapter.close()

            self.assertEqual(1, backend.closed)
            self.assertIsNone(adapter._backend)

    def test_close_is_noop_when_backend_was_never_initialized(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self.write_models_yaml(tmp_dir)
            adapter = LLMAdapter(models_config_path=config_path)

            with patch("gc.collect") as gc_collect:
                adapter.close()

            gc_collect.assert_not_called()
