import os
import sys
import tempfile
import textwrap
import types
import unittest
from unittest.mock import patch

sys.modules.setdefault("ollama", types.SimpleNamespace(Client=lambda: object()))
sys.modules.setdefault("llama_cpp", types.SimpleNamespace(Llama=object))

from src.config.models import get_active_model_profile, load_models_config
from src.infrastructure.llm.adapter import LLMAdapter


class ActiveRuntimeModelConfigTests(unittest.TestCase):
    def write_models_yaml(self) -> str:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = os.path.join(tmpdir.name, "models.yaml")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(
                textwrap.dedent(
                    """
                    default_model: primary
                    profiles:
                      primary:
                        backend: ollama
                        name: llama3
                        description: Primary runtime model
                        params:
                          temperature: 0.1
                      fallback:
                        backend: llama_cpp
                        path: /tmp/model.gguf
                        description: Fallback runtime model
                        params:
                          max_tokens: 512
                    """
                ).strip()
            )
        return path

    def test_canonical_loader_uses_default_profile_for_active_runtime(self):
        config_path = self.write_models_yaml()

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ACTIVE_MODEL_PROFILE", None)
            config = load_models_config(config_path)
            profile = get_active_model_profile(config)

        self.assertEqual(profile.key, "primary")
        self.assertEqual(profile.backend, "ollama")
        self.assertEqual(profile.to_backend_config()["profile_name"], "primary")

    def test_adapter_uses_canonical_loader_with_env_override(self):
        config_path = self.write_models_yaml()

        with patch.dict(os.environ, {"ACTIVE_MODEL_PROFILE": "fallback"}, clear=False):
            adapter = LLMAdapter(models_config_path=config_path)

        self.assertEqual(adapter.models_config.default_model, "primary")
        self.assertEqual(adapter.profile.key, "fallback")
        self.assertEqual(adapter.profile.backend, "llama_cpp")
        self.assertEqual(adapter.profile.to_backend_config()["path"], "/tmp/model.gguf")


if __name__ == "__main__":
    unittest.main()
