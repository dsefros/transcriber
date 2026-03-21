import os
import sys
import tempfile
import textwrap
import types
import unittest
from unittest.mock import patch

sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda: None))
sys.modules.setdefault("qdrant_client", types.SimpleNamespace(QdrantClient=object))
sys.modules.setdefault(
    "qdrant_client.http.models",
    types.SimpleNamespace(
        Distance=types.SimpleNamespace(COSINE="cosine"), VectorParams=object
    ),
)

from src.config.models import load_models_config as load_canonical_models_config
from src.infrastructure.llm.config import (
    get_active_model_profile,
    load_models_config,
)
from src.infrastructure.transcription.legacy_adapter import (
    LegacyTranscriptionAdapter,
    WhisperXTranscriptionAdapter,
)
from src.legacy.v1.storage import (
    Fragment,
    Meeting,
    Speaker,
    create_collections_if_not_exists,
    get_db_session,
    init_db,
    init_qdrant_client,
)


class LegacyCompatShimsTests(unittest.TestCase):
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

    def test_legacy_llm_config_wraps_canonical_typed_loader(self):
        config_path = self.write_models_yaml()

        compat_config = load_models_config(config_path)
        self.assertEqual(compat_config["default_model"], "primary")
        self.assertEqual(
            compat_config["profiles"]["primary"]["profile_name"], "primary"
        )

        canonical_config = load_canonical_models_config(config_path)
        with patch.dict(os.environ, {"ACTIVE_MODEL_PROFILE": "fallback"}, clear=False):
            compat_profile = get_active_model_profile(canonical_config)

        self.assertEqual(compat_profile["profile_name"], "fallback")
        self.assertEqual(compat_profile["backend"], "llama_cpp")

    def test_legacy_storage_package_and_adapter_export_canonical_symbols(self):
        self.assertIs(LegacyTranscriptionAdapter, WhisperXTranscriptionAdapter)
        self.assertTrue(
            all(
                symbol is not None
                for symbol in [
                    Meeting,
                    Speaker,
                    Fragment,
                    init_db,
                    get_db_session,
                    init_qdrant_client,
                    create_collections_if_not_exists,
                ]
            )
        )


if __name__ == "__main__":
    unittest.main()
