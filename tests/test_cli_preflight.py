import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from src.app.preflight import PreflightError, run_preflight


class CliPreflightTests(unittest.TestCase):
    def write_models_yaml(self, directory: str, *, backend: str = "ollama") -> str:
        payload = """
        default_model: primary
        profiles:
          primary:
            backend: {backend}
            {backend_fields}
            description: Primary runtime model
        """
        backend_fields = "name: llama3" if backend == "ollama" else "path: /tmp/model.gguf"
        path = Path(directory) / "models.yaml"
        path.write_text(
            textwrap.dedent(payload.format(backend=backend, backend_fields=backend_fields)).strip(),
            encoding="utf-8",
        )
        return str(path)

    def test_preflight_rejects_missing_source_before_runtime_startup(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self.write_models_yaml(tmp_dir)
            missing_source = Path(tmp_dir) / "missing.wav"
            with self.assertRaisesRegex(PreflightError, "Input source path does not exist"):
                run_preflight(missing_source, source_type="json", models_config_path=config_path)

    def test_preflight_surfaces_missing_database_url(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self.write_models_yaml(tmp_dir)
            source = Path(tmp_dir) / "input.json"
            source.write_text("{}", encoding="utf-8")
            with patch("src.app.preflight._module_available", return_value=True):
                with patch.dict(os.environ, {}, clear=True):
                    with self.assertRaisesRegex(PreflightError, "DATABASE_URL is required"):
                        run_preflight(source, source_type="json", models_config_path=config_path)

    def test_preflight_checks_ml_dependencies_for_audio_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self.write_models_yaml(tmp_dir)
            source = Path(tmp_dir) / "input.wav"
            source.write_bytes(b"audio")

            def fake_module_available(name: str) -> bool:
                return name == "ollama"

            with patch("src.app.preflight._module_available", side_effect=fake_module_available):
                with patch.dict(os.environ, {"DATABASE_URL": "postgresql://example"}, clear=True):
                    with self.assertRaisesRegex(PreflightError, "WhisperX transcription runtime requires missing dependencies"):
                        run_preflight(source, source_type="audio", models_config_path=config_path)
