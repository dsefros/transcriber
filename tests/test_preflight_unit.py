from __future__ import annotations

import pytest

from src.app.preflight import PreflightError, run_preflight

pytestmark = pytest.mark.unit


def test_preflight_validates_database_models_profile_and_audio_dependencies(models_config_factory, runtime_env, tmp_path):
    config_path = models_config_factory()
    source = tmp_path / "input.wav"
    source.write_bytes(b"audio")
    runtime_env()

    def available(name: str) -> bool:
        return True

    from unittest.mock import patch
    with patch("src.app.preflight._module_available", side_effect=available):
        summary = run_preflight(source, source_type="audio", models_config_path=str(config_path))

    assert summary["source_type"] == "audio"
    assert summary["model_profile"] == "primary"
    assert summary["model_backend"] == "ollama"


def test_preflight_invalid_model_configuration_fails(models_config_factory, runtime_env, tmp_path):
    config_path = models_config_factory(default_model="missing")
    source = tmp_path / "input.json"
    source.write_text("{}", encoding="utf-8")
    runtime_env()

    from unittest.mock import patch
    with patch("src.app.preflight._module_available", return_value=True):
        with pytest.raises(PreflightError, match="Model configuration preflight failed"):
            run_preflight(source, source_type="json", models_config_path=str(config_path))
