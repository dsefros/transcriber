import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.app.preflight import PreflightError, run_preflight
from src.config.env import load_env_file_if_present

pytestmark = pytest.mark.unit


def test_preflight_rejects_missing_source_before_runtime_startup(models_config_factory, tmp_path):
    config_path = models_config_factory()
    missing_source = tmp_path / 'missing.wav'

    with pytest.raises(PreflightError, match='Input source path does not exist'):
        run_preflight(missing_source, source_type='json', models_config_path=str(config_path))


def test_preflight_surfaces_missing_database_url(models_config_factory, tmp_path):
    config_path = models_config_factory()
    source = tmp_path / 'input.json'
    source.write_text('{}', encoding='utf-8')

    with patch('src.app.preflight._module_available', return_value=True):
        with pytest.raises(PreflightError, match='DATABASE_URL is required'):
            run_preflight(source, source_type='json', models_config_path=str(config_path))


def test_preflight_checks_ml_dependencies_for_audio_path(models_config_factory, runtime_env, tmp_path):
    config_path = models_config_factory()
    source = tmp_path / 'input.wav'
    source.write_bytes(b'audio')

    def fake_module_available(name: str) -> bool:
        return name == 'ollama'

    runtime_env()
    with patch('src.app.preflight._module_available', side_effect=fake_module_available):
        with pytest.raises(PreflightError, match='WhisperX transcription runtime requires missing dependencies'):
            run_preflight(source, source_type='audio', models_config_path=str(config_path))


def test_env_loader_populates_database_url_without_overriding_existing_env(env_file_factory, monkeypatch):
    env_path = env_file_factory(
        '''
        DATABASE_URL=postgresql://from-env-file
        ACTIVE_MODEL_PROFILE=file-profile
        '''
    )
    monkeypatch.setenv('ACTIVE_MODEL_PROFILE', 'already-exported')

    loaded = load_env_file_if_present(str(env_path))

    assert loaded is True
    assert 'postgresql://from-env-file' == os.environ['DATABASE_URL']
    assert 'already-exported' == os.environ['ACTIVE_MODEL_PROFILE']
