import pytest

from src.config.models import get_active_model_profile, load_models_config
from src.infrastructure.llm.adapter import LLMAdapter

pytestmark = pytest.mark.unit


def test_canonical_loader_uses_default_profile_for_active_runtime(models_config_factory):
    config_path = models_config_factory()

    config = load_models_config(str(config_path))
    profile = get_active_model_profile(config)

    assert profile.key == 'primary'
    assert profile.backend == 'ollama'
    assert profile.to_backend_config()['profile_name'] == 'primary'


def test_adapter_uses_canonical_loader_with_env_override(models_config_factory, monkeypatch):
    config_path = models_config_factory()
    monkeypatch.setenv('ACTIVE_MODEL_PROFILE', 'fallback')

    adapter = LLMAdapter(models_config_path=str(config_path))

    assert adapter.models_config.default_model == 'primary'
    assert adapter.profile.key == 'fallback'
    assert adapter.profile.backend == 'llama_cpp'
    assert adapter.profile.to_backend_config()['path'] == '/tmp/model.gguf'


def test_runtime_model_config_imports_remain_lightweight():
    from src.infrastructure.llm.adapter import LLMAdapter
    from src.infrastructure.transcription.whisperx_adapter import WhisperXTranscriptionAdapter

    assert LLMAdapter is not None
    assert WhisperXTranscriptionAdapter is not None
