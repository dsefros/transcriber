import pytest

from src.config.models import load_models_config as load_canonical_models_config
from src.infrastructure.llm.config import get_active_model_profile, load_models_config
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

pytestmark = pytest.mark.unit


def test_legacy_llm_config_wraps_canonical_typed_loader(models_config_factory, monkeypatch):
    config_path = models_config_factory()

    compat_config = load_models_config(str(config_path))
    assert compat_config['default_model'] == 'primary'
    assert compat_config['profiles']['primary']['profile_name'] == 'primary'

    canonical_config = load_canonical_models_config(str(config_path))
    monkeypatch.setenv('ACTIVE_MODEL_PROFILE', 'fallback')
    compat_profile = get_active_model_profile(canonical_config)

    assert compat_profile['profile_name'] == 'fallback'
    assert compat_profile['backend'] == 'llama_cpp'


def test_legacy_storage_package_and_adapter_export_canonical_symbols():
    assert LegacyTranscriptionAdapter is WhisperXTranscriptionAdapter
    assert all(
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
