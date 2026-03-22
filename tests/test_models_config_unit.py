from __future__ import annotations

import pytest

from src.config.models import ModelConfigError, ModelProfile, get_active_model_profile, load_models_config

pytestmark = pytest.mark.unit


def test_load_models_config_and_active_profile_override(models_config_factory, monkeypatch):
    config_path = models_config_factory(default_model="primary")
    monkeypatch.setenv("ACTIVE_MODEL_PROFILE", "fallback")

    config = load_models_config(str(config_path))
    profile = get_active_model_profile(config)

    assert config.default_model == "primary"
    assert set(config.profiles) == {"primary", "fallback"}
    assert profile.key == "fallback"


def test_missing_active_profile_fails(models_config_factory, monkeypatch):
    config_path = models_config_factory(default_model="missing")
    monkeypatch.delenv("ACTIVE_MODEL_PROFILE", raising=False)

    with pytest.raises(ModelConfigError, match="Профиль 'missing' не найден"):
        get_active_model_profile(config_path=str(config_path))


@pytest.mark.parametrize(
    "config,match",
    [({"name": "x"}, "backend"), ({"backend": "ollama"}, "name"), ({"backend": "llama_cpp"}, "path")],
)
def test_invalid_profile_shapes_fail(config, match):
    with pytest.raises(ModelConfigError, match=match):
        ModelProfile("broken", config)


def test_invalid_profile_is_skipped_but_valid_profiles_remain(models_config_factory, capsys):
    config_path = models_config_factory(
        profiles={
            "valid": {"backend": "ollama", "name": "llama3"},
            "broken": {"backend": "ollama"},
        },
        default_model="valid",
    )

    config = load_models_config(str(config_path))

    assert set(config.profiles) == {"valid"}
    assert "Пропущен некорректный профиль 'broken'" in capsys.readouterr().out
    assert config.profiles["valid"].to_backend_config()["backend"] == "ollama"


def test_ollama_profile_reads_num_ctx_from_config(models_config_factory):
    config_path = models_config_factory(
        profiles={
            "primary": {
                "backend": "ollama",
                "name": "llama3",
                "params": {"num_ctx": 16384},
            }
        }
    )

    config = load_models_config(str(config_path))

    assert config.profiles["primary"].params["num_ctx"] == 16384


@pytest.mark.parametrize(
    "params,match",
    [
        ({"context_size": 4096}, "context_size"),
        ({"context_size": 4096, "num_ctx": 8192}, "используйте 'num_ctx'"),
    ],
)
def test_ollama_profile_rejects_legacy_context_size_param(params, match):
    with pytest.raises(ModelConfigError, match=match):
        ModelProfile("broken", {"backend": "ollama", "name": "llama3", "params": params})


def test_models_config_uses_legacy_analysis_prompt_when_key_is_missing(models_config_factory):
    config_path = models_config_factory()

    config = load_models_config(str(config_path))

    assert config.get_default_analysis_prompt() == "analysis/v1.yaml"


def test_models_config_reads_default_analysis_prompt_when_configured(models_config_factory):
    config_path = models_config_factory(default_analysis_prompt="analysis/v2.yaml")

    config = load_models_config(str(config_path))

    assert config.get_default_analysis_prompt() == "analysis/v2.yaml"
