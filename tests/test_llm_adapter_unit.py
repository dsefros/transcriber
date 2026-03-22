from __future__ import annotations

import types

import pytest

from src.infrastructure.llm.adapter import LLMAdapter

pytestmark = pytest.mark.unit


class FakeBackend:
    instances = []

    def __init__(self, config):
        self.config = config
        self.generate_calls = []
        self.closed = 0
        FakeBackend.instances.append(self)

    def generate(self, prompt, params):
        self.generate_calls.append((prompt, params))
        return "generated"

    @property
    def meta(self):
        return {"backend": self.config["backend"], "profile": self.config["profile_name"]}

    def close(self):
        self.closed += 1


def test_llm_adapter_lazy_initializes_backend_once(models_config_factory, monkeypatch):
    config_path = models_config_factory(profiles={"primary": {"backend": "ollama", "name": "llama3", "params": {"temperature": 0.2}}})
    FakeBackend.instances.clear()
    monkeypatch.setattr("importlib.import_module", lambda name: types.SimpleNamespace(OllamaBackend=FakeBackend, LlamaCppBackend=FakeBackend))

    adapter = LLMAdapter(models_config_path=str(config_path))
    assert adapter._backend is None

    assert adapter.generate("hello") == "generated"
    assert adapter.meta == {"backend": "ollama", "profile": "primary"}
    assert len(FakeBackend.instances) == 1
    assert FakeBackend.instances[0].generate_calls == [("hello", {"temperature": 0.2})]


def test_llm_adapter_supports_llama_cpp_and_unknown_backend_failure(models_config_factory, monkeypatch):
    llama_path = models_config_factory(profiles={"primary": {"backend": "llama_cpp", "path": "/tmp/model.gguf"}})
    monkeypatch.setattr("importlib.import_module", lambda name: types.SimpleNamespace(OllamaBackend=FakeBackend, LlamaCppBackend=FakeBackend))
    adapter = LLMAdapter(models_config_path=str(llama_path))
    assert adapter.meta["backend"] == "llama_cpp"

    bad_path = models_config_factory(profiles={"primary": {"backend": "custom", "name": "x"}})
    bad_adapter = LLMAdapter(models_config_path=str(bad_path))
    with pytest.raises(ValueError, match="Unsupported backend"):
        bad_adapter.generate("hello")


def test_llm_adapter_close_is_safe_without_backend(models_config_factory):
    config_path = models_config_factory(profiles={"primary": {"backend": "ollama", "name": "llama3"}})
    adapter = LLMAdapter(models_config_path=str(config_path))

    adapter.close()

    assert adapter._backend is None
