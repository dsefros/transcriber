from __future__ import annotations

import sys
import types

import pytest

from src.infrastructure.llm.backends.llama_cpp import LlamaCppBackend
from src.infrastructure.llm.backends.ollama import OllamaBackend
from src.infrastructure.transcription.whisperx_adapter import WhisperXTranscriptionAdapter

pytestmark = pytest.mark.unit


def test_ollama_backend_generate_and_meta(monkeypatch):
    class Client:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            return {"message": {"content": " hi "}}

    client = Client()
    monkeypatch.setattr("src.infrastructure.llm.backends.ollama.ollama.Client", lambda: client)
    backend = OllamaBackend(
        {
            "profile_name": "primary",
            "name": "llama3",
            "params": {"temperature": 0.5, "num_ctx": 8192},
        }
    )

    assert backend.generate("prompt") == "hi"
    assert client.calls[0]["stream"] is False
    assert client.calls[0]["options"]["num_ctx"] == 8192
    assert client.calls[0]["options"]["temperature"] == 0.5
    assert backend.meta["backend"] == "ollama"
    assert backend.meta["context_size"] == 8192


def test_llama_cpp_backend_requires_path_and_forwards_n_ctx(monkeypatch):
    created = {}

    class FakeLlama:
        def __init__(self, **kwargs):
            created.update(kwargs)

        def __call__(self, prompt, **kwargs):
            return {"choices": [{"text": " done "}]}

        def close(self):
            created["closed"] = True

    monkeypatch.setattr("src.infrastructure.llm.backends.llama_cpp.Llama", FakeLlama)

    with pytest.raises(ValueError, match="requires 'path'"):
        LlamaCppBackend({"profile_name": "broken"})

    backend = LlamaCppBackend({"profile_name": "primary", "path": "/tmp/model.gguf", "params": {"n_ctx": 8192}})
    assert created["n_ctx"] == 8192
    assert backend.generate("prompt") == "done"
    backend.close()
    assert created["closed"] is True


def test_whisperx_adapter_lazy_imports_runtime_and_propagates_errors(monkeypatch):
    runtime_module = types.SimpleNamespace(transcribe_and_diarize=lambda path: [{"speaker": "A", "text": path, "start": 0, "end": 1}])
    sys.modules.pop("src.infrastructure.transcription.whisperx_runtime", None)
    adapter = WhisperXTranscriptionAdapter()
    assert "src.infrastructure.transcription.whisperx_runtime" not in sys.modules
    monkeypatch.setitem(sys.modules, "src.infrastructure.transcription.whisperx_runtime", runtime_module)

    assert adapter.transcribe("audio.wav")[0]["text"] == "audio.wav"

    monkeypatch.setitem(sys.modules, "src.infrastructure.transcription.whisperx_runtime", types.SimpleNamespace(transcribe_and_diarize=lambda path: (_ for _ in ()).throw(RuntimeError("boom"))))
    with pytest.raises(RuntimeError, match="boom"):
        adapter.transcribe("audio.wav")
