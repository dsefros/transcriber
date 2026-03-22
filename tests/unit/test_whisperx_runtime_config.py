import types

import pytest

from src.infrastructure.transcription import whisperx_runtime

pytestmark = pytest.mark.unit


class _FakeAudioSegmentFile:
    def set_channels(self, channels: int):
        return self

    def set_frame_rate(self, frame_rate: int):
        return self

    def export(self, path: str, format: str):
        return None


class _FakeAudioSegment:
    @staticmethod
    def from_file(path: str):
        return _FakeAudioSegmentFile()


class _FakeModel:
    def transcribe(self, audio_data, language=None):
        return {"segments": [{"speaker": "SPEAKER_00", "text": " hello ", "start": 0.0, "end": 1.0}]}


class _FakeDiarizationPipeline:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, audio_data):
        return {"segments": []}


def test_get_transcription_settings_defaults(monkeypatch):
    for name in (
        'TRANSCRIPTION_MODEL_NAME',
        'TRANSCRIPTION_DEVICE',
        'ALIGNMENT_LANGUAGE_CODE',
        'ALIGNMENT_MODEL_NAME',
    ):
        monkeypatch.delenv(name, raising=False)

    settings = whisperx_runtime.get_transcription_settings()

    assert settings == {
        'model_name': 'large-v3',
        'device': 'cuda',
        'alignment_language_code': 'ru',
        'alignment_model_name': 'facebook/wav2vec2-base-960h',
    }


def test_get_transcription_settings_honors_env_overrides(monkeypatch):
    monkeypatch.setenv('TRANSCRIPTION_MODEL_NAME', 'small')
    monkeypatch.setenv('TRANSCRIPTION_DEVICE', 'cpu')
    monkeypatch.setenv('ALIGNMENT_LANGUAGE_CODE', 'en')
    monkeypatch.setenv('ALIGNMENT_MODEL_NAME', 'custom-aligner')

    settings = whisperx_runtime.get_transcription_settings()

    assert settings == {
        'model_name': 'small',
        'device': 'cpu',
        'alignment_language_code': 'en',
        'alignment_model_name': 'custom-aligner',
    }


def test_runtime_transcribe_uses_env_driven_defaults(monkeypatch, tmp_path):
    audio_path = tmp_path / 'audio.wav'
    audio_path.write_bytes(b'audio')
    captures = {}

    monkeypatch.setenv('TRANSCRIPTION_MODEL_NAME', 'small')
    monkeypatch.setenv('TRANSCRIPTION_DEVICE', 'cpu')
    monkeypatch.setenv('ALIGNMENT_LANGUAGE_CODE', 'en')
    monkeypatch.setenv('ALIGNMENT_MODEL_NAME', 'custom-aligner')
    monkeypatch.setattr(whisperx_runtime, 'log_memory', lambda **kwargs: None)
    monkeypatch.setattr(whisperx_runtime, 'debug_dump_segments', lambda segments, filename: None)
    monkeypatch.setattr(whisperx_runtime, 'debug_validate_segments', lambda segments, label: None)
    monkeypatch.setattr(whisperx_runtime, 'free_gpu_memory', lambda: None)
    monkeypatch.setattr(whisperx_runtime, '_load_audio_segment_class', lambda: _FakeAudioSegment)

    def fake_load_model(model_name, device, compute_type):
        captures['load_model'] = (model_name, device, compute_type)
        return _FakeModel()

    def fake_load_align_model(language_code, device, model_name):
        captures['align_model'] = (language_code, device, model_name)
        return 'aligner', {'meta': True}

    fake_whisperx = types.SimpleNamespace(
        load_model=fake_load_model,
        load_audio=lambda wav_path: 'audio-data',
        load_align_model=fake_load_align_model,
        align=lambda segments, model_a, metadata, audio_data, device: {'segments': segments},
        DiarizationPipeline=lambda **kwargs: _FakeDiarizationPipeline(**kwargs),
        assign_word_speakers=lambda diarized, aligned_result: aligned_result,
    )
    monkeypatch.setattr(whisperx_runtime, '_load_whisperx_module', lambda: fake_whisperx)

    result = whisperx_runtime.WhisperXTranscriptionRuntime(hf_token='token').transcribe(str(audio_path))

    assert captures['load_model'] == ('small', 'cpu', 'int8')
    assert captures['align_model'] == ('en', 'cpu', 'custom-aligner')
    assert result == [{'speaker': 'SPEAKER_00', 'text': 'hello', 'start': 0.0, 'end': 1.0}]
