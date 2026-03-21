from typing import Dict, List

from src.core.transcription.port import TranscriptionPort
from src.infrastructure.transcription.whisperx_runtime import transcribe_and_diarize


class WhisperXTranscriptionAdapter(TranscriptionPort):
    """Canonical adapter for the active WhisperX-backed transcription runtime."""

    def transcribe(self, audio_path: str) -> List[Dict]:
        return transcribe_and_diarize(audio_path)
