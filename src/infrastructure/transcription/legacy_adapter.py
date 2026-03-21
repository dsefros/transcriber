from typing import Dict, List

from src.core.transcription.port import TranscriptionPort
from src.infrastructure.transcription.whisperx_runtime import transcribe_and_diarize


class LegacyTranscriptionAdapter(TranscriptionPort):
    """Compatibility adapter kept while the worker still uses the legacy-named entrypoint."""

    def transcribe(self, audio_path: str) -> List[Dict]:
        return transcribe_and_diarize(audio_path)
