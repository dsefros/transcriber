from typing import List, Dict

from src.core.transcription.port import TranscriptionPort
from src.legacy.v1.pipeline.main import transcribe_and_diarize


class LegacyTranscriptionAdapter(TranscriptionPort):
    def transcribe(self, audio_path: str) -> List[Dict]:
        return transcribe_and_diarize(audio_path)
