from abc import ABC, abstractmethod
from typing import List

from src.core.transcription.contracts import TranscriptionSegment


class TranscriptionPort(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> List[TranscriptionSegment]:
        """
        Выполняет транскрипцию аудио-файла.

        Возвращает список сегментов в активном pipeline-контракте:
        [{"speaker": str, "text": str, "start": float, "end": float}, ...]
        Core не знает и не должен знать, как именно они получены.
        """
        raise NotImplementedError
