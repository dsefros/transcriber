from abc import ABC, abstractmethod
from typing import List, Dict


class TranscriptionPort(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> List[Dict]:
        """
        Выполняет транскрипцию аудио-файла.

        Возвращает список сегментов в согласованном формате.
        Core не знает и не должен знать, как именно они получены.
        """
        raise NotImplementedError
