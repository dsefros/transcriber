import copy
import gc
import importlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.infrastructure.logging.setup import log_memory
from src.infrastructure.logging.stages import MemoryStage


class TranscriptionDependencyError(RuntimeError):
    """Raised when the WhisperX runtime is executed without required packages."""


def _load_torch_module():
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        raise TranscriptionDependencyError(
            "WhisperX backend selected but torch is not installed. "
            "Install with: pip install -r requirements-ml.txt"
        ) from exc


def _load_whisperx_module():
    try:
        return importlib.import_module("whisperx")
    except ModuleNotFoundError as exc:
        raise TranscriptionDependencyError(
            "WhisperX backend selected but whisperx is not installed. "
            "Install with: pip install -r requirements-ml.txt"
        ) from exc


def _load_audio_segment_class():
    try:
        module = importlib.import_module("pydub")
    except ModuleNotFoundError as exc:
        raise TranscriptionDependencyError(
            "WhisperX backend selected but pydub is not installed. "
            "Install with: pip install -r requirements-ml.txt"
        ) from exc
    return module.AudioSegment


def _load_env_file_if_present(env_path: str = ".env") -> None:
    env_file = Path(env_path)
    if not env_file.exists():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        cleaned_value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), cleaned_value)


_load_env_file_if_present()

HF_TOKEN = os.getenv("HF_TOKEN")


class WhisperXTranscriptionRuntime:
    """Canonical home for the active WhisperX transcription runtime."""

    def __init__(self, hf_token: Optional[str] = None):
        self._hf_token = HF_TOKEN if hf_token is None else hf_token

    def transcribe(self, audio_path: str, device: str = "cuda") -> List[Dict[str, Any]]:
        whisperx = _load_whisperx_module()
        AudioSegment = _load_audio_segment_class()
        job_id = None
        wav_path = "temp_audio.wav"

        log_memory(
            stage=MemoryStage.BEFORE_WHISPER_LOAD,
            component="transcription",
            job_id=job_id,
        )

        print(f"[DEBUG] WhisperX load (device={device})")

        model = whisperx.load_model(
            "large-v3",
            device,
            compute_type="float16" if device == "cuda" else "int8",
        )

        log_memory(
            stage=MemoryStage.AFTER_WHISPER_LOAD,
            component="transcription",
            job_id=job_id,
        )

        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(wav_path, format="wav")

        audio_data = whisperx.load_audio(wav_path)

        raw_result = model.transcribe(audio_data, language=None)
        raw_segments = copy.deepcopy(raw_result["segments"])
        debug_dump_segments(raw_segments, "debug_raw_segments.json")
        debug_validate_segments(raw_segments, "RAW")

        model_a, metadata = whisperx.load_align_model(
            language_code="ru",
            device=device,
            model_name="facebook/wav2vec2-base-960h",
        )

        aligned_result = whisperx.align(
            raw_result["segments"],
            model_a,
            metadata,
            audio_data,
            device,
        )

        aligned_segments = copy.deepcopy(aligned_result["segments"])
        debug_dump_segments(aligned_segments, "debug_aligned_segments.json")
        debug_validate_segments(aligned_segments, "ALIGNED")

        del model_a

        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=self._hf_token,
            device=device,
        )

        diarized = diarize_model(audio_data)
        diarized_result = whisperx.assign_word_speakers(diarized, aligned_result)

        diarized_segments = copy.deepcopy(diarized_result["segments"])
        debug_dump_segments(diarized_segments, "debug_diarized_segments.json")
        debug_validate_segments(diarized_segments, "DIARIZED")

        segments = [
            {
                "speaker": seg.get("speaker", "SPEAKER_00"),
                "text": seg.get("text", "").strip(),
                "start": round(seg.get("start", 0), 2),
                "end": round(seg.get("end", 0), 2),
            }
            for seg in diarized_segments
        ]

        log_memory(
            stage=MemoryStage.AFTER_WHISPER_INFERENCE,
            component="transcription",
            job_id=job_id,
        )

        del model
        del diarize_model
        del audio_data
        del diarized
        del raw_result
        del aligned_result
        del diarized_result

        free_gpu_memory()

        if os.path.exists(wav_path):
            os.remove(wav_path)

        log_memory(
            stage=MemoryStage.AFTER_WHISPER_CLEANUP,
            component="transcription",
            job_id=job_id,
        )

        print(f"[DEBUG] WhisperX done: {len(segments)} segments")

        return segments


def transcribe_and_diarize(audio_path: str, device: str = "cuda") -> List[Dict[str, Any]]:
    return WhisperXTranscriptionRuntime().transcribe(audio_path=audio_path, device=device)


def free_gpu_memory() -> None:
    try:
        torch = _load_torch_module()
    except TranscriptionDependencyError:
        gc.collect()
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def debug_validate_segments(segments: List[Dict[str, Any]], label: str) -> None:
    print(f"\n=== VALIDATION: {label} ===")

    inversions = 0
    overlaps = 0
    unsorted = 0
    empty = 0

    for index, segment in enumerate(segments):
        start = segment.get("start", 0)
        end = segment.get("end", 0)
        text = segment.get("text", "")

        if not text:
            empty += 1

        if end < start:
            inversions += 1

        if index > 0:
            previous = segments[index - 1]
            if start < previous.get("start", 0):
                unsorted += 1
            if start < previous.get("end", 0):
                overlaps += 1

    print("count:", len(segments))
    print("empty_text:", empty)
    print("inversions:", inversions)
    print("unsorted:", unsorted)
    print("overlaps:", overlaps)


def debug_dump_segments(segments: List[Dict[str, Any]], filename: str) -> None:
    Path(filename).write_text(
        json.dumps(segments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
