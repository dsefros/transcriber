from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, TypedDict


class TranscriptionSegment(TypedDict):
    """Stable segment shape consumed by the active pipeline boundary."""

    speaker: str
    text: str
    start: float
    end: float


class TranscriptionArtifact(TypedDict, total=False):
    """Stable transcription artifact written by ``TranscriptionStep``."""

    segments_path: str
    segment_count: int
    contract_warnings: List[str]
    normalized_segment_count: int


_REQUIRED_SEGMENT_KEYS = ("speaker", "text", "start", "end")


def normalize_transcription_segments(segments: Any) -> tuple[List[TranscriptionSegment], List[str]]:
    """Validate the active transcription segment contract.

    Hard failures:
    - artifact is not a non-empty list
    - a segment is not a dict
    - required keys are missing
    - text/speaker are empty after stripping
    - start/end are not numeric

    Soft compatibility handling:
    - if ``end < start`` for a segment produced by the active runtime, clamp
      ``end`` to ``start`` and emit a warning instead of failing the pipeline
    """

    if not isinstance(segments, list) or not segments:
        raise ValueError("Transcription segments must be a non-empty list")

    normalized: List[TranscriptionSegment] = []
    warnings: List[str] = []

    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"Segment {index} must be a dict")

        missing_keys = [key for key in _REQUIRED_SEGMENT_KEYS if key not in segment]
        if missing_keys:
            raise ValueError(
                f"Segment {index} missing required keys: {', '.join(missing_keys)}"
            )

        speaker = str(segment["speaker"]).strip()
        text = str(segment["text"]).strip()

        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Segment {index} start/end must be numeric") from exc

        if not speaker:
            raise ValueError(f"Segment {index} speaker must be non-empty")
        if not text:
            raise ValueError(f"Segment {index} text must be non-empty")

        if end < start:
            warnings.append(
                f"Segment {index} had inverted timing; normalized end from {end} to {start}"
            )
            end = start

        normalized.append(
            TranscriptionSegment(
                speaker=speaker,
                text=text,
                start=start,
                end=end,
            )
        )

    return normalized, warnings



def validate_transcription_segments(segments: Any) -> List[TranscriptionSegment]:
    """Return normalized segments while preserving the explicit contract API."""

    normalized_segments, _ = normalize_transcription_segments(segments)
    return normalized_segments



def build_segments_artifact_path(job_id: str | Any, output_dir: Path | str = "output") -> Path:
    """Canonical naming used by the active pipeline for transcription segments."""

    return Path(output_dir) / f"{job_id}_segments.json"



def write_transcription_segments(
    segments: Any,
    job_id: str | Any,
    output_dir: Path | str = "output",
) -> TranscriptionArtifact:
    """Validate and persist the canonical transcription artifact."""

    normalized_segments, warnings = normalize_transcription_segments(segments)
    segments_path = build_segments_artifact_path(job_id=job_id, output_dir=output_dir)
    segments_path.parent.mkdir(parents=True, exist_ok=True)
    segments_path.write_text(
        json.dumps(normalized_segments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    artifact: TranscriptionArtifact = TranscriptionArtifact(
        segments_path=str(segments_path),
        segment_count=len(normalized_segments),
    )
    if warnings:
        artifact["contract_warnings"] = warnings
        artifact["normalized_segment_count"] = len(warnings)

    return artifact



def load_transcription_segments(segments_path: Path | str) -> List[TranscriptionSegment]:
    """Load and validate the canonical transcription artifact for analysis."""

    path = Path(segments_path)
    if not path.exists():
        raise FileNotFoundError(f"Segments file not found: {path}")

    with path.open("r", encoding="utf-8") as file_obj:
        segments = json.load(file_obj)

    return validate_transcription_segments(segments)
