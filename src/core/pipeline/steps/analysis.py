import json
import logging
import re
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.contracts.analysis_result import AnalysisResult
from src.core.pipeline.steps.base import Step, StepResult
from src.prompts.registry import PromptRegistry


class AnalysisStep(Step):
    name = "analysis"

    PROMPT_PATH = "analysis/v1.yaml"
    MAX_TRANSCRIPT_CHARS = 12000
    MIN_TEXT_LEN = 2
    SHORT_SEGMENT_CHARS = 24
    SHORT_SEGMENT_DURATION = 1.6
    MERGE_GAP_SECONDS = 0.75
    MAX_MERGED_CHARS = 160

    def __init__(self):
        self.prompt_registry = PromptRegistry()
        self.logger = logging.getLogger("analysis")

    def run(self, ctx) -> StepResult:
        job_id = str(ctx.job_id)
        t0 = time.monotonic()

        transcription = ctx.artifacts.get("transcription")
        if not transcription:
            return StepResult(status="failed", error="Missing transcription artifacts")

        segments_path = transcription.get("segments_path")
        if not segments_path:
            return StepResult(
                status="failed",
                error="Missing segments_path in transcription artifacts",
            )

        segments_path = Path(segments_path)
        if not segments_path.exists():
            return StepResult(
                status="failed",
                error=f"Segments file not found: {segments_path}",
            )

        with open(segments_path, "r", encoding="utf-8") as f:
            segments = json.load(f)

        normalized_segments, normalization_stats = self._normalize_segments(segments)
        transcript = self._build_transcript(normalized_segments)
        if not transcript:
            return StepResult(status="failed", error="Empty transcription text")

        prompt = self.prompt_registry.render(
            self.PROMPT_PATH,
            transcript=transcript,
            segment_count=len(normalized_segments),
            dropped_segments=normalization_stats["dropped_segments"],
            merged_segments=normalization_stats["merged_segments"],
        )

        try:
            llm_response = ctx.services.llm.generate(prompt)
        except Exception as e:
            return StepResult(status="failed", error=f"LLM inference failed: {e}")

        meta = ctx.services.llm.meta
        parsed_output, parse_error = self._parse_analysis_response(llm_response)

        result = AnalysisResult(
            prompt_id="analysis.v1",
            generated_at=datetime.utcnow(),
            summary_raw=llm_response,
            segment_count=len(normalized_segments),
            model_backend=meta["backend"],
            model_profile=meta["profile"],
            summary=parsed_output.get("summary", []),
            decisions=parsed_output.get("decisions", []),
            action_items=parsed_output.get("action_items", []),
            uncertainties=parsed_output.get("uncertainties", []),
            transcript_preview=transcript[:1000],
            normalization=normalization_stats,
            parse_error=parse_error,
        )

        output_path = segments_path.with_name(
            segments_path.stem.replace("_segments", "_analysis") + ".json"
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2, default=str)

        total_ms = int((time.monotonic() - t0) * 1000)
        self.logger.info(
            "analysis_completed",
            extra={
                "extra": {
                    "event": "analysis_completed",
                    "job_id": job_id,
                    "analysis_path": str(output_path),
                    "total_ms": total_ms,
                    "normalized_segment_count": len(normalized_segments),
                    "dropped_segments": normalization_stats["dropped_segments"],
                    "merged_segments": normalization_stats["merged_segments"],
                    "parse_error": parse_error,
                }
            },
        )

        return StepResult(
            status="completed",
            artifacts={
                "analysis_path": str(output_path),
                "prompt_id": result.prompt_id,
                "segment_count": result.segment_count,
            },
        )

    def _normalize_segments(self, segments: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        cleaned: List[Dict[str, Any]] = []
        dropped_invalid_timing = 0
        dropped_empty_text = 0

        for index, segment in enumerate(segments):
            text = self._normalize_text(segment.get("text", ""))
            if not text or len(text) < self.MIN_TEXT_LEN:
                dropped_empty_text += 1
                continue

            start = self._safe_float(segment.get("start"))
            end = self._safe_float(segment.get("end"))
            if start is None or end is None or start < 0 or end <= start:
                dropped_invalid_timing += 1
                continue

            cleaned.append(
                {
                    "speaker": self._normalize_speaker(segment.get("speaker")),
                    "text": text,
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "_source_index": index,
                }
            )

        cleaned.sort(key=lambda seg: (seg["start"], seg["end"], seg["_source_index"]))

        merged: List[Dict[str, Any]] = []
        merged_segments = 0
        for segment in cleaned:
            if merged and self._can_merge_segments(merged[-1], segment):
                merged[-1]["text"] = self._join_text(merged[-1]["text"], segment["text"])
                merged[-1]["end"] = max(merged[-1]["end"], segment["end"])
                merged_segments += 1
                continue
            merged.append(segment.copy())

        for segment in merged:
            segment.pop("_source_index", None)

        stats = {
            "input_segments": len(segments),
            "output_segments": len(merged),
            "dropped_segments": dropped_invalid_timing + dropped_empty_text,
            "dropped_invalid_timing": dropped_invalid_timing,
            "dropped_empty_text": dropped_empty_text,
            "merged_segments": merged_segments,
        }
        return merged, stats

    def _build_transcript(self, segments: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        total_chars = 0
        for segment in segments:
            line = f"{segment['speaker']}: {segment['text']}"
            if total_chars + len(line) > self.MAX_TRANSCRIPT_CHARS:
                break
            lines.append(line)
            total_chars += len(line) + 1
        return "\n".join(lines).strip()

    def _parse_analysis_response(self, response: str) -> Tuple[Dict[str, Any], str | None]:
        try:
            payload = json.loads(response)
        except json.JSONDecodeError as exc:
            return self._fallback_payload(response), f"json_decode_error: {exc.msg}"

        normalized = {
            "summary": self._ensure_string_list(payload.get("summary")),
            "decisions": self._ensure_string_list(payload.get("decisions")),
            "uncertainties": self._ensure_string_list(payload.get("uncertainties")),
            "action_items": self._normalize_action_items(payload.get("action_items")),
        }
        return normalized, None

    def _fallback_payload(self, response: str) -> Dict[str, Any]:
        text = response.strip()
        return {
            "summary": [text] if text else [],
            "decisions": [],
            "action_items": [],
            "uncertainties": ["Модель вернула неструктурированный ответ; требуется ручная проверка."] if text else [],
        }

    def _normalize_action_items(self, items: Any) -> List[Dict[str, str]]:
        normalized: List[Dict[str, str]] = []
        if not isinstance(items, list):
            return normalized

        for item in items:
            if isinstance(item, dict):
                task = self._normalize_text(item.get("task", ""))
                owner = self._normalize_text(item.get("owner", "не указан")) or "не указан"
                deadline = self._normalize_text(item.get("deadline", "не указан")) or "не указан"
                if task:
                    normalized.append({"task": task, "owner": owner, "deadline": deadline})
            elif isinstance(item, str):
                task = self._normalize_text(item)
                if task:
                    normalized.append({"task": task, "owner": "не указан", "deadline": "не указан"})
        return normalized

    def _ensure_string_list(self, value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        normalized: List[str] = []
        for item in value:
            if isinstance(item, str):
                text = self._normalize_text(item)
                if text:
                    normalized.append(text)
        return normalized

    def _can_merge_segments(self, left: Dict[str, Any], right: Dict[str, Any]) -> bool:
        if left["speaker"] != right["speaker"]:
            return False
        gap = right["start"] - left["end"]
        if gap < 0 or gap > self.MERGE_GAP_SECONDS:
            return False
        left_duration = left["end"] - left["start"]
        right_duration = right["end"] - right["start"]
        if max(left_duration, right_duration) > self.SHORT_SEGMENT_DURATION:
            return False
        if max(len(left["text"]), len(right["text"])) > self.SHORT_SEGMENT_CHARS:
            return False
        return len(left["text"]) + len(right["text"]) <= self.MAX_MERGED_CHARS

    def _normalize_speaker(self, speaker: Any) -> str:
        speaker_text = self._normalize_text(str(speaker or "SPEAKER_UNKNOWN"))
        return speaker_text or "SPEAKER_UNKNOWN"

    def _normalize_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        return text.strip("-–— ")

    def _join_text(self, left: str, right: str) -> str:
        if not left:
            return right
        if not right:
            return left
        if left.endswith(('.', '!', '?', ':', ';', ',')):
            return f"{left} {right}"
        return f"{left} {right}"

    def _safe_float(self, value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
