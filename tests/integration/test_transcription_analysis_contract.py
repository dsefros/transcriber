import json
from pathlib import Path
from uuid import uuid4

import pytest

from src.core.pipeline.context import PipelineContext
from src.core.pipeline.steps.analysis import AnalysisStep
from src.core.pipeline.steps.transcription import TranscriptionStep
from src.core.transcription.contracts import (
    build_segments_artifact_path,
    load_transcription_segments,
    normalize_transcription_segments,
    validate_transcription_segments,
)
from src.prompts.registry import PromptRegistry

pytestmark = [pytest.mark.integration, pytest.mark.smoke]


class FakeTranscriptionService:
    def __init__(self, segments):
        self._segments = segments

    def transcribe(self, audio_path: str):
        return list(self._segments)


class FakeLLM:
    def __init__(self):
        self.meta = {'backend': 'fake', 'profile': 'test'}
        self.prompts = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return 'Summary output'


class FakeServices:
    def __init__(self, llm, transcription):
        self.llm = llm
        self.transcription = transcription


def test_validate_transcription_segments_accepts_active_shape():
    segments = [
        {'speaker': 'SPEAKER_00', 'text': ' Hello ', 'start': 0, 'end': 1.25},
        {'speaker': 'SPEAKER_01', 'text': 'World', 'start': 1.25, 'end': 2},
    ]

    validated = validate_transcription_segments(segments)

    assert len(validated) == 2
    assert validated[0]['text'] == 'Hello'
    assert validated[1]['end'] == 2.0


def test_normalize_transcription_segments_clamps_inverted_timing_and_warns():
    normalized, warnings = normalize_transcription_segments(
        [
            {
                'speaker': 'SPEAKER_00',
                'text': 'Closing note',
                'start': 2699.08,
                'end': 2698.43,
            }
        ]
    )

    assert normalized[0]['start'] == 2699.08
    assert normalized[0]['end'] == 2699.08
    assert len(warnings) == 1
    assert 'inverted timing' in warnings[0]


def test_validate_transcription_segments_rejects_missing_required_keys():
    with pytest.raises(ValueError, match='missing required keys: end'):
        validate_transcription_segments([{'speaker': 'SPEAKER_00', 'text': 'Hello', 'start': 0.0}])


def test_transcription_step_writes_canonical_artifact_and_warning_metadata(isolated_workspace):
    audio_path = isolated_workspace / 'sample.wav'
    audio_path.write_bytes(b'audio')

    ctx = PipelineContext(
        job_id=uuid4(),
        source_type='audio',
        source_path=audio_path,
        services=FakeServices(
            llm=FakeLLM(),
            transcription=FakeTranscriptionService(
                [
                    {
                        'speaker': 'SPEAKER_00',
                        'text': 'Hello',
                        'start': 2699.08,
                        'end': 2698.43,
                    }
                ]
            ),
        ),
    )

    result = TranscriptionStep().run(ctx)

    assert result.status == 'completed'
    assert result.artifacts['segment_count'] == 1
    assert result.artifacts['normalized_segment_count'] == 1
    assert len(result.artifacts['contract_warnings']) == 1
    expected_path = build_segments_artifact_path(ctx.job_id)
    assert result.artifacts['segments_path'] == str(expected_path)
    assert json.loads((isolated_workspace / expected_path).read_text(encoding='utf-8')) == [
        {
            'speaker': 'SPEAKER_00',
            'text': 'Hello',
            'start': 2699.08,
            'end': 2699.08,
        }
    ]


def test_analysis_step_consumes_transcription_artifact_path(isolated_workspace):
    segments_path = isolated_workspace / 'job_segments.json'
    segments_path.write_text(
        json.dumps(
            [
                {'speaker': 'SPEAKER_00', 'text': 'Hello', 'start': 0.0, 'end': 1.0},
                {'speaker': 'SPEAKER_01', 'text': 'World', 'start': 1.0, 'end': 2.0},
            ]
        ),
        encoding='utf-8',
    )
    audio_path = isolated_workspace / 'sample.wav'
    audio_path.write_bytes(b'audio')
    llm = FakeLLM()
    ctx = PipelineContext(
        job_id=uuid4(),
        source_type='audio',
        source_path=audio_path,
        services=FakeServices(llm=llm, transcription=FakeTranscriptionService([])),
        artifacts={'transcription': {'segments_path': str(segments_path), 'segment_count': 2}},
    )
    step = AnalysisStep()
    step.prompt_registry = PromptRegistry(base_dir=Path(__file__).resolve().parents[2] / 'src/prompts')

    result = step.run(ctx)

    assert result.status == 'completed'
    assert Path(result.artifacts['analysis_path']).exists()
    assert 'Hello\nWorld' in llm.prompts[0]


def test_transcription_to_analysis_boundary_smoke_tolerates_inverted_timing(isolated_workspace):
    audio_path = isolated_workspace / 'sample.wav'
    audio_path.write_bytes(b'audio')
    llm = FakeLLM()
    ctx = PipelineContext(
        job_id=uuid4(),
        source_type='audio',
        source_path=audio_path,
        services=FakeServices(
            llm=llm,
            transcription=FakeTranscriptionService(
                [
                    {'speaker': 'SPEAKER_00', 'text': 'Hello', 'start': 0.0, 'end': 1.0},
                    {'speaker': 'SPEAKER_01', 'text': 'there', 'start': 2699.08, 'end': 2698.43},
                ]
            ),
        ),
    )
    step = AnalysisStep()
    step.prompt_registry = PromptRegistry(base_dir=Path(__file__).resolve().parents[2] / 'src/prompts')

    transcription_result = TranscriptionStep().run(ctx)
    ctx.artifacts['transcription'] = transcription_result.artifacts
    analysis_result = step.run(ctx)

    assert transcription_result.status == 'completed'
    assert analysis_result.status == 'completed'
    analysis_path = isolated_workspace / analysis_result.artifacts['analysis_path']
    segments_path = isolated_workspace / transcription_result.artifacts['segments_path']

    assert json.loads(analysis_path.read_text(encoding='utf-8'))['segment_count'] == 2
    assert load_transcription_segments(segments_path) == [
        {'speaker': 'SPEAKER_00', 'text': 'Hello', 'start': 0.0, 'end': 1.0},
        {'speaker': 'SPEAKER_01', 'text': 'there', 'start': 2699.08, 'end': 2699.08},
    ]
    assert transcription_result.artifacts['normalized_segment_count'] == 1


def test_validate_transcription_segments_still_rejects_unusable_artifact():
    with pytest.raises(ValueError, match='speaker must be non-empty'):
        validate_transcription_segments(
            [{'speaker': ' ', 'text': 'Hello', 'start': 0.0, 'end': 1.0}]
        )
