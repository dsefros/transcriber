"""Legacy-only adapter alias kept for older/manual import paths.

The active runtime should use
:class:`src.infrastructure.transcription.whisperx_adapter.WhisperXTranscriptionAdapter`
directly. This module exists only as a quarantine shim so legacy callers keep
resolving to the canonical implementation without any adapter-specific logic
living here.
"""

from src.infrastructure.transcription.whisperx_adapter import (
    WhisperXTranscriptionAdapter,
)

# Backward-compatible name retained for migration-era imports.
LegacyTranscriptionAdapter = WhisperXTranscriptionAdapter

__all__ = ["LegacyTranscriptionAdapter", "WhisperXTranscriptionAdapter"]
