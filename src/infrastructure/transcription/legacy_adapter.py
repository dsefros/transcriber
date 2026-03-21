"""Compatibility shim for older transcription adapter imports.

The active runtime should use :class:`src.infrastructure.transcription.whisperx_adapter.WhisperXTranscriptionAdapter`.
This alias remains only so migration-era or manual callers importing the old
name keep resolving to the same canonical adapter implementation.
"""

from src.infrastructure.transcription.whisperx_adapter import WhisperXTranscriptionAdapter

LegacyTranscriptionAdapter = WhisperXTranscriptionAdapter

__all__ = ["LegacyTranscriptionAdapter"]
