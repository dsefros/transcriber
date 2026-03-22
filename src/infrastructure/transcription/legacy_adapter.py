"""Compatibility facade for legacy/manual import paths.

The active runtime uses ``src.infrastructure.transcription.whisperx_adapter``
directly and must not import the quarantined legacy tree. This module remains as a tiny
quarantine shim so migration-era callers can resolve the old adapter symbol
without keeping any duplicate runtime logic here.
"""

from src.infrastructure.transcription.whisperx_adapter import WhisperXTranscriptionAdapter

LegacyTranscriptionAdapter = WhisperXTranscriptionAdapter

__all__ = ["LegacyTranscriptionAdapter"]

# Intentionally do not re-export ``WhisperXTranscriptionAdapter`` via ``__all__``.
# New code should import the canonical adapter from
# ``src.infrastructure.transcription.whisperx_adapter`` directly.
