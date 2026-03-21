from enum import Enum


class MemoryStage(str, Enum):
    # --- transcription / whisper ---
    BEFORE_WHISPER_LOAD = "before_whisper_load"
    AFTER_WHISPER_LOAD = "after_whisper_load"
    AFTER_WHISPER_INFERENCE = "after_whisper_inference"
    AFTER_WHISPER_CLEANUP = "after_whisper_cleanup"

    # --- analysis / llm ---
    BEFORE_LLM_LOAD = "before_llm_load"
    AFTER_LLM_LOAD = "after_llm_load"
    AFTER_LLM_INFERENCE = "after_llm_inference"
    AFTER_LLM_CLEANUP = "after_llm_cleanup"
