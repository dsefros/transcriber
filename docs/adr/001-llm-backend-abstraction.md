# ADR-001: LLM Backend Abstraction and Metadata Contract

## Status
Accepted

## Context

The project uses Large Language Models (LLMs) for meeting analysis.
Multiple LLM backends are required (local llama.cpp, Ollama, potentially others).

Without a strict abstraction, pipeline and core logic would become tightly
coupled to specific backends and configuration details.

## Decision

We introduce a strict LLM abstraction with the following principles:

1. **LLMAdapter is the single entry point** for inference.
2. **All LLM backends implement a unified contract**:
   - `generate(prompt) -> str`
   - `meta -> LLMMetadata`
3. **Pipeline and core do not know backend details**:
   - no access to model profiles
   - no backend-specific branching
4. **LLMMetadata is the only allowed way** to expose backend information.

The canonical metadata contract includes:
- backend type
- model identifier
- profile name
- context size
- supported capabilities (chat, system prompt)

## Consequences

### Positive
- Adding a new backend requires **no changes to core or pipeline**
- Enables future routing, fallback, and observability
- Prevents architectural leakage from infrastructure to core

### Negative
- Slightly more boilerplate for new backends
- Requires discipline when extending LLM functionality

## Non-goals

- Streaming inference (future work)
- Chat/completion unification (future work)
- Retry/fallback logic (future work)

## Guardrails

The following are considered **architecture violations**:
- Accessing model profiles from pipeline or core
- Backend-specific logic outside infrastructure layer
- Returning raw backend objects instead of `LLMMetadata`
