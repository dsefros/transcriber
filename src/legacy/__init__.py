"""Quarantined legacy compatibility tree.

The active runtime must stay on the canonical path:
``src.app.cli -> src.worker -> JobRunner -> PipelineOrchestrator``.
Modules under ``src.legacy`` exist only to preserve a narrow set of older import
paths and manual migration workflows.
"""
