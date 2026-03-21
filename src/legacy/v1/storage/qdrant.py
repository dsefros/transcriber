"""Legacy-only compatibility shim for the retired v1 Qdrant storage path.

The active runtime does not use this module directly.
It re-exports the canonical infrastructure helpers for compatibility with the
legacy v1 pipeline and any manual maintenance scripts that still import the old
module path.
"""

from src.infrastructure.storage.qdrant import (
    create_collections_if_not_exists,
    init_qdrant_client,
)

__all__ = [
    "init_qdrant_client",
    "create_collections_if_not_exists",
]
