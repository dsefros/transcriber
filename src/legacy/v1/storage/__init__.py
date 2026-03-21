"""Legacy-only storage compatibility facade.

Use ``src.infrastructure.storage`` for canonical runtime code.
This package re-exports the canonical storage symbols so older/manual imports can
resolve through a single explicit quarantine layer.
"""

from src.legacy.v1.storage.postgres import (
    Base,
    Fragment,
    Meeting,
    Speaker,
    get_db_session,
    init_db,
)
from src.legacy.v1.storage.qdrant import (
    create_collections_if_not_exists,
    init_qdrant_client,
)

__all__ = [
    "Base",
    "Meeting",
    "Speaker",
    "Fragment",
    "init_db",
    "get_db_session",
    "init_qdrant_client",
    "create_collections_if_not_exists",
]
