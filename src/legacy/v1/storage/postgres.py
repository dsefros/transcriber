"""Legacy-only compatibility shim for the retired v1 Postgres storage path.

The active runtime does not import ``src.legacy``.
This module intentionally re-exports the canonical infrastructure storage models
and helpers so older/manual v1 flows keep working without maintaining a second
copy of the schema glue.
"""

from src.infrastructure.storage.postgres import (
    Base,
    Fragment,
    Meeting,
    Speaker,
    get_db_session,
    init_db,
)

__all__ = [
    "Base",
    "Meeting",
    "Speaker",
    "Fragment",
    "init_db",
    "get_db_session",
]
