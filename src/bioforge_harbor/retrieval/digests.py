"""Content digests: bind every retrieved artifact to its bytes.

Provenance without a digest is just a citation that can drift. We record the
SHA256 of the exact bytes we consumed so a later reproduction can prove it
read the *same* abstract / supplement, and so an HTML error page served with a
200 (a real Europe PMC / ChEMBL failure mode) can be caught and rejected.
"""

from __future__ import annotations

from ..util.io import sha256_bytes, sha256_text


def bind_digest(data: str | bytes) -> str:
    """Return the sha256 hex of *data* (text is UTF-8 encoded)."""
    if isinstance(data, str):
        return sha256_text(data)
    return sha256_bytes(data)


def verify_digest(data: str | bytes, expected: str) -> bool:
    return bind_digest(data) == expected
