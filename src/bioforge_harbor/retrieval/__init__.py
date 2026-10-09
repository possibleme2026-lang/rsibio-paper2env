"""Literature retrieval: Europe PMC REST client with offline caching."""

from __future__ import annotations

from .europepmc import EuropePmcClient, RetrievalError
from .digests import bind_digest, verify_digest

__all__ = ["EuropePmcClient", "RetrievalError", "bind_digest", "verify_digest"]
