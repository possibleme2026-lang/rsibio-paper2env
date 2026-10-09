"""The backend protocol every stage depends on."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class BackendError(RuntimeError):
    """Raised when a backend cannot produce a usable answer."""


@runtime_checkable
class AgentBackend(Protocol):
    """Minimal contract: prompt in, text/JSON out, tagged for auditability.

    ``tag`` is a short stage label (e.g. ``"constructor.build"``). MockBackend
    keys its replay table on it; real backends log it. Keeping the tag in the
    signature means every stage call is traceable without extra plumbing.

    Note: only *methods* may appear in a runtime_checkable protocol, so the
    ``name`` attribute is an implementation convention (every backend defines
    it), not a protocol member.
    """

    def ask_text(self, tag: str, prompt: str, *, timeout_sec: float = 600.0) -> str:
        ...

    def ask_json(
        self,
        tag: str,
        prompt: str,
        *,
        schema_name: str | None = None,
        timeout_sec: float = 600.0,
    ) -> dict:
        ...

    def close(self) -> None:  # pragma: no cover - trivial
        ...
