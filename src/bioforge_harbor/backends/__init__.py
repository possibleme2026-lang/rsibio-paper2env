"""Agent backends: how a pipeline stage talks to a model.

Four implementations share one protocol so stages never know which is active:
- MockBackend    : replays canned responses (offline tests, deterministic CI).
- DryRunBackend  : synthesizes schema-valid placeholders (full-chain smoke, no model).
- RealPiBackend  : drives the local `pi` coding agent as a headless subprocess.
- TierflowBackend: OpenAI-compatible HTTP gateway (tierflow.cn/v1).
"""

from __future__ import annotations

from .base import AgentBackend, BackendError
from .mock_backend import MockBackend, DryRunBackend
from .pi_backend import RealPiBackend, PiUnavailable
from .tierflow_backend import TierflowBackend, TierflowUnavailable

__all__ = [
    "AgentBackend",
    "BackendError",
    "MockBackend",
    "DryRunBackend",
    "RealPiBackend",
    "PiUnavailable",
    "TierflowBackend",
    "TierflowUnavailable",
    "make_backend",
]


def make_backend(kind: str = "mock", **kw) -> AgentBackend:
    kind = (kind or "mock").lower()
    if kind == "mock":
        return MockBackend(kw.get("responses", {}), strict=kw.get("strict", True))
    if kind == "dryrun":
        return DryRunBackend()
    if kind in ("pi", "real", "pi-agent"):
        return RealPiBackend(
            provider=kw.get("provider"),
            model=kw.get("model"),
            thinking=kw.get("thinking"),
            timeout_sec=kw.get("timeout_sec", 240.0),
        )
    if kind in ("tierflow", "openai", "http"):
        return TierflowBackend(
            api_key=kw.get("api_key"),
            base_url=kw.get("base_url"),
            model=kw.get("model"),
            timeout_sec=kw.get("timeout_sec", 120.0),
        )
    raise ValueError(f"unknown backend kind {kind!r} (want mock|dryrun|pi|tierflow)")
