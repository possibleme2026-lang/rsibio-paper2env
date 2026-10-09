"""Pipeline stages: the five roles from BioForge-Harbor paper section 3.2."""

from __future__ import annotations

from . import curator  # noqa: F401

# Registry mapping stage name -> callable(state, ctx) -> StageResult.
# Populated incrementally as stages are added (M3: curate; M5: the rest).
STAGE_ORDER = ("curate", "construct", "reproduce", "review", "package")

_REGISTRY: dict[str, object] = {}


def register(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def get_stage(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"stage {name!r} not registered (have {sorted(_REGISTRY)})")
    return _REGISTRY[name]


def registered() -> list[str]:
    return sorted(_REGISTRY)
