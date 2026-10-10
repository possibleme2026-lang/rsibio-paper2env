"""Pipeline stages: the five roles from rsibio-paper2env paper section 3.2.

Import order matters: the registry primitives are defined *first*, then each
stage module is imported and self-registers via @register at import time.
"""

from __future__ import annotations

# Registry mapping stage name -> callable(ctx, record) -> StageResult.
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


# --- self-registering stage modules (must come after register is defined) ---
from . import curator      # noqa: E402,F401  (curate is corpus-level; orchestrated by runner)
from . import constructor  # noqa: E402,F401
from . import reproducer   # noqa: E402,F401
from . import reviewer     # noqa: E402,F401
from . import packager     # noqa: E402,F401
