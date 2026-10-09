"""Harbor task emission + verification (the compiler core).

Mirrors EnvForge's discipline: emit -> report every dropped semantic ->
verify through four gates. Self-contained (no import of the harbor package at
emit time); the real harbor model is used only as an *optional* cross-check in
gate 3, and its absence is reported as UNVERIFIED, never as a pass.
"""

from __future__ import annotations

from .emitter import emit_task
from .tomlx import dump_toml
from .verifier import verify_task_dir, VerificationOutcome

__all__ = ["emit_task", "dump_toml", "verify_task_dir", "VerificationOutcome"]
