"""Schema package: load the embedded strict schemas."""

from __future__ import annotations

from .validator import (
    SchemaValidator,
    ValidationResult,
    ValidationError,
    validate_against,
)

__all__ = [
    "SchemaValidator",
    "ValidationResult",
    "ValidationError",
    "validate_against",
    "default_validator",
]

_CACHE: SchemaValidator | None = None


def default_validator() -> SchemaValidator:
    global _CACHE
    if _CACHE is None:
        _CACHE = SchemaValidator.load_dir()
    return _CACHE
