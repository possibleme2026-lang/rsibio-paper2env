"""Embedded JSON Schemas + a minimal stdlib validator.

Why embed a schema at all when Harbor ships pydantic models? Because
``TaskConfig`` defaults to ``extra='ignore'``: a typo'd or unsupported key in a
generated ``task.toml`` is *silently dropped*, so the task still validates and
still runs — with the intended resource limit / timeout quietly absent. The
first line of defense here is a strict schema with ``additionalProperties:
false`` at every level, which turns that silent drop into a loud error. The
real Harbor model validation (when the package is importable) is a *second*,
cross-checking gate, never the only one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_DIR = Path(__file__).parent


@dataclass
class ValidationError:
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path or '<root>'}: {self.message}"


@dataclass
class ValidationResult:
    errors: list[ValidationError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self, limit: int = 20) -> str:
        if self.ok:
            return "OK"
        lines = [str(e) for e in self.errors[:limit]]
        if len(self.errors) > limit:
            lines.append(f"... (+{len(self.errors) - limit} more)")
        return "\n".join(lines)


_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "null": type(None),
}


def _type_ok(value: Any, t: str) -> bool:
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "boolean":
        return isinstance(value, bool)
    py = _TYPES.get(t)
    if py is None:
        return True
    return isinstance(value, py)


def _resolve_ref(root: dict, ref: str) -> dict:
    """Resolve a local ``#/definitions/...`` or ``#/$defs/...`` JSON pointer."""
    if not ref.startswith("#/"):
        return {}
    node: Any = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return {}
    return node if isinstance(node, dict) else {}


def _validate(value: Any, schema: dict, path: str, out: list[ValidationError], root: dict | None = None) -> None:
    if root is None:
        root = schema
    if schema is True or schema == {}:
        return
    if schema is False:
        out.append(ValidationError(path, "schema forbids this value"))
        return

    if "$ref" in schema and root is not None:
        target = _resolve_ref(root, schema["$ref"])
        if target:
            _validate(value, target, path, out, root)
            # $ref may be combined with sibling keywords; fall through to them.
            schema = {k: v for k, v in schema.items() if k != "$ref"}
            if schema == {}:
                return


    # anyOf / oneOf / allOf
    if "allOf" in schema:
        for sub in schema["allOf"]:
            _validate(value, sub, path, out, root)
    if "anyOf" in schema:
        branch_errors: list[list[ValidationError]] = []
        for sub in schema["anyOf"]:
            tmp: list[ValidationError] = []
            _validate(value, sub, path, tmp, root)
            if not tmp:
                branch_errors = []
                break
            branch_errors.append(tmp)
        if branch_errors:
            out.append(ValidationError(path, "matches none of anyOf branches"))
        return
    if "oneOf" in schema:
        matches = 0
        for sub in schema["oneOf"]:
            tmp: list[ValidationError] = []
            _validate(value, sub, path, tmp, root)
            if not tmp:
                matches += 1
        if matches != 1:
            out.append(ValidationError(path, f"oneOf matched {matches} branches (need 1)"))
        return

    if "const" in schema and value != schema["const"]:
        out.append(ValidationError(path, f"expected const {schema['const']!r}"))
        return
    if "enum" in schema and value not in schema["enum"]:
        out.append(ValidationError(path, f"{value!r} not in enum {schema['enum']}"))
        return

    types = schema.get("type")
    if types is not None:
        tl = types if isinstance(types, list) else [types]
        if not any(_type_ok(value, t) for t in tl):
            out.append(ValidationError(path, f"expected type {tl}, got {type(value).__name__}"))
            return

    if isinstance(value, dict):
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in value:
                out.append(ValidationError(path, f"missing required key {req!r}"))
        addl = schema.get("additionalProperties", True)
        for k, v in value.items():
            if k in props:
                _validate(v, props[k], f"{path}/{k}" if path else k, out, root)
            elif addl is False:
                out.append(ValidationError(path, f"unknown key {k!r} (additionalProperties:false)"))
            elif isinstance(addl, dict):
                _validate(v, addl, f"{path}/{k}" if path else k, out, root)
        if "minProperties" in schema and len(value) < schema["minProperties"]:
            out.append(ValidationError(path, f"needs >= {schema['minProperties']} properties"))

    if isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for i, it in enumerate(value):
                _validate(it, items, f"{path}[{i}]", out, root)
        if "minItems" in schema and len(value) < schema["minItems"]:
            out.append(ValidationError(path, f"needs >= {schema['minItems']} items"))

    if isinstance(value, str):
        if "pattern" in schema:
            import re
            if not re.search(schema["pattern"], value):
                out.append(ValidationError(path, f"does not match pattern {schema['pattern']!r}"))
        if "minLength" in schema and len(value) < schema["minLength"]:
            out.append(ValidationError(path, f"shorter than minLength {schema['minLength']}"))

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            out.append(ValidationError(path, f"{value} < minimum {schema['minimum']}"))
        if "maximum" in schema and value > schema["maximum"]:
            out.append(ValidationError(path, f"{value} > maximum {schema['maximum']}"))


class SchemaValidator:
    """Wraps a loaded schema set; validates named schemas by key."""

    def __init__(self, schemas: dict[str, dict]):
        self._schemas = schemas

    @classmethod
    def load_dir(cls, directory: Path | str | None = None) -> "SchemaValidator":
        d = Path(directory) if directory else SCHEMA_DIR
        schemas: dict[str, dict] = {}
        for f in sorted(d.glob("*.json")):
            data = json.loads(f.read_text(encoding="utf-8"))
            # Each file is either a single schema (has $id) or a map of schemas.
            if isinstance(data, dict) and "$id" in data:
                schemas[data["$id"]] = data
            elif isinstance(data, dict):
                schemas.update(data)
        return cls(schemas)

    def get(self, name: str) -> dict:
        if name not in self._schemas:
            raise KeyError(f"unknown schema {name!r}; have {sorted(self._schemas)}")
        return self._schemas[name]

    def validate(self, name: str, value: Any) -> ValidationResult:
        out: list[ValidationError] = []
        _validate(value, self.get(name), "", out)
        return ValidationResult(errors=out)


def validate_against(schema: dict, value: Any) -> ValidationResult:
    out: list[ValidationError] = []
    _validate(value, schema, "", out)
    return ValidationResult(errors=out)
