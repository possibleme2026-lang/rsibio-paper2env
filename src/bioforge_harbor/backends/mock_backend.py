"""Offline backends: MockBackend (replay) and DryRunBackend (synthesize).

These exist so the *entire* pipeline is runnable and testable with no model and
no network. That is not a convenience — it is the property that lets CI be
structurally green and lets a reviewer reproduce a run deterministically.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .base import BackendError
from .jsonx import extract_json


class MockBackend:
    """Replay a ``tag -> response`` table.

    ``strict=True`` (default) raises on an unknown tag. This is important: a
    permissive mock would let a stage silently call a tag nobody wired up and
    still "pass", which is exactly the kind of fake-green we refuse.
    """

    name = "mock"

    def __init__(self, responses: dict[str, Any] | None = None, *, strict: bool = True):
        self._responses: dict[str, Any] = dict(responses or {})
        self._strict = strict
        self.calls: list[str] = []

    @classmethod
    def from_dir(cls, directory: Path | str, *, strict: bool = True) -> "MockBackend":
        """Load ``<tag>.json`` / ``<tag>.txt`` files, tag = filename stem."""
        d = Path(directory)
        table: dict[str, Any] = {}
        if d.is_dir():
            for f in sorted(d.iterdir()):
                tag = f.stem
                if f.suffix == ".json":
                    table[tag] = extract_json(f.read_text(encoding="utf-8"))
                elif f.suffix in (".txt", ".md"):
                    table[tag] = f.read_text(encoding="utf-8")
        return cls(table, strict=strict)

    def set(self, tag: str, response: Any) -> None:
        self._responses[tag] = response

    def _lookup(self, tag: str) -> Any:
        self.calls.append(tag)
        if tag in self._responses:
            return self._responses[tag]
        if self._strict:
            raise BackendError(
                f"MockBackend has no response for tag {tag!r} "
                f"(known: {sorted(self._responses)})"
            )
        return {}

    def ask_text(self, tag: str, prompt: str, *, timeout_sec: float = 600.0) -> str:
        resp = self._lookup(tag)
        if isinstance(resp, str):
            return resp
        import json
        return json.dumps(resp, ensure_ascii=False)

    def ask_json(
        self,
        tag: str,
        prompt: str,
        *,
        schema_name: str | None = None,
        timeout_sec: float = 600.0,
    ) -> dict:
        resp = self._lookup(tag)
        if isinstance(resp, str):
            return extract_json(resp)
        if isinstance(resp, dict):
            return copy.deepcopy(resp)
        raise BackendError(f"MockBackend response for {tag!r} is not JSON-able")

    def close(self) -> None:
        pass


class DryRunBackend:
    """Synthesize a schema-valid example so a full chain runs with no model.

    ``ask_json`` builds a minimal object from the embedded artifact schema;
    ``ask_text`` returns a short placeholder. Nothing here calls a network.
    """

    name = "dryrun"

    def __init__(self):
        self.calls: list[str] = []

    def ask_text(self, tag: str, prompt: str, *, timeout_sec: float = 600.0) -> str:
        self.calls.append(tag)
        return f"[dryrun:{tag}] placeholder output"

    def close(self) -> None:
        pass

    def ask_json(
        self,
        tag: str,
        prompt: str,
        *,
        schema_name: str | None = None,
        timeout_sec: float = 600.0,
    ) -> dict:
        self.calls.append(tag)
        if not schema_name:
            return {}
        from bioforge_harbor.schemas import default_validator
        v = default_validator()
        try:
            schema = v.get(schema_name)
        except KeyError:
            return {}
        return _example(schema, schema)


_TYPE_EXAMPLE = {
    "string": "dryrun",
    "integer": 1,
    "number": 1.0,
    "boolean": True,
    "object": dict,
    "array": list,
}


def _example(schema: dict, root: dict, key_hint: str = "") -> Any:
    if "$ref" in schema:
        from bioforge_harbor.schemas.validator import _resolve_ref
        schema = _resolve_ref(root, schema["$ref"])
    if "const" in schema:
        return schema["const"]
    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]
    t = schema.get("type")
    if isinstance(t, list):
        t = t[0]
    if t == "object" or "properties" in schema:
        out: dict[str, Any] = {}
        for k, sub in schema.get("properties", {}).items():
            out[k] = _example(sub, root, key_hint=k)
        # A map-style object (only additionalProperties) still needs at least
        # one entry when minProperties demands it — invent a placeholder key.
        if not out and isinstance(schema.get("additionalProperties"), dict):
            sub = schema["additionalProperties"]
            need = schema.get("minProperties", 0)
            if need >= 1:
                if key_hint == "test_files":
                    out["tests/test.sh"] = "#!/bin/bash\nset -euo pipefail\n"
                else:
                    out["k0"] = _example(sub, root)
        return out
    if t == "array":
        items = schema.get("items")
        if isinstance(items, dict):
            return [_example(items, root)]
        return []
    if t == "string":
        if key_hint in ("name", "task_name") and "pattern" in schema:
            return "bioforge/dryrun-task"
        if key_hint == "dockerfile":
            return "FROM python:3.11-slim\nWORKDIR /app\n"
        return "dryrun"
    return _TYPE_EXAMPLE.get(t, "dryrun")
