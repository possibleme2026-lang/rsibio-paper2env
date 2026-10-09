"""Deterministic TOML serializer (stdlib-only; tomllib only *reads*).

Scope is exactly what Harbor ``task.toml`` needs: scalars, arrays of scalars,
sub-tables, arrays of tables, and inline tables for short records. Key order is
insertion order (dicts are ordered), so output is byte-stable across runs —
which is what lets golden fixtures be diffed and digests be reproducible.
"""

from __future__ import annotations

import json
from typing import Any


def _scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        # Harbor uses floats for timeouts; repr keeps the decimal (120.0).
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=True)  # valid TOML basic string
    raise TypeError(f"unsupported scalar {type(v).__name__}")


def _is_table(v: Any) -> bool:
    return isinstance(v, dict) and len(v) > 0


def _is_aot(v: Any) -> bool:
    return isinstance(v, list) and len(v) > 0 and all(isinstance(x, dict) for x in v)


def _inline(v: Any) -> str:
    if isinstance(v, list):
        return "[" + ", ".join(_inline(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k} = {_inline(x)}" for k, x in v.items()) + "}"
    return _scalar(v)


def dump_toml(data: dict[str, Any]) -> str:
    lines: list[str] = []

    def emit_table(prefix: str, table: dict[str, Any]) -> None:
        # 1) scalars & simple arrays first (must precede sub-tables in TOML)
        for k, v in table.items():
            if _is_table(v) or _is_aot(v):
                continue
            key = k if _bare_ok(k) else json.dumps(k)
            lines.append(f"{key} = {_inline(v)}")
        # 2) sub-tables
        for k, v in table.items():
            if not _is_table(v):
                continue
            name = (prefix + "." if prefix else "") + (k if _bare_ok(k) else json.dumps(k))
            lines.append("")
            lines.append(f"[{name}]")
            emit_table(name, v)
        # 3) arrays of tables
        for k, v in table.items():
            if not _is_aot(v):
                continue
            name = (prefix + "." if prefix else "") + (k if _bare_ok(k) else json.dumps(k))
            for item in v:
                lines.append("")
                lines.append(f"[[{name}]]")
                emit_table(name, item)

    # root-level scalars before any table header
    root_scalars = {k: v for k, v in data.items() if not _is_table(v) and not _is_aot(v)}
    for k, v in root_scalars.items():
        key = k if _bare_ok(k) else json.dumps(k)
        lines.append(f"{key} = {_inline(v)}")

    # tables / AOTs at root
    for k, v in data.items():
        if _is_table(v):
            lines.append("")
            lines.append(f"[{k if _bare_ok(k) else json.dumps(k)}]")
            emit_table(k, v)
        elif _is_aot(v):
            for item in v:
                lines.append("")
                lines.append(f"[[{k if _bare_ok(k) else json.dumps(k)}]]")
                emit_table(k, item)

    text = "\n".join(lines).strip("\n") + "\n"
    # collapse the leading blank line if root had no scalars
    return text.lstrip("\n")


def _bare_ok(k: str) -> bool:
    return bool(k) and all(c.isalnum() or c in "_-" for c in k)
