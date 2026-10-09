"""Tolerant JSON extraction from model output.

LLMs (and pi) frequently wrap JSON in markdown fences or add a sentence before
and after. This pulls the first balanced ``{...}`` or ``[...]`` out of arbitrary
text. It is deliberately a small hand-rolled scanner rather than ``json.loads``
on the whole blob, because trailing prose after the object is common.
"""

from __future__ import annotations

import json
from typing import Any


def extract_json(text: str) -> Any:
    if text is None:
        raise ValueError("empty output")
    # Strip a leading ```json fence if present.
    s = text.strip()
    if s.startswith("```"):
        # drop fence line
        nl = s.find("\n")
        if nl >= 0:
            s = s[nl + 1:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    start = min((i for i in (s.find("{"), s.find("[")) if i >= 0), default=-1)
    if start < 0:
        raise ValueError("no JSON object/array found in output")
    open_ch = s[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                blob = s[start:i + 1]
                return json.loads(blob)
    raise ValueError("unbalanced JSON in output")
