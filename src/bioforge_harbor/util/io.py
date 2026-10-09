"""Filesystem helpers: LF-exact writes, atomic replace, JSON, digests.

Two non-negotiable invariants encoded here:

1. Every text file this package writes uses LF line endings explicitly
   (``newline="\\n"``). On Windows the default ``newline=None`` translates
   ``\\n`` -> ``\\r\\n``, which silently breaks shell scripts that run inside
   Linux containers (``$'\\r': command not found``).

2. State files are written via tmp + ``os.replace`` so a crash mid-write can
   never leave a half-written ``state.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_text_lf(path: Path | str, text: str) -> None:
    """Write *text* with LF endings and UTF-8, creating parents as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def atomic_write_text(path: Path | str, text: str) -> None:
    """Atomically replace *path* with *text* (LF endings, UTF-8)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".tmp-", suffix=p.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_json(path: Path | str, obj: Any, *, indent: int = 2) -> None:
    """Write JSON atomically with LF endings and a trailing newline."""
    text = json.dumps(obj, ensure_ascii=False, indent=indent, sort_keys=False) + "\n"
    atomic_write_text(path, text)


def read_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def short_digest(text: str, n: int = 12) -> str:
    """A short, human-checkable digest prefix (for provenance labels)."""
    return sha256_text(text)[:n]
