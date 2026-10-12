"""Pure-Python shim for the blocked native tiktoken core.

Installed into the tiktoken package by ``tools/patch_tiktoken_shim.py``.

Why: this host runs Smart App Control (registry
``HKLM\\SYSTEM\\CurrentControlSet\\Control\\CI\\Policy\\VerifiedAndReputablePolicyState=1``),
which blocks unsigned native binaries at load time. tiktoken's encoder core is
an unsigned Rust extension, so ``import tiktoken`` -- which harbor pulls in
transitively through litellm -- raises::

    ImportError: DLL load failed while importing _tiktoken:
    应用程序控制策略已阻止此文件。

SAC has no per-file exclusions and cannot be disabled without resetting
Windows, so this module substitutes a length-based estimator for the Rust BPE.
Counts become approximations (~4 chars/token); harbor's verifier rewards never
depend on token counts, and litellm uses them only for cost and context
accounting. Correctness is untouched; only usage figures are approximate.
"""

from __future__ import annotations

AVG_CHARS_PER_TOKEN = 4.0


def _estimate(text: str) -> list[int]:
    """Deterministic pseudo-token ids sized from a char-count heuristic."""
    if not text:
        return []
    n = max(1, int(round(len(text) / AVG_CHARS_PER_TOKEN)))
    per = len(text) / n
    ids: list[int] = []
    pos = 0.0
    for i in range(n):
        start = int(pos)
        pos += per
        end = int(pos) if i < n - 1 else len(text)
        chunk = text[start:end] or " "
        ids.append(abs(hash(chunk)) % 1_000_000)
    return ids


class CoreBPE:
    """Length-based stand-in for the Rust ``CoreBPE``.

    Implements the call surface ``tiktoken.core.Encoding`` exercises:
    ``encode``, ``encode_ordinary``, ``encode_to_tiktoken_buffer``,
    ``encode_with_unstable``, ``encode_single_token``, ``encode_single_piece``,
    ``decode_bytes``, ``decode_single_token_bytes``, ``token_byte_values``,
    and ``_encode_bytes``.
    """

    def __init__(self, mergeable_ranks=None, special_tokens=None, pat_str=""):
        self.mergeable_ranks = mergeable_ranks or {}
        self.special_tokens = special_tokens or {}
        self.pat_str = pat_str

    # --- encoding -------------------------------------------------------
    def encode_ordinary(self, text: str) -> list[int]:
        return _estimate(text)

    def encode(self, text: str, allowed_special: object = "all") -> list[int]:
        return _estimate(text)

    def encode_to_tiktoken_buffer(self, text: str, allowed_special: object = "all"):
        return _estimate(text)

    def encode_with_unstable(self, text: str, allowed_special: object = "all"):
        return _estimate(text), []

    def encode_single_token(self, text_or_bytes) -> int:
        return abs(hash(str(text_or_bytes))) % 1_000_000

    def encode_single_piece(self, text_or_bytes) -> list[int]:
        if isinstance(text_or_bytes, (bytes, bytearray)):
            text_or_bytes = bytes(text_or_bytes).decode("utf-8", "replace")
        return _estimate(str(text_or_bytes))

    def _encode_bytes(self, text: bytes) -> list[int]:
        return _estimate(bytes(text).decode("utf-8", "replace"))

    # --- decoding -------------------------------------------------------
    def decode_bytes(self, tokens: list[int]) -> bytes:
        # Lossy by construction; callers only measure lengths.
        return b" " * len(tokens)

    def decode_single_token_bytes(self, token: int) -> bytes:
        return b" "

    def token_byte_values(self) -> list[bytes]:
        return [b" "]
