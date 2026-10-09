"""Deadlock-safe subprocess execution.

The trap this module exists to avoid (learned the hard way on Windows and
POSIX): ``subprocess.run(capture_output=True, timeout=T)`` hangs *forever*
when the child spawns a grandchild that inherits the stdout/stderr pipe write
end. ``communicate()`` never sees EOF, and the timeout path only kills the
*direct* child — the grandchild keeps the pipe open, so even
``TimeoutExpired`` may never fire.

Fixes encoded here:

1. Output goes to a **temp file**, not an OS pipe. A lingering grandchild
   writing to an inherited file descriptor cannot block us from reading what
   was already flushed.
2. ``stdin=DEVNULL`` always. A CLI agent that blocks reading stdin (e.g.
   ``pi`` without EOF) otherwise hangs identically to a missing API key.
3. On timeout we kill the **whole process tree**, not just the direct child,
   and we refuse to kill our own process group.
4. On POSIX the child is started with ``start_new_session=True`` so ``killpg``
   targets the child's group, never the caller's (a past incident SIGKILLed
   the CI runner's own shell).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProcResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def _kill_tree(posix: bool, proc: subprocess.Popen) -> None:
    """Best-effort kill of *proc* and its descendants."""
    if posix:
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, OSError):
            return
        # Guard: never signal our own group — that would kill the caller.
        if pgid != os.getpgid(0):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass
        return
    # Windows: taskkill /T walks the tree.
    try:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
    except Exception:
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass


def run_to_file(
    argv: list[str],
    *,
    timeout_sec: float = 600.0,
    cwd: str | os.PathLike[str] | None = None,
    env: dict[str, str] | None = None,
    stdin_text: str | None = None,
) -> ProcResult:
    """Run *argv* with file-backed stdout/stderr; kill the tree on timeout.

    Never uses OS pipes for output, so a grandchild holding a write end cannot
    deadlock the call.
    """
    posix = os.name == "posix"
    out_fd, out_path = tempfile.mkstemp(prefix="bioforge-out-")
    err_fd, err_path = tempfile.mkstemp(prefix="bioforge-err-")
    # Popen needs *file objects* (a path string is not accepted on Windows),
    # and the child must be able to inherit the handles -> open in binary.
    out_fh = os.fdopen(out_fd, "wb")
    err_fh = os.fdopen(err_fd, "wb")

    popen_kw: dict = dict(
        cwd=cwd,
        env=env,
        stdout=out_fh,
        stderr=err_fh,
    )
    if stdin_text is None:
        popen_kw["stdin"] = subprocess.DEVNULL
    else:
        popen_kw["stdin"] = subprocess.PIPE
    if posix:
        popen_kw["start_new_session"] = True

    timed_out = False
    try:
        proc = subprocess.Popen(argv, **popen_kw)  # noqa: S603 (argv is caller-built)
        # Parent's copies of the write handles can go now; the child has its own.
        out_fh.close()
        err_fh.close()
        try:
            if stdin_text is not None:
                try:
                    proc.communicate(input=stdin_text.encode("utf-8"), timeout=timeout_sec)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _kill_tree(posix, proc)
                    proc.wait(timeout=30)
            else:
                try:
                    proc.wait(timeout=timeout_sec)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _kill_tree(posix, proc)
                    proc.wait(timeout=30)
        except Exception:
            # Last-resort cleanup so we never leak a process tree.
            _kill_tree(posix, proc)
            raise
        rc = proc.returncode if proc.returncode is not None else -1
    finally:
        for fh in (out_fh, err_fh):
            try:
                fh.close()
            except OSError:
                pass
        stdout = Path(out_path).read_text(encoding="utf-8", errors="replace")
        stderr = Path(err_path).read_text(encoding="utf-8", errors="replace")
        for p in (out_path, err_path):
            try:
                os.unlink(p)
            except OSError:
                pass

    return ProcResult(
        argv=argv,
        returncode=rc,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
    )
