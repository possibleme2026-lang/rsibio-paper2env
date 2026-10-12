#!/usr/bin/env python
"""Install a pure-Python tiktoken shim over the blocked native extension.

Why this exists
---------------
harbor imports litellm, which eagerly imports tiktoken; tiktoken's encoder core
is a native Rust extension (``_tiktoken.cp313-win_amd64.pyd``). This machine
runs Smart App Control (registry
``HKLM:\\SYSTEM\\CurrentControlSet\\Control\\CI\\Policy\\VerifiedAndReputablePolicyState=1``),
which blocks unsigned native binaries at load time::

    ImportError: DLL load failed while importing _tiktoken:
    应用程序控制策略已阻止此文件。

SAC has no per-file exclusions and cannot be turned off without a Windows
reset, so the practical fix is to make ``import tiktoken`` succeed without
native code. litellm uses tiktoken only to *estimate* token counts (cost
prediction, context trimming) -- harbor's verifier rewards never depend on it.
A length-based estimator is therefore a faithful-enough drop-in for this
project: usage figures become approximations, correctness is untouched.

What this script does (idempotent)
----------------------------------
1. Moves ``tiktoken/_tiktoken*.pyd`` aside to ``*.pyd.blocked``.
2. Copies ``tools/_tiktoken_shim.py`` to ``tiktoken/_tiktoken.py``.
3. Verifies ``import tiktoken; tiktoken.get_encoding("o200k_base")`` works.

Run:  .rlvenv/Scripts/python.exe tools/patch_tiktoken_shim.py [--check]
"""

from __future__ import annotations

import argparse
import glob
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / ".rlvenv" / "Lib" / "site-packages" / "tiktoken"
SOURCE = Path(__file__).with_name("_tiktoken_shim.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report state without writing")
    a = ap.parse_args()

    if not PKG.is_dir():
        print(f"ERROR: {PKG} not found", file=sys.stderr)
        return 2
    if not SOURCE.is_file():
        print(f"ERROR: {SOURCE} not found", file=sys.stderr)
        return 2

    shim_path = PKG / "_tiktoken.py"
    blocked = sorted(glob.glob(str(PKG / "_tiktoken*.pyd")))
    sidelined = sorted(glob.glob(str(PKG / "_tiktoken*.pyd.blocked")))

    if shim_path.is_file() and sidelined and not blocked:
        print("shim already installed")
        return 0
    if a.check:
        print("shim NOT installed")
        return 1

    for pyd in blocked:
        target = pyd + ".blocked"
        Path(pyd).rename(target)
        print(f"sidelined {Path(pyd).name} -> {Path(target).name}")

    shutil.copyfile(SOURCE, shim_path)
    print(f"installed {shim_path}")

    # Self-verify in a fresh interpreter (this process already imported the
    # real tiktoken chain via the parent venv, so reload would be misleading).
    probe = (
        "import tiktoken;"
        "enc=tiktoken.get_encoding('o200k_base');"
        "print('tiktoken OK ->', len(enc.encode('hello world, token estimate')), 'tokens')"
    )
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    return 0 if r.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
