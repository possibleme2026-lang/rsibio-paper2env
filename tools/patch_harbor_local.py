#!/usr/bin/env python
"""Apply the local harbor patches this project depends on, idempotently.

Why this exists
---------------
``.rlvenv`` is a disposable virtualenv: a reinstall (or a harbor upgrade) would
silently drop the patches below and the eval harness would fail with confusing
native errors. This script reapplies them so the fix is reproducible, not a
one-off edit nobody remembers.

Patches
-------
1. ``harbor/environments/factory.py`` — upstream eagerly imports every sandbox
   backend (docker, e2b, modal, gke, daytona, runloop, apple-container). On this
   Windows host an application-control policy blocks ``_pyqwest.pyd`` (pulled in
   by the ``e2b`` SDK), so the *import itself* raises::

       ImportError: DLL load failed while importing _pyqwest:
       应用程序控制策略已阻止此文件。

   That kills ``harbor run`` before it can build a Docker sandbox, even though
   E2B is never used. The patch makes backend resolution lazy so an
   unimportable cloud backend is reported as unavailable instead of fatal.

Usage
-----
    .rlvenv/Scripts/python.exe tools/patch_harbor_local.py [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FACTORY = REPO / ".rlvenv" / "Lib" / "site-packages" / "harbor" / "environments" / "factory.py"

MARKER = "NOTE (rsibio-paper2env local patch"

ORIGINAL_HEAD = """import importlib
import logging
from pathlib import Path

from harbor.environments.apple_container import AppleContainerSandbox
from harbor.environments.base import BaseSandbox
from harbor.environments.daytona import DaytonaSandbox
from harbor.environments.docker.docker import DockerSandbox
from harbor.environments.e2b import E2BSandbox
from harbor.environments.gke import GKESandbox
from harbor.environments.modal import ModalSandbox
from harbor.environments.runloop import RunloopSandbox
from harbor.models.environment_type import SandboxType
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.config import EnvironmentConfig as TrialEnvironmentConfig
from harbor.models.trial.paths import TrialPaths


class SandboxFactory:
    _SANDBOXES: list[type[BaseSandbox]] = [
        AppleContainerSandbox,
        DockerSandbox,
        DaytonaSandbox,
        E2BSandbox,
        GKESandbox,
        ModalSandbox,
        RunloopSandbox,
    ]
    _SANDBOX_MAP: dict[SandboxType, type[BaseSandbox]] = {
        sandbox.type(): sandbox for sandbox in _SANDBOXES
    }
"""

PATCHED_HEAD = '''import importlib
import logging
from pathlib import Path

from harbor.environments.base import BaseSandbox
from harbor.models.environment_type import SandboxType
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.config import EnvironmentConfig as TrialEnvironmentConfig
from harbor.models.trial.paths import TrialPaths

# NOTE (rsibio-paper2env local patch, 2026-10-12)
# Upstream imports every sandbox backend eagerly. On a Windows host with an
# application-control policy, importing the cloud SDKs raises a native
# ImportError/DLL-block *at import time*, so `harbor run` dies before it can
# build a Docker sandbox -- even though E2B/Modal/GKE are never used.
# We therefore resolve the backend modules lazily: a backend that cannot be
# imported on this machine is reported as unavailable instead of fatal, and
# Docker (the only backend this project uses) keeps working.
_BACKEND_MODULES: list[tuple[SandboxType, str, str]] = [
    (SandboxType.APPLE_CONTAINER, "harbor.environments.apple_container",
     "AppleContainerSandbox"),
    (SandboxType.DOCKER, "harbor.environments.docker.docker", "DockerSandbox"),
    (SandboxType.DAYTONA, "harbor.environments.daytona", "DaytonaSandbox"),
    (SandboxType.E2B, "harbor.environments.e2b", "E2BSandbox"),
    (SandboxType.GKE, "harbor.environments.gke", "GKESandbox"),
    (SandboxType.MODAL, "harbor.environments.modal", "ModalSandbox"),
    (SandboxType.RUNLOOP, "harbor.environments.runloop", "RunloopSandbox"),
]

_UNAVAILABLE: dict[SandboxType, str] = {}


def _load_backend(sandbox_type: SandboxType) -> type[BaseSandbox] | None:
    """Import one backend on demand, remembering failures."""
    if sandbox_type in _UNAVAILABLE:
        return None
    for stype, module_path, class_name in _BACKEND_MODULES:
        if stype is not sandbox_type:
            continue
        try:
            module = importlib.import_module(module_path)
        except Exception as exc:  # noqa: BLE001 - native DLL blocks, not just ImportError
            _UNAVAILABLE[sandbox_type] = f"{type(exc).__name__}: {exc}"
            return None
        return getattr(module, class_name)
    return None


class SandboxFactory:
    @classmethod
    def _sandbox_map(cls) -> dict[SandboxType, type[BaseSandbox]]:
        out: dict[SandboxType, type[BaseSandbox]] = {}
        for sandbox_type, _, _ in _BACKEND_MODULES:
            klass = _load_backend(sandbox_type)
            if klass is not None:
                out[sandbox_type] = klass
        return out
'''

PATCHES: list[tuple[str, str]] = [
    (ORIGINAL_HEAD, PATCHED_HEAD),
    (
        """        if type not in cls._SANDBOX_MAP:
            raise ValueError(
                f"Unsupported sandbox type: {type}. This could be because the "
                "sandbox is not registered in the SandboxFactory or because "
                "the sandbox type is invalid."
            )

        sandbox_class = cls._SANDBOX_MAP[type]""",
        """        sandbox_map = cls._sandbox_map()
        if type not in sandbox_map:
            reason = _UNAVAILABLE.get(type, "not registered")
            raise ValueError(
                f"Unsupported sandbox type: {type}. This could be because the "
                "sandbox is not registered in the SandboxFactory or because "
                f"the sandbox type is invalid. Details: {reason}"
            )

        sandbox_class = sandbox_map[type]""",
    ),
    (
        """        if type is None or type not in cls._SANDBOX_MAP:
            return

        cls._SANDBOX_MAP[type].preflight()""",
        """        if type is None:
            return

        sandbox_map = cls._sandbox_map()
        if type not in sandbox_map:
            return

        sandbox_map[type].preflight()""",
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report state without writing")
    a = ap.parse_args()

    if not FACTORY.is_file():
        print(f"ERROR: {FACTORY} not found", file=sys.stderr)
        return 2

    text = FACTORY.read_text(encoding="utf-8")
    if MARKER in text:
        print("already patched (marker present)")
        return 0
    if a.check:
        print("NOT patched")
        return 1

    applied = 0
    for old, new in PATCHES:
        if old not in text:
            print(f"WARN: anchor not found, skipping patch #{applied + 1}",
                  file=sys.stderr)
            continue
        text = text.replace(old, new, 1)
        applied += 1

    FACTORY.write_text(text, encoding="utf-8")
    print(f"patched {FACTORY} ({applied}/{len(PATCHES)} hunks applied)")
    return 0 if applied == len(PATCHES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
