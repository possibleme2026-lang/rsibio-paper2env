"""Post-reboot Docker bootstrap for rsibio agentic RL.

Run this FIRST after machine reboot. It:
  1. Cleans stale socket files (handles are released by reboot).
  2. Launches Docker Desktop and waits for the engine.
  3. Runs a smoke container test.

Usage (Git Bash):
  /c/Python314/python.exe tools/post_reboot_bootstrap.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

LOCAL = Path.home() / "AppData" / "Local"
DOCKER_EXE = LOCAL / "Programs" / "DockerDesktop" / "Docker Desktop.exe"
RUN_DIR = LOCAL / "Docker" / "run"
SECRETS_DIR = LOCAL / "docker-secrets-engine"


def kill_docker():
    for exe in ("Docker Desktop.exe", "com.docker.backend.exe", "com.docker.build.exe"):
        subprocess.run(["taskkill", "/F", "/IM", exe], capture_output=True)
    time.sleep(3)


def clean_stale(protected: set[str]):
    """Move stale socket files aside into *.locked-<ts> sibling dirs."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    for d in (RUN_DIR, SECRETS_DIR):
        if not d.exists():
            continue
        blocking = [p for p in d.iterdir() if p.suffix == ".sock" or p.name.endswith(".stale")]
        # test renamability individually
        stuck = []
        for p in blocking:
            probe = p.with_name(p.name + ".probe")
            try:
                p.rename(probe)
                probe.rename(p)  # rename back; file is free
            except OSError:
                stuck.append(p)
        if stuck:
            locked = d.with_name(d.name + f".locked-{ts}")
            print(f"[clean] {len(stuck)} stuck files in {d.name} -> renaming dir to {locked.name}")
            try:
                d.rename(locked)
                d.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                print(f"[clean] FAILED to rename {d}: {e}")
                return False
        else:
            print(f"[clean] {d.name}: no stuck files")
    return True


def wait_engine(timeout_s: int = 300) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        r = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            print(f"[engine] ready, server={r.stdout.strip()}")
            return True
        time.sleep(10)
    return False


def smoke_container() -> bool:
    print("[smoke] docker run hello-world ...")
    r = subprocess.run(
        ["docker", "run", "--rm", "hello-world"],
        capture_output=True, text=True, timeout=300)
    ok = "Hello from Docker!" in (r.stdout + r.stderr)
    print("[smoke]", "OK" if ok else f"FAILED\n{r.stdout}\n{r.stderr}")
    return ok


def main():
    print("== post-reboot docker bootstrap ==")
    kill_docker()
    if not clean_stale(set()):
        print("FATAL: could not clear stuck sockets even after clean")
        sys.exit(1)
    print("[launch] Docker Desktop ...")
    subprocess.Popen([str(DOCKER_EXE)], close_fds=True)
    if not wait_engine():
        print("FATAL: engine not ready in 300s")
        sys.exit(2)
    if not smoke_container():
        sys.exit(3)
    print("== bootstrap OK ==")


if __name__ == "__main__":
    main()
